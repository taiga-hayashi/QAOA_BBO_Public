"""mlfilter-scale closed-loop comparison: XY-FMQAOA vs mis-tuned penalty FMQA (exploratory).

  init    --layers L --truth-from <phase0 run> --phase1-run <phase1 explore run of the same L>
  run     --run-dir D --seed S --condition XY-FMQAOA|alpha0.3
  analyze --run-dir D        new runs + reused Phase 1 runs, first 100 evaluations, pre-fixed criterion
"""
import argparse
import hashlib
import json
import secrets
import sys
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_penalty as P1  # noqa: E402  (sets sys.path; provides metrics())
import mlfilter_fmqa as M  # noqa: E402
import mlfilter_fmqa_xy as MX  # noqa: E402

STUDY_DIR, PROJECT = P1.STUDY_DIR, P1.PROJECT
CONFIG = json.loads((STUDY_DIR/'bbo_xy.json').read_text())
CODE = [Path(__file__).resolve(), STUDY_DIR/'bbo_xy.json', PROJECT/'src/mlfilter_fmqa_xy.py', PROJECT/'src/qarp_pair_xy.py', *P1.CODE]


def code_hashes():
    return {str(p): P1.sha(p) for p in CODE}


def init(layers, truth_from, phase1_run):
    truth_from, phase1_run = truth_from.resolve(), phase1_run.resolve()
    enum = json.loads((truth_from/'raw/enumeration.json').read_text())
    assert enum['status']=='completed' and enum['validation']=='passed' and enum['layers']==layers
    p1meta = json.loads((phase1_run/'metadata.json').read_text()); assert p1meta['layers']==layers and p1meta['phase']=='phase1-penalty'
    assert p1meta['truth_from']['truth_table_sha256']==enum['truth_table_sha256']
    reused = {}
    for cid in CONFIG['reused_conditions']['conditions']:
        for seed in CONFIG['seeds']:
            kind = 'fixed100' if cid=='Random' or layers==6 else 'primary'
            if layers==6: kind = 'primary'
            f = phase1_run/f'raw/bbo/{cid}_{kind}_seed{seed}.json'; d = json.loads(f.read_text()); assert d['status']=='completed'
            reused[str(f.relative_to(PROJECT))] = P1.sha(f)
    canon = json.dumps({'config':CONFIG,'layers':layers,'truth':enum['truth_table_sha256']},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    config_hash = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+config_hash[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    run_dir = base/run_id; (run_dir/'raw/bbo').mkdir(parents=True); (run_dir/'logs').mkdir()
    P1.save(run_dir/'metadata.json', {'study':'mlfilter-scale','phase':'bbo-xy','protocol_phase':'exploratory','layers':layers,
            'config_hash':config_hash,'config_id':base.name,'run_id':run_id,'created_utc':P1.now(),'code_sha256':code_hashes(),
            'config_hash_method':'sha256 of canonical JSON {config: bbo_xy.json, layers, truth_table_sha256}',
            'truth_from':{'run_dir':str(truth_from.relative_to(PROJECT)),'truth_table_sha256':enum['truth_table_sha256']},
            'phase1_run':str(phase1_run.relative_to(PROJECT)),'reused_sha256':reused})
    print(run_dir)


def load(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    table = PROJECT/meta['truth_from']['run_dir']/'raw/truth_table.npz'; assert P1.sha(table)==meta['truth_from']['truth_table_sha256']
    return meta, np.load(table)['fom']


def run(run_dir, seed, cid):
    import torch
    torch.set_num_threads(1)
    meta, truth = load(run_dir); L = meta['layers']
    cond = next(c for c in CONFIG['new_conditions'] if c['id']==cid)
    out = run_dir/f'raw/bbo/{cid}_seed{seed}.json'; assert not out.exists(), 'Preserve prior runs; do not overwrite.'
    d = {'status':'running','started_utc':P1.now(),'seed':seed,'condition':cond,'budget':CONFIG['budget']}; P1.save(out, d)
    try:
        if cond['kind']=='xy':
            cfg = {k:cond[k] for k in ('depth','objective','first_evals','cycle_evals','local_maxfev')}
            r = MX.trajectory_xy(seed, L, truth, CONFIG['budget'], CONFIG['proposal_cycle_cap'], cfg)
        else:
            r = M.trajectory(seed, cond, L, truth, CONFIG['budget'], CONFIG['proposal_cycle_cap'])
        d.update(r, status='completed', ended_utc=P1.now()); P1.save(out, d)
        print(cid, seed, 'evals', r['actual_evaluations'], 'best', r['final_best'], f"{r['total_seconds']:.0f}s", flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc(), ended_utc=P1.now()); P1.save(out, d); raise


def analyze(run_dir):
    meta, truth = load(run_dir); L = meta['layers']; cap = CONFIG['budget']; rows = []; unfinished = []
    for f in sorted((run_dir/'raw/bbo').glob('*.json')):
        d = json.loads(f.read_text())
        if d['status']!='completed': unfinished.append(f.name); continue
        rows.append({'condition':d['condition']['id'],'seed':d['seed'],'source':'new',**P1.metrics(d, truth, cap)})
    for rel, digest in meta['reused_sha256'].items():
        assert P1.sha(PROJECT/rel)==digest, rel
        d = json.loads((PROJECT/rel).read_text()); rows.append({'condition':d['condition']['id'],'seed':d['seed'],'source':rel,**P1.metrics(d, truth, cap)})
    order = np.argsort(truth, kind='stable'); rank = np.empty(len(truth), int); rank[order] = np.arange(1, len(truth)+1)
    for r in rows:   # first evaluation reaching the top 10
        src = run_dir/f"raw/bbo/{r['condition']}_seed{r['seed']}.json" if r['source']=='new' else PROJECT/r['source']
        ids = json.loads(src.read_text())['final_ids'][:cap]; h = np.flatnonzero(rank[ids] <= 10); r['first_top10'] = int(h[0]+1) if len(h) else cap+1
    def q(v): v = np.asarray(v, float); return [float(np.median(v)), float(np.percentile(v,25)), float(np.percentile(v,75))]
    groups = {}
    for r in rows: groups.setdefault(r['condition'], []).append(r)
    summary = {c:{'n':len(v),'successes':sum(x['success'] for x in v),'best_rank':q([x['best_rank'] for x in v]),
                  'first_top10':q([x['first_top10'] for x in v]),'regret':q([x['regret'] for x in v]),
                  **({'feasible':q([x['feasible'] for x in v])} if all('feasible' in x for x in v) else {})} for c, v in groups.items()}
    verdict = None
    if all(c in summary and summary[c]['n']==len(CONFIG['seeds']) for c in ('XY-FMQAOA','alpha0.3','alpha100')):
        xy = summary['XY-FMQAOA']['best_rank'][0]
        verdict = {'xy_median_rank':xy,'alpha0.3':summary['alpha0.3']['best_rank'][0],'alpha100':summary['alpha100']['best_rank'][0],
                   'better_than_both':bool(xy < summary['alpha0.3']['best_rank'][0] and xy < summary['alpha100']['best_rank'][0]),
                   'applies_to_criterion':L==CONFIG['main_layer']}
    P1.save(run_dir/'raw/summary.json', {'layers':L,'budget':cap,'generated_utc':P1.now(),'summary':summary,'verdict':verdict,
                                         'criterion':CONFIG['criterion'],'unfinished':unfinished,'rows':rows})
    print(f'== L={L} budget {cap} (median [IQR] over seeds)')
    for c in ['XY-FMQAOA','alpha0.3','alpha1','alpha3','alpha10','alpha100','Adaptive','RandomFM','Random']:
        if c not in summary: continue
        v = summary[c]
        print(f"  {c:10s} n={v['n']} success {v['successes']}  best_rank {v['best_rank'][0]:.0f} [{v['best_rank'][1]:.0f},{v['best_rank'][2]:.0f}]"
              f"  first_top10 {v['first_top10'][0]:.0f}  regret {v['regret'][0]:.4f}" + (f"  feas {v['feasible'][0]:.2f}" if 'feasible' in v else ''))
    print('verdict', verdict)
    if unfinished: print('unfinished', unfinished)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init','run','analyze']); ap.add_argument('--layers', type=int)
    ap.add_argument('--truth-from', type=Path); ap.add_argument('--phase1-run', type=Path)
    ap.add_argument('--run-dir', type=Path); ap.add_argument('--seed', type=int); ap.add_argument('--condition')
    a = ap.parse_args()
    if a.command=='init': init(a.layers, a.truth_from, a.phase1_run)
    elif a.command=='run': run(a.run_dir.resolve(), a.seed, a.condition)
    else: analyze(a.run_dir.resolve())
