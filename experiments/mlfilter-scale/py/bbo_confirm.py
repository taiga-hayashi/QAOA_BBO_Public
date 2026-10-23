"""mlfilter-scale confirmation (15 new seeds, L=7, budget 100): XY-FMQAOA vs mis-tuned penalty FMQA.

  init    --truth-from <phase0 L=7 run>
  run     --run-dir D --seed S --condition ID
  analyze --run-dir D        pre-registered one-sided paired Wilcoxon tests (Bonferroni x2)
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
import phase1_penalty as P1  # noqa: E402  (sets sys.path; metrics())
import mlfilter_fmqa as M  # noqa: E402
import mlfilter_fmqa_xy as MX  # noqa: E402

STUDY_DIR, PROJECT = P1.STUDY_DIR, P1.PROJECT
CONFIG = json.loads((STUDY_DIR/'bbo_confirm.json').read_text())
CODE = [Path(__file__).resolve(), STUDY_DIR/'bbo_confirm.json', PROJECT/'src/mlfilter_fmqa_xy.py', PROJECT/'src/qarp_pair_xy.py', *P1.CODE]


def code_hashes():
    return {str(p): P1.sha(p) for p in CODE}


def init(truth_from):
    truth_from = truth_from.resolve(); enum = json.loads((truth_from/'raw/enumeration.json').read_text())
    assert enum['status']=='completed' and enum['validation']=='passed' and enum['layers']==CONFIG['layers']
    canon = json.dumps({'config':CONFIG,'truth':enum['truth_table_sha256']},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    config_hash = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+config_hash[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    run_dir = base/run_id; (run_dir/'raw/bbo').mkdir(parents=True); (run_dir/'logs').mkdir()
    P1.save(run_dir/'metadata.json', {'study':'mlfilter-scale','phase':'bbo-confirm','protocol_phase':'confirmatory','layers':CONFIG['layers'],
            'config_hash':config_hash,'config_id':base.name,'run_id':run_id,'created_utc':P1.now(),'code_sha256':code_hashes(),
            'truth_from':{'run_dir':str(truth_from.relative_to(PROJECT)),'truth_table_sha256':enum['truth_table_sha256']}})
    print(run_dir)


def load(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    table = PROJECT/meta['truth_from']['run_dir']/'raw/truth_table.npz'; assert P1.sha(table)==meta['truth_from']['truth_table_sha256']
    return meta, np.load(table)['fom']


def run(run_dir, seed, cid):
    import torch
    torch.set_num_threads(1)
    meta, truth = load(run_dir); L = CONFIG['layers']; B = CONFIG['budget']; C = CONFIG['proposal_cycle_cap']
    cond = next(c for c in CONFIG['conditions'] if c['id']==cid)
    out = run_dir/f'raw/bbo/{cid}_seed{seed}.json'; assert not out.exists(), 'Preserve prior runs; do not overwrite.'
    d = {'status':'running','started_utc':P1.now(),'seed':seed,'condition':cond,'budget':B}; P1.save(out, d)
    try:
        if cond['kind']=='xy':
            r = MX.trajectory_xy(seed, L, truth, B, C, {k:cond[k] for k in ('depth','objective','first_evals','cycle_evals','local_maxfev')})
        elif cond['kind']=='random':
            r = M.random_run(seed, L, truth, B)
        else:
            r = M.trajectory(seed, cond, L, truth, B, C)
        d.update(r, status='completed', ended_utc=P1.now()); P1.save(out, d)
        print(cid, seed, 'best', r['final_best'], flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc(), ended_utc=P1.now()); P1.save(out, d); raise


def analyze(run_dir):
    from scipy.stats import wilcoxon
    meta, truth = load(run_dir); B = CONFIG['budget']; seeds = CONFIG['seeds']
    order = np.argsort(truth, kind='stable'); rank = np.empty(len(truth), int); rank[order] = np.arange(1, len(truth)+1)
    table, unfinished = {}, []
    for c in CONFIG['conditions']:
        for s in seeds:
            f = run_dir/f"raw/bbo/{c['id']}_seed{s}.json"
            if not f.exists(): unfinished.append(f.name); continue
            d = json.loads(f.read_text())
            if d['status']!='completed': unfinished.append(f.name); continue
            m = P1.metrics(d, truth, B); h = np.flatnonzero(rank[d['final_ids'][:B]] <= 10); m['first_top10'] = int(h[0]+1) if len(h) else B+1
            table[(c['id'], s)] = m
    def q(v): v = np.asarray(v, float); return [float(np.median(v)), float(np.percentile(v,25)), float(np.percentile(v,75))]
    summary = {}
    for c in CONFIG['conditions']:
        rs = [table[(c['id'], s)] for s in seeds if (c['id'], s) in table]
        if rs: summary[c['id']] = {'n':len(rs),'successes':sum(r['success'] for r in rs),'best_rank':q([r['best_rank'] for r in rs]),
                                   'first_top10':q([r['first_top10'] for r in rs]),'regret':q([r['regret'] for r in rs])}
    def test(a, b, alternative):
        pairs = [(table[(a,s)]['best_rank'], table[(b,s)]['best_rank']) for s in seeds if (a,s) in table and (b,s) in table]
        diff = [x-y for x, y in pairs]
        p = 1.0 if all(v==0 for v in diff) else float(wilcoxon(diff, zero_method='wilcox', alternative=alternative).pvalue)
        return {'n_pairs':len(pairs),'xy_better':sum(v<0 for v in diff),'ties':sum(v==0 for v in diff),'xy_worse':sum(v>0 for v in diff),
                'median_difference':float(np.median(diff)) if diff else None,'p_raw':p}
    tests = {}
    if not unfinished:
        for h, other in (('H1','alpha0.3'),('H2','alpha100')):
            t = test('XY-FMQAOA', other, 'less'); t['p_bonferroni'] = min(1., 2*t['p_raw']); t['supported'] = t['p_bonferroni'] < 0.05; tests[h] = t
        tests['secondary_vs_alpha1'] = test('XY-FMQAOA', 'alpha1', 'two-sided')
        tests['conclusion_supported'] = tests['H1']['supported'] and tests['H2']['supported']
    P1.save(run_dir/'raw/summary.json', {'generated_utc':P1.now(),'summary':summary,'tests':tests,'hypotheses':CONFIG['hypotheses'],
             'unfinished':unfinished,'per_seed':{f'{k[0]}|{k[1]}':v for k, v in table.items()}})
    print('== L=7 budget 100, 15 seeds (median [IQR])')
    for c, v in summary.items():
        print(f"  {c:10s} n={v['n']} success {v['successes']}  best_rank {v['best_rank'][0]:.0f} [{v['best_rank'][1]:.0f},{v['best_rank'][2]:.0f}]  first_top10 {v['first_top10'][0]:.0f}")
    print(json.dumps(tests, indent=1))
    if unfinished: print('unfinished', len(unfinished))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init','run','analyze']); ap.add_argument('--truth-from', type=Path)
    ap.add_argument('--run-dir', type=Path); ap.add_argument('--seed', type=int); ap.add_argument('--condition')
    a = ap.parse_args()
    if a.command=='init': init(a.truth_from)
    elif a.command=='run': run(a.run_dir.resolve(), a.seed, a.condition)
    else: analyze(a.run_dir.resolve())
