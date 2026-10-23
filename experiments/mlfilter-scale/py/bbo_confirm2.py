"""Second confirmation (30 new seeds, L=7, budget 100): proposal-distribution shape (see bbo_confirm2.json).

  init    --truth-from <phase0 L=7 run>
  run     --run-dir D --seed S --condition ID
  analyze --run-dir D        pre-registered one-sided paired Wilcoxon tests (Bonferroni x3) + secondary
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
import phase1_penalty as P1  # noqa: E402
import mlfilter_fmqa as M  # noqa: E402
import mlfilter_fmqa_xy as MX  # noqa: E402
import mlfilter_fmqa_ext as E  # noqa: E402
import mlfilter_classical as C  # noqa: E402

STUDY_DIR, PROJECT = P1.STUDY_DIR, P1.PROJECT
CONFIG = json.loads((STUDY_DIR/'bbo_confirm2.json').read_text())
CODE = [Path(__file__).resolve(), STUDY_DIR/'bbo_confirm2.json', *[PROJECT/'src'/f for f in
        ('mlfilter_fmqa_xy.py','mlfilter_fmqa_ext.py','mlfilter_classical.py','qarp_pair_xy.py')], *P1.CODE]


def code_hashes():
    return {str(p): P1.sha(p) for p in dict.fromkeys(CODE)}


def init(truth_from):
    truth_from = truth_from.resolve(); enum = json.loads((truth_from/'raw/enumeration.json').read_text())
    assert enum['status']=='completed' and enum['validation']=='passed' and enum['layers']==CONFIG['layers']
    canon = json.dumps({'config':CONFIG,'truth':enum['truth_table_sha256']},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    h = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+h[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    d = base/run_id; (d/'raw/bbo').mkdir(parents=True); (d/'logs').mkdir()
    P1.save(d/'metadata.json', {'study':'mlfilter-scale','phase':'bbo-confirm2','protocol_phase':'confirmatory','layers':CONFIG['layers'],'config_hash':h,
            'config_id':base.name,'run_id':run_id,'created_utc':P1.now(),'code_sha256':code_hashes(),
            'truth_from':{'run_dir':str(truth_from.relative_to(PROJECT)),'truth_table_sha256':enum['truth_table_sha256']}})
    print(d)


def load(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    table = PROJECT/meta['truth_from']['run_dir']/'raw/truth_table.npz'; assert P1.sha(table)==meta['truth_from']['truth_table_sha256']
    return meta, np.load(table)['fom']


def run(run_dir, seed, cid):
    import torch
    torch.set_num_threads(1)
    meta, truth = load(run_dir); L, B, Cc = CONFIG['layers'], CONFIG['budget'], CONFIG['proposal_cycle_cap']
    cond = next(c for c in CONFIG['conditions'] if c['id']==cid); x_all = M.onehot(M.labels_for(L))
    out = run_dir/f'raw/bbo/{cid}_seed{seed}.json'; assert not out.exists()
    d = {'status':'running','seed':seed,'condition':cond,'budget':B}; P1.save(out, d)
    try:
        k = cond['kind']
        if k=='xy': r = MX.trajectory_xy(seed, L, truth, B, Cc, {q:cond[q] for q in ('depth','objective','first_evals','cycle_evals','local_maxfev')}, x_all=x_all)
        elif k in ('sa','randomfm'): r = M.trajectory(seed, cond, L, truth, B, Cc, x_all=x_all)
        else:
            def sampler(qd, b, s, layers, sd):
                pred = M.predict_all(x_all, M.matrix(qd, 4*layers), b)
                return C.boltzmann(pred, layers, sd, cond['tau']) if k=='boltzmann' else C.categorical_sa(pred, layers, sd)
            r = E.trajectory_custom(seed, L, truth, B, Cc, sampler, cid, x_all=x_all)
        d.update(r, condition=cond, status='completed', ended_utc=P1.now()); P1.save(out, d); print(cid, seed, 'best', r['final_best'], flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc(), ended_utc=P1.now()); P1.save(out, d); raise


def analyze(run_dir):
    from scipy.stats import wilcoxon, binomtest
    meta, truth = load(run_dir); B = CONFIG['budget']; seeds = CONFIG['seeds']
    order = np.argsort(truth, kind='stable'); rank = np.empty(len(truth), int); rank[order] = np.arange(1, len(truth)+1)
    table, unfinished = {}, []
    for c in CONFIG['conditions']:
        for s in seeds:
            f = run_dir/f"raw/bbo/{c['id']}_seed{s}.json"
            d = json.loads(f.read_text()) if f.exists() else {'status':'missing'}
            if d['status']!='completed': unfinished.append(f.name); continue
            m = P1.metrics(d, truth, B); h = np.flatnonzero(rank[d['final_ids'][:B]] <= 10); m['first_top10'] = int(h[0]+1) if len(h) else B+1
            table[(c['id'], s)] = m
    q = lambda v: [float(np.median(v)), float(np.percentile(v,25)), float(np.percentile(v,75))]
    summary = {}
    for c in CONFIG['conditions']:
        rs = [table[(c['id'], s)] for s in seeds if (c['id'], s) in table]
        if rs: summary[c['id']] = {'n':len(rs),'successes':sum(r['success'] for r in rs),'best_rank':q([r['best_rank'] for r in rs]),
                                   'first_top10':q([r['first_top10'] for r in rs])}
    def test(a, b, alt):
        pr = [(table[(a,s)], table[(b,s)]) for s in seeds if (a,s) in table and (b,s) in table]
        diff = [x['best_rank']-y['best_rank'] for x, y in pr]
        p = 1.0 if all(v==0 for v in diff) else float(wilcoxon(diff, zero_method='wilcox', alternative=alt).pvalue)
        b10 = sum(x['success'] and not y['success'] for x, y in pr); b01 = sum(y['success'] and not x['success'] for x, y in pr)
        mc = float(binomtest(b10, b10+b01, 0.5).pvalue) if b10+b01 else 1.0
        return {'n_pairs':len(pr),'first_better':sum(v<0 for v in diff),'ties':sum(v==0 for v in diff),'first_worse':sum(v>0 for v in diff),
                'median_difference':float(np.median(diff)) if diff else None,'p_raw':p,'mcnemar_exact_p':mc,'success_only_first':b10,'success_only_second':b01}
    tests = {}
    if not unfinished:
        for h, a, b in (('H1','XY-FMQAOA','CatSA'),('H2','Boltzmann-t0.2','CatSA'),('H3','XY-FMQAOA','alpha100')):
            t = test(a, b, 'less'); t.update(pair=f'{a} vs {b}', p_bonferroni=min(1., 3*t['p_raw'])); t['supported'] = t['p_bonferroni'] < 0.05; tests[h] = t
        for name, a, b in (('XY_vs_Boltzmann','XY-FMQAOA','Boltzmann-t0.2'),('XY_vs_alpha1','XY-FMQAOA','alpha1')):
            tests[name] = {**test(a, b, 'two-sided'), 'pair':f'{a} vs {b}'}
    P1.save(run_dir/'raw/summary.json', {'generated_utc':P1.now(),'summary':summary,'tests':tests,'hypotheses':CONFIG['hypotheses'],
                                         'unfinished':unfinished,'per_seed':{f'{k[0]}|{k[1]}':v for k, v in table.items()}})
    for c, v in summary.items():
        print(f"  {c:15s} n={v['n']} optimum {v['successes']:2d}  best_rank {v['best_rank'][0]:.0f} [{v['best_rank'][1]:.0f},{v['best_rank'][2]:.0f}]  top10 {v['first_top10'][0]:.0f}")
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
