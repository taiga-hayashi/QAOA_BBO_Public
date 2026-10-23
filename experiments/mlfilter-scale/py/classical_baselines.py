"""Classical penalty-free baselines (CatSA, Boltzmann) vs XY-FMQAOA (exploratory; see classical_baselines.json).

  init     --confirm-run C --solver-runs S6 S7 S8
  fixedfm  --run-dir D --layers L --seed S
  loop     --run-dir D --seed S --condition CatSA|Boltzmann-t<tau>
  analyze  --run-dir D
"""
import argparse
import hashlib
import json
import secrets
import sys
import time
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_fixedfm as SF  # noqa: E402
import phase1_penalty as P1  # noqa: E402
import mlfilter_fmqa as M  # noqa: E402
import mlfilter_fmqa_ext as E  # noqa: E402
import mlfilter_classical as C  # noqa: E402

STUDY_DIR, PROJECT = SF.STUDY_DIR, SF.PROJECT
CONFIG = json.loads((STUDY_DIR/'classical_baselines.json').read_text())
CODE = [Path(__file__).resolve(), STUDY_DIR/'classical_baselines.json', PROJECT/'src/mlfilter_classical.py', PROJECT/'src/mlfilter_fmqa_ext.py', *SF.CODE, *P1.CODE]
CONDS = ['CatSA'] + [f'Boltzmann-t{t:g}' for t in CONFIG['taus']]


def code_hashes():
    return {str(p): SF.sha(p) for p in dict.fromkeys(CODE)}


def init(confirm_run, solver_runs):
    rel = lambda p: str(p.resolve().relative_to(PROJECT))
    cm = json.loads((confirm_run/'metadata.json').read_text()); assert cm['phase']=='bbo-confirm'
    links = {'confirm_run':rel(confirm_run),'truth_7':cm['truth_from'],'solver_runs':{}}
    for r in solver_runs:
        m = json.loads((r/'metadata.json').read_text()); assert m['phase']=='solver-fixedfm'; links['solver_runs'][str(m['layers'])] = rel(r)
    canon = json.dumps({'config':CONFIG,'links':links},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    h = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+h[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    d = base/run_id
    for sub in ['raw/loop','raw/fixedfm','logs']: (d/sub).mkdir(parents=True)
    SF.save(d/'metadata.json', {'study':'mlfilter-scale','phase':'classical-baselines','protocol_phase':'exploratory','config_hash':h,'config_id':base.name,
                                'run_id':run_id,'created_utc':SF.now(),'code_sha256':code_hashes(),'links':links})
    print(d)


def load(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    return meta


def sampler_for(cid, pred, L, seed):
    if cid=='CatSA': return C.categorical_sa(pred, L, seed)
    return C.boltzmann(pred, L, seed, float(cid.split('-t')[1]))


def fixedfm(run_dir, L, seed):
    meta = load(run_dir); solver_run = PROJECT/meta['links']['solver_runs'][str(L)]; SF.load_meta(solver_run)
    out = run_dir/f'raw/fixedfm/L{L}_seed{seed}.json'; assert not out.exists()
    d, qd, q, b, s, x, pred = SF.frozen(solver_run, seed, L); ctx = SF.Context(pred, d['initial_ids'], L)
    res = {'status':'running','layers':L,'seed':seed,'batches':[]}; SF.save(out, res)
    try:
        for cid in CONDS:
            for rep in range(CONFIG['fixed_fm']['reps']):
                t = time.perf_counter(); raw, meta_s = sampler_for(cid, pred, L, seed*100000+rep)
                res['batches'].append({'sampler':cid,'rep':rep,'seconds':time.perf_counter()-t,**ctx.batch(raw, x)})
        res.update(status='completed', ended_utc=SF.now()); SF.save(out, res); print('fixedfm', L, seed, 'done', flush=True)
    except Exception:
        res.update(status='failed', traceback=traceback.format_exc()); SF.save(out, res); raise


def loop(run_dir, seed, cid):
    import torch
    torch.set_num_threads(1)
    meta = load(run_dir); cfg = CONFIG['closed_loop']; L = cfg['layers']
    tf = meta['links']['truth_7']; table = PROJECT/tf['run_dir']/'raw/truth_table.npz'; assert SF.sha(table)==tf['truth_table_sha256']
    truth = np.load(table)['fom']; x_all = M.onehot(M.labels_for(L))
    out = run_dir/f'raw/loop/{cid}_seed{seed}.json'; assert not out.exists()
    d = {'status':'running','seed':seed,'condition':{'id':cid}}; SF.save(out, d)
    def sampler(qd, b, s, layers, sd):
        pred = M.predict_all(x_all, M.matrix(qd, 4*layers), b)
        return sampler_for(cid, pred, layers, sd)
    try:
        r = E.trajectory_custom(seed, L, truth, cfg['budget'], cfg['proposal_cycle_cap'], sampler, cid, x_all=x_all)
        d.update(r, status='completed', ended_utc=SF.now()); SF.save(out, d); print(cid, seed, 'best', r['final_best'], flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc()); SF.save(out, d); raise


def analyze(run_dir):
    from scipy.stats import wilcoxon
    meta = load(run_dir); def_q = lambda v: [float(np.median(v)), float(np.percentile(v,25)), float(np.percentile(v,75))]
    tf = meta['links']['truth_7']; truth = np.load(PROJECT/tf['run_dir']/'raw/truth_table.npz')['fom']
    rows = []
    for f in sorted((run_dir/'raw/loop').glob('*.json')):
        d = json.loads(f.read_text())
        if d['status']=='completed': rows.append({'condition':d['condition']['id'],'seed':d['seed'],**P1.metrics(d, truth, 100)})
    confirm = PROJECT/meta['links']['confirm_run']
    for cid in ('XY-FMQAOA','alpha1','alpha100','alpha0.3','RandomFM'):
        for s in CONFIG['closed_loop']['seeds']:
            d = json.loads((confirm/f'raw/bbo/{cid}_seed{s}.json').read_text()); rows.append({'condition':cid,'seed':s,**P1.metrics(d, truth, 100)})
    g = {}
    for r in rows: g.setdefault(r['condition'], {})[r['seed']] = r
    loop = {c:{'n':len(v),'successes':sum(x['success'] for x in v.values()),'best_rank':def_q([x['best_rank'] for x in v.values()])} for c, v in g.items()}
    tests = {}
    for c in g:
        if c=='XY-FMQAOA': continue
        diff = [g['XY-FMQAOA'][s]['best_rank']-g[c][s]['best_rank'] for s in g[c] if s in g['XY-FMQAOA']]
        p = 1.0 if all(v==0 for v in diff) else float(wilcoxon(diff, zero_method='wilcox').pvalue)
        tests[c] = {'xy_better':sum(v<0 for v in diff),'ties':sum(v==0 for v in diff),'xy_worse':sum(v>0 for v in diff),'p_two_sided_uncorrected':p}
    fixed = {}
    for f in sorted((run_dir/'raw/fixedfm').glob('*.json')):
        d = json.loads(f.read_text())
        if d['status']!='completed': continue
        gg = {}
        for r in d['batches']: gg.setdefault(r['sampler'], []).append(r)
        for smp, rs in gg.items():
            fixed.setdefault(f"L{d['layers']}|{smp}", []).append({k:float(np.mean([r[k] for r in rs if r.get(k) is not None])) for k in ('capture','norm_gap','sample_fm_pct','feasible')})
    fixed_s = {k:{m:def_q([x[m] for x in v]) for m in ('capture','norm_gap','sample_fm_pct','feasible')} for k, v in fixed.items()}
    SF.save(run_dir/'raw/summary.json', {'generated_utc':SF.now(),'loop':loop,'loop_tests_vs_xy':tests,'fixed':fixed_s,'rows':rows})
    for c, v in sorted(loop.items(), key=lambda kv: kv[1]['best_rank'][0]):
        t = tests.get(c, {}); print(f"loop {c:16s} opt {v['successes']:2d}/{v['n']} rank {v['best_rank'][0]:5.0f} [{v['best_rank'][1]:.0f},{v['best_rank'][2]:.0f}]"
                                       + (f"  XY W/T/L {t['xy_better']}/{t['ties']}/{t['xy_worse']} p={t['p_two_sided_uncorrected']:.3f}" if t else ''))
    for k, v in sorted(fixed_s.items()): print(f"fixed {k:22s} pct {v['sample_fm_pct'][0]:.3f} gap {v['norm_gap'][0]:.4f} cap {v['capture'][0]:.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init','fixedfm','loop','analyze'])
    ap.add_argument('--confirm-run', type=Path); ap.add_argument('--solver-runs', type=Path, nargs='+')
    ap.add_argument('--run-dir', type=Path); ap.add_argument('--seed', type=int); ap.add_argument('--condition'); ap.add_argument('--layers', type=int)
    a = ap.parse_args()
    if a.command=='init': init(a.confirm_run, a.solver_runs)
    elif a.command=='fixedfm': fixedfm(a.run_dir.resolve(), a.layers, a.seed)
    elif a.command=='loop': loop(a.run_dir.resolve(), a.seed, a.condition)
    else: analyze(a.run_dir.resolve())
