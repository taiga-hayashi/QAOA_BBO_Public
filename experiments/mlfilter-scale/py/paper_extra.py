"""Additional runs for the paper figures (exploratory; see paper_extra.json).

  init      --confirm-run C --solver-runs S6 S7 S8 --interp-runs I6 I7 I8
  loop      --run-dir D --seed S --condition ID          closed loop, L=7, budget 100
  fixedfm   --run-dir D --layers L --seed S              SA schedules x alpha, RandomFM, XY p=4 on one frozen FM
  analyze   --run-dir D
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
import solver_fixedfm as SF  # noqa: E402  (sets sys.path)
import phase1_penalty as P1  # noqa: E402
import mlfilter_fmqa as M  # noqa: E402
import mlfilter_fmqa_ext as E  # noqa: E402

STUDY_DIR, PROJECT = SF.STUDY_DIR, SF.PROJECT
CONFIG = json.loads((STUDY_DIR/'paper_extra.json').read_text())
CODE = [Path(__file__).resolve(), STUDY_DIR/'paper_extra.json', PROJECT/'src/mlfilter_fmqa_ext.py', *SF.CODE, *P1.CODE]


def code_hashes():
    return {str(p): SF.sha(p) for p in dict.fromkeys(CODE)}


def init(confirm_run, solver_runs, interp_runs):
    rel = lambda p: str(p.resolve().relative_to(PROJECT))
    cmeta = json.loads((confirm_run/'metadata.json').read_text()); assert cmeta['phase']=='bbo-confirm'
    links = {'confirm_run':rel(confirm_run),'truth_7':cmeta['truth_from'],'solver_runs':{},'interp_runs':{}}
    for r in solver_runs:
        m = json.loads((r/'metadata.json').read_text()); assert m['phase']=='solver-fixedfm'; links['solver_runs'][str(m['layers'])] = rel(r)
    for r in interp_runs:
        m = json.loads((r/'metadata.json').read_text()); assert m['phase']=='solver-xyinterp'; links['interp_runs'][str(m['layers'])] = rel(r)
    canon = json.dumps({'config':CONFIG,'links':links},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    config_hash = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+config_hash[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    run_dir = base/run_id
    for sub in ['raw/loop','raw/fixedfm','logs']: (run_dir/sub).mkdir(parents=True)
    SF.save(run_dir/'metadata.json', {'study':'mlfilter-scale','phase':'paper-extra','protocol_phase':'exploratory','config_hash':config_hash,
            'config_id':base.name,'run_id':run_id,'created_utc':SF.now(),'code_sha256':code_hashes(),'links':links})
    print(run_dir)


def load(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    return meta


def loop(run_dir, seed, cid):
    import torch
    torch.set_num_threads(1)
    meta = load(run_dir); cfg = CONFIG['closed_loop']; L = cfg['layers']
    tf = meta['links']['truth_7']; table = PROJECT/tf['run_dir']/'raw/truth_table.npz'; assert SF.sha(table)==tf['truth_table_sha256']
    truth = np.load(table)['fom']; cond = next(c for c in cfg['conditions'] if c['id']==cid)
    out = run_dir/f'raw/loop/{cid}_seed{seed}.json'; assert not out.exists()
    d = {'status':'running','seed':seed,'condition':cond}; SF.save(out, d)
    try:
        if cond['sampler']=='adaptive':
            r = M.trajectory(seed, {'id':'Adaptive','kind':'sa','alpha':None,'adaptive':True}, L, truth, cfg['budget'], cfg['proposal_cycle_cap'])
        else:
            a, sch = cond['alpha'], cond['schedule']
            r = E.trajectory_custom(seed, L, truth, cfg['budget'], cfg['proposal_cycle_cap'],
                                    lambda qd, b, s, layers, sd: E.sample_sa_schedule(qd, b, s, a, layers, sd, sch), cid)
        d.update(r, condition=cond, status='completed', ended_utc=SF.now()); SF.save(out, d); print(cid, seed, 'best', r['final_best'], flush=True)
    except Exception:
        d.update(status='failed', traceback=traceback.format_exc()); SF.save(out, d); raise


def fixedfm(run_dir, L, seed):
    import torch
    from qarp_pair_xy import OpenQARPPairXYQAOA
    torch.set_num_threads(1)
    meta = load(run_dir); cfg = CONFIG['fixed_fm']
    solver_run = PROJECT/meta['links']['solver_runs'][str(L)]; SF.load_meta(solver_run)
    out = run_dir/f'raw/fixedfm/L{L}_seed{seed}.json'; assert not out.exists()
    d, qd, q, b, s, x, pred = SF.frozen(solver_run, seed, L); ctx = SF.Context(pred, d['initial_ids'], L)
    res = {'status':'running','layers':L,'seed':seed,'batches':[]}; SF.save(out, res); dists = {}
    try:
        def record(label, raw, extra):
            res['batches'].append({'sampler':label, **extra, **ctx.batch(raw, x)})
            ids = [i for i in M.bits_to_ids(raw, L) if i is not None]; dists.setdefault(label, []).extend(ctx.pct[ids].tolist())
        for sch in cfg['schedules']:
            for a in cfg['sa_alphas']:
                for rep in range(cfg['reps']):
                    t = time.perf_counter(); raw, sm = E.sample_sa_schedule(qd, b, s, float(a), L, seed*100000+rep, sch)
                    record(f'SA-{sch}-a{a:g}', raw, {'alpha':a,'schedule':sch,'rep':rep,'seconds':time.perf_counter()-t,
                                                    'beta_range':sm['beta_range'],'num_sweeps':sm['num_sweeps']})
        for rep in range(cfg['reps']):
            raw, _ = M.sample_random_feasible(L, seed*100000+rep); record('RandomFM', raw, {'rep':rep})
        ir = PROJECT/meta['links']['interp_runs'][str(L)]/f'raw/samplers/xy_cvar10_seed{seed}.json'
        best = next(x for x in json.loads(ir.read_text())['searches'] if x['sampler']=='XY-p4-cvar10')['best']['angles']
        prob = OpenQARPPairXYQAOA(q/s, [4]*L).feasible_probabilities(best[:4], best[4:]); cdf = np.cumsum(prob)
        for rep in range(cfg['reps']):
            draw = np.minimum(np.searchsorted(cdf, np.random.default_rng(seed*100000+rep).random(100)*cdf[-1], side='left'), len(prob)-1)
            record('XY-p4-cvar10', x[draw], {'rep':rep})
        res.update(status='completed', ended_utc=SF.now(), xy_source=str(ir.relative_to(PROJECT)))
        np.savez_compressed(run_dir/f'raw/fixedfm/L{L}_seed{seed}_pct.npz', **{k:np.asarray(v, np.float32) for k, v in dists.items()})
        SF.save(out, res); print('fixedfm L', L, 'seed', seed, 'done', flush=True)
    except Exception:
        res.update(status='failed', traceback=traceback.format_exc()); SF.save(out, res); raise


def analyze(run_dir):
    meta = load(run_dir); out = {'loop':{}, 'fixedfm':{}}
    def q(v): v = np.asarray(v, float); return [float(np.median(v)), float(np.percentile(v,25)), float(np.percentile(v,75))]
    tf = meta['links']['truth_7']; truth = np.load(PROJECT/tf['run_dir']/'raw/truth_table.npz')['fom']
    for f in sorted((run_dir/'raw/loop').glob('*.json')):
        d = json.loads(f.read_text())
        if d['status']=='completed': out['loop'].setdefault(d['condition']['id'], []).append({'seed':d['seed'], **P1.metrics(d, truth, 100)})
    out['loop_summary'] = {c:{'n':len(v),'successes':sum(x['success'] for x in v),'best_rank':q([x['best_rank'] for x in v])} for c, v in out['loop'].items()}
    for f in sorted((run_dir/'raw/fixedfm').glob('*.json')):
        d = json.loads(f.read_text())
        if d['status']!='completed': continue
        g = {}
        for row in d['batches']: g.setdefault(row['sampler'], []).append(row)
        for smp, rows in g.items():
            out['fixedfm'].setdefault(f"L{d['layers']}|{smp}", []).append({'seed':d['seed'], **{k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None
                                                                                          for k in ('capture','norm_gap','feasible','sample_fm_pct')}})
    out['fixedfm_summary'] = {k:{m:(q([x[m] for x in v if x[m] is not None]) if any(x[m] is not None for x in v) else None)
                                 for m in ('capture','norm_gap','feasible','sample_fm_pct')} for k, v in out['fixedfm'].items()}
    SF.save(run_dir/'raw/summary.json', {'generated_utc':SF.now(), **out})
    for c, v in sorted(out['loop_summary'].items()): print(f"loop {c:20s} n={v['n']} opt {v['successes']} rank {v['best_rank'][0]:.0f} [{v['best_rank'][1]:.0f},{v['best_rank'][2]:.0f}]")
    for k, v in sorted(out['fixedfm_summary'].items()):
        if 'a1000' in k or 'a100' in k or 'a10' in k or 'XY' in k or 'Random' in k:
            print(f"fixed {k:28s} pct {v['sample_fm_pct'][0] if v['sample_fm_pct'] else float('nan'):.3f} gap {v['norm_gap'][0]:.3f} feas {v['feasible'][0]:.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init','loop','fixedfm','analyze'])
    ap.add_argument('--confirm-run', type=Path); ap.add_argument('--solver-runs', type=Path, nargs='+'); ap.add_argument('--interp-runs', type=Path, nargs='+')
    ap.add_argument('--run-dir', type=Path); ap.add_argument('--seed', type=int); ap.add_argument('--condition'); ap.add_argument('--layers', type=int)
    a = ap.parse_args()
    if a.command=='init': init(a.confirm_run, a.solver_runs, a.interp_runs)
    elif a.command=='loop': loop(a.run_dir.resolve(), a.seed, a.condition)
    elif a.command=='fixedfm': fixedfm(a.run_dir.resolve(), a.layers, a.seed)
    else: analyze(a.run_dir.resolve())
