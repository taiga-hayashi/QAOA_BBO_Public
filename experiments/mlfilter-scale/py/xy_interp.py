"""XY-FMQAOA p=1..6 with a stronger angle search (INTERP starts, 4096 evaluations per depth) on frozen FMs (exploratory).

  init    --solver-run D --deep-run E      new run dir linked to finished solver-fixedfm / solver-xydeep runs (same L)
  run     --run-dir R --seed S --objective mean|cvar10
  analyze --run-dir R                      summary.json (solver-fixedfm schema) + the pre-fixed 'comparable' check
"""
import argparse
import hashlib
import itertools
import json
import secrets
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_fixedfm as SF  # noqa: E402

STUDY_DIR, PROJECT = SF.STUDY_DIR, SF.PROJECT
CONFIG = json.loads((STUDY_DIR/'xy_interp.json').read_text())
CODE = [Path(__file__).resolve(), STUDY_DIR/'xy_interp.json', *SF.CODE]


def code_hashes():
    return {str(p): SF.sha(p) for p in CODE}


def interp(angles, p):
    """INTERP (Zhou et al. 2020): p angles per family -> p+1, with x_0 = x_{p+1} = 0."""
    a = np.asarray(angles, float); g, b = a[:p], a[p:]
    def up(x):
        x = np.concatenate([[0.], x, [0.]])
        return np.array([(i/p)*x[i] + ((p-i)/p)*x[i+1] for i in range(p+1)])
    return np.concatenate([up(g), up(b)])


def init(solver_run, deep_run):
    solver_run, deep_run = solver_run.resolve(), deep_run.resolve()
    smeta, _ = SF.load_meta(solver_run); dmeta = json.loads((deep_run/'metadata.json').read_text())
    assert dmeta['solver_run']==str(solver_run.relative_to(PROJECT)) and dmeta['layers']==smeta['layers']
    inputs = {}
    for run in (solver_run, deep_run):
        for seed in SF.CONFIG['fm']['seeds']:
            for obj in CONFIG['objectives']:
                f = run/f'raw/samplers/xy_{obj}_seed{seed}.json'; assert json.loads(f.read_text())['status']=='completed'
                inputs[str(f.relative_to(PROJECT))] = SF.sha(f)
    canon = json.dumps({'config':CONFIG,'layers':smeta['layers'],'solver_config_hash':smeta['config_hash'],'deep_config_hash':dmeta['config_hash']},
                       sort_keys=True,ensure_ascii=False,separators=(',',':'))
    config_hash = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+config_hash[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    run_dir = base/run_id
    for sub in ['raw/samplers','logs']: (run_dir/sub).mkdir(parents=True)
    SF.save(run_dir/'metadata.json', {'study':'mlfilter-scale','phase':'solver-xyinterp','protocol_phase':'exploratory','layers':smeta['layers'],
            'config_hash':config_hash,'config_id':base.name,'run_id':run_id,'created_utc':SF.now(),'code_sha256':code_hashes(),
            'config_hash_method':'sha256 of canonical JSON {config: xy_interp.json, layers, solver_config_hash, deep_config_hash}',
            'solver_run':str(solver_run.relative_to(PROJECT)),'deep_run':str(deep_run.relative_to(PROJECT)),'inputs_sha256':inputs,
            'environment':smeta['environment']})
    print(run_dir)


def load(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    solver_run = PROJECT/meta['solver_run']; SF.load_meta(solver_run)
    return meta, solver_run, PROJECT/meta['deep_run']


def previous_best(meta, seed, objective, depth):
    run = meta['solver_run'] if depth <= 3 else meta['deep_run']
    f = f'{run}/raw/samplers/xy_{objective}_seed{seed}.json'; assert SF.sha(PROJECT/f)==meta['inputs_sha256'][f]
    d = json.loads((PROJECT/f).read_text())
    return next(s['best'] for s in d['searches'] if s['sampler']==f'XY-p{depth}-{objective}')


def search(backend, ctx, scale, seed, depth, objective, starts):
    from qarp_backend import OpenQARPCompactXYQAOA
    cfg = CONFIG['search']; budget = cfg['evaluations_per_depth']
    trace = []; best = None; bestprob = None
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, 2020, depth]))
    maxima = np.array([2*np.pi]*depth+[np.pi]*depth); bounds = [(0., v) for v in maxima]
    def value(prob):
        return float(prob@ctx.pred) if objective=='mean' else SF.M_cvar(ctx.pred, prob, .1)
    def evaluate(a, stop=budget, phase='global'):
        nonlocal best, bestprob
        if len(trace) >= stop: raise SF.BudgetReached()
        if len(trace) % 32 == 0: OpenQARPCompactXYQAOA._mixer_cache.clear()
        a = np.clip(np.asarray(a, float), 0., maxima); prob = backend.feasible_probabilities(a[:depth], a[depth:]); v = value(prob)
        trace.append({'angles':a.tolist(),'phase':phase,'objective':v})
        if best is None or v < best['objective']: best, bestprob = trace[-1].copy(), prob.copy()
        return v/scale
    for name, a in starts: evaluate(a, phase=name)
    if depth == 1:
        for g, bt in itertools.product(SF.GRID['gammas'], SF.GRID['betas']): evaluate([g, bt], phase='grid49')
    while len(trace) < 512: evaluate(rng.random(2*depth)*maxima, phase='global_to512')
    for a in [r['angles'] for r in sorted(trace, key=lambda r: r['objective'])[:8]]:
        stop = min(budget, len(trace)+384)
        try: minimize(lambda v: evaluate(v, stop, 'Powell'), a, method='Powell', bounds=bounds, options={'maxfev':384,'xtol':1e-6,'ftol':1e-9})
        except SF.BudgetReached: pass
        if len(trace) >= budget: break
    while len(trace) < budget: evaluate(rng.random(2*depth)*maxima, phase='remaining_global')
    return best, bestprob, [r['objective'] for r in trace], {n:float(r['objective']) for n, r in zip([s[0] for s in starts], trace)}


def run(run_dir, seed, objective):
    from qarp_pair_xy import OpenQARPPairXYQAOA
    meta, solver_run, _ = load(run_dir); L = meta['layers']
    out = run_dir/f'raw/samplers/xy_{objective}_seed{seed}.json'; assert not out.exists()
    d, qd, q, b, s, x, pred = SF.frozen(solver_run, seed, L); ctx = SF.Context(pred, d['initial_ids'], L)
    res = {'status':'running','seed':seed,'layers':L,'objective':objective,'S':s,'searches':[],'batches':[]}; SF.save(out, res)
    try:
        backend = OpenQARPPairXYQAOA(q/s, [4]*L); prev = None
        for depth in CONFIG['depths']:
            old = previous_best(meta, seed, objective, depth)
            starts = [('previous_search_best', old['angles'])]
            if prev is not None:
                starts += [('interp', interp(prev['angles'], depth-1)), ('embed', SF.embed(prev['angles'], depth-1, depth))]
            t = time.perf_counter(); best, prob, objs, start_values = search(backend, ctx, s, seed, depth, objective, starts); sec = time.perf_counter()-t
            assert abs(prob.sum()-1) < 1e-10 and best['objective'] <= old['objective']+1e-10
            res['searches'].append({'sampler':f'XY-p{depth}-{objective}','depth':depth,'budget':len(objs),'seconds':sec,'best':best,
                                    'previous_search_objective':old['objective'],'start_objectives':start_values,**ctx.exact(prob),
                                    'objective_trace_min_at':int(np.argmin(objs))})
            res['batches'] += SF.sample_from(prob, x, ctx, seed, SF.CONFIG['samplers']['SA']['reps'], f'XY-p{depth}-{objective}', {'depth':depth})
            prev = best; SF.save(out, res); print('XY', objective, 'seed', seed, 'p', depth, f'{sec:.0f}s', 'obj', best['objective'], 'old', old['objective'], flush=True)
        res.update(status='completed', ended_utc=SF.now()); SF.save(out, res)
    except Exception:
        res.update(status='failed', traceback=traceback.format_exc()); SF.save(out, res); raise


def analyze(run_dir):
    meta, solver_run, _ = load(run_dir); L = meta['layers']; per_fm = {}; exact = {}; unfinished = []
    for f in sorted((run_dir/'raw/samplers').glob('*.json')):
        r = json.loads(f.read_text())
        if r['status']!='completed': unfinished.append(f.name); continue
        for row in r['batches']: per_fm.setdefault((row['sampler'], r['seed']), []).append(row)
        for srch in r['searches']:
            exact.setdefault(srch['sampler'], []).append({'seed':r['seed'], **{k:srch[k] for k in ('P_unseen_min','ideal_capture100','expected_fm_pct','seconds')},
                                                         'improvement_over_previous':srch['previous_search_objective']-srch['best']['objective']})
    keys = ['capture','norm_gap','fm_rank_pct','feasible','sample_fm_pct']
    fm_means = {}
    for (sampler, seed), rows in per_fm.items():
        fm_means.setdefault(sampler, []).append({'seed':seed, **{k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys}})
    def q(v): v = np.asarray(v, float); return [float(np.median(v)), float(np.percentile(v,25)), float(np.percentile(v,75))]
    summary = {s:{'n_fm':len(v), **{k:q([x[k] for x in v]) for k in keys}} for s, v in fm_means.items()}
    exact_summary = {s:{'n_fm':len(v), **{k:q([x[k] for x in v]) for k in v[0] if k!='seed'}} for s, v in exact.items()}
    sa = json.loads((solver_run/'raw/summary.json').read_text())['summary']
    gap_bound = max(sa['SA-a1']['norm_gap'][2], sa['SA-a3']['norm_gap'][2]); pct_bound = sa['SA-a3']['sample_fm_pct'][0]
    verdict = {s:{'norm_gap':v['norm_gap'][0],'sample_fm_pct':v['sample_fm_pct'][0],'gap_ok':v['norm_gap'][0] <= gap_bound,
                  'pct_ok':v['sample_fm_pct'][0] <= pct_bound} for s, v in summary.items() if v['n_fm']==len(SF.CONFIG['fm']['seeds'])}
    for v in verdict.values(): v['comparable'] = v['gap_ok'] and v['pct_ok']
    best = min(verdict, key=lambda s: verdict[s]['norm_gap']) if verdict else None
    SF.save(run_dir/'raw/summary.json', {'layers':L,'generated_utc':SF.now(),'summary':summary,'exact':exact_summary,'per_fm':fm_means,
             'per_fm_exact':exact,'unfinished':unfinished,'criterion':CONFIG['comparable_criterion'],
             'criterion_bounds':{'gap_bound':gap_bound,'pct_bound':pct_bound,'SA_a1':sa['SA-a1'],'SA_a3':sa['SA-a3']},'verdict':verdict,'best_condition':best})
    print(f'== L={L}  bounds: gap <= {gap_bound:.4f}, sample_pct <= {pct_bound:.3f}')
    for s in sorted(verdict, key=lambda k: (k.split('-')[-1], int(k.split('-')[1][1:]))):
        v = verdict[s]; e = exact_summary[s]
        print(f"  {s:16s} gap {v['norm_gap']:.4f} {'OK' if v['gap_ok'] else '--'}  pct {v['sample_fm_pct']:.3f} {'OK' if v['pct_ok'] else '--'}"
              f"  capture {summary[s]['capture'][0]:.2f}  ideal_cap100 {e['ideal_capture100'][0]:.3f}  improve {e['improvement_over_previous'][0]:.3g}"
              + ('  <== comparable' if v['comparable'] else ''))
    print('best condition by gap:', best, verdict.get(best))
    if unfinished: print('unfinished', unfinished)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init','run','analyze']); ap.add_argument('--solver-run', type=Path); ap.add_argument('--deep-run', type=Path)
    ap.add_argument('--run-dir', type=Path); ap.add_argument('--seed', type=int); ap.add_argument('--objective', choices=['mean','cvar10'])
    a = ap.parse_args()
    import torch; torch.set_num_threads(1)
    if a.command=='init': init(a.solver_run, a.deep_run)
    elif a.command=='run': run(a.run_dir.resolve(), a.seed, a.objective)
    else: analyze(a.run_dir.resolve())
