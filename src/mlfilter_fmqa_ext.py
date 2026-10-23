"""Extensions of mlfilter_fmqa for the paper figures (mlfilter_fmqa.py itself is unchanged).

- SA with an explicit temperature schedule (beta_range, num_sweeps) instead of Neal's automatic range
- trajectory_custom(): the same closed loop as mlfilter_fmqa.trajectory() with a pluggable proposal sampler
  (with sampler = mlfilter_fmqa.sample_sa it reproduces trajectory() exactly; checked against stored runs)
"""
from __future__ import annotations

import time

import numpy as np

import mlfilter_fmqa as M


def beta_ranges(qd, b, s, alpha, layers):
    """Neal's automatic beta range for the penalised BQM and for the FM-only BQM (lambda = 0)."""
    import neal.sampler as NS
    from fmqa_solver import build_penalty_bqm
    pen = NS.default_beta_range(build_penalty_bqm(qd, [4]*layers, s*alpha, b))
    fm = NS.default_beta_range(build_penalty_bqm(qd, [4]*layers, 0.0, b))
    return [float(pen[0]), float(pen[1])], [float(fm[0]), float(fm[1])]


def sample_sa_schedule(qd, b, s, alpha, layers, seed, schedule):
    """schedule: 'auto' (Neal default, 1000 sweeps), 'fmscale' (beta range of the FM-only BQM, 1000 sweeps),
    'wide10x' (hot end of the penalised BQM, cold end of the FM-only BQM, 10000 sweeps)."""
    import neal
    from fmqa_solver import build_penalty_bqm
    n = 4*layers; lam = s*alpha
    pen, fm = beta_ranges(qd, b, s, alpha, layers)
    if schedule == 'auto': br, sweeps = None, M.SA['num_sweeps']
    elif schedule == 'fmscale': br, sweeps = fm, M.SA['num_sweeps']
    elif schedule == 'wide10x': br, sweeps = [pen[0], fm[1]], 10*M.SA['num_sweeps']
    elif schedule == 'fmscale10x': br, sweeps = fm, 10*M.SA['num_sweeps']
    else: raise ValueError(schedule)
    res = neal.SimulatedAnnealingSampler().sample(build_penalty_bqm(qd, [4]*layers, lam, b), num_reads=M.SA['num_reads'],
                                                  num_sweeps=sweeps, beta_range=br, beta_schedule_type='geometric', seed=seed)
    arr = res.record.sample[:, [list(res.variables).index(i) for i in range(n)]]
    raw = np.repeat(arr, res.record.num_occurrences, axis=0).astype(np.int8)
    assert raw.shape == (M.SA['num_reads'], n)
    return raw, {'alpha': alpha, 'lambda': lam, 'S': s, 'schedule': schedule, 'beta_range': list(res.info['beta_range']),
                 'num_sweeps': sweeps, 'beta_range_penalised_auto': pen, 'beta_range_fm_only_auto': fm, 'sampler_seed': seed}


def trajectory_custom(seed, layers, truth, budget, cycle_cap, sampler, label, x_all=None):
    """sampler(qd, b, s, layers, sampler_seed) -> (raw, meta). Same loop and seeds as mlfilter_fmqa.trajectory()."""
    x_all = M.onehot(M.labels_for(layers)) if x_all is None else x_all
    ids = M.initial_ids(seed, layers); ys = [float(truth[i]) for i in ids]
    events = []; start = time.perf_counter()
    for cycle in range(1, cycle_cap + 1):
        if len(ids) >= budget:
            break
        modelseed = seed + 100000*(cycle-1)
        t = time.perf_counter(); m, qd, q, b, s = M.fit(x_all[ids], ys, modelseed); training = time.perf_counter() - t
        pred = M.predict_all(x_all, q, b); sampler_seed = seed*100000 + (cycle-1)*1000
        t = time.perf_counter(); raw, meta = sampler(qd, b, s, layers, sampler_seed); sampling = time.perf_counter() - t
        seen = set(ids); chosen, filters = M.select(raw, pred, x_all, seen, layers)
        unseen_mask = np.ones(len(pred), bool); unseen_mask[ids] = False; unseen_min = float(pred[unseen_mask].min())
        event = {'cycle': cycle, 'model_seed': modelseed, 'model_sha256': M.model_hash(m), 'S_fm': s, **meta, **filters,
                 'selected_id': chosen, 'selected_value': None,
                 'selected_FM_gap': None if chosen is None else float(pred[chosen] - unseen_min),
                 'unseen_FM_min_hit': bool(chosen is not None and abs(pred[chosen] - unseen_min) < 1e-8),
                 'evaluations_before': len(ids), 'training_seconds': training, 'sampling_seconds': sampling}
        if chosen is not None:
            assert chosen not in seen and len(ids) < budget
            value = float(truth[chosen]); ids.append(chosen); ys.append(value); event['selected_value'] = value
        event.update(evaluations_after=len(ids), best_so_far=float(min(ys)))
        events.append(event)
    return {'seed': seed, 'condition': {'id': label}, 'layers': layers, 'budget': budget, 'cycle_cap': cycle_cap,
            'initial_ids': ids[:40], 'final_ids': ids, 'final_values': ys, 'actual_evaluations': len(ids),
            'final_best': float(min(ys)), 'events': events, 'total_seconds': time.perf_counter() - start,
            'stopping_reason': 'evaluation_cap' if len(ids) >= budget else 'proposal_cycle_cap'}
