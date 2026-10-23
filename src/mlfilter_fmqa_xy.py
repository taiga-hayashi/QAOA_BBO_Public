"""FMQA closed loop whose proposal step is XY-FMQAOA (lambda = 0) instead of penalty SA.

Everything except the proposal is identical to mlfilter_fmqa.trajectory(): same initial points, fresh FM per
cycle (seed + 100000*(cycle-1)), same selection rule (minimum-FM new feasible candidate, ties by one-hot tuple),
same evaluation from the validated truth table, 100 proposals per cycle.

Proposal (per cycle): OpenQARPPairXYQAOA(Q/S) at fixed depth p, angles chosen by minimising the CVaR_0.1 (or mean)
of the FM energy under the exact feasible distribution; 100 shots drawn from that distribution with
default_rng(seed*100000 + (cycle-1)*1000) (the SA sampler-seed rule).
  cycle 1: INTERP ladder p = 1..P, `first_evals` evaluations per depth (49-grid at p = 1)
  later:   warm start from the previous cycle's best angles: that point + Powell (<= `local_maxfev`) +
           uniform random fill, `cycle_evals` evaluations in total
mlfilter_fmqa.py itself is not modified (its results stay reproducible).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.optimize import minimize

import mlfilter_fmqa as M

GRID = {'gammas': [0.0, 0.05, 0.2, 0.4, 0.8, 1.6, 3.2], 'betas': [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6]}


class _Budget(Exception):
    pass


def cvar(energy, prob, alpha=0.1):
    # Same as intern_1009/perovskites/xy_cvar_sampling/py/experiment.py:cvar
    order = np.argsort(energy, kind='stable'); p = prob[order]/prob.sum(); before = np.cumsum(p)-p
    w = np.maximum(0, np.minimum(p, alpha-before)); return float(w@energy[order]/alpha)


def interp(angles, p):
    """INTERP (Zhou et al. 2020): p angles per family -> p+1, with x_0 = x_{p+1} = 0."""
    a = np.asarray(angles, float)
    def up(x):
        x = np.concatenate([[0.], x, [0.]]); return np.array([(i/p)*x[i] + ((p-i)/p)*x[i+1] for i in range(p+1)])
    return np.concatenate([up(a[:p]), up(a[p:])])


def angle_search(backend, pred, scale, depth, objective, starts, budget, rng, n_global, n_local, local_maxfev, grid=False):
    from qarp_backend import OpenQARPCompactXYQAOA
    trace = []; best = [None, None]
    maxima = np.array([2*np.pi]*depth + [np.pi]*depth); bounds = [(0., v) for v in maxima]
    def value(prob):
        return float(prob@pred) if objective == 'mean' else cvar(pred, prob, .1)
    def evaluate(a, stop=budget):
        if len(trace) >= stop: raise _Budget()
        if len(trace) % 32 == 0: OpenQARPCompactXYQAOA._mixer_cache.clear()
        a = np.clip(np.asarray(a, float), 0., maxima); prob = backend.feasible_probabilities(a[:depth], a[depth:]); v = value(prob)
        trace.append((a, v))
        if best[0] is None or v < best[0][1]: best[0], best[1] = (a, v), prob
        return v/scale
    try:
        for a in starts: evaluate(a)
        if grid:
            for g, b in itertools.product(GRID['gammas'], GRID['betas']): evaluate([g, b])
        while len(trace) < n_global: evaluate(rng.random(2*depth)*maxima)
        for a in [t[0] for t in sorted(trace, key=lambda t: t[1])[:n_local]]:
            stop = min(budget, len(trace)+local_maxfev)
            try: minimize(lambda v: evaluate(v, stop), a, method='Powell', bounds=bounds, options={'maxfev': local_maxfev, 'xtol': 1e-6, 'ftol': 1e-9})
            except _Budget: pass
        while len(trace) < budget: evaluate(rng.random(2*depth)*maxima)
    except _Budget:
        pass
    return best[0][0], best[0][1], best[1], len(trace)


def trajectory_xy(seed, layers, truth, budget, cycle_cap, cfg, x_all=None):
    """cfg: depth, objective, first_evals, cycle_evals, local_maxfev."""
    from qarp_pair_xy import OpenQARPPairXYQAOA
    x_all = M.onehot(M.labels_for(layers)) if x_all is None else x_all
    ids = M.initial_ids(seed, layers); ys = [float(truth[i]) for i in ids]
    events = []; start = time.perf_counter(); angles = None; P = cfg['depth']
    for cycle in range(1, cycle_cap + 1):
        if len(ids) >= budget:
            break
        modelseed = seed + 100000*(cycle-1)
        t = time.perf_counter(); m, qd, q, b, s = M.fit(x_all[ids], ys, modelseed); training = time.perf_counter() - t
        pred = M.predict_all(x_all, q, b)
        t = time.perf_counter(); backend = OpenQARPPairXYQAOA(q/s, [4]*layers)
        rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 7, 2026]))
        if angles is None:   # cycle 1: INTERP ladder
            a_prev = None; evals = 0
            for d in range(1, P+1):
                starts = [] if a_prev is None else [interp(a_prev, d-1), np.concatenate([a_prev[:d-1], [0.], a_prev[d-1:], [0.]])]
                a_prev, obj, prob, n = angle_search(backend, pred, s, d, cfg['objective'], starts, cfg['first_evals'], rng,
                                                    n_global=512, n_local=8, local_maxfev=384, grid=(d == 1))
                evals += n
        else:                # warm start from the previous cycle
            a_prev, obj, prob, evals = angle_search(backend, pred, s, P, cfg['objective'], [angles], cfg['cycle_evals'], rng,
                                                    n_global=1, n_local=1, local_maxfev=cfg['local_maxfev'])
        angles = a_prev
        cdf = np.cumsum(prob); sampler_seed = seed*100000 + (cycle-1)*1000
        draw = np.minimum(np.searchsorted(cdf, np.random.default_rng(sampler_seed).random(M.SA['num_reads'])*cdf[-1], side='left'), len(prob)-1)
        raw = x_all[draw]; sampling = time.perf_counter() - t
        seen = set(ids); chosen, filters = M.select(raw, pred, x_all, seen, layers)
        unseen_mask = np.ones(len(pred), bool); unseen_mask[ids] = False; unseen_min = float(pred[unseen_mask].min())
        targets = np.flatnonzero(unseen_mask & (np.abs(pred - unseen_min) < 1e-8))
        event = {'cycle': cycle, 'model_seed': modelseed, 'model_sha256': M.model_hash(m), 'S_fm': s, 'sampler': 'XY-FMQAOA',
                 'depth': P, 'objective': cfg['objective'], 'angles': angles.tolist(), 'angle_objective': obj, 'angle_evaluations': evals,
                 'P_unseen_min': float(prob[targets].sum()), 'sampler_seed': sampler_seed, **filters,
                 'selected_id': chosen, 'selected_value': None,
                 'selected_FM_gap': None if chosen is None else float(pred[chosen] - unseen_min),
                 'unseen_FM_min_hit': bool(chosen is not None and abs(pred[chosen] - unseen_min) < 1e-8),
                 'evaluations_before': len(ids), 'training_seconds': training, 'sampling_seconds': sampling}
        if chosen is not None:
            assert chosen not in seen and len(ids) < budget
            value = float(truth[chosen]); ids.append(chosen); ys.append(value); event['selected_value'] = value
        event.update(evaluations_after=len(ids), best_so_far=float(min(ys)))
        events.append(event)
    return {'seed': seed, 'condition': {'id': 'XY-FMQAOA', 'kind': 'xy', **cfg}, 'layers': layers, 'budget': budget, 'cycle_cap': cycle_cap,
            'initial_ids': ids[:40], 'final_ids': ids, 'final_values': ys, 'actual_evaluations': len(ids),
            'final_best': float(min(ys)), 'events': events, 'total_seconds': time.perf_counter() - start,
            'stopping_reason': 'evaluation_cap' if len(ids) >= budget else 'proposal_cycle_cap'}
