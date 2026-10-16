"""QAOA sampler for Penalty-FMQAOA (Standard X-Mixer) and XY-FMQAOA (W-state + Ring XY-Mixer)."""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from shared_numerics import _build_qubo_energies
from qubo import build_onehot_penalty_bqm

from qarp_backend import (
    OpenQARPCompactXYQAOA,
    OpenQARPStandardQAOA,
    OpenQARPXYQAOA,
    all_bitstrings_lsb,
    one_hot_penalty,
)

LEGACY_GAMMAS = (0.05, 0.40, 0.80)
LEGACY_BETAS = (0.10, 0.40, 0.80)
WIDE_GAMMAS = tuple(float(x) for x in np.linspace(0.0, 2.0 * np.pi, 7))
WIDE_BETAS = tuple(float(x) for x in np.linspace(0.0, np.pi, 7))


class _BudgetReached(Exception):
    pass


def compute_cvar(
    energies: np.ndarray,
    probs: np.ndarray,
    alpha: float = 0.10,
    sorted_order: np.ndarray | None = None,
) -> float:
    """Compute exact lower-tail CVaR_alpha from energy values and probability mass."""
    if alpha >= 1.0:
        return float(probs @ energies)
    order = np.argsort(energies, kind="stable") if sorted_order is None else sorted_order
    p = probs[order] / probs.sum()
    before = np.cumsum(p) - p
    weights = np.maximum(0.0, np.minimum(p, alpha - before))
    return float((weights @ energies[order]) / alpha)


def _score_from_probs(
    probs: np.ndarray,
    energies: np.ndarray,
    objective: str,
    cvar_alpha: float,
    sorted_order: np.ndarray,
) -> float:
    if objective == "mean":
        return float(probs @ energies)
    if objective == "cvar":
        return compute_cvar(energies, probs, alpha=cvar_alpha, sorted_order=sorted_order)
    raise ValueError(f"Unknown QAOA objective '{objective}'")


def optimize_qaoa_angles(
    prob_fn: Callable[[Sequence[float], Sequence[float]], np.ndarray],
    energies: np.ndarray,
    depth: int,
    objective: str,
    cvar_alpha: float,
    angle_search_mode: str,
    seed: int = 42,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], np.ndarray, Dict[str, Any]]:
    """Optimize QAOA angles (gammas, betas) matching the verified protocol in matched_sampler_budget."""
    sorted_order = np.argsort(energies, kind="stable")
    eval_count = 0
    best_score = float("inf")
    best_gammas: Tuple[float, ...] = ()
    best_betas: Tuple[float, ...] = ()
    best_probs: np.ndarray | None = None
    trace_scores: list[ Tuple[float, list[float]] ] = []

    def evaluate(
        gammas: Sequence[float],
        betas: Sequence[float],
        limit: int | None = None,
    ) -> float:
        nonlocal eval_count, best_score, best_gammas, best_betas, best_probs
        if limit is not None and eval_count >= limit:
            raise _BudgetReached()
        if len(OpenQARPCompactXYQAOA._mixer_cache) > 512:
            OpenQARPCompactXYQAOA._mixer_cache.clear()
        probs = prob_fn(gammas, betas)
        eval_count += 1
        score = _score_from_probs(probs, energies, objective, cvar_alpha, sorted_order)
        vec = [float(x) for x in gammas] + [float(x) for x in betas]
        trace_scores.append((score, vec))
        if score < best_score:
            best_score = score
            best_gammas = tuple(float(x) for x in gammas)
            best_betas = tuple(float(x) for x in betas)
            best_probs = probs.copy()
        return score

    def embed_p1(g0: float, b0: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        gs = (float(g0),) + (0.0,) * (depth - 1)
        bs = (float(b0),) + (0.0,) * (depth - 1)
        return gs, bs

    if angle_search_mode in ("legacy9", "wide49"):
        g_grid = LEGACY_GAMMAS if angle_search_mode == "legacy9" else WIDE_GAMMAS
        b_grid = LEGACY_BETAS if angle_search_mode == "legacy9" else WIDE_BETAS
        for g0, b0 in itertools.product(g_grid, b_grid):
            gs, bs = embed_p1(g0, b0)
            evaluate(gs, bs)

    elif angle_search_mode == "powell49":
        limit = 49
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 49, 231]))
        bounds = [(0.0, 2.0 * np.pi)] * depth + [(0.0, np.pi)] * depth
        # 1) 9 embedded legacy9 evaluations
        for g0, b0 in itertools.product(LEGACY_GAMMAS, LEGACY_BETAS):
            gs, bs = embed_p1(g0, b0)
            evaluate(gs, bs, limit=limit)
        # 2) 16 random global evaluations (up to 25)
        while eval_count < 25:
            gs_r = rng.uniform(0.0, 2.0 * np.pi, depth)
            bs_r = rng.uniform(0.0, np.pi, depth)
            evaluate(gs_r, bs_r, limit=limit)
        # 3) 24 Powell evaluations from best start (up to 49)
        start = sorted(trace_scores, key=lambda t: t[0])[0][1]
        try:
            minimize(
                lambda a: evaluate(a[:depth], a[depth:], limit=limit),
                np.asarray(start, dtype=float),
                method="Powell",
                bounds=bounds,
                options={"maxfev": 24, "xtol": 1e-5, "ftol": 1e-7},
            )
        except _BudgetReached:
            pass
        while eval_count < limit:
            gs_r = rng.uniform(0.0, 2.0 * np.pi, depth)
            bs_r = rng.uniform(0.0, np.pi, depth)
            evaluate(gs_r, bs_r, limit=limit)

    elif angle_search_mode == "powell1024":
        limit = 1024
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 20, 231]))
        bounds = [(0.0, 2.0 * np.pi)] * depth + [(0.0, np.pi)] * depth
        # 1) 49 embedded wide49 evaluations
        for g0, b0 in itertools.product(WIDE_GAMMAS, WIDE_BETAS):
            gs, bs = embed_p1(g0, b0)
            evaluate(gs, bs, limit=limit)
        # 2) Random global evaluations up to 256
        while eval_count < 256:
            gs_r = rng.uniform(0.0, 2.0 * np.pi, depth)
            bs_r = rng.uniform(0.0, np.pi, depth)
            evaluate(gs_r, bs_r, limit=limit)
        # 3) Multi-start Powell from top 4 starts (192 evals each)
        starts = [vec for _, vec in sorted(trace_scores, key=lambda t: t[0])[:4]]
        for start in starts:
            stop = min(limit, eval_count + 192)
            try:
                minimize(
                    lambda a: evaluate(a[:depth], a[depth:], limit=stop),
                    np.asarray(start, dtype=float),
                    method="Powell",
                    bounds=bounds,
                    options={"maxfev": 192, "xtol": 1e-5, "ftol": 1e-7},
                )
            except _BudgetReached:
                pass
        while eval_count < limit:
            gs_r = rng.uniform(0.0, 2.0 * np.pi, depth)
            bs_r = rng.uniform(0.0, np.pi, depth)
            evaluate(gs_r, bs_r, limit=limit)
    else:
        raise ValueError(
            f"Unsupported angle_search_mode '{angle_search_mode}'. "
            "Choose from: 'legacy9', 'wide49', 'powell49', 'powell1024'."
        )

    assert best_probs is not None
    info = {
        "depth": int(depth),
        "objective": objective,
        "cvar_alpha": float(cvar_alpha) if objective == "cvar" else None,
        "angle_search_mode": angle_search_mode,
        "angle_evaluations": int(eval_count),
        "best_objective_score": float(best_score),
        "best_gammas": list(best_gammas),
        "best_betas": list(best_betas),
    }
    return best_gammas, best_betas, best_probs, info


def sample_standard_qaoa(
    q_mat: np.ndarray,
    group_sizes: Sequence[int],
    valid_basis_indices: np.ndarray,
    penalty_lambda: float,
    penalty_alpha: float,
    ising_scale: float,
    qaoa_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run Penalty-FMQAOA (Standard X-Mixer QAOA with One-Hot penalty in Cost Hamiltonian)."""
    t0 = time.perf_counter()
    n_qubits = q_mat.shape[0]
    shots = int(qaoa_cfg["shots"])
    depth = int(qaoa_cfg.get("standard_depth", 1))
    objective = str(qaoa_cfg.get("standard_objective", "mean"))
    cvar_alpha = float(qaoa_cfg.get("cvar_alpha", 0.10))
    search_mode = str(qaoa_cfg.get("standard_angle_search_mode", "legacy9"))

    scale = max(float(ising_scale), 1e-8)
    q_norm = q_mat / scale
    lam_norm = float(penalty_lambda) / scale

    circuit = OpenQARPStandardQAOA(q_norm, group_sizes, lambda_penalty=lam_norm)
    # Reuse the canonical recurrence; never allocate the full (2**N, N) bit matrix.
    q_dict = {(i, j): float(q_norm[i, j]) for i in range(n_qubits) for j in range(i, n_qubits)}
    bqm = build_onehot_penalty_bqm(q_dict, group_sizes, lam_norm)
    total, energy_offset = bqm.to_qubo()
    total_matrix = np.zeros_like(q_norm)
    for (i, j), value in total.items():
        total_matrix[min(i, j), max(i, j)] += value
    penalized = _build_qubo_energies(total_matrix, energy_offset, np.dtype(np.float64))

    _, _, best_probs, angle_info = optimize_qaoa_angles(
        prob_fn=circuit.probabilities,
        energies=penalized,
        depth=depth,
        objective=objective,
        cvar_alpha=cvar_alpha,
        angle_search_mode=search_mode,
        seed=seed,
    )

    p_feasible = float(best_probs[valid_basis_indices].sum())
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(1 << n_qubits, size=shots, replace=True, p=best_probs).astype(np.uint64)
    positions = np.arange(n_qubits, dtype=np.uint64)[None, :]
    raw_bits = ((sampled_idx[:, None] >> positions) & 1).astype(np.int8)
    elapsed_sec = time.perf_counter() - t0

    meta = {
        "sampler_type": "Penalty-FMQAOA",
        "shots": shots,
        "penalty_alpha": float(penalty_alpha),
        "penalty_lambda": float(penalty_lambda),
        "statevector_feasible_prob": p_feasible,
        "elapsed_sec": float(elapsed_sec),
        **angle_info,
    }
    return raw_bits, meta


def sample_xy_qaoa(
    q_mat: np.ndarray,
    group_sizes: Sequence[int],
    feasible_patterns: np.ndarray,
    valid_basis_indices: np.ndarray,
    ising_scale: float,
    qaoa_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run XY-FMQAOA (W-state initialization + Ring XY-Mixer, strictly penalty-free lambda=0)."""
    t0 = time.perf_counter()
    shots = int(qaoa_cfg["shots"])
    depth = int(qaoa_cfg["depth"])
    objective = str(qaoa_cfg["objective"])
    cvar_alpha = float(qaoa_cfg.get("cvar_alpha", 0.10))
    search_mode = str(qaoa_cfg["angle_search_mode"])
    xy_sim = str(qaoa_cfg.get("xy_simulator", "compact"))

    scale = max(float(ising_scale), 1e-8)
    q_norm = q_mat / scale

    if xy_sim == "compact":
        circuit = OpenQARPCompactXYQAOA(q_norm, group_sizes)
        prob_fn = circuit.feasible_probabilities
        energies = circuit.energies
    elif xy_sim == "full23":
        circuit = OpenQARPXYQAOA(q_norm, group_sizes)
        energies = np.einsum("bi,ij,bj->b", feasible_patterns, q_norm, feasible_patterns)

        def prob_fn(gs: Sequence[float], bs: Sequence[float]) -> np.ndarray:
            full_p = circuit.probabilities(gs, bs)
            return full_p[valid_basis_indices]
    else:
        raise ValueError(f"Unknown xy_simulator '{xy_sim}'. Choose 'compact' or 'full23'.")

    if getattr(circuit, "onehot_penalty_lambda", 0.0) != 0.0:
        raise RuntimeError("Strict rule violation: XY-FMQAOA must have onehot_penalty_lambda == 0.0")

    _, _, best_probs, angle_info = optimize_qaoa_angles(
        prob_fn=prob_fn,
        energies=energies,
        depth=depth,
        objective=objective,
        cvar_alpha=cvar_alpha,
        angle_search_mode=search_mode,
        seed=seed,
    )

    p_feasible = float(best_probs.sum())
    if abs(1.0 - p_feasible) >= 1e-10:
        raise RuntimeError(
            f"XY-FMQAOA subspace leak detected: |1 - P_feasible| = {abs(1.0 - p_feasible):.3e} >= 1e-10"
        )
    probs_norm = best_probs / p_feasible

    rng = np.random.default_rng(seed)
    chosen_rows = rng.choice(len(feasible_patterns), size=shots, replace=True, p=probs_norm)
    raw_bits = np.asarray(feasible_patterns[chosen_rows], dtype=np.int8)
    elapsed_sec = time.perf_counter() - t0

    meta = {
        "sampler_type": "XY-FMQAOA",
        "xy_simulator": xy_sim,
        "shots": shots,
        "penalty_alpha": 0.0,
        "penalty_lambda": 0.0,
        "statevector_feasible_prob": p_feasible,
        "subspace_leak": float(abs(1.0 - p_feasible)),
        "elapsed_sec": float(elapsed_sec),
        **angle_info,
    }
    return raw_bits, meta
