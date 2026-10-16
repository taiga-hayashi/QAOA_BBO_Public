"""Simulated Annealing (SA) sampler for Adaptive-FMQA and LargePenalty-FMQA."""
from __future__ import annotations

import time
from typing import Any, Dict, Sequence, Tuple

import neal
import numpy as np

from qubo import build_onehot_penalty_bqm


def sample_sa(
    q_dict: Dict[Tuple[int, int], float],
    group_sizes: Sequence[int],
    penalty_lambda: float,
    penalty_alpha: float,
    offset: float,
    sa_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run D-Wave Neal Simulated Annealing on a One-Hot penalized QUBO BQM.

    Returns:
        raw_bits: np.ndarray of shape (num_reads, n_bits) with values in {0, 1}.
        meta: dict of execution metadata (reads, sweeps, penalty, runtime).
    """
    n_bits = sum(group_sizes)
    num_reads = int(sa_cfg["num_reads"])
    num_sweeps = int(sa_cfg["num_sweeps"])
    beta_schedule_type = str(sa_cfg.get("beta_schedule_type", "geometric"))

    t0 = time.perf_counter()
    bqm = build_onehot_penalty_bqm(
        q_dict=q_dict,
        group_sizes=group_sizes,
        penalty_lambda=penalty_lambda,
        offset=offset,
    )
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_schedule_type=beta_schedule_type,
        seed=seed,
    )
    variables = list(sampleset.variables)
    var_to_col = {v: idx for idx, v in enumerate(variables)}
    col_order = [var_to_col[i] for i in range(n_bits)]

    records = []
    for datum in sampleset.data(["sample", "num_occurrences"]):
        row = [datum.sample[i] for i in range(n_bits)]
        for _ in range(int(datum.num_occurrences)):
            records.append(row)
    raw_bits = np.asarray(records[:num_reads], dtype=np.int8)
    elapsed_sec = time.perf_counter() - t0

    meta = {
        "sampler_type": "SA",
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "beta_schedule_type": beta_schedule_type,
        "beta_range": [float(v) for v in sampleset.info["beta_range"]],
        "sampler_seed": int(seed),
        "penalty_alpha": float(penalty_alpha),
        "penalty_lambda": float(penalty_lambda),
        "elapsed_sec": float(elapsed_sec),
    }
    return raw_bits, meta
