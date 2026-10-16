"""FM-to-QUBO conversion, Ising coefficient scale S_t, and One-Hot penalty BQM."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import dimod
import numpy as np
import torch


def fm_to_qubo(model: torch.nn.Module) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, float]:
    """Convert trained TorchFM parameters into an upper-triangular QUBO dict, matrix, and scalar offset."""
    linear_weights = model.lin.weight.detach().cpu().numpy().flatten()
    bias = float(model.lin.bias.detach().cpu().numpy().item())
    v_matrix = model.V.detach().cpu().numpy()
    d = v_matrix.shape[0]

    q_dict: Dict[Tuple[int, int], float] = {}
    q_mat = np.zeros((d, d), dtype=np.float64)

    for i in range(d):
        val = float(linear_weights[i])
        q_dict[(i, i)] = val
        q_mat[i, i] = val
        for j in range(i + 1, d):
            inter = float(np.dot(v_matrix[i], v_matrix[j]))
            q_dict[(i, j)] = inter
            q_mat[i, j] = inter

    return q_dict, q_mat, bias


def qubo_to_ising_scale(q_mat: np.ndarray) -> float:
    """Compute internal Ising coefficient scale S_t = max(max_i |h_i|, max_{j<i} |J_{ij}|)
    under x_i = (1 - z_i) / 2.
    """
    n = q_mat.shape[0]
    q_sym = np.triu(q_mat)
    diag = np.diag(q_sym)
    off = q_sym - np.diag(diag)

    h = {}
    j_couplings = {}
    for i in range(n):
        row_sum = float(off[i, :].sum() + off[:, i].sum())
        h[i] = -0.5 * float(diag[i]) - 0.25 * row_sum
        for k in range(i + 1, n):
            if off[i, k] != 0.0:
                j_couplings[(i, k)] = 0.25 * float(off[i, k])

    max_h = max((abs(v) for v in h.values()), default=0.0)
    max_j = max((abs(v) for v in j_couplings.values()), default=0.0)
    return float(max(max_h, max_j, 0.0))


def build_onehot_penalty_bqm(
    q_dict: Dict[Tuple[int, int], float],
    group_sizes: Sequence[int],
    penalty_lambda: float,
    offset: float = 0.0,
) -> dimod.BinaryQuadraticModel:
    """Build BQM for E(x) = x^T Q x + offset + lambda_t * sum_g (sum_{i in g} x_i - 1)^2."""
    bqm = dimod.BinaryQuadraticModel.from_qubo(q_dict, offset=offset)
    start = 0
    for size in group_sizes:
        group_vars = list(range(start, start + size))
        # (sum_{i in g} x_i - 1)^2 = 1 - sum_i x_i + 2 sum_{i<j} x_i x_j
        bqm.offset += penalty_lambda
        for u in group_vars:
            bqm.add_linear(u, -penalty_lambda)
        for idx, u in enumerate(group_vars):
            for v in group_vars[idx + 1:]:
                bqm.add_quadratic(u, v, 2.0 * penalty_lambda)
        start += size
    return bqm
