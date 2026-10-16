"""Vendored shared helpers from intern_0924; no problem-specific implementation.

See shared_numerics_sources.json for source hashes and symbols.
"""
from __future__ import annotations
from typing import List
import numpy as np

def _build_qubo_energies(
    Q: np.ndarray,
    offset: float,
    real_dtype: np.dtype,
) -> np.ndarray:
    """Build x.T Q x + offset in LSB order without a (2**N, N) bit matrix."""
    N = Q.shape[0]
    if Q.shape != (N, N):
        raise ValueError("Q must be square")
    energies = np.empty(1 << N, dtype=real_dtype)
    energies[0] = offset
    populated = 1
    for j in range(N):
        upper = energies[populated : 2 * populated]
        np.add(energies[:populated], Q[j, j], out=upper, casting="unsafe")
        for i in range(j):
            coefficient = float(Q[i, j] + Q[j, i])
            if coefficient == 0.0:
                continue
            step = 1 << i
            upper.reshape(-1, 2 * step)[:, step:] += coefficient
        populated *= 2
    return energies

class AdaptivePenaltyTracker:
    """
    Manages and updates alpha_t dynamically across FMQA iterations.
    """
    def __init__(self, initial_alpha: float = 1.0, min_alpha: float = 0.01, max_alpha: float = 100.0):
        self.alpha_t = initial_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.history: List[float] = []

    def get_current_alpha(self) -> float:
        return self.alpha_t

    def update(self, raw_feasible_rate: float) -> float:
        """
        Adaptively update alpha_t based on constraint feasibility.
        If feasible rate is low, increase alpha_t to enforce constraint.
        If feasible rate is 100%, slightly decrease alpha_t to favor objective exploration.
        """
        self.history.append(self.alpha_t)
        if raw_feasible_rate < 0.7:
            self.alpha_t = min(self.max_alpha, self.alpha_t * 2.0)
        elif raw_feasible_rate < 1.0:
            self.alpha_t = min(self.max_alpha, self.alpha_t * 1.25)
        else:
            self.alpha_t = max(self.min_alpha, self.alpha_t * 0.85)
        return self.alpha_t
