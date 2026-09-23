"""Compatibility solver for the baseline reduced-subspace XY-QAOA method.

Hamiltonian construction is shared with the enhanced solver so the two paths
cannot drift in their One-Hot basis or ring-XY definition.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

try:
    from .enhanced_subspace_xy_qaoa_solver import build_subspace_basis, build_subspace_hamiltonians
except ImportError:  # pragma: no cover - legacy direct script execution
    from enhanced_subspace_xy_qaoa_solver import build_subspace_basis, build_subspace_hamiltonians


def solve_subspace_xy_qaoa(
    qubo_matrix: np.ndarray, block_sizes: List[int], reps: int = 1, maxiter: int = 30
) -> Dict[str, Any]:
    """Run the historical p-layer ring-XY baseline in the feasible subspace."""
    energies, mixer, _ = build_subspace_hamiltonians(qubo_matrix, block_sizes, mixer_type="ring")
    dimension = len(energies)
    initial_state = np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.complex128)

    def get_state(parameters: np.ndarray) -> np.ndarray:
        gammas, betas = parameters[:reps], parameters[reps:]
        state = initial_state.copy()
        for gamma, beta in zip(gammas, betas):
            state *= np.exp(-1j * gamma * energies)
            state = expm_multiply(-1j * beta * mixer, state)
        return state

    def objective(parameters: np.ndarray) -> float:
        probabilities = np.abs(get_state(parameters)) ** 2
        return float(np.sum(probabilities * energies))

    np.random.seed(42)  # Preserves the historical baseline trajectory.
    initial_parameters = np.random.uniform(0, np.pi, size=2 * reps)
    result = minimize(objective, initial_parameters, method="COBYLA", options={"maxiter": maxiter})

    probabilities = np.abs(get_state(result.x)) ** 2
    optimum_index = int(np.argmin(energies))
    sampled_indices = np.random.choice(dimension, size=500, p=probabilities / probabilities.sum())
    best_sample_index = sampled_indices[np.argmin(energies[sampled_indices])]
    return {
        "opt_energy": result.fun,
        "best_sampled_energy": energies[best_sample_index],
        "exact_min_energy": energies[optimum_index],
        "success_probability": probabilities[optimum_index],
        "feasibility_rate": 1.0,
        "dim_subspace": dimension,
        "params": result.x,
    }
