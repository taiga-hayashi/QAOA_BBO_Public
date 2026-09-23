"""OpenQARP-only QAOA solver API for FMQA/BBO workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

try:
    from .qarp_backend import OpenQARPStandardQAOA, OpenQARPXYQAOA, all_bitstrings_lsb, one_hot_penalty, one_hot_patterns
except ImportError:  # pragma: no cover - direct script execution
    from qarp_backend import OpenQARPStandardQAOA, OpenQARPXYQAOA, all_bitstrings_lsb, one_hot_penalty, one_hot_patterns


@dataclass(frozen=True)
class QAOASample:
    x: tuple[int, ...]
    fval: float
    probability: float


@dataclass(frozen=True)
class QAOASolveResult:
    samples: list[QAOASample]
    backend: str = "OpenQARP/qarpx"


def _validate_blocks(block_sizes: Sequence[int]) -> None:
    if not block_sizes or any(size < 1 for size in block_sizes):
        raise ValueError("block_sizes must be a non-empty sequence of positive integers.")


def _matrix_from_dict(qubo_dict: Dict[Tuple[int, int], float], n_qubits: int) -> np.ndarray:
    matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for (left, right), value in qubo_dict.items():
        if not 0 <= left < n_qubits or not 0 <= right < n_qubits:
            raise ValueError("QUBO index is outside the declared problem size.")
        matrix[min(left, right), max(left, right)] += float(value)
    return matrix


def _parameter_candidates(reps: int, maxiter: int) -> list[np.ndarray]:
    if reps < 1 or maxiter < 1:
        raise ValueError("reps and maxiter must be positive integers.")
    if reps == 1:
        side = max(2, int(np.sqrt(maxiter)))
        return [np.asarray([gamma, beta]) for gamma in np.linspace(0.05, 0.8, side) for beta in np.linspace(0.05, 0.8, side)]
    rng = np.random.default_rng(42)
    return [np.full(2 * reps, 0.3), *[rng.uniform(0.05, 0.8, size=2 * reps) for _ in range(maxiter - 1)]]


def _optimize(
    probability_function: Callable[[Sequence[float], Sequence[float]], np.ndarray], energies: np.ndarray, reps: int, maxiter: int
) -> tuple[np.ndarray, np.ndarray, float]:
    best_parameters: np.ndarray | None = None
    best_probabilities: np.ndarray | None = None
    best_value = float("inf")
    for parameters in _parameter_candidates(reps, maxiter):
        probabilities = probability_function(parameters[:reps], parameters[reps:])
        value = float(np.dot(probabilities, energies))
        if value < best_value:
            best_parameters, best_probabilities, best_value = parameters, probabilities, value
    assert best_parameters is not None and best_probabilities is not None
    return best_parameters, best_probabilities, best_value


def _summarize(
    probabilities: np.ndarray, qubo: np.ndarray, block_sizes: Sequence[int], offset: float,
    lambda_penalty: float, solver: str, parameters: np.ndarray,
) -> Dict[str, Any]:
    bits = all_bitstrings_lsb(qubo.shape[0])
    objective = np.einsum("ni,ij,nj->n", bits, qubo, bits) + offset
    penalty = one_hot_penalty(bits, block_sizes)
    penalized = objective + lambda_penalty * penalty
    feasible = penalty == 0.0
    best_overall_index = int(np.argmin(penalized))
    best_feasible = None
    if np.any(feasible):
        feasible_indices = np.flatnonzero(feasible)
        index = int(feasible_indices[np.argmin(objective[feasible_indices])])
        best_feasible = {"bitstring": "".join(map(str, bits[index].astype(int))), "x": tuple(bits[index].astype(int)), "fval": float(objective[index]), "probability": float(probabilities[index])}
    state_probabilities: Dict[str, float] = {}
    for pattern in one_hot_patterns(block_sizes):
        index = int(np.asarray(pattern, dtype=np.uint64) @ (1 << np.arange(qubo.shape[0], dtype=np.uint64)))
        state_probabilities["".join(map(str, pattern.astype(int)))] = float(probabilities[index])
    overall_bits = bits[best_overall_index].astype(int)
    return {
        "solver": solver,
        "backend": "OpenQARP/qarpx",
        "parameters": parameters.tolist(),
        "state_probabilities": state_probabilities,
        "feasibility_rate": float(probabilities[feasible].sum()),
        "best_feasible_sample": best_feasible,
        "best_overall_sample": {"bitstring": "".join(map(str, overall_bits)), "x": tuple(overall_bits), "fval": float(objective[best_overall_index]), "raw_fval": float(penalized[best_overall_index]), "probability": float(probabilities[best_overall_index])},
    }


def solve_xy_qaoa(qubo_dict: Dict[Tuple[int, int], float], block_sizes: List[int], offset: float = 0.0, reps: int = 1, maxiter: int = 50, optimizer_name: str = "grid") -> Dict[str, Any]:
    _validate_blocks(block_sizes)
    qubo = _matrix_from_dict(qubo_dict, sum(block_sizes))
    backend = OpenQARPXYQAOA(qubo, block_sizes)
    bits = all_bitstrings_lsb(sum(block_sizes))
    objective = np.einsum("ni,ij,nj->n", bits, qubo, bits) + offset
    parameters, probabilities, _ = _optimize(backend.probabilities, objective, reps, maxiter)
    return _summarize(probabilities, qubo, block_sizes, offset, 0.0, "FM-XY-QAOA", parameters)


def solve_standard_qaoa(qubo_dict: Dict[Tuple[int, int], float], block_sizes: List[int], lambda_penalty: float = 5.0, offset: float = 0.0, reps: int = 1, maxiter: int = 50, optimizer_name: str = "grid") -> Dict[str, Any]:
    _validate_blocks(block_sizes)
    qubo = _matrix_from_dict(qubo_dict, sum(block_sizes))
    backend = OpenQARPStandardQAOA(qubo, block_sizes, lambda_penalty)
    bits = all_bitstrings_lsb(sum(block_sizes))
    objective = np.einsum("ni,ij,nj->n", bits, qubo, bits) + offset
    parameters, probabilities, _ = _optimize(backend.probabilities, objective + lambda_penalty * one_hot_penalty(bits, block_sizes), reps, maxiter)
    return _summarize(probabilities, qubo, block_sizes, offset, lambda_penalty, "Standard QAOA", parameters)


def solve_qubo_qaoa(qubo_dict: Dict[Tuple[int, int], float], offset: float = 0.0, reps: int = 1, maxiter: int = 100, optimizer_name: str = "grid") -> QAOASolveResult:
    if not qubo_dict:
        raise ValueError("QUBO dictionary is empty.")
    n_qubits = max(max(pair) for pair in qubo_dict) + 1
    qubo = _matrix_from_dict(qubo_dict, n_qubits)
    backend = OpenQARPStandardQAOA(qubo, [1] * n_qubits)
    bits = all_bitstrings_lsb(n_qubits)
    objective = np.einsum("ni,ij,nj->n", bits, qubo, bits) + offset
    _, probabilities, _ = _optimize(backend.probabilities, objective, reps, maxiter)
    best_index = int(np.argmin(objective))
    return QAOASolveResult(samples=[QAOASample(tuple(bits[best_index].astype(int)), float(objective[best_index]), float(probabilities[best_index]))])
