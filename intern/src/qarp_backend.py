"""Canonical OpenQARP state-vector circuits for the project QAOA solvers."""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np


def all_bitstrings_lsb(n_qubits: int) -> np.ndarray:
    values = np.arange(1 << n_qubits, dtype=np.uint64)[:, None]
    positions = np.arange(n_qubits, dtype=np.uint64)[None, :]
    return ((values >> positions) & 1).astype(np.float64)


def one_hot_penalty(bits: np.ndarray, block_sizes: Sequence[int]) -> np.ndarray:
    penalty = np.zeros(len(bits), dtype=np.float64)
    start = 0
    for size in block_sizes:
        penalty += (bits[:, start : start + size].sum(axis=1) - 1.0) ** 2
        start += size
    return penalty


def one_hot_patterns(block_sizes: Sequence[int]) -> np.ndarray:
    return np.asarray([np.concatenate(choice) for choice in product(*(np.eye(s) for s in block_sizes))], dtype=np.float64)


def qubo_to_ising_coefficients(
    qubo: np.ndarray, block_sizes: Sequence[int] | None = None, lambda_penalty: float = 0.0
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    matrix = np.asarray(qubo, dtype=np.float64).copy()
    if block_sizes and lambda_penalty:
        start = 0
        for size in block_sizes:
            for index in range(start, start + size):
                matrix[index, index] -= lambda_penalty
            for left in range(start, start + size):
                for right in range(left + 1, start + size):
                    matrix[left, right] += 2.0 * lambda_penalty
            start += size
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}
    for left in range(matrix.shape[0]):
        linear[left] = linear.get(left, 0.0) - float(matrix[left, left]) / 2.0
        for right in range(left + 1, matrix.shape[0]):
            coefficient = float(matrix[left, right])
            if coefficient:
                linear[left] -= coefficient / 4.0
                linear[right] = linear.get(right, 0.0) - coefficient / 4.0
                quadratic[(left, right)] = coefficient / 4.0
    return linear, quadratic


class OpenQARPStandardQAOA:
    def __init__(self, qubo: np.ndarray, block_sizes: Sequence[int], lambda_penalty: float = 0.0):
        self.qubo = np.asarray(qubo, dtype=np.float64)
        self.block_sizes = tuple(block_sizes)
        self.n_qubits = self.qubo.shape[0]
        self.linear, self.quadratic = qubo_to_ising_coefficients(self.qubo, self.block_sizes, lambda_penalty)

    def probabilities(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        if len(gammas) != len(betas) or len(gammas) == 0:
            raise ValueError("gammas and betas must be non-empty sequences of equal length.")
        import qarpx as qx

        circuit = qx.SimpleBlock(self.n_qubits, "OpenQARP Standard QAOA")
        for qubit in range(self.n_qubits):
            circuit.h(qubit)
        for gamma, beta in zip(gammas, betas):
            for qubit, coefficient in self.linear.items():
                circuit.rz(qubit, 2.0 * coefficient * float(gamma))
            for (left, right), coefficient in self.quadratic.items():
                circuit.rzz(left, right, 2.0 * coefficient * float(gamma))
            for qubit in range(self.n_qubits):
                circuit.rx(qubit, 2.0 * float(beta))
        circuit.build()
        state = qx.QarpSimulator().statevector(circuit.commands(), self.n_qubits)
        probabilities = np.abs(state) ** 2
        return probabilities / probabilities.sum()


class OpenQARPXYQAOA:
    def __init__(self, qubo: np.ndarray, block_sizes: Sequence[int]):
        self.qubo = np.asarray(qubo, dtype=np.float64)
        self.block_sizes = tuple(block_sizes)
        self.n_qubits = self.qubo.shape[0]
        self.linear, self.quadratic = qubo_to_ising_coefficients(self.qubo)
        self.blocks: list[list[int]] = []
        start = 0
        for size in self.block_sizes:
            self.blocks.append(list(range(start, start + size)))
            start += size
        patterns = one_hot_patterns(self.block_sizes)
        indices = np.asarray(patterns, dtype=np.uint64) @ (1 << np.arange(self.n_qubits, dtype=np.uint64))
        self.initial_state = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        self.initial_state[indices] = 1.0 / np.sqrt(len(patterns))

    def probabilities(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        if len(gammas) != len(betas) or len(gammas) == 0:
            raise ValueError("gammas and betas must be non-empty sequences of equal length.")
        import qarpx as qx

        circuit = qx.SimpleBlock(self.n_qubits, "OpenQARP One-Hot XY-QAOA")
        for gamma, beta in zip(gammas, betas):
            for qubit, coefficient in self.linear.items():
                circuit.rz(qubit, 2.0 * coefficient * float(gamma))
            for (left, right), coefficient in self.quadratic.items():
                circuit.rzz(left, right, 2.0 * coefficient * float(gamma))
            for block in self.blocks:
                for offset, left in enumerate(block):
                    for right in block[offset + 1 :]:
                        circuit.rxx(left, right, float(beta))
                        circuit.ryy(left, right, float(beta))
        circuit.build()
        state = qx.QarpSimulator().statevector(circuit.commands(), self.n_qubits, initial_state=self.initial_state)
        probabilities = np.abs(state) ** 2
        return probabilities / probabilities.sum()
