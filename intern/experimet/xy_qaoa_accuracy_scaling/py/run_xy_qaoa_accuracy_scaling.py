#!/usr/bin/env python3
"""Benchmark the current OpenQARP FM-XY-QAOA across two QUBO families and N.

This is deliberately an exact state-vector experiment.  It measures the
current project's parameter-search policies rather than estimating outcomes
from shots: p=1 uses deterministic grids and p=2 uses the current fixed-seed
candidate search in :mod:`src.qaoa_solver`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qaoa_solver import solve_xy_qaoa
from src.qubo import matrix_to_qubo_dict


K = 4
SETTINGS = (
    {
        "id": "p1_grid_25",
        "label": "p=1, 5x5 grid",
        "reps": 1,
        "maxiter": 25,
        "search": "deterministic 5x5 grid on gamma,beta in [0.05, 0.8]",
    },
    {
        "id": "p1_grid_49",
        "label": "p=1, 7x7 grid",
        "reps": 1,
        "maxiter": 49,
        "search": "deterministic 7x7 grid on gamma,beta in [0.05, 0.8]",
    },
    {
        "id": "p2_random_49",
        "label": "p=2, 49 candidates",
        "reps": 2,
        "maxiter": 49,
        "search": "one [0.3,...,0.3] candidate plus 48 fixed-seed (42) uniform candidates in [0.05, 0.8]^(4)",
    },
)


def materials_qubo(n_blocks: int, seed: int = 123) -> np.ndarray:
    """Scalable BB-1 family; its N=16 definition matches the direct BB-1 model."""
    rng = np.random.RandomState(seed)
    n_qubits = n_blocks * K
    qubo = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for left in range(n_qubits):
        for right in range(left, n_qubits):
            if left == right:
                qubo[left, right] = rng.uniform(-2.0, 1.0)
                continue
            left_block, right_block = left // K, right // K
            if (left_block, right_block) in {(0, 1), (2, 3)}:
                qubo[left, right] = rng.uniform(-6.0, 2.0)
            elif left_block != right_block:
                qubo[left, right] = rng.uniform(-1.0, 1.0)
    return qubo


def random_qubo(n_qubits: int, seed: int = 42) -> np.ndarray:
    """Scalable BB-2 family; its N=16 definition matches the direct BB-2 model."""
    rng = np.random.RandomState(seed)
    qubo = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for left in range(n_qubits):
        for right in range(left, n_qubits):
            qubo[left, right] = rng.uniform(-1.5, 1.5) if left == right else rng.uniform(-1.0, 1.0)
    return qubo


def feasible_patterns(block_sizes: list[int]) -> np.ndarray:
    return np.asarray(
        [np.concatenate(parts) for parts in product(*(np.eye(size) for size in block_sizes))], dtype=np.float64
    )


def pattern_key(pattern: np.ndarray) -> str:
    return "".join(str(int(value)) for value in pattern)


def atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate_setting(qubo: np.ndarray, block_sizes: list[int], setting: dict[str, Any]) -> dict[str, Any]:
    patterns = feasible_patterns(block_sizes)
    energies = np.einsum("ni,ij,nj->n", patterns, qubo, patterns)
    exact_minimum = float(energies.min())
    optimum_mask = np.isclose(energies, exact_minimum, rtol=0.0, atol=1e-10)

    start = time.perf_counter()
    result = solve_xy_qaoa(
        matrix_to_qubo_dict(qubo), block_sizes, reps=int(setting["reps"]), maxiter=int(setting["maxiter"])
    )
    runtime = time.perf_counter() - start
    probabilities = np.asarray([result["state_probabilities"][pattern_key(pattern)] for pattern in patterns])
    probability_sum = float(probabilities.sum())
    expected_objective = float(np.dot(probabilities, energies))
    return {
        "reps": setting["reps"],
        "maxiter": setting["maxiter"],
        "search": setting["search"],
        "selected_parameters": result["parameters"],
        "exact_minimum": exact_minimum,
        "number_of_optima": int(optimum_mask.sum()),
        "p_opt": float(probabilities[optimum_mask].sum()),
        "expected_objective": expected_objective,
        "expected_gap": float(expected_objective - exact_minimum),
        "feasibility_rate": float(result["feasibility_rate"]),
        "feasible_probability_sum": probability_sum,
        "runtime_sec": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20], help="multiples of four, up to N=20")
    args = parser.parse_args()
    sizes = list(dict.fromkeys(args.sizes))
    if any(n < 4 or n % K for n in sizes):
        raise ValueError("--sizes must contain positive multiples of four.")
    if any(n > 20 for n in sizes):
        raise ValueError("This exact full-state benchmark is deliberately limited to N <= 20.")

    experiment_root = Path(__file__).resolve().parents[1]
    output = experiment_root / "json" / "results_xy_qaoa_accuracy_scaling.json"
    data: dict[str, Any] = {
        "metadata": {
            "experiment": "OpenQARP FM-XY-QAOA parameter-search and qubit scaling",
            "backend": "OpenQARP/qarpx exact statevector",
            "mixer": "complete-graph RXX/RYY within each One-Hot block",
            "initial_state": "uniform superposition over One-Hot feasible patterns",
            "shots": None,
            "seed_policy": {"BB1_materials": 123, "BB2_random": 42, "p2_candidate_search": 42},
            "measurement_scope": "N <= 20 only; every reported result is an executed full-state simulation.",
            "warning": "p=2 uses the current fixed-seed random candidate policy, not a continuous optimizer; it is an implementation baseline, not an optimized p=2 result.",
        },
        "settings": list(SETTINGS),
        "results": {"BB1_materials": {}, "BB2_random": {}},
    }

    problems = (("BB1_materials", materials_qubo, 123), ("BB2_random", random_qubo, 42))
    for problem_id, generator, seed in problems:
        for n_qubits in sizes:
            block_sizes = [K] * (n_qubits // K)
            qubo = generator(n_qubits // K, seed) if problem_id == "BB1_materials" else generator(n_qubits, seed)
            record: dict[str, Any] = {
                "n_qubits": n_qubits,
                "block_sizes": block_sizes,
                "feasible_dimension": int(K ** len(block_sizes)),
                "full_state_dimension": int(2**n_qubits),
                "statevector_bytes": int((2**n_qubits) * np.dtype(np.complex128).itemsize),
                "settings": {},
            }
            print(f"[{problem_id}] N={n_qubits} (feasible={record['feasible_dimension']}, full={record['full_state_dimension']})", flush=True)
            for setting in SETTINGS:
                print(f"  - {setting['id']}", flush=True)
                measured = evaluate_setting(qubo, block_sizes, setting)
                record["settings"][setting["id"]] = measured
                print(
                    f"    Popt={measured['p_opt']:.6%}, gap={measured['expected_gap']:.6f}, "
                    f"feas={measured['feasibility_rate']:.8%}, time={measured['runtime_sec']:.2f}s",
                    flush=True,
                )
                data["results"][problem_id][str(n_qubits)] = record
                atomic_json_write(output, data)

    atomic_json_write(output, data)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
