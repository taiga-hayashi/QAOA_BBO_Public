#!/usr/bin/env python3
"""Measure coefficient-normalization effects in the current OpenQARP XY-QAOA."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCALING_PY = ROOT / "experimet" / "xy_qaoa_accuracy_scaling" / "py"
if str(SCALING_PY) not in sys.path:
    sys.path.insert(0, str(SCALING_PY))

from run_xy_qaoa_accuracy_scaling import feasible_patterns, materials_qubo, pattern_key, random_qubo
from src.qaoa_solver import solve_xy_qaoa
from src.qubo import matrix_to_qubo_dict, normalize_upper_triangular_qubo


K = 4
SETTINGS = (
    {"id": "p1_grid_49", "label": "p=1, 7x7 grid", "reps": 1, "maxiter": 49},
    {"id": "p2_random_49", "label": "p=2, 49 candidates", "reps": 2, "maxiter": 49},
)
NORMALIZATIONS = (
    {"id": "none", "label": "No normalization"},
    {"id": "max_abs", "label": "max |Q_ij| = 1"},
    {"id": "rms", "label": "RMS(nonzero Q_ij) = 1"},
)


def atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate(
    original_qubo: np.ndarray, block_sizes: list[int], setting: dict[str, Any], normalization: dict[str, str]
) -> dict[str, Any]:
    circuit_qubo, scale = normalize_upper_triangular_qubo(original_qubo, normalization["id"])
    patterns = feasible_patterns(block_sizes)
    raw_energies = np.einsum("ni,ij,nj->n", patterns, original_qubo, patterns)
    exact_minimum = float(raw_energies.min())
    optimum_mask = np.isclose(raw_energies, exact_minimum, rtol=0.0, atol=1e-10)

    started = time.perf_counter()
    result = solve_xy_qaoa(
        matrix_to_qubo_dict(circuit_qubo), block_sizes, reps=int(setting["reps"]), maxiter=int(setting["maxiter"])
    )
    runtime = time.perf_counter() - started
    probabilities = np.asarray([result["state_probabilities"][pattern_key(pattern)] for pattern in patterns])
    expected_raw = float(np.dot(probabilities, raw_energies))
    return {
        "normalization_scale": scale,
        "selected_parameters": result["parameters"],
        "exact_minimum_raw": exact_minimum,
        "p_opt": float(probabilities[optimum_mask].sum()),
        "expected_objective_raw": expected_raw,
        "expected_gap_raw": float(expected_raw - exact_minimum),
        "feasibility_rate": float(result["feasibility_rate"]),
        "feasible_probability_sum": float(probabilities.sum()),
        "runtime_sec": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20])
    args = parser.parse_args()
    sizes = list(dict.fromkeys(args.sizes))
    if any(n < 4 or n % K or n > 20 for n in sizes):
        raise ValueError("--sizes must be multiples of four in the exact N <= 20 range.")

    experiment_root = Path(__file__).resolve().parents[1]
    output = experiment_root / "json" / "results_xy_qaoa_normalization.json"
    data: dict[str, Any] = {
        "metadata": {
            "backend": "OpenQARP/qarpx exact statevector",
            "mixer": "current complete-graph within-block RXX/RYY XY mixer",
            "objective_evaluation": "always the original, unnormalized QUBO objective",
            "normalization_meaning": "only the QUBO passed to the cost Hamiltonian is divided by the reported positive scale",
            "scope": "N <= 20; all entries are executed simulations, without shot noise.",
            "p2_warning": "p=2 uses the current fixed-seed candidate search, not continuous parameter optimization.",
        },
        "settings": list(SETTINGS),
        "normalizations": list(NORMALIZATIONS),
        "results": {"BB1_materials": {}, "BB2_random": {}},
    }
    problems = (("BB1_materials", materials_qubo, 123), ("BB2_random", random_qubo, 42))
    for problem_id, generator, seed in problems:
        for n_qubits in sizes:
            block_sizes = [K] * (n_qubits // K)
            original_qubo = generator(n_qubits // K, seed) if problem_id == "BB1_materials" else generator(n_qubits, seed)
            record: dict[str, Any] = {"n_qubits": n_qubits, "block_sizes": block_sizes, "results": {}}
            print(f"[{problem_id}] N={n_qubits}", flush=True)
            for setting in SETTINGS:
                setting_results = {}
                for normalization in NORMALIZATIONS:
                    print(f"  - {setting['id']} / {normalization['id']}", flush=True)
                    measured = evaluate(original_qubo, block_sizes, setting, normalization)
                    setting_results[normalization["id"]] = measured
                    print(
                        f"    Popt={measured['p_opt']:.6%}, raw-gap={measured['expected_gap_raw']:.6f}, "
                        f"scale={measured['normalization_scale']:.6f}, time={measured['runtime_sec']:.2f}s",
                        flush=True,
                    )
                    record["results"][setting["id"]] = setting_results
                    data["results"][problem_id][str(n_qubits)] = record
                    atomic_json_write(output, data)
    atomic_json_write(output, data)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
