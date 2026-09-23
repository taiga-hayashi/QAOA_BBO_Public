#!/usr/bin/env python3
"""Rerun the direct N=16 comparison and update only ``part1_direct``.

This separates the short, 1,000-shot direct comparison from the resumable
100-cycle BBO runs so both are measured rather than inherited from a prior
backend-labelled JSON file.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "optimization" / "py" / "run_optimization_experiment.py"
spec = importlib.util.spec_from_file_location("optimization_experiment", SOURCE)
exp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp)


def record_direct(qubo: np.ndarray, optimum_x: np.ndarray, optimum: float, lam: float) -> dict:
    blocks = [4, 4, 4, 4]
    result: dict = {}

    solver = exp.FMQASolver(blocks, lambda_penalty=5.0)
    sample = solver.solve(qubo, num_reads=1000, seed=42)
    result["FMQA_default"] = {
        "lambda": 5.0, "feasibility_rate": sample["feasibility_rate"],
        "success_probability": float(sum(np.array_equal(x, optimum_x) for x in sample["samples"]) / 1000),
        "best_energy": sample["best_energy"], "optimality_gap": float(sample["best_energy"] - optimum),
        "mean_energy": sample["mean_energy"], "unique_ratio": sample["unique_ratio"],
    }

    start = time.perf_counter()
    solver = exp.FMQASolver(blocks, lambda_penalty=lam)
    sample = solver.solve(qubo, num_reads=1000, seed=42)
    result["FMQA_adaptive"] = {
        "lambda": lam, "feasibility_rate": sample["feasibility_rate"],
        "success_probability": float(sum(np.array_equal(x, optimum_x) for x in sample["samples"]) / 1000),
        "best_energy": sample["best_energy"], "optimality_gap": float(sample["best_energy"] - optimum),
        "mean_energy": sample["mean_energy"], "unique_ratio": sample["unique_ratio"],
        "runtime_sec": time.perf_counter() - start,
    }

    start = time.perf_counter()
    solver = exp.StandardQAOASolver(blocks, lambda_penalty=lam)
    sample = solver.solve(qubo, num_shots=1000, seed=42)
    index = np.where(np.all(sample["all_bits"] == optimum_x, axis=1))[0][0]
    result["Standard_QAOA"] = {
        "lambda": lam, "feasibility_rate": sample["feasibility_rate"],
        "success_probability": float(sample["probabilities"][index]),
        "best_energy": sample["best_energy"], "optimality_gap": float(sample["best_energy"] - optimum),
        "mean_energy": sample["mean_energy"], "unique_ratio": sample["unique_ratio"],
        "runtime_sec": time.perf_counter() - start,
    }

    start = time.perf_counter()
    solver = exp.FMXYQAOASolver(blocks)
    sample = solver.solve(qubo, num_shots=1000, seed=42)
    index = np.where(np.all(sample["patterns"] == optimum_x, axis=1))[0][0]
    result["FM_XY_QAOA"] = {
        "lambda": 0.0, "feasibility_rate": sample["feasibility_rate"],
        "success_probability": float(sample["probabilities"][index]),
        "best_energy": sample["best_energy"], "optimality_gap": float(sample["best_energy"] - optimum),
        "mean_energy": sample["mean_energy"], "unique_ratio": sample["unique_ratio"],
        "runtime_sec": time.perf_counter() - start,
    }
    return result


def main() -> None:
    blocks = [4, 4, 4, 4]
    patterns, _ = exp.get_all_feasible_patterns(blocks)
    problems = {
        "BB1_materials": exp.create_materials_design_ground_truth(blocks, seed=123),
        "BB2_random": exp.create_random_qubo_bb(16, seed=42),
    }
    output = ROOT / "optimization" / "json" / "results_two_bb_optimization.json"
    data = json.loads(output.read_text(encoding="utf-8"))
    for name, qubo in problems.items():
        optimum_x, optimum = exp.get_exact_minimum(qubo, patterns)
        lam = exp.compute_theoretical_adaptive_lambda(qubo, blocks, safety_factor=1.25)
        data[name]["part1_direct"] = record_direct(qubo, optimum_x, optimum, lam)
        print(f"completed {name}", flush=True)
    data["metadata"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["metadata"]["direct_execution"] = "Direct comparison rerun with OpenQARP-native p=1 circuits and 1,000 sampled shots."
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
