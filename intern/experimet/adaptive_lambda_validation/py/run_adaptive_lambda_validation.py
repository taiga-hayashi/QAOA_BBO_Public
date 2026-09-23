#!/usr/bin/env python3
import importlib.util
import json
import os

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OPT_SCRIPT = os.path.join(os.path.dirname(BASE), "optimization", "py", "run_optimization_experiment.py")
spec = importlib.util.spec_from_file_location("optimization_experiment", OPT_SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def exact_required_lambda(Q, blocks, patterns, all_bits, penalties):
    _, optimum = exp.get_exact_minimum(Q, patterns)
    energies = np.sum((all_bits @ Q) * all_bits, axis=1)
    invalid = penalties > 0
    required = np.maximum(0.0, (optimum - energies[invalid]) / penalties[invalid])
    return float(required.max()), float(optimum)


def main():
    blocks = [4, 4, 4, 4]
    patterns, _ = exp.get_all_feasible_patterns(blocks)
    all_bits = np.array([[(i >> (15-b)) & 1 for b in range(16)] for i in range(2**16)], dtype=float)
    penalties = np.zeros(len(all_bits))
    for start in range(0, 16, 4):
        penalties += (all_bits[:, start:start+4].sum(axis=1) - 1.0) ** 2
    seeds = list(range(50))
    records = []
    for seed in seeds:
        Q = exp.create_random_qubo_bb(16, seed)
        exact_required, optimum = exact_required_lambda(Q, blocks, patterns, all_bits, penalties)
        heuristic = exp.compute_theoretical_adaptive_lambda(Q, blocks, 1.25)
        records.append({
            "seed": seed,
            "exact_required_lambda": exact_required,
            "heuristic_lambda": heuristic,
            "ratio": heuristic / exact_required if exact_required > 0 else None,
            "is_sufficient": bool(heuristic + 1e-12 >= exact_required),
            "exact_min": optimum,
        })
    ratios = np.array([r["ratio"] for r in records if r["ratio"] is not None])
    summary = {
        "num_instances": len(records),
        "sufficient_count": sum(r["is_sufficient"] for r in records),
        "ratio_mean": float(ratios.mean()),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
    }
    output = {"metadata": {"block_sizes": blocks, "seeds": seeds}, "summary": summary, "records": records}
    with open(os.path.join(BASE, "json", "results_adaptive_lambda_validation.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    exact = np.array([r["exact_required_lambda"] for r in records])
    heuristic = np.array([r["heuristic_lambda"] for r in records])
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=250)
    ax.scatter(exact, heuristic, color="#1565c0", alpha=0.8)
    lim = max(exact.max(), heuristic.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", linestyle=":", label="heuristic = exact lower bound")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Exact minimum sufficient penalty")
    ax.set_ylabel("Heuristic adaptive penalty")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(BASE, "png", "adaptive_lambda_validation.png"), bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "pdf", "adaptive_lambda_validation.pdf"), bbox_inches="tight")
    plt.close(fig)

    failed = [r for r in records if not r["is_sufficient"]]
    failed_text = ", ".join(
        f"seed {r['seed']} (exact={r['exact_required_lambda']:.3f}, heuristic={r['heuristic_lambda']:.3f})"
        for r in failed
    ) or "none"
    text = f"""# Adaptive penalty validation

The heuristic penalty was compared with the exact minimum sufficient penalty over {summary['num_instances']} random N=16 instances.

- Sufficient instances: {summary['sufficient_count']}/{summary['num_instances']}
- Mean heuristic/exact ratio: {summary['ratio_mean']:.3f}
- Minimum ratio: {summary['ratio_min']:.3f}
- Maximum ratio: {summary['ratio_max']:.3f}
- Insufficient cases: {failed_text}

A ratio above 1 means that the heuristic is sufficient but conservative. This experiment validates sufficiency only for the tested instance family and scale; it does not prove a universal bound.
"""
    with open(os.path.join(BASE, "md", "adaptive_lambda_validation_summary.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
