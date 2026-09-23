#!/usr/bin/env python3
import importlib.util
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OPT_SCRIPT = os.path.join(os.path.dirname(BASE), "optimization", "py", "run_optimization_experiment.py")
spec = importlib.util.spec_from_file_location("optimization_experiment", OPT_SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def initial_surrogate(Q, blocks, opt_x, seed=42):
    patterns, _ = exp.get_all_feasible_patterns(blocks)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(patterns), size=10, replace=False)
    X, y = [], []
    for idx in indices:
        p = patterns[idx]
        if not np.array_equal(p, opt_x):
            X.append(p)
            y.append(exp.evaluate_bb(p, Q))
        if len(X) == 5:
            break
    return exp.SimpleSurrogateModel(16, alpha=0.05).fit(np.array(X), np.array(y)), X


def main():
    blocks = [4, 4, 4, 4]
    patterns, _ = exp.get_all_feasible_patterns(blocks)
    shots = [100, 500, 1000, 5000, 10000, 50000]
    problems = {
        "BB1_materials": exp.create_materials_design_ground_truth(blocks, 123),
        "BB2_random": exp.create_random_qubo_bb(16, 42),
    }
    output = {
        "metadata": {
            "shots": shots,
            "initial_seed": 42,
            "qaoa_backend": "OpenQARP 0.1.0 (via experimet/optimization/py/run_optimization_experiment.py)",
        },
        "problems": {},
    }

    for name, Q in problems.items():
        opt_x, _ = exp.get_exact_minimum(Q, patterns)
        Qs, seen = initial_surrogate(Q, blocks, opt_x)
        lam = exp.compute_theoretical_adaptive_lambda(Qs, blocks, 1.25)
        solver = exp.StandardQAOASolver(blocks, lam)
        res = solver.solve(Qs, num_shots=1, seed=1)
        unseen = np.array([tuple(x) not in {tuple(v) for v in seen} for x in res["all_bits"]])
        valid_unseen = res["feasible_mask"] & unseen
        p_valid = float(res["probabilities"][valid_unseen].sum())
        records = []
        for s in shots:
            records.append({
                "shots": s,
                "prob_at_least_one_unseen_feasible": float(1.0 - (1.0 - p_valid) ** s),
                "expected_unseen_feasible_samples": float(s * p_valid),
            })
        output["problems"][name] = {"lambda": lam, "single_shot_probability": p_valid, "records": records}

    with open(os.path.join(BASE, "json", "results_shot_budget_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if os.environ.get("OPENQARP_SKIP_PLOT") == "1":
        print("OpenQARP shot-budget data saved; plotting deferred.")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=250)
    for name, result in output["problems"].items():
        ax.plot(shots, [r["prob_at_least_one_unseen_feasible"] * 100 for r in result["records"]], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Shot budget")
    ax.set_ylabel("Probability of at least one unseen feasible candidate (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(os.path.join(BASE, "png", "shot_budget_sensitivity.png"), bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "pdf", "shot_budget_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)

    lines = ["# Standard QAOA shot-budget sensitivity", "", "Analytical probability based on the optimized initial-cycle state distribution.", ""]
    for name, result in output["problems"].items():
        lines += [f"## {name}", "", f"Single-shot probability of an unseen feasible candidate: {result['single_shot_probability']:.8f}", "", "| Shots | Expected valid unseen samples | Probability of at least one |", "|---:|---:|---:|"]
        for r in result["records"]:
            lines.append(f"| {r['shots']} | {r['expected_unseen_feasible_samples']:.4f} | {100*r['prob_at_least_one_unseen_feasible']:.3f}% |")
        lines.append("")
    with open(os.path.join(BASE, "md", "shot_budget_sensitivity_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
