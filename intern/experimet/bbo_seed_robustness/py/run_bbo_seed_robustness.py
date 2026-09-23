#!/usr/bin/env python3
import importlib.util
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
OPT_SCRIPT = os.path.join(os.path.dirname(BASE), "optimization", "py", "run_optimization_experiment.py")
spec = importlib.util.spec_from_file_location("optimization_experiment", OPT_SCRIPT)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def main():
    seeds = [11, 22, 33]
    cycles = 100
    blocks = [4, 4, 4, 4]
    patterns, _ = exp.get_all_feasible_patterns(blocks)
    problems = {
        "BB1_materials": exp.create_materials_design_ground_truth(blocks, seed=123),
        "BB2_random": exp.create_random_qubo_bb(16, seed=42),
    }
    methods = ["FMQA", "Standard QAOA", "FM-XY-QAOA"]
    results = {
        "metadata": {
            "seeds": seeds,
            "cycles": cycles,
            "block_sizes": blocks,
            "qaoa_backend": "OpenQARP 0.1.0 (via experimet/optimization/py/run_optimization_experiment.py)",
            "seed_scope": "Each outer seed changes both the initial design and the solver sampling seeds (seed*1000 + cycle)."
        },
        "problems": {}
    }

    for problem_name, Q in problems.items():
        opt_x, opt_val = exp.get_exact_minimum(Q, patterns)
        problem_result = {"exact_min": opt_val, "methods": {m: [] for m in methods}}
        for seed in seeds:
            for method in methods:
                t0 = time.time()
                result = exp.run_bbo_closed_loop(method, Q, blocks, opt_x, opt_val, num_cycles=cycles, seed=seed)
                result["runtime_sec"] = time.time() - t0
                result["seed"] = seed
                problem_result["methods"][method].append(result)
                print(problem_name, seed, method, result["final_best"], result["infeasible_count"])
        results["problems"][problem_name] = problem_result

    for problem in results["problems"].values():
        for runs in problem["methods"].values():
            finals = np.array([r["final_best"] for r in runs])
            failures = np.array([r["infeasible_count"] for r in runs])
            problem.setdefault("summary", {})[runs[0]["solver"]] = {
                "mean_final_best": float(finals.mean()),
                "std_final_best": float(finals.std(ddof=1)),
                "mean_candidate_failures": float(failures.mean()),
                "success_count": int(np.sum(np.isclose(finals, problem["exact_min"], atol=1e-10))),
            }

    json_path = os.path.join(BASE, "json", "results_bbo_seed_robustness.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    colors = {"FMQA": "#1565c0", "Standard QAOA": "#e53935", "FM-XY-QAOA": "#2e7d32"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=250)
    for ax, (problem_name, problem) in zip(axes, results["problems"].items()):
        x = np.arange(cycles + 1)
        for method, runs in problem["methods"].items():
            histories = np.array([r["best_history"] for r in runs])
            mean = histories.mean(axis=0)
            std = histories.std(axis=0, ddof=1)
            ax.plot(x, mean, label=method, color=colors[method], marker="o", markevery=20)
            ax.fill_between(x, mean - std, mean + std, color=colors[method], alpha=0.15)
        ax.axhline(problem["exact_min"], color="black", linestyle=":", label="Exact minimum")
        ax.set_xlabel("BBO cycle (0-100)")
        ax.set_xlim(0, cycles)
        ax.set_xticks(np.arange(0, cycles + 1, 20))
        ax.set_ylabel("Best objective value (mean ± SD)")
        ax.set_title(problem_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.subplots_adjust(bottom=0.28, wspace=0.28)
    fig.savefig(os.path.join(BASE, "png", "bbo_seed_robustness.png"), bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "pdf", "bbo_seed_robustness.pdf"), bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# BBO seed robustness validation",
        "",
        f"Seeds: {seeds}; cycles: {cycles}. Curves show mean ± sample SD.",
        "",
        "| Problem | Method | Mean final best | SD | Optimum hits | Mean candidate failures |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for problem_name, problem in results["problems"].items():
        for method, summary in problem["summary"].items():
            lines.append(f"| {problem_name} | {method} | {summary['mean_final_best']:.6f} | {summary['std_final_best']:.6f} | {summary['success_count']}/{len(seeds)} | {summary['mean_candidate_failures']:.2f} |")
    lines += ["", "This is a preliminary three-seed check. Each seed changes the initial design and solver sampling stream; more seeds are required for inferential statistics."]
    with open(os.path.join(BASE, "md", "bbo_seed_robustness_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
