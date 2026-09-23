#!/usr/bin/env python3
"""Re-render the BBO seed-robustness figure from its validated result JSON."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def main():
    with open(os.path.join(BASE, "json", "results_bbo_seed_robustness.json"), encoding="utf-8") as f:
        data = json.load(f)
    cycles = data["metadata"]["cycles"]
    colors = {"FMQA": "#1565c0", "Standard QAOA": "#e53935", "FM-XY-QAOA": "#2e7d32"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=250)
    for ax, (problem_name, problem) in zip(axes, data["problems"].items()):
        x = np.arange(cycles + 1)
        for method, runs in problem["methods"].items():
            histories = np.asarray([run["best_history"] for run in runs])
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


if __name__ == "__main__":
    main()
