#!/usr/bin/env python3
"""Render the shot-budget figure from the OpenQARP-updated JSON result."""

import json
import os

import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)


def main():
    with open(os.path.join(BASE, "json", "results_shot_budget_sensitivity.json"), encoding="utf-8") as f:
        data = json.load(f)
    if "OpenQARP" not in data.get("metadata", {}).get("qaoa_backend", ""):
        raise RuntimeError("Refusing to plot: shot-budget JSON was not generated with OpenQARP.")

    shots = data["metadata"]["shots"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=250)
    for name, result in data["problems"].items():
        ax.plot(
            shots,
            [record["prob_at_least_one_unseen_feasible"] * 100 for record in result["records"]],
            marker="o",
            label=name,
        )
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


if __name__ == "__main__":
    main()
