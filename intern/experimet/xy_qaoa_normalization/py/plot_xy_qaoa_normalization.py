#!/usr/bin/env python3
"""Render separate normalization-sensitivity plots from executed XY-QAOA data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "none": {"label": "No normalization", "color": "#1f77b4", "marker": "o"},
    "max_abs": {"label": "max-abs(Q_ij) = 1", "color": "#ff7f0e", "marker": "s"},
    "rms": {"label": "RMS(nonzero Q_ij) = 1", "color": "#2ca02c", "marker": "^"},
}


def save(fig: plt.Figure, root: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(root / extension / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric(data: dict, problem_id: str, setting_id: str, metric: str, ylabel: str, stem: str, logarithmic: bool) -> None:
    records = data["results"][problem_id]
    sizes = np.asarray(sorted(map(int, records)), dtype=int)
    fig, ax = plt.subplots(figsize=(6.7, 4.25), constrained_layout=True)
    for normalization_id, style in STYLES.items():
        values = [records[str(size)]["results"][setting_id][normalization_id][metric] for size in sizes]
        ax.plot(sizes, values, linewidth=2.0, markersize=6.5, **style)
    ax.set_xlabel("Number of qubits, N")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.28)
    if logarithmic:
        ax.set_yscale("log")
    ax.legend(loc="best", frameon=True)
    save(fig, Path(__file__).resolve().parents[1], stem)


def write_summary(data: dict) -> None:
    root = Path(__file__).resolve().parents[1]
    lines = [
        "# XY-QAOA QUBO-coefficient normalization sensitivity",
        "",
        "Each QAOA cost Hamiltonian used either the raw QUBO, Q/max|Q_ij|, or Q/RMS(nonzero Q_ij). Every reported P(opt) and expected gap is evaluated afterwards using the same original, unnormalized QUBO. A positive global scaling does not change the exact discrete optimum; it can change this finite angle search's state distribution.",
        "",
        "All entries are noise-free OpenQARP/`qarpx` full-state simulations with the current complete-graph within-block RXX/RYY mixer and One-Hot initial state. `p=2` means the present fixed-seed 49-candidate policy, not a continuously optimized p=2 circuit.",
        "",
    ]
    for problem_id, records in data["results"].items():
        lines += [f"## {problem_id}", ""]
        for setting in data["settings"]:
            setting_id = setting["id"]
            lines += [f"### {setting['label']}", "", "| N | normalization | scale | P(opt) | raw expected gap | feasibility | runtime (s) |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
            for size in sorted(map(int, records)):
                result_group = records[str(size)]["results"][setting_id]
                for normalization_id, style in STYLES.items():
                    result = result_group[normalization_id]
                    lines.append(
                        f"| {size} | {style['label']} | {result['normalization_scale']:.6f} | {result['p_opt']:.6%} | "
                        f"{result['expected_gap_raw']:.6f} | {result['feasibility_rate']:.6%} | {result['runtime_sec']:.3f} |"
                    )
            lines.append("")
    lines += [
        "## Interpretation boundary",
        "",
        "Compare normalization choices only within the same problem and N. Objective magnitudes across N are not comparable because each N is a separately generated QUBO. This is a parameter-scale sensitivity result for the current classical state-vector implementation, not a hardware result or a general claim for other encodings.",
    ]
    (root / "md" / "xy_qaoa_normalization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "json" / "results_xy_qaoa_normalization.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    for problem_id, short_problem in (("BB1_materials", "bb1_materials"), ("BB2_random", "bb2_random")):
        for setting in data["settings"]:
            short_setting = setting["id"]
            plot_metric(data, problem_id, short_setting, "p_opt", "Ground-state probability, P(opt)", f"xy_qaoa_normalization_popt_{short_problem}_{short_setting}", False)
            plot_metric(data, problem_id, short_setting, "expected_gap_raw", "Expected objective gap (raw QUBO)", f"xy_qaoa_normalization_gap_{short_problem}_{short_setting}", True)
    write_summary(data)


if __name__ == "__main__":
    main()
