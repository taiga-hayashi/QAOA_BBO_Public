#!/usr/bin/env python3
"""Render one figure per metric from the XY-QAOA accuracy/scaling JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STYLES = {
    "p1_grid_25": {"label": "p=1, 5x5 grid", "color": "#1f77b4", "marker": "o"},
    "p1_grid_49": {"label": "p=1, 7x7 grid", "color": "#ff7f0e", "marker": "s"},
    "p2_random_49": {"label": "p=2, 49 candidates", "color": "#2ca02c", "marker": "^"},
}


def save(fig: plt.Figure, root: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(root / extension / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric(data: dict, problem_id: str, metric: str, ylabel: str, stem: str, logarithmic: bool = False) -> None:
    records = data["results"][problem_id]
    sizes = np.asarray(sorted(map(int, records)), dtype=int)
    fig, ax = plt.subplots(figsize=(6.7, 4.25), constrained_layout=True)
    for setting_id, style in STYLES.items():
        values = [records[str(size)]["settings"][setting_id][metric] for size in sizes]
        ax.plot(sizes, values, linewidth=2.0, markersize=6.5, **style)
    ax.set_xlabel("Number of qubits, N")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sizes)
    ax.grid(True, alpha=0.28)
    if logarithmic:
        ax.set_yscale("log")
    ax.legend(loc="best", frameon=True)
    save(fig, Path(__file__).resolve().parents[1], stem)


def plot_runtime(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.25), constrained_layout=True)
    for problem_id, linestyle in (("BB1_materials", "-"), ("BB2_random", "--")):
        records = data["results"][problem_id]
        sizes = np.asarray(sorted(map(int, records)), dtype=int)
        for setting_id, style in STYLES.items():
            values = [records[str(size)]["settings"][setting_id]["runtime_sec"] for size in sizes]
            ax.plot(sizes, values, linestyle=linestyle, linewidth=1.8, markersize=5.5, color=style["color"], marker=style["marker"], label=f"{problem_id}: {style['label']}")
    ax.set_xlabel("Number of qubits, N")
    ax.set_ylabel("Wall-clock time per solve (s)")
    ax.set_xticks(sorted({int(size) for records in data["results"].values() for size in records}))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", fontsize=7.7, ncols=2, frameon=True)
    save(fig, Path(__file__).resolve().parents[1], "xy_qaoa_runtime_vs_qubits")


def plot_memory(data: dict) -> None:
    records = data["results"]["BB1_materials"]
    sizes = np.asarray(sorted(map(int, records)), dtype=int)
    bytes_used = np.asarray([records[str(size)]["statevector_bytes"] for size in sizes], dtype=float)
    fig, ax = plt.subplots(figsize=(6.7, 4.25), constrained_layout=True)
    ax.plot(sizes, bytes_used / 2**20, color="#6a3d9a", marker="D", linewidth=2.0, label="complex128 statevector")
    ax.set_xlabel("Number of qubits, N")
    ax.set_ylabel("One statevector memory (MiB)")
    ax.set_xticks(sizes)
    ax.set_yscale("log", base=2)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", frameon=True)
    save(fig, Path(__file__).resolve().parents[1], "xy_qaoa_statevector_memory_vs_qubits")


def write_summary(data: dict) -> None:
    root = Path(__file__).resolve().parents[1]
    lines = [
        "# Current OpenQARP FM-XY-QAOA: accuracy and qubit scaling",
        "",
        "All entries are executed, noise-free OpenQARP/`qarpx` full-state simulations. The mixer is the current complete-graph within-block RXX/RYY mixer and the initial state is uniform over the One-Hot feasible subspace.",
        "",
        "`p=2, 49 candidates` is not a continuous p=2 optimizer: it is the project's current fixed-seed random candidate policy. Thus it tests the current implementation as-is, not the best attainable p=2 performance.",
        "",
    ]
    for problem_id, records in data["results"].items():
        lines += [f"## {problem_id}", "", "| N | setting | P(opt) | expected gap | feasibility | runtime (s) |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
        for size in sorted(map(int, records)):
            record = records[str(size)]
            for setting_id, style in STYLES.items():
                result = record["settings"][setting_id]
                lines.append(
                    f"| {size} | {style['label']} | {result['p_opt']:.6%} | {result['expected_gap']:.6f} | "
                    f"{result['feasibility_rate']:.6%} | {result['runtime_sec']:.3f} |"
                )
        lines.append("")
    lines += [
        "## Scope",
        "",
        "The reported scaling is implementation/runtime scaling for this Mac and this full-state backend. It is not hardware scaling, a proof of quantum advantage, or a claim about other encodings. All comparisons at a fixed N use the same generated instance; cross-N objective magnitudes are not compared because the generated QUBO changes with N.",
        "",
        "The full statevector is retained even though the XY mixer preserves the One-Hot subspace. Therefore the memory curve is `16 * 2^N` bytes for a complex128 statevector; the feasible-space dimension is only `4^(N/4)`.",
    ]
    (root / "md" / "xy_qaoa_accuracy_scaling_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "json" / "results_xy_qaoa_accuracy_scaling.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    for problem_id, short in (("BB1_materials", "bb1_materials"), ("BB2_random", "bb2_random")):
        plot_metric(data, problem_id, "p_opt", "Ground-state probability, P(opt)", f"xy_qaoa_popt_vs_qubits_{short}")
        plot_metric(data, problem_id, "expected_gap", "Expected objective gap", f"xy_qaoa_expected_gap_vs_qubits_{short}", logarithmic=True)
    plot_runtime(data)
    plot_memory(data)
    write_summary(data)


if __name__ == "__main__":
    main()
