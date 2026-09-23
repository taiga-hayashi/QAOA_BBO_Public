#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_xy_qaoa_bbo_scaling.py
100-cycle BBO スケーリング実験のプロット生成スクリプト

生成する図:
  1. Trajectory 個別プロット:
     - BB-1: N=8, 12, 16, 20 それぞれの 100 サイクル最適化推移
     - BB-2: N=8, 12, 16, 20 それぞれの 100 サイクル最適化推移
     - 統合図 (2x2) および独立パネル図 (pdf/panels/, png/panels/)
  2. N スケーリング比較プロット:
     - 最終到達エネルギー (Final Best Objective Value vs N)
     - 最適値ギャップ (Optimality Gap vs N)
     - 候補探索失敗数 (Candidate Failures vs N)
     - 平均計算時間 (Mean Runtime vs N)
     - 統合図 (2x2) および独立パネル図 (pdf/panels/, png/panels/)

agent.md 規約遵守:
  - タイトルなし (plt.title 禁止)
  - 不要な注釈・コメントボックス・テキストバナーの完全排除
  - 独立パネルは新規 Matplotlib Figure で生成
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# intern/ およびプロジェクトルートのパスを通す
CURRENT_FILE = Path(__file__).resolve()
if CURRENT_FILE.parents[3].name == "intern":
    INTERN_DIR = CURRENT_FILE.parents[3]
    ROOT = CURRENT_FILE.parents[4]
else:
    ROOT = CURRENT_FILE.parents[3]
    INTERN_DIR = ROOT / "intern"

EXP_DIR = INTERN_DIR / "experimet" / "xy_qaoa_bbo_scaling"

# Matplotlib のスタイル設定 (論文品質、クリーン)
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
})

# 手法別プロット設定
METHOD_STYLES = {
    "FM-XY-QAOA": {
        "label": "FM-XY-QAOA (p=2, 49 cands)",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
    },
    "Standard QAOA": {
        "label": "Standard QAOA (p=1, adapt λ)",
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "s",
    },
    "FMQA": {
        "label": "FMQA (SA, adapt λ)",
        "color": "#2ca02c",
        "linestyle": "-.",
        "marker": "^",
    },
}


def load_results(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"Result JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# 1. Trajectory プロット (個別パネル & 統合図)
# ==============================================================================

def plot_single_trajectory(
    ax: plt.Axes,
    scale_data: Dict[str, Any],
    methods: list[str],
    show_legend: bool = True,
) -> None:
    N = scale_data["N"]
    exact_min = scale_data["exact_min"]

    # 真の大域最適値
    ax.axhline(exact_min, color="#d62728", linestyle=":", linewidth=1.5, label="Exact Minimum")

    for method in methods:
        if method not in scale_data["methods"]:
            continue
        res = scale_data["methods"][method]
        history = res["best_history"]
        cycles = np.arange(len(history))
        style = METHOD_STYLES.get(method, {"label": method, "color": "black", "linestyle": "-"})
        ax.plot(cycles, history, label=style["label"], color=style["color"], linestyle=style["linestyle"])

    ax.set_xlabel("BBO Cycle")
    ax.set_ylabel("Best Objective Value")
    ax.grid(True)
    if show_legend:
        ax.legend(frameon=True, loc="upper right")


def generate_trajectory_plots(results: Dict[str, Any]) -> None:
    pdf_dir = EXP_DIR / "pdf"
    png_dir = EXP_DIR / "png"
    pdf_panels = pdf_dir / "panels"
    png_panels = png_dir / "panels"

    for prob_name, prob_data in results["problems"].items():
        prob_short = "bb1" if "materials" in prob_name.lower() else "bb2"
        scales = sorted([int(k) for k in prob_data["scales"].keys()])

        # (1) 独立パネル図の出力 (新規 Figure)
        for N in scales:
            scale_data = prob_data["scales"][str(N)]
            fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
            plot_single_trajectory(ax, scale_data, list(METHOD_STYLES.keys()), show_legend=True)

            panel_base = f"trajectory_{prob_short}_N{N}"
            fig.savefig(pdf_panels / f"{panel_base}.pdf")
            fig.savefig(png_panels / f"{panel_base}.png", dpi=300)
            plt.close(fig)

        # (2) 2x2 統合図の出力
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        axes_flat = axes.flatten()

        for idx, N in enumerate(scales[:4]):
            ax = axes_flat[idx]
            scale_data = prob_data["scales"][str(N)]
            plot_single_trajectory(ax, scale_data, list(METHOD_STYLES.keys()), show_legend=(idx == 0))
            # サブプロット識別のため軸内テキスト（注釈ボックスではなくラベルのみ）
            ax.text(0.04, 0.08, f"N = {N}", transform=ax.transAxes, fontsize=12, fontweight="bold")

        int_base = f"integrated_trajectory_{prob_short}"
        fig.savefig(pdf_dir / f"{int_base}.pdf")
        fig.savefig(png_dir / f"{int_base}.png", dpi=300)
        plt.close(fig)
        print(f"  [Trajectory] {prob_name} 統合図・独立パネル生成完了")


# ==============================================================================
# 2. N スケーリング比較プロット (個別パネル & 統合図)
# ==============================================================================

def generate_scaling_plots(results: Dict[str, Any]) -> None:
    pdf_dir = EXP_DIR / "pdf"
    png_dir = EXP_DIR / "png"
    pdf_panels = pdf_dir / "panels"
    png_panels = png_dir / "panels"

    for prob_name, prob_data in results["problems"].items():
        prob_short = "bb1" if "materials" in prob_name.lower() else "bb2"
        scales = sorted([int(k) for k in prob_data["scales"].keys()])
        methods = list(METHOD_STYLES.keys())

        # 指標データの収集
        final_bests: Dict[str, list[float]] = {m: [] for m in methods}
        opt_gaps: Dict[str, list[float]] = {m: [] for m in methods}
        failures: Dict[str, list[float]] = {m: [] for m in methods}
        runtimes: Dict[str, list[float]] = {m: [] for m in methods}
        exact_mins = []

        for N in scales:
            scale_data = prob_data["scales"][str(N)]
            exact_mins.append(scale_data["exact_min"])
            for m in methods:
                if m in scale_data["methods"]:
                    res = scale_data["methods"][m]
                    final_bests[m].append(res["final_best"])
                    opt_gaps[m].append(res["optimality_gap"])
                    failures[m].append(res["infeasible_count"])
                    runtimes[m].append(res["mean_cycle_time"])
                else:
                    final_bests[m].append(np.nan)
                    opt_gaps[m].append(np.nan)
                    failures[m].append(np.nan)
                    runtimes[m].append(np.nan)

        # ----------------------------------------------------------------------
        # 独立パネル 1: 最終到達エネルギー (Final Best vs N)
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
        ax.plot(scales, exact_mins, color="#d62728", linestyle=":", marker="x", label="Exact Minimum")
        for m in methods:
            style = METHOD_STYLES[m]
            ax.plot(scales, final_bests[m], label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"])
        ax.set_xlabel("Number of Qubits N")
        ax.set_ylabel("Final Best Objective Value")
        ax.set_xticks(scales)
        ax.grid(True)
        ax.legend(frameon=True)
        fig.savefig(pdf_panels / f"scaling_final_best_{prob_short}.pdf")
        fig.savefig(png_panels / f"scaling_final_best_{prob_short}.png", dpi=300)
        plt.close(fig)

        # ----------------------------------------------------------------------
        # 独立パネル 2: 最適値ギャップ (Optimality Gap vs N)
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
        for m in methods:
            style = METHOD_STYLES[m]
            ax.plot(scales, opt_gaps[m], label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"])
        ax.set_xlabel("Number of Qubits N")
        ax.set_ylabel("Optimality Gap")
        ax.set_xticks(scales)
        ax.grid(True)
        ax.legend(frameon=True)
        fig.savefig(pdf_panels / f"scaling_optimality_gap_{prob_short}.pdf")
        fig.savefig(png_panels / f"scaling_optimality_gap_{prob_short}.png", dpi=300)
        plt.close(fig)

        # ----------------------------------------------------------------------
        # 独立パネル 3: 探索失敗数 (Candidate Failures vs N)
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
        bar_width = 0.8
        x_indices = np.arange(len(scales))
        n_methods = len(methods)
        for idx, m in enumerate(methods):
            style = METHOD_STYLES[m]
            offset = (idx - (n_methods - 1) / 2) * (bar_width / n_methods)
            ax.bar(x_indices + offset, failures[m], width=bar_width / n_methods,
                   label=style["label"], color=style["color"], alpha=0.85)
        ax.set_xlabel("Number of Qubits N")
        ax.set_ylabel("Candidate Failures (out of 100 cycles)")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(scales)
        ax.grid(True, axis="y")
        ax.legend(frameon=True)
        fig.savefig(pdf_panels / f"scaling_candidate_failures_{prob_short}.pdf")
        fig.savefig(png_panels / f"scaling_candidate_failures_{prob_short}.png", dpi=300)
        plt.close(fig)

        # ----------------------------------------------------------------------
        # 独立パネル 4: 平均計算時間 (Runtime vs N)
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
        for m in methods:
            style = METHOD_STYLES[m]
            ax.plot(scales, runtimes[m], label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"])
        ax.set_xlabel("Number of Qubits N")
        ax.set_ylabel("Runtime per Cycle (s)")
        ax.set_yscale("log")
        ax.set_xticks(scales)
        ax.grid(True)
        ax.legend(frameon=True)
        fig.savefig(pdf_panels / f"scaling_runtime_{prob_short}.pdf")
        fig.savefig(png_panels / f"scaling_runtime_{prob_short}.png", dpi=300)
        plt.close(fig)

        # ----------------------------------------------------------------------
        # 2x2 統合図 (Scaling Comparison 4-in-1)
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

        # (0,0) Final Best
        ax = axes[0, 0]
        ax.plot(scales, exact_mins, color="#d62728", linestyle=":", marker="x", label="Exact Minimum")
        for m in methods:
            style = METHOD_STYLES[m]
            ax.plot(scales, final_bests[m], label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"])
        ax.set_xlabel("Number of Qubits N")
        ax.set_ylabel("Final Best Objective Value")
        ax.set_xticks(scales)
        ax.grid(True)
        ax.legend(frameon=True)

        # (0,1) Gap
        ax = axes[0, 1]
        for m in methods:
            style = METHOD_STYLES[m]
            ax.plot(scales, opt_gaps[m], label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"])
        ax.set_xlabel("Number of Qubits N")
        ax.set_ylabel("Optimality Gap")
        ax.set_xticks(scales)
        ax.grid(True)
        ax.legend(frameon=True)

        # (1,0) Failures
        ax = axes[1, 0]
        for idx, m in enumerate(methods):
            style = METHOD_STYLES[m]
            offset = (idx - (n_methods - 1) / 2) * (bar_width / n_methods)
            ax.bar(x_indices + offset, failures[m], width=bar_width / n_methods,
                   label=style["label"], color=style["color"], alpha=0.85)
        ax.set_xlabel("Number of Qubits N")
        ax.set_ylabel("Candidate Failures (out of 100 cycles)")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(scales)
        ax.grid(True, axis="y")
        ax.legend(frameon=True)

        # (1,1) Runtime
        ax = axes[1, 1]
        for m in methods:
            style = METHOD_STYLES[m]
            ax.plot(scales, runtimes[m], label=style["label"], color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"])
        ax.set_xlabel("Number of Qubits N")
        ax.set_ylabel("Runtime per Cycle (s)")
        ax.set_yscale("log")
        ax.set_xticks(scales)
        ax.grid(True)
        ax.legend(frameon=True)

        int_base = f"integrated_scaling_{prob_short}"
        fig.savefig(pdf_dir / f"{int_base}.pdf")
        fig.savefig(png_dir / f"{int_base}.png", dpi=300)
        plt.close(fig)
        print(f"  [Scaling] {prob_name} 統合図・独立パネル生成完了")


# ==============================================================================
# 3. メインルーチン
# ==============================================================================

def main():
    json_path = EXP_DIR / "json" / "results_xy_qaoa_bbo_scaling.json"
    if not json_path.exists():
        print(f"Error: {json_path} が存在しません。先に実験を実行してください。")
        sys.exit(1)

    print("=" * 80)
    print(" 100-cycle BBO スケーリング プロット生成")
    print(f" 読み込み元: {json_path}")
    print("=" * 80)

    results = load_results(json_path)

    generate_trajectory_plots(results)
    generate_scaling_plots(results)

    # 正規ツリー側の experimet/ にもコピー
    canonical_pdf = ROOT / "experimet" / "xy_qaoa_bbo_scaling" / "pdf"
    canonical_png = ROOT / "experimet" / "xy_qaoa_bbo_scaling" / "png"
    import shutil
    shutil.copytree(EXP_DIR / "pdf", canonical_pdf, dirs_exist_ok=True)
    shutil.copytree(EXP_DIR / "png", canonical_png, dirs_exist_ok=True)

    print("=" * 80)
    print(" 全プロットの描画および正規ツリーへのコピーが完了しました！")
    print("=" * 80)


if __name__ == "__main__":
    main()
