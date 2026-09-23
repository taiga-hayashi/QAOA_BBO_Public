#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_optimization_results.py
experimet/optimization フォルダ内の実験結果 (results_two_bb_optimization.json) を読み込み、
学術論文グレードの 6 パネル総合比較プロットを生成するスクリプト。
文字被りを完全に排除し、バーおよび折れ線と注釈テキストが美しく調和するようレイアウトを最適化。
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "json", "results_two_bb_optimization.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Matplotlib スタイル設定 (英語フォントで完全統一)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11.5,
        'xtick.labelsize': 9.0,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.0,
        'figure.titlesize': 13.5,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
        'grid.alpha': 0.35,
        'grid.linestyle': '--'
    })

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 16.5), dpi=300)
    plt.subplots_adjust(hspace=0.58, wspace=0.25)

    # 配色設定
    c_fmqa_def = "#90caf9"   # FMQA Default (Light Blue)
    c_fmqa_adp = "#1565c0"   # FMQA Adaptive (Deep Blue)
    c_std = "#e53935"        # Standard QAOA (Red)
    c_xy = "#2e7d32"         # FM-XY-QAOA 提案手法 (Deep Green)
    c_opt = "#212121"        # Exact Optimum (Dark Charcoal)

    colors4 = [c_fmqa_def, c_fmqa_adp, c_std, c_xy]

    # =========================================================================
    # ROW 1: 制約充足率 (Feasibility Rate)
    # =========================================================================
    # (a) BB-1 多元触媒
    ax = axes[0, 0]
    d1 = data["BB1_materials"]["part1_direct"]
    lam1_adp = data["BB1_materials"]["lambda_adaptive"]
    methods_bb1 = [
        "FMQA (SA)\nDef $\\lambda=5.0$",
        f"FMQA (SA)\n[Opt $\\lambda={lam1_adp}$]",
        f"Std QAOA\n$\\lambda={lam1_adp}$",
        "FM-XY-QAOA\n[Proposed, $\\lambda=0$]"
    ]
    feas_bb1 = [
        d1["FMQA_default"]["feasibility_rate"] * 100,
        d1["FMQA_adaptive"]["feasibility_rate"] * 100,
        d1["Standard_QAOA"]["feasibility_rate"] * 100,
        d1["FM_XY_QAOA"]["feasibility_rate"] * 100
    ]
    bars = ax.bar(methods_bb1, feas_bb1, color=colors4, width=0.55, edgecolor='black', linewidth=0.8)
    ax.set_title("(a) [BB-1 Materials Catalyst] Feasibility Rate (%)", fontweight='bold')
    ax.set_ylabel("One-Hot Feasibility Rate (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis='y')

    labels_feas1 = [f"{v:.2f}%" if v < 1.0 else f"{v:.1f}%" for v in feas_bb1]
    for bar, lab in zip(bars, labels_feas1):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 2.0, lab, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # (b) BB-2 全結合ランダム
    ax = axes[0, 1]
    d2 = data["BB2_random"]["part1_direct"]
    lam2_adp = data["BB2_random"]["lambda_adaptive"]
    methods_bb2 = [
        "FMQA (SA)\nDef $\\lambda=5.0$",
        f"FMQA (SA)\n[Opt $\\lambda={lam2_adp}$]",
        f"Std QAOA\n$\\lambda={lam2_adp}$",
        "FM-XY-QAOA\n[Proposed, $\\lambda=0$]"
    ]
    feas_bb2 = [
        d2["FMQA_default"]["feasibility_rate"] * 100,
        d2["FMQA_adaptive"]["feasibility_rate"] * 100,
        d2["Standard_QAOA"]["feasibility_rate"] * 100,
        d2["FM_XY_QAOA"]["feasibility_rate"] * 100
    ]
    bars = ax.bar(methods_bb2, feas_bb2, color=colors4, width=0.55, edgecolor='black', linewidth=0.8)
    ax.set_title("(b) [BB-2 Fully-Connected Random] Feasibility Rate (%)", fontweight='bold')
    ax.set_ylabel("One-Hot Feasibility Rate (%)")
    ax.set_ylim(0, 115)
    ax.grid(axis='y')

    labels_feas2 = [f"{v:.2f}%" if v < 1.0 else f"{v:.1f}%" for v in feas_bb2]
    for bar, lab in zip(bars, labels_feas2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 2.0, lab, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # =========================================================================
    # ROW 2: 大域最適解到達確率 (Success Probability)
    # =========================================================================
    # (c) BB-1 多元触媒
    ax = axes[1, 0]
    p_opt_bb1 = [
        d1["FMQA_default"]["success_probability"] * 100,
        d1["FMQA_adaptive"]["success_probability"] * 100,
        d1["Standard_QAOA"]["success_probability"] * 100,
        d1["FM_XY_QAOA"]["success_probability"] * 100
    ]
    bars = ax.bar(methods_bb1, p_opt_bb1, color=colors4, width=0.55, edgecolor='black', linewidth=0.8)
    ax.set_title("(c) [BB-1 Materials Catalyst] Success Probability P(x*)", fontweight='bold')
    ax.set_ylabel("Ground State Hit Probability (%)")
    ax.set_ylim(0, 9.5)
    ax.grid(axis='y')

    labels_p1 = [f"{v:.3f}%" if v < 0.01 else f"{v:.2f}%" for v in p_opt_bb1]
    for bar, lab in zip(bars, labels_p1):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.2, lab, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # (d) BB-2 全結合ランダム
    ax = axes[1, 1]
    p_opt_bb2 = [
        d2["FMQA_default"]["success_probability"] * 100,
        d2["FMQA_adaptive"]["success_probability"] * 100,
        d2["Standard_QAOA"]["success_probability"] * 100,
        d2["FM_XY_QAOA"]["success_probability"] * 100
    ]
    bars = ax.bar(methods_bb2, p_opt_bb2, color=colors4, width=0.55, edgecolor='black', linewidth=0.8)
    ax.set_title("(d) [BB-2 Fully-Connected Random] Success Probability P(x*)", fontweight='bold')
    ax.set_ylabel("Ground State Hit Probability (%)")
    ax.set_ylim(0, 9.5)
    ax.grid(axis='y')

    labels_p2 = [f"{v:.3f}%" if v < 0.01 else f"{v:.2f}%" for v in p_opt_bb2]
    for bar, lab in zip(bars, labels_p2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.2, lab, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # =========================================================================
    # ROW 3: BBO 探索推移 (Best-found value curve)
    # =========================================================================
    # (e) BB-1 多元触媒 BBO
    ax = axes[2, 0]
    b1 = data["BB1_materials"]["part2_bbo"]
    cycles1 = list(range(len(b1["FMQA"]["best_history"])))
    n_cyc1 = len(cycles1) - 1
    marker_every1 = max(1, n_cyc1 // 5)
    ax.plot(cycles1, b1["FMQA"]["best_history"], marker='s', markevery=marker_every1, color=c_fmqa_adp, label="FMQA (SA, dynamic $\\lambda$)", linestyle='-')
    ax.plot(cycles1, b1["Standard_QAOA"]["best_history"], marker='^', markevery=marker_every1, color=c_std, label="Std QAOA (dynamic $\\lambda$)", linestyle='--')
    ax.plot(cycles1, b1["FM_XY_QAOA"]["best_history"], marker='o', markevery=marker_every1, color=c_xy, label='FM-XY-QAOA (Proposed)', linestyle='-')
    ax.axhline(data["BB1_materials"]["exact_min"], color=c_opt, linestyle=':', label=f"Global Optimum ({data['BB1_materials']['exact_min']:.2f})")
    ax.set_title(f"(e) [BB-1 Materials Catalyst] {n_cyc1}-Cycle BBO Trajectory", fontweight='bold')
    ax.set_xlabel(f"BBO Iteration Cycle (0-{n_cyc1})")
    ax.set_xlim(0, n_cyc1)
    ax.set_ylabel("Best Found Objective Value (Lower is Better)")
    ax.set_ylim(-17.5, -6.5)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.23), ncol=2, framealpha=0.95)

    # (f) BB-2 全結合ランダム BBO
    ax = axes[2, 1]
    b2 = data["BB2_random"]["part2_bbo"]
    cycles2 = list(range(len(b2["FMQA"]["best_history"])))
    n_cyc2 = len(cycles2) - 1
    marker_every2 = max(1, n_cyc2 // 5)
    ax.plot(cycles2, b2["FMQA"]["best_history"], marker='s', markevery=marker_every2, color=c_fmqa_adp, label="FMQA (SA, dynamic $\\lambda$)", linestyle='-')
    ax.plot(cycles2, b2["Standard_QAOA"]["best_history"], marker='^', markevery=marker_every2, color=c_std, label="Std QAOA (dynamic $\\lambda$)", linestyle='--')
    ax.plot(cycles2, b2["FM_XY_QAOA"]["best_history"], marker='o', markevery=marker_every2, color=c_xy, label='FM-XY-QAOA (Proposed)', linestyle='-')
    ax.axhline(data["BB2_random"]["exact_min"], color=c_opt, linestyle=':', label=f"Global Optimum ({data['BB2_random']['exact_min']:.2f})")
    ax.set_title(f"(f) [BB-2 Fully-Connected Random] {n_cyc2}-Cycle BBO Trajectory", fontweight='bold')
    ax.set_xlabel(f"BBO Iteration Cycle (0-{n_cyc2})")
    ax.set_xlim(0, n_cyc2)
    ax.set_ylabel("Best Found Objective Value (Lower is Better)")
    ax.set_ylim(-6.4, -3.8)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.23), ncol=2, framealpha=0.95)

    # 全体タイトル
    fig.suptitle("Comprehensive Benchmark on 2 BB Functions: FMQA (Default vs Adaptive $\\lambda$) vs Standard QAOA vs FM-XY-QAOA", 
                 fontsize=12.5, fontweight='bold', y=0.992)

    out_pdf = os.path.join(base_dir, "pdf", "two_bb_optimization_comparison.pdf")
    out_png = os.path.join(base_dir, "png", "two_bb_optimization_comparison.png")
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    print(f"★ グラフ生成完了 (文字被り修正済み):")
    print(f"   PDF: {out_pdf}")
    print(f"   PNG: {out_png}")

if __name__ == "__main__":
    generate_plots()
