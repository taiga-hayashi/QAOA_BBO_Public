#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_std_qaoa_sensitivity.py
Standard QAOA におけるペナルティ係数 λ 感度解析結果
(results_std_qaoa_sensitivity.json) を読み込み、
学術論文仕様の 6 パネル総合比較プロットを生成するスクリプト。
文字被りを完全に排除し、見やすく洗練されたレイアウトを実現。
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "json", "results_std_qaoa_sensitivity.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # スタイル設定
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11.5,
        'xtick.labelsize': 9.0,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 8.8,
        'figure.titlesize': 13.0,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'lines.markersize': 5.5,
        'grid.alpha': 0.35,
        'grid.linestyle': '--'
    })

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 14.5), dpi=300)
    plt.subplots_adjust(hspace=0.78, wspace=0.25)

    # 配色
    c_std = "#d32f2f"      # Standard QAOA (Deep Red)
    c_xy = "#2e7d32"       # FM-XY-QAOA 提案手法 (Deep Green)
    c_fmqa = "#1976d2"     # FMQA Reference (Blue)
    c_opt = "#212121"      # Global Optimum (Dark Charcoal)
    c_random = "#9e9e9e"   # Random Sampling Ratio 0.39%

    d1 = data["BB1_materials"]
    d2 = data["BB2_random"]

    lams1 = [item["lambda"] for item in d1["scan_data"]]
    lams2 = [item["lambda"] for item in d2["scan_data"]]

    # =========================================================================
    # ROW 1: 制約充足率 (Feasibility Rate % vs λ)
    # =========================================================================
    # (a) BB-1 Materials Catalyst
    ax = axes[0, 0]
    feas_theo1 = [item["theoretical_feasibility"] * 100 for item in d1["scan_data"]]
    feas_emp1 = [item["empirical_feasibility"] * 100 for item in d1["scan_data"]]
    ax.plot(lams1, feas_theo1, color=c_std, marker='o', label='Std QAOA (Theoretical State Vector)')
    ax.scatter(lams1, feas_emp1, color="#ef5350", marker='^', alpha=0.8, s=40, label='Std QAOA (1500 Shots Sampled)')
    ax.axhline(100.0, color=c_xy, linestyle='-', linewidth=2.0, label='FM-XY-QAOA [Proposed] (100% Inherent, $\\lambda=0$)')
    ax.axhline(0.3906, color=c_random, linestyle=':', linewidth=1.2, label=r'Uniform Random Baseline ($256 / 2^{16} = 0.39\%$)')
    ax.set_title("(a) [BB-1 Materials Catalyst] Feasibility Rate vs Penalty $\\lambda$", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("One-Hot Feasibility Rate (%)")
    ax.set_ylim(-2, 118)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2, framealpha=0.92)

    # (b) BB-2 Random Interactions
    ax = axes[0, 1]
    feas_theo2 = [item["theoretical_feasibility"] * 100 for item in d2["scan_data"]]
    feas_emp2 = [item["empirical_feasibility"] * 100 for item in d2["scan_data"]]
    ax.plot(lams2, feas_theo2, color=c_std, marker='o', label='Std QAOA (Theoretical State Vector)')
    ax.scatter(lams2, feas_emp2, color="#ef5350", marker='^', alpha=0.8, s=40, label='Std QAOA (1500 Shots Sampled)')
    ax.axhline(100.0, color=c_xy, linestyle='-', linewidth=2.0, label='FM-XY-QAOA [Proposed] (100% Inherent, $\\lambda=0$)')
    ax.axhline(0.3906, color=c_random, linestyle=':', linewidth=1.2, label=r'Uniform Random Baseline ($256 / 2^{16} = 0.39\%$)')
    ax.set_title("(b) [BB-2 Fully-Connected Random] Feasibility Rate vs Penalty $\\lambda$", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("One-Hot Feasibility Rate (%)")
    ax.set_ylim(-2, 118)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2, framealpha=0.92)

    # =========================================================================
    # ROW 2: 真の大域最適解到達率 P(x*) vs λ
    # =========================================================================
    # (c) BB-1 Materials Catalyst P(x*)
    ax = axes[1, 0]
    p_opt_theo1 = [item["theoretical_p_opt"] * 100 for item in d1["scan_data"]]
    p_opt_emp1 = [item["empirical_p_opt"] * 100 for item in d1["scan_data"]]
    xy_p_opt1 = d1["fm_xy_qaoa_reference"]["success_probability"] * 100
    fmqa_p_opt1 = d1["fmqa_adaptive_reference"]["success_probability"] * 100

    ax.plot(lams1, p_opt_theo1, color=c_std, marker='o', label='Std QAOA $P(x^*)$ (Theoretical)')
    ax.scatter(lams1, p_opt_emp1, color="#ef5350", marker='^', alpha=0.8, s=40, label='Std QAOA (Sampled Hits)')
    ax.axhline(xy_p_opt1, color=c_xy, linestyle='-', linewidth=2.0, label=f'FM-XY-QAOA [Proposed] ({xy_p_opt1:.2f}%, $\\lambda=0$)')
    ax.axhline(fmqa_p_opt1, color=c_fmqa, linestyle='--', linewidth=1.5, label=f'FMQA (SA, Adaptive $\\lambda={d1["lambda_adaptive"]:.2f}$, {fmqa_p_opt1:.2f}%)')
    ax.set_title("(c) [BB-1 Materials Catalyst] Ground State Hit Probability $P(x^*)$", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("Ground State Probability $P(x^*)$ (%)")
    ax.set_ylim(-0.3, 9.2)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=1, framealpha=0.92)

    # (d) BB-2 Random Interactions P(x*)
    ax = axes[1, 1]
    p_opt_theo2 = [item["theoretical_p_opt"] * 100 for item in d2["scan_data"]]
    p_opt_emp2 = [item["empirical_p_opt"] * 100 for item in d2["scan_data"]]
    xy_p_opt2 = d2["fm_xy_qaoa_reference"]["success_probability"] * 100
    fmqa_p_opt2 = d2["fmqa_adaptive_reference"]["success_probability"] * 100

    ax.plot(lams2, p_opt_theo2, color=c_std, marker='o', label='Std QAOA $P(x^*)$ (Theoretical)')
    ax.scatter(lams2, p_opt_emp2, color="#ef5350", marker='^', alpha=0.8, s=40, label='Std QAOA (Sampled Hits)')
    ax.axhline(xy_p_opt2, color=c_xy, linestyle='-', linewidth=2.0, label=f'FM-XY-QAOA [Proposed] ({xy_p_opt2:.2f}%, $\\lambda=0$)')
    ax.axhline(fmqa_p_opt2, color=c_fmqa, linestyle='--', linewidth=1.5, label=f'FMQA (SA, Adaptive $\\lambda={d2["lambda_adaptive"]:.2f}$, {fmqa_p_opt2:.2f}%)')
    ax.set_title("(d) [BB-2 Fully-Connected Random] Ground State Hit Probability $P(x^*)$", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("Ground State Probability $P(x^*)$ (%)")
    ax.set_ylim(-0.3, 9.6)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=1, framealpha=0.92)

    # =========================================================================
    # ROW 3: 最良エネルギーおよび平均エネルギー推移
    # =========================================================================
    # (e) BB-1 Materials Catalyst Energy
    ax = axes[2, 0]
    best_e1 = [np.nan if item["best_feas_energy"] is None else item["best_feas_energy"] for item in d1["scan_data"]]
    mean_e1 = [np.nan if item["mean_feas_energy"] is None else item["mean_feas_energy"] for item in d1["scan_data"]]
    ax.plot(lams1, best_e1, color=c_std, marker='s', label='Std QAOA Best Feasible Found')
    ax.plot(lams1, mean_e1, color="#ff7043", linestyle='--', marker='^', alpha=0.8, label='Std QAOA Mean Feasible Energy')
    ax.axhline(d1["exact_min"], color=c_opt, linestyle=':', linewidth=1.8, label=f"Global Optimum $f^* = {d1['exact_min']:.2f}$")
    ax.set_title("(e) [BB-1 Materials Catalyst] Energy Spectrum of Feasible Samples", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("Objective Value (Lower is Better)")
    ax.set_ylim(-17.5, 5.0)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=1, framealpha=0.92)

    # (f) BB-2 Random Interactions Energy
    ax = axes[2, 1]
    best_e2 = [np.nan if item["best_feas_energy"] is None else item["best_feas_energy"] for item in d2["scan_data"]]
    mean_e2 = [np.nan if item["mean_feas_energy"] is None else item["mean_feas_energy"] for item in d2["scan_data"]]
    ax.plot(lams2, best_e2, color=c_std, marker='s', label='Std QAOA Best Feasible Found')
    ax.plot(lams2, mean_e2, color="#ff7043", linestyle='--', marker='^', alpha=0.8, label='Std QAOA Mean Feasible Energy')
    ax.axhline(d2["exact_min"], color=c_opt, linestyle=':', linewidth=1.8, label=f"Global Optimum $f^* = {d2['exact_min']:.2f}$")
    ax.set_title("(f) [BB-2 Fully-Connected Random] Energy Spectrum of Feasible Samples", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("Objective Value (Lower is Better)")
    ax.set_ylim(-7.0, 4.0)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=1, framealpha=0.92)

    # 全体タイトル
    fig.suptitle("Standard QAOA Penalty Sensitivity Benchmark: Full Hilbert Space Analysis vs FM-XY-QAOA", 
                 fontsize=12.5, fontweight='bold', y=0.992)

    out_pdf = os.path.join(base_dir, "pdf", "std_qaoa_sensitivity_comparison.pdf")
    out_png = os.path.join(base_dir, "png", "std_qaoa_sensitivity_comparison.png")
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    print(f"★ グラフ生成完了 (文字被り完全排除):")
    print(f"   PDF: {out_pdf}")
    print(f"   PNG: {out_png}")

if __name__ == "__main__":
    generate_plots()
