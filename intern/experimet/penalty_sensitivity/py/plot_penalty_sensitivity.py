#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_penalty_sensitivity.py
FMQA のペナルティ係数感度解析結果 (results_penalty_sensitivity.json) を可視化するスクリプト。
横軸: ペナルティ係数 λ vs 縦軸: 最適化性能 (制約充足率, 最適解到達確率, エネルギー)
FM-XY-QAOA の性能基準線 (Proposed baseline, λ=0) を併記。
文字被りとプロットはみ出しを完全に解消し、制約崩壊域と健全稼働域を帯状表示。
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "json", "results_penalty_sensitivity.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} が存在しません。")
        return

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
        'figure.titlesize': 13.5,
        'axes.linewidth': 0.8,
        'lines.linewidth': 2.0,
        'lines.markersize': 5.5,
        'grid.alpha': 0.35,
        'grid.linestyle': '--'
    })

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 15.0), dpi=300)
    plt.subplots_adjust(hspace=0.50, wspace=0.26)

    # 配色
    c_fmqa = "#1565c0"       # FMQA (Deep Blue)
    c_xy = "#2e7d32"         # FM-XY-QAOA (Green)
    c_crit = "#d32f2f"       # Critical Penalty Threshold (Red dashed)
    c_adp = "#e65100"        # Adaptive Optimal Penalty (Deep Orange dashed)
    c_opt = "#212121"        # Exact Optimum (Dark Charcoal)
    c_mean = "#0288d1"       # Mean Energy (Cyan)

    d1 = data["BB1_materials"]
    d2 = data["BB2_random"]

    sw1 = d1["sweep_data"]
    sw2 = d2["sweep_data"]

    lams1 = [pt["lambda"] for pt in sw1]
    lams2 = [pt["lambda"] for pt in sw2]

    # =========================================================================
    # ROW 1: 制約充足率 (%) vs λ
    # =========================================================================
    # (a) BB-1 多元触媒
    ax = axes[0, 0]
    feas1 = [pt["feasibility_rate"] * 100 for pt in sw1]
    
    # The gain-based values are visual guides, not theoretical phase boundaries.
    ax.axvspan(0, d1["lambda_gain_heuristic"], color='#ffebee', alpha=0.5, label='Low-$\\lambda$ region (guide)')
    ax.axvspan(d1["lambda_gain_heuristic"], 26, color='#e8f5e9', alpha=0.3, label='Higher-$\\lambda$ region (guide)')

    ax.plot(lams1, feas1, marker='o', color=c_fmqa, label='FMQA (SA Feasibility)')
    ax.axvline(d1["lambda_gain_heuristic"], color=c_crit, linestyle=':', linewidth=1.8, 
               label=f'Gain heuristic $\\lambda_h={d1["lambda_gain_heuristic"]:.2f}$')
    ax.axvline(d1["lambda_safety_heuristic"], color=c_adp, linestyle='--', linewidth=1.8, 
               label=f'Safety heuristic $\\lambda_h={d1["lambda_safety_heuristic"]:.2f}$')
    ax.axhline(100.0, color=c_xy, linestyle='-', linewidth=1.2, alpha=0.7,
               label='FM-XY-QAOA (100% Inherent, $\\lambda=0$)')

    ax.set_title("(a) [BB-1 Materials Catalyst] One-Hot Feasibility Rate (%)", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("One-Hot Feasibility Rate (%)")
    ax.set_ylim(-5, 115)
    ax.set_xlim(0, 26)
    ax.grid(True)
    ax.legend(loc='center right', framealpha=0.92)

    # (b) BB-2 全結合ランダム
    ax = axes[0, 1]
    feas2 = [pt["feasibility_rate"] * 100 for pt in sw2]

    ax.axvspan(0, d2["lambda_gain_heuristic"], color='#ffebee', alpha=0.5, label='Low-$\\lambda$ region (guide)')
    ax.axvspan(d2["lambda_gain_heuristic"], 26, color='#e8f5e9', alpha=0.3, label='Higher-$\\lambda$ region (guide)')

    ax.plot(lams2, feas2, marker='o', color=c_fmqa, label='FMQA (SA Feasibility)')
    ax.axvline(d2["lambda_gain_heuristic"], color=c_crit, linestyle=':', linewidth=1.8, 
               label=f'Gain heuristic $\\lambda_h={d2["lambda_gain_heuristic"]:.2f}$')
    ax.axvline(d2["lambda_safety_heuristic"], color=c_adp, linestyle='--', linewidth=1.8, 
               label=f'Safety heuristic $\\lambda_h={d2["lambda_safety_heuristic"]:.2f}$')
    ax.axhline(100.0, color=c_xy, linestyle='-', linewidth=1.2, alpha=0.7,
               label='FM-XY-QAOA (100% Inherent, $\\lambda=0$)')

    ax.set_title("(b) [BB-2 Fully-Connected Random] One-Hot Feasibility Rate (%)", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("One-Hot Feasibility Rate (%)")
    ax.set_ylim(-5, 115)
    ax.set_xlim(0, 26)
    ax.grid(True)
    ax.legend(loc='center right', framealpha=0.92)

    # =========================================================================
    # ROW 2: 大域最適解到達確率 P(x*) (%) vs λ
    # =========================================================================
    # (c) BB-1 多元触媒
    ax = axes[1, 0]
    p_opt1 = [pt["success_probability"] * 100 for pt in sw1]

    ax.axvspan(0, d1["lambda_gain_heuristic"], color='#ffebee', alpha=0.5)
    ax.axvspan(d1["lambda_gain_heuristic"], 26, color='#e8f5e9', alpha=0.3)

    ax.plot(lams1, p_opt1, marker='s', color=c_fmqa, label='FMQA (SA Optimization)')
    ax.axhline(d1["fm_xy_qaoa_baseline"]["success_probability"] * 100, color=c_xy, linestyle='--', linewidth=2.0, 
               label=f'FM-XY-QAOA Proposed ({d1["fm_xy_qaoa_baseline"]["success_probability"]*100:.2f}%, $\\lambda=0$)')
    ax.axvline(d1["lambda_safety_heuristic"], color=c_adp, linestyle=':', linewidth=1.8, 
               label=f'Safety heuristic $\\lambda_h={d1["lambda_safety_heuristic"]:.2f}$')

    ax.set_title("(c) [BB-1 Materials Catalyst] Ground State Hit Probability P(x*) (%)", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("Ground State Probability (%)")
    ax.set_ylim(-1.0, 26.0)
    ax.set_xlim(0, 26)
    ax.grid(True)
    ax.legend(loc='upper right', framealpha=0.92)

    # (d) BB-2 全結合ランダム
    ax = axes[1, 1]
    p_opt2 = [pt["success_probability"] * 100 for pt in sw2]

    ax.axvspan(0, d2["lambda_gain_heuristic"], color='#ffebee', alpha=0.5)
    ax.axvspan(d2["lambda_gain_heuristic"], 26, color='#e8f5e9', alpha=0.3)

    ax.plot(lams2, p_opt2, marker='s', color=c_fmqa, label='FMQA (SA Optimization)')
    ax.axhline(d2["fm_xy_qaoa_baseline"]["success_probability"] * 100, color=c_xy, linestyle='--', linewidth=2.0, 
               label=f'FM-XY-QAOA Proposed ({d2["fm_xy_qaoa_baseline"]["success_probability"]*100:.2f}%, $\\lambda=0$)')
    ax.axvline(d2["lambda_safety_heuristic"], color=c_adp, linestyle=':', linewidth=1.8, 
               label=f'Safety heuristic $\\lambda_h={d2["lambda_safety_heuristic"]:.2f}$')

    ax.set_title("(d) [BB-2 Fully-Connected Random] Ground State Hit Probability P(x*) (%)", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("Ground State Probability (%)")
    ax.set_ylim(-1.0, 26.0)
    ax.set_xlim(0, 26)
    ax.grid(True)
    ax.legend(loc='upper right', framealpha=0.92)

    # =========================================================================
    # ROW 3: 最良エネルギー / 平均有効エネルギー vs λ
    # =========================================================================
    # (e) BB-1 多元触媒
    ax = axes[2, 0]
    best_e1 = [np.nan if pt["best_feasible_energy"] is None else pt["best_feasible_energy"] for pt in sw1]
    mean_e1 = [np.nan if pt["mean_feasible_energy"] is None else pt["mean_feasible_energy"] for pt in sw1]

    ax.axvspan(0, d1["lambda_gain_heuristic"], color='#ffebee', alpha=0.5)
    ax.axvspan(d1["lambda_gain_heuristic"], 26, color='#e8f5e9', alpha=0.3)

    ax.plot(lams1, best_e1, marker='^', color=c_fmqa, label='FMQA Best Feasible Energy')
    ax.plot(lams1, mean_e1, marker='v', color=c_mean, linestyle='--', label='FMQA Mean Feasible Energy')
    ax.axhline(d1["exact_min"], color=c_opt, linestyle=':', linewidth=1.8, 
               label=f'Global Optimum $f^* = {d1["exact_min"]:.2f}$')
    ax.axvline(d1["lambda_safety_heuristic"], color=c_adp, linestyle=':', linewidth=1.5,
               label=f'Safety heuristic $\\lambda_h={d1["lambda_safety_heuristic"]:.2f}$')

    ax.set_title("(e) [BB-1 Materials Catalyst] Energy Profile vs $\\lambda$", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("Objective Value (Lower is Better)")
    ax.set_xlim(0, 26)
    ax.set_ylim(-90, 0)
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.92)

    # (f) BB-2 全結合ランダム
    ax = axes[2, 1]
    best_e2 = [np.nan if pt["best_feasible_energy"] is None else pt["best_feasible_energy"] for pt in sw2]
    mean_e2 = [np.nan if pt["mean_feasible_energy"] is None else pt["mean_feasible_energy"] for pt in sw2]

    ax.axvspan(0, d2["lambda_gain_heuristic"], color='#ffebee', alpha=0.5)
    ax.axvspan(d2["lambda_gain_heuristic"], 26, color='#e8f5e9', alpha=0.3)

    ax.plot(lams2, best_e2, marker='^', color=c_fmqa, label='FMQA Best Feasible Energy')
    ax.plot(lams2, mean_e2, marker='v', color=c_mean, linestyle='--', label='FMQA Mean Feasible Energy')
    ax.axhline(d2["exact_min"], color=c_opt, linestyle=':', linewidth=1.8, 
               label=f'Global Optimum $f^* = {d2["exact_min"]:.2f}$')
    ax.axvline(d2["lambda_safety_heuristic"], color=c_adp, linestyle=':', linewidth=1.5,
               label=f'Safety heuristic $\\lambda_h={d2["lambda_safety_heuristic"]:.2f}$')

    ax.set_title("(f) [BB-2 Fully-Connected Random] Energy Profile vs $\\lambda$", fontweight='bold')
    ax.set_xlabel("Penalty Coefficient $\\lambda$")
    ax.set_ylabel("Objective Value (Lower is Better)")
    ax.set_xlim(0, 26)
    ax.set_ylim(-11, 2)
    ax.grid(True)
    ax.legend(loc='lower right', framealpha=0.92)

    # 全体タイトル
    fig.suptitle("FMQA Penalty Sensitivity Analysis: Penalty Coefficient $\\lambda$ vs Optimization Performance\n(Comparison against Subspace-Preserving FM-XY-QAOA Baseline)", 
                 fontsize=13.0, fontweight='bold', y=0.993)

    out_pdf = os.path.join(base_dir, "pdf", "penalty_sensitivity_comparison.pdf")
    out_png = os.path.join(base_dir, "png", "penalty_sensitivity_comparison.png")
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    print(f"★ 感度解析プロット生成完了:")
    print(f"   PDF: {out_pdf}")
    print(f"   PNG: {out_png}")

if __name__ == "__main__":
    generate_plots()
