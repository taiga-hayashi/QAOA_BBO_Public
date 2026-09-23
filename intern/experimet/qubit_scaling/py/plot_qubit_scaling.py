#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_qubit_scaling.py
量子ビット数 N のスケーリング特性解析結果 (results_qubit_scaling.json) を読み込み、
学術論文仕様の 6 パネル総合スケーリング比較プロットを生成するスクリプト。
文字被りを完全に排除し、見やすく洗練されたレイアウトを実現。
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "json", "results_qubit_scaling.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    # スタイル設定
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 10.5,
        'axes.titlesize': 11.2,
        'xtick.labelsize': 9.0,
        'ytick.labelsize': 9.2,
        'legend.fontsize': 8.5,
        'figure.titlesize': 13.0,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'lines.markersize': 5.5,
        'grid.alpha': 0.35,
        'grid.linestyle': '--'
    })

    fig, axes = plt.subplots(3, 2, figsize=(14.0, 18.0), dpi=300)
    plt.subplots_adjust(hspace=0.82, wspace=0.25)

    # 配色
    c_std = "#d32f2f"      # Standard QAOA (Red)
    c_xy = "#2e7d32"       # FM-XY-QAOA (Green)
    c_fmqa = "#1976d2"     # FMQA (Blue)
    c_sub = "#7b1fa2"      # Subspace (Purple)
    c_lim = "#e65100"      # Classical Limit (Orange)
    c_gray = "#757575"

    N_vals = [d["N"] for d in data]
    sub_ratios = [d["subspace_ratio_percent"] for d in data]
    hilbert_dims = [d["hilbert_dim"] for d in data]
    sub_dims = [d["subspace_dim"] for d in data]

    # =========================================================================
    # ROW 1: (a) 制約充足率スケーリング (Semi-log) & (b) 探索空間次元の乖離 (Log-log)
    # =========================================================================
    # (a) 制約充足率
    ax = axes[0, 0]
    xy_records = [d for d in data if d["FM_XY_QAOA"]["feasibility_rate"] is not None]
    std_records = [d for d in data if d["Standard_QAOA"]["feasibility_rate"] is not None]
    feas_xy = [d["FM_XY_QAOA"]["feasibility_rate"] * 100 for d in xy_records]
    feas_std = [d["Standard_QAOA"]["feasibility_rate"] * 100 for d in std_records]
    feas_fmqa = [d["FMQA"]["feasibility_rate"] * 100 for d in data]

    ax.plot([d["N"] for d in xy_records], feas_xy, color=c_xy, marker='o', linewidth=2.2, label='FM-XY-QAOA (simulated range)')
    ax.plot(N_vals, feas_fmqa, color=c_fmqa, marker='s', linestyle='--', label='FMQA (Simulated Annealing, Adaptive $\\lambda$)')
    ax.plot([d["N"] for d in std_records], feas_std, color=c_std, marker='^', linestyle='-', label='Standard QAOA (simulated range)')
    ax.plot(N_vals, sub_ratios, color=c_gray, linestyle=':', label=r'Random Feasibility Baseline ($2^{-N/2} \times 100\%$)')

    ax.set_yscale('log')
    ax.set_title("(a) Constraint Feasibility Rate vs Number of Qubits $N$", fontweight='bold')
    ax.set_xlabel("Number of Qubits $N$ ($4M$, One-Hot Blocks)")
    ax.set_ylabel("Feasibility Rate (%) [Log Scale]")
    ax.set_ylim(1e-4, 250)
    ax.grid(True, which='both')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, framealpha=0.92)

    # (b) 探索空間次元の乖離
    ax = axes[0, 1]
    ax.plot(N_vals, hilbert_dims, color=c_std, marker='^', linewidth=2.0, label='Full Hilbert Space ($2^N$, Standard QAOA)')
    ax.plot(N_vals, sub_dims, color=c_xy, marker='o', linewidth=2.0, label='Valid Subspace ($4^{N/4} = 2^{N/2}$, FM-XY-QAOA)')

    ax.set_yscale('log')
    ax.set_title("(b) State Space Explosion: Full Hilbert vs Valid Subspace", fontweight='bold')
    ax.set_xlabel("Number of Qubits $N$")
    ax.set_ylabel("Dimension of Search Space [Log Scale]")
    ax.set_ylim(1e1, 1e20)
    ax.grid(True, which='both')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, framealpha=0.92)

    # =========================================================================
    # ROW 2: (c) 真の大域最適解到達率 & (d) 最適値ギャップ (Optimality Gap)
    # =========================================================================
    # (c) 最適解到達率 P(x*)
    ax = axes[1, 0]
    p_opt_xy = [d["FM_XY_QAOA"]["p_opt"] * 100 for d in data if d["FM_XY_QAOA"]["p_opt"] is not None]
    N_xy = [d["N"] for d in data if d["FM_XY_QAOA"]["p_opt"] is not None]

    p_opt_std = [d["Standard_QAOA"]["p_opt"] * 100 for d in data if d["Standard_QAOA"]["p_opt"] is not None]
    N_std = [d["N"] for d in data if d["Standard_QAOA"]["p_opt"] is not None]

    fmqa_exact_records = [d for d in data if d["FMQA"]["p_opt"] is not None]
    p_opt_fmqa = [d["FMQA"]["p_opt"] * 100 for d in fmqa_exact_records]

    ax.plot(N_xy, p_opt_xy, color=c_xy, marker='o', linewidth=2.0, label='FM-XY-QAOA (Subspace Ground State Hit)')
    ax.plot([d["N"] for d in fmqa_exact_records], p_opt_fmqa, color=c_fmqa, marker='s', linestyle='--', label='FMQA (SA hits / 600 reads; exact-reference range)')
    ax.plot(N_std, p_opt_std, color=c_std, marker='^', linestyle='-', label='Standard QAOA (Full Space Hit)')

    ax.set_yscale('log')
    ax.set_title("(c) Ground State Hit Probability $P(x^*)$ vs Scale $N$", fontweight='bold')
    ax.set_xlabel("Number of Qubits $N$")
    ax.set_ylabel("Ground State Probability (%) [Log Scale]")
    ax.set_ylim(1e-3, 25)
    ax.grid(True, which='both')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, framealpha=0.92)

    # (d) 最適値ギャップ (Optimality Gap)
    ax = axes[1, 1]
    gap_xy = [d["FM_XY_QAOA"]["opt_gap"] for d in data if d["FM_XY_QAOA"]["opt_gap"] is not None]
    gap_std = [d["Standard_QAOA"]["opt_gap"] for d in data if d["Standard_QAOA"]["opt_gap"] is not None]
    fmqa_gap_records = [d for d in data if d["FMQA"]["opt_gap"] is not None]
    gap_fmqa = [d["FMQA"]["opt_gap"] for d in fmqa_gap_records]

    ax.plot(N_xy, gap_xy, color=c_xy, marker='o', linewidth=2.0, label='FM-XY-QAOA Optimality Gap')
    ax.plot([d["N"] for d in fmqa_gap_records], gap_fmqa, color=c_fmqa, marker='s', linestyle='--', label='FMQA (SA) Optimality Gap')
    ax.plot(N_std, gap_std, color=c_std, marker='^', linestyle='-', label='Standard QAOA Optimality Gap')
    ax.axhline(0.0, color=c_gray, linestyle=':', label='Exact Global Minimum (Gap = 0)')

    ax.set_title("(d) Optimality Gap to Ground Truth vs Scale $N$", fontweight='bold')
    ax.set_xlabel("Number of Qubits $N$")
    ax.set_ylabel("Energy Gap ($E_{\\mathrm{found}} - f^*$)")
    ax.set_ylim(-0.5, 12.0)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, framealpha=0.92)

    # =========================================================================
    # ROW 3: (e) 計算時間スケーリング & (f) 理論 RAM 容量と古典限界
    # =========================================================================
    # (e) 計算時間スケーリング
    ax = axes[2, 0]
    t_xy = [d["FM_XY_QAOA"]["runtime_sec"] for d in data if d["FM_XY_QAOA"]["runtime_sec"] is not None]
    t_std = [d["Standard_QAOA"]["runtime_sec"] for d in data if d["Standard_QAOA"]["runtime_sec"] is not None]
    t_fmqa = [d["FMQA"]["runtime_sec"] for d in data]

    ax.plot(N_xy, t_xy, color=c_xy, marker='o', linewidth=2.0, label='FM-XY-QAOA (OpenQARP)')
    ax.plot(N_std, t_std, color=c_std, marker='^', linestyle='-', label='Standard QAOA (OpenQARP, full space)')
    ax.plot(N_vals, t_fmqa, color=c_fmqa, marker='s', linestyle='--', label='FMQA (SA, 600 Reads)')

    ax.set_yscale('log')
    ax.set_title("(e) Simulation Runtime Scaling vs Scale $N$", fontweight='bold')
    ax.set_xlabel("Number of Qubits $N$")
    ax.set_ylabel("Execution Time (Seconds) [Log Scale]")
    ax.set_ylim(5e-4, 50)
    ax.grid(True, which='both')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, framealpha=0.92)

    # (f) 理論 RAM 容量と古典計算限界境界
    ax = axes[2, 1]
    ram_full_gb = [d["ram_bytes_full"] / (1024**3) for d in data]
    ram_sub_gb = [d["ram_bytes_sub"] / (1024**3) for d in data]

    ax.plot(N_vals, ram_full_gb, color=c_std, marker='^', linewidth=2.2, label='Full State Vector RAM ($2^N \\times 16$B)')
    ax.plot(N_vals, ram_sub_gb, color=c_xy, marker='o', linewidth=2.2, label='Subspace State Vector RAM ($2^{N/2} \\times 16$B)')

    # 古典限界ライン
    ax.axhline(16.0, color=c_lim, linestyle='--', label='Laptop RAM Limit (16 GB, $N=30$)')
    ax.axhline(128.0, color="#b71c1c", linestyle='--', label='High-End Compute Server (128 GB, $N=33$)')
    ax.axhline(1e6, color="#4a148c", linestyle=':', label='Exascale Supercomputer (1 Petabyte, $N=46$)')

    ax.set_yscale('log')
    ax.set_title("(f) Theoretical RAM Footprint & Classical Limits", fontweight='bold')
    ax.set_xlabel("Number of Qubits $N$")
    ax.set_ylabel("Memory Required (Gigabytes) [Log Scale]")
    ax.set_ylim(1e-7, 1e11)
    ax.grid(True, which='both')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=1, framealpha=0.92)

    # 全体タイトル
    fig.suptitle("Multi-Scale Quantum Scaling Benchmark: Standard QAOA vs FMQA vs FM-XY-QAOA ($N=8 \\sim 64$)", 
                 fontsize=12.5, fontweight='bold', y=0.992)

    out_pdf = os.path.join(base_dir, "pdf", "qubit_scaling_comparison.pdf")
    out_png = os.path.join(base_dir, "png", "qubit_scaling_comparison.png")
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    # 目的関数値の比較（値が実測されている範囲のみを表示）
    fig_obj, ax_obj = plt.subplots(figsize=(8.2, 5.2), dpi=300)
    exact_data = [d for d in data if d["exact_min_available"]]
    xy_data = [d for d in data if d["FM_XY_QAOA"]["best_energy"] is not None]
    std_data = [d for d in data if d["Standard_QAOA"]["best_energy"] is not None]
    fmqa_data = [d for d in data if d["FMQA"]["best_energy"] is not None and d["exact_min_available"]]

    ax_obj.plot(
        [d["N"] for d in exact_data],
        [d["exact_min"] for d in exact_data],
        color="black", marker="D", linestyle=":", linewidth=1.8,
        label="Exact minimum (available for $N \\leq 20$)",
    )
    ax_obj.plot(
        [d["N"] for d in xy_data],
        [d["FM_XY_QAOA"]["best_energy"] for d in xy_data],
        color=c_xy, marker="o", linewidth=2.0,
        label="FM-XY-QAOA best found ($N \\leq 20$)",
    )
    ax_obj.plot(
        [d["N"] for d in std_data],
        [d["Standard_QAOA"]["best_energy"] for d in std_data],
        color=c_std, marker="^", linewidth=2.0,
        label="Standard QAOA best found ($N \\leq 20$)",
    )
    ax_obj.plot(
        [d["N"] for d in fmqa_data],
        [d["FMQA"]["best_energy"] for d in fmqa_data],
        color=c_fmqa, marker="s", linestyle="--", linewidth=2.0,
        label="FMQA best feasible found (exact-reference range)",
    )

    ax_obj.set_xlabel("Number of Qubits $N$")
    ax_obj.set_ylabel("Best Objective Value (lower is better)")
    ax_obj.set_xticks(N_vals)
    ax_obj.grid(True)
    ax_obj.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.92)

    out_obj_pdf = os.path.join(base_dir, "pdf", "objective_value_vs_qubits.pdf")
    out_obj_png = os.path.join(base_dir, "png", "objective_value_vs_qubits.png")
    fig_obj.savefig(out_obj_pdf, bbox_inches="tight")
    fig_obj.savefig(out_obj_png, bbox_inches="tight")
    plt.close(fig_obj)

    print(f"★ スケーリングプロット生成完了:")
    print(f"   PDF: {out_pdf}")
    print(f"   PNG: {out_png}")
    print(f"   Objective PDF: {out_obj_pdf}")
    print(f"   Objective PNG: {out_obj_png}")

if __name__ == "__main__":
    generate_plots()
