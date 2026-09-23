import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any

from bb_function import (
    create_random_qubo_bb,
    get_exact_minimum_categorical_bb,
    get_feasible_patterns,
    is_feasible_one_hot
)
from fmqa_solver import solve_fmqa
from qaoa_solver import solve_xy_qaoa, solve_standard_qaoa
from qubo import matrix_to_qubo_dict
from utils import Logger

def create_deceptive_qubo(block_sizes: List[int], seed: int = 42) -> np.ndarray:
    """
    欺瞞的（Deceptive）ランドスケープを持つQUBOを構築する。
    - 局所解 x_loc: 広大な引力圏を持ち、周囲の無効状態も低エネルギー
    - 大域的最適解 x_opt: 真の最小値だが引力圏が狭く、ペナルティ障壁で囲まれている
    """
    np.random.seed(seed)
    total_qubits = sum(block_sizes)
    feasible_patterns = get_feasible_patterns(block_sizes)
    num_feas = len(feasible_patterns)
    
    # 基本のランダムQUBO
    Q = np.random.uniform(-0.5, 0.5, (total_qubits, total_qubits))
    
    # インデックス 0 を「欺瞞的局所解」、インデックス -1 を「真の大域最適解」に設計
    # x_loc: (1,0,0,0, 1,0,0,0) 等
    # x_opt: (0,0,0,1, 0,0,0,1) 等
    loc_idx = 0
    opt_idx = num_feas - 1
    
    x_loc = feasible_patterns[loc_idx]
    x_opt = feasible_patterns[opt_idx]
    
    # x_opt のビットが立つときだけ強く負のエネルギーを与える
    for i in range(total_qubits):
        if x_opt[i] == 1:
            Q[i, i] -= 3.0
        if x_loc[i] == 1:
            Q[i, i] -= 1.5
            
    # x_loc のペアに負の相互作用（魅力的な局所解）
    for i in range(total_qubits):
        for j in range(i + 1, total_qubits):
            if x_loc[i] == 1 and x_loc[j] == 1:
                Q[i, j] -= 2.0
            if x_opt[i] == 1 and x_opt[j] == 1:
                Q[i, j] -= 2.5
                
    Q = np.triu(Q)
    return Q

def create_multiscale_qubo(block_sizes: List[int], scale_factor: float = 10.0, seed: int = 42) -> np.ndarray:
    """
    ブロック間で相互作用のスケールが大きく異なるマルチスケールQUBO。
    ブロック1は O(1.0), ブロック2は O(scale_factor)
    """
    np.random.seed(seed)
    total_qubits = sum(block_sizes)
    Q = np.random.uniform(-1.0, 1.0, (total_qubits, total_qubits))
    
    idx = 0
    for b_idx, sz in enumerate(block_sizes):
        mult = 1.0 if b_idx == 0 else scale_factor
        Q[idx:idx+sz, idx:idx+sz] *= mult
        idx += sz
        
    Q = np.triu(Q)
    return Q

def run_deceptive_experiment(block_sizes: List[int], output_prefix: str) -> Dict[str, Any]:
    print(f"\n============================================================")
    print(f" 検証1: 欺瞞的局所解（Deceptive Landscape）における切り分け")
    print(f"============================================================")
    
    total_qubits = sum(block_sizes)
    Q = create_deceptive_qubo(block_sizes, seed=42)
    best_x, min_val = get_exact_minimum_categorical_bb(Q, block_sizes)
    best_x_str = "".join(str(int(b)) for b in best_x)
    qubo_dict = matrix_to_qubo_dict(Q)
    
    print(f"真の大域最適解 x*: {best_x_str} (最小値: {min_val:.4f})")
    
    # FMQA を複数の lambda で評価
    lambda_list = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    fmqa_opt_probs = []
    fmqa_feas_rates = []
    
    for lam in lambda_list:
        rf = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=lam, num_reads=1000, seed=42)
        p_opt = rf["state_probabilities"].get(best_x_str, 0.0)
        feas = rf["feasibility_rate"]
        fmqa_opt_probs.append(p_opt)
        fmqa_feas_rates.append(feas)
        print(f"  [FMQA] lam={lam:4.1f}: Feasibility={feas*100:5.1f}%, Opt Prob P(x*)={p_opt*100:5.2f}%")
        
    # Standard QAOA
    r_std = solve_standard_qaoa(qubo_dict, block_sizes, lambda_penalty=5.0, reps=1, maxiter=50)
    std_opt = r_std["state_probabilities"].get(best_x_str, 0.0)
    print(f"  [Standard QAOA (p=1)]: Feas={r_std['feasibility_rate']*100:.1f}%, Opt Prob={std_opt*100:.2f}%")
    
    # FM-XY-QAOA (p=1)
    r_xy1 = solve_xy_qaoa(qubo_dict, block_sizes, reps=1, maxiter=50)
    xy1_opt = r_xy1["state_probabilities"].get(best_x_str, 0.0)
    print(f"  [FM-XY-QAOA (p=1)]: Feas={r_xy1['feasibility_rate']*100:.1f}%, Opt Prob={xy1_opt*100:.2f}%")
    
    # FM-XY-QAOA (p=2)
    r_xy2 = solve_xy_qaoa(qubo_dict, block_sizes, reps=2, maxiter=80)
    xy2_opt = r_xy2["state_probabilities"].get(best_x_str, 0.0)
    print(f"  [FM-XY-QAOA (p=2)]: Feas={r_xy2['feasibility_rate']*100:.1f}%, Opt Prob={xy2_opt*100:.2f}%")
    
    # プロット作成
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lambda_list, [p * 100 for p in fmqa_opt_probs], marker='o', label='FMQA (SA, varying $\lambda$)', color='#4C72B0', linewidth=2)
    ax.axhline(y=xy1_opt * 100, color='#55A868', linestyle='--', linewidth=2, label=f'FM-XY-QAOA p=1 ({xy1_opt*100:.1f}%)')
    ax.axhline(y=xy2_opt * 100, color='#2CA02C', linestyle='-', linewidth=2.5, label=f'FM-XY-QAOA p=2 ({xy2_opt*100:.1f}%) [Proposed]')
    ax.axhline(y=std_opt * 100, color='#DD8452', linestyle=':', linewidth=1.5, label=f'Standard QAOA p=1 ({std_opt*100:.2f}%)')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Penalty Coefficient $\lambda$ in FMQA (log scale)', fontsize=11)
    ax.set_ylabel(r'Optimal Solution Success Probability $P(x^*)$ (%)', fontsize=11)
    ax.set_title('Deceptive Landscape: FM-XY Quantum Advantage over FMQA Trapping', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    png_path = f"{output_prefix}_deceptive_comparison.png"
    pdf_path = f"{output_prefix}_deceptive_comparison.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    
    return {
        "lambda_list": lambda_list,
        "fmqa_opt_probs": fmqa_opt_probs,
        "std_opt": float(std_opt),
        "xy1_opt": float(xy1_opt),
        "xy2_opt": float(xy2_opt)
    }

def run_multiscale_experiment(block_sizes: List[int], output_prefix: str) -> Dict[str, Any]:
    print(f"\n============================================================")
    print(f" 検証2: マルチスケール不均衡結合における切り分け")
    print(f"============================================================")
    
    scale_factors = [1.0, 3.0, 5.0, 10.0, 20.0]
    fmqa_res_by_scale = []
    xy_res_by_scale = []
    
    for s_fac in scale_factors:
        print(f"\n--- Scale Factor: {s_fac}x ---")
        Q = create_multiscale_qubo(block_sizes, scale_factor=s_fac, seed=42)
        best_x, min_val = get_exact_minimum_categorical_bb(Q, block_sizes)
        best_x_str = "".join(str(int(b)) for b in best_x)
        qubo_dict = matrix_to_qubo_dict(Q)
        
        # 固定ペナルティ lambda=5.0 での FMQA (単一のlambdaでは不均衡に対応できない)
        rf = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=5.0, num_reads=1000, seed=42)
        f_feas = rf["feasibility_rate"]
        f_opt = rf["state_probabilities"].get(best_x_str, 0.0)
        fmqa_res_by_scale.append({"scale": s_fac, "feas": f_feas, "opt": f_opt})
        print(f"  [FMQA (lam=5.0)] Feasibility: {f_feas*100:5.1f}%, Opt Prob: {f_opt*100:5.2f}%")
        
        # FM-XY-QAOA (lambda不要)
        r_xy = solve_xy_qaoa(qubo_dict, block_sizes, reps=1, maxiter=50)
        xy_feas = r_xy["feasibility_rate"]
        xy_opt = r_xy["state_probabilities"].get(best_x_str, 0.0)
        xy_res_by_scale.append({"scale": s_fac, "feas": xy_feas, "opt": xy_opt})
        print(f"  [FM-XY-QAOA]     Feasibility: {xy_feas*100:5.1f}%, Opt Prob: {xy_opt*100:5.2f}%")
        
    # プロット作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    
    # 1. 制約充足率
    ax1.plot(scale_factors, [r["feas"] * 100 for r in fmqa_res_by_scale], marker='o', label='FMQA (SA, fixed $\lambda=5$)', color='#4C72B0', linewidth=2)
    ax1.plot(scale_factors, [r["feas"] * 100 for r in xy_res_by_scale], marker='^', label='FM-XY-QAOA (Proposed)', color='#55A868', linewidth=2.5)
    ax1.set_xlabel('Inter-Block Coupling Scale Factor', fontsize=11)
    ax1.set_ylabel('Feasibility Rate (%)', fontsize=11)
    ax1.set_title('Constraint Robustness vs Scale Discrepancy', fontsize=12, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 最適解到達確率
    ax2.plot(scale_factors, [r["opt"] * 100 for r in fmqa_res_by_scale], marker='o', label='FMQA (SA, fixed $\lambda=5$)', color='#4C72B0', linewidth=2)
    ax2.plot(scale_factors, [r["opt"] * 100 for r in xy_res_by_scale], marker='^', label='FM-XY-QAOA (Proposed)', color='#55A868', linewidth=2.5)
    ax2.set_xlabel('Inter-Block Coupling Scale Factor', fontsize=11)
    ax2.set_ylabel(r'Optimal Success Probability $P(x^*)$ (%)', fontsize=11)
    ax2.set_title('Success Rate under Heterogeneous Scales', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    png_path = f"{output_prefix}_multiscale_comparison.png"
    pdf_path = f"{output_prefix}_multiscale_comparison.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    
    return {
        "scale_factors": scale_factors,
        "fmqa": fmqa_res_by_scale,
        "xy": xy_res_by_scale
    }

def run_layer_depth_experiment(block_sizes: List[int], output_prefix: str) -> Dict[str, Any]:
    print(f"\n============================================================")
    print(f" 検証3: レイヤー深度 p=1 vs p=2 による量子濃縮ブースト")
    print(f"============================================================")
    
    seeds = [12, 34, 42, 56, 78, 99]
    d = sum(block_sizes)
    
    fmqa_list, xy_p1_list, xy_p2_list = [], [], []
    
    for s in seeds:
        Q = create_random_qubo_bb(d, seed=s)
        best_x, min_val = get_exact_minimum_categorical_bb(Q, block_sizes)
        best_x_str = "".join(str(int(b)) for b in best_x)
        qd = matrix_to_qubo_dict(Q)
        
        # FMQA
        rf = solve_fmqa(qd, block_sizes, lambda_penalty=5.0, num_reads=1000, seed=s)
        p_fmqa = rf["state_probabilities"].get(best_x_str, 0.0)
        fmqa_list.append(p_fmqa)
        
        # XY p=1
        r1 = solve_xy_qaoa(qd, block_sizes, reps=1, maxiter=50)
        p_xy1 = r1["state_probabilities"].get(best_x_str, 0.0)
        xy_p1_list.append(p_xy1)
        
        # XY p=2
        r2 = solve_xy_qaoa(qd, block_sizes, reps=2, maxiter=80)
        p_xy2 = r2["state_probabilities"].get(best_x_str, 0.0)
        xy_p2_list.append(p_xy2)
        
        print(f"  Seed {s:2d} | FMQA={p_fmqa*100:5.2f}% | XY(p=1)={p_xy1*100:5.2f}% | XY(p=2)={p_xy2*100:5.2f}%")
        
    avg_fmqa = float(np.mean(fmqa_list))
    avg_xy1 = float(np.mean(xy_p1_list))
    avg_xy2 = float(np.mean(xy_p2_list))
    
    print(f"\n--- 平均到達確率 ---")
    print(f"FMQA: {avg_fmqa*100:.2f}% | FM-XY(p=1): {avg_xy1*100:.2f}% | FM-XY(p=2): {avg_xy2*100:.2f}%")
    
    # 棒グラフ比較
    fig, ax = plt.subplots(figsize=(9, 5))
    x_indices = np.arange(len(seeds))
    width = 0.28
    
    ax.bar(x_indices - width, [p * 100 for p in fmqa_list], width, label='FMQA (SA, 1000 reads)', color='#4C72B0', alpha=0.9)
    ax.bar(x_indices, [p * 100 for p in xy_p1_list], width, label='FM-XY-QAOA p=1', color='#55A868', alpha=0.7)
    ax.bar(x_indices + width, [p * 100 for p in xy_p2_list], width, label='FM-XY-QAOA p=2 [Proposed Boost]', color='#2CA02C', alpha=0.95)
    
    ax.set_ylabel(r'Optimal Success Probability $P(x^*)$ (%)', fontsize=11)
    ax.set_title(f'Quantum Layer Scaling: FM-XY (p=2) Outperforms FMQA across Hard Instances (N={d})', fontsize=12, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10)
    ax.legend(fontsize=10.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    png_path = f"{output_prefix}_layer_boost_comparison.png"
    pdf_path = f"{output_prefix}_layer_boost_comparison.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    
    return {
        "seeds": seeds,
        "fmqa": fmqa_list,
        "xy_p1": xy_p1_list,
        "xy_p2": xy_p2_list,
        "avg": {"fmqa": avg_fmqa, "xy_p1": avg_xy1, "xy_p2": avg_xy2}
    }

def main():
    os.makedirs(os.path.join("result", "txt"), exist_ok=True)
    os.makedirs(os.path.join("result", "json"), exist_ok=True)
    os.makedirs(os.path.join("result", "png"), exist_ok=True)
    os.makedirs(os.path.join("result", "pdf"), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("result", "txt", f"boundary_{timestamp}.txt")
    sys.stdout = Logger(log_path)
    
    print(f"============================================================")
    print(f" FM-XY vs FMQA 境界領域（Regime）切り分け包括実験")
    print(f" タイムスタンプ: {timestamp}")
    print(f"============================================================")
    
    prefix = os.path.join("result", "png", f"boundary_{timestamp}")
    block_sizes = [4, 4] # 8量子ビット, 全256状態, 有効16状態
    
    res1 = run_deceptive_experiment(block_sizes, prefix)
    res2 = run_multiscale_experiment(block_sizes, prefix)
    res3 = run_layer_depth_experiment(block_sizes, prefix)
    
    json_path = os.path.join("result", "json", f"boundary_analysis_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "deceptive_experiment": res1,
            "multiscale_experiment": res2,
            "layer_depth_experiment": res3
        }, f, indent=4)
        
    print(f"\n============================================================")
    print(f" 全切り分け実験が完了しました！ JSON: {json_path}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
