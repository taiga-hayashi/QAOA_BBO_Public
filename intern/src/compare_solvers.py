import os
import sys
import json
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any

from bb_function import (
    create_random_qubo_bb,
    get_exact_minimum_categorical_bb,
    get_feasible_patterns,
    is_feasible_one_hot,
    evaluate_bb
)
from fmqa_solver import solve_fmqa
from qaoa_solver import solve_standard_qaoa, solve_xy_qaoa
from qubo import matrix_to_qubo_dict
from utils import Logger

def run_distribution_comparison(
    qubo_dict: Dict[tuple, float],
    block_sizes: List[int],
    lambda_val: float,
    exact_best_x: np.ndarray,
    exact_min_val: float,
    output_prefix: str,
    qaoa_reps: int = 1,
    qaoa_maxiter: int = 50
) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f" 実験1: 状態確率分布の直接比較 (lambda = {lambda_val})")
    print(f"==========================================")
    
    # 1. FMQA (Simulated Annealing)
    print("\n[1/3] Running FMQA (Simulated Annealing)...")
    t0 = time.time()
    res_fmqa = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=lambda_val, num_reads=1000, seed=42)
    t_fmqa = time.time() - t0
    print(f"FMQA 完了 ({t_fmqa:.2f}s) | Feasibility: {res_fmqa['feasibility_rate']*100:.1f}%")
    
    # 2. Standard FM-QAOA
    print("\n[2/3] Running Standard FM-QAOA (X-Mixer + Penalty)...")
    t0 = time.time()
    res_std = solve_standard_qaoa(qubo_dict, block_sizes, lambda_penalty=lambda_val, reps=qaoa_reps, maxiter=qaoa_maxiter)
    t_std = time.time() - t0
    print(f"Standard QAOA 完了 ({t_std:.2f}s) | Feasibility: {res_std['feasibility_rate']*100:.1f}%")
    
    # 3. FM-XY-QAOA (Proposed)
    print("\n[3/3] Running FM-XY-QAOA (Block-XY Mixer, Penalty-free)...")
    t0 = time.time()
    res_xy = solve_xy_qaoa(qubo_dict, block_sizes, reps=qaoa_reps, maxiter=qaoa_maxiter)
    t_xy = time.time() - t0
    print(f"FM-XY-QAOA 完了 ({t_xy:.2f}s) | Feasibility: {res_xy['feasibility_rate']*100:.1f}%")
    
    # 最適解のビット文字列
    best_x_tuple = tuple(int(xi) for xi in exact_best_x)
    best_x_str = "".join(str(b) for b in best_x_tuple)
    
    # 全実行可能パターンの文字列リスト
    feasible_patterns = get_feasible_patterns(block_sizes)
    feasible_strs = ["".join(str(int(b)) for b in pat) for pat in feasible_patterns]
    
    # 各手法での最適解到達確率
    p_opt_fmqa = res_fmqa["state_probabilities"].get(best_x_str, 0.0)
    p_opt_std = res_std["state_probabilities"].get(best_x_str, 0.0)
    p_opt_xy = res_xy["state_probabilities"].get(best_x_str, 0.0)
    
    print("\n--- 比較サマリー ---")
    print(f"真の最適解 x*: {best_x_str} (目的値: {exact_min_val:.4f})")
    print(f"制約充足率:  FMQA={res_fmqa['feasibility_rate']*100:.1f}%,  Standard QAOA={res_std['feasibility_rate']*100:.1f}%,  XY-QAOA={res_xy['feasibility_rate']*100:.1f}%")
    print(f"最適解到達率: FMQA={p_opt_fmqa*100:.2f}%,  Standard QAOA={p_opt_std*100:.2f}%,  XY-QAOA={p_opt_xy*100:.2f}%")
    
    # プロットの作成
    # 有効解ごとの確率 + 無効解の合計確率
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # X軸のラベル: 各有効解 + 'Infeasible (Total)'
    labels = [s if s != best_x_str else f"{s}\n(Optimal)" for s in feasible_strs] + ["Infeasible\n(Sum)"]
    
    fmqa_probs = [res_fmqa["state_probabilities"].get(s, 0.0) for s in feasible_strs]
    fmqa_infeasible = max(0.0, 1.0 - sum(fmqa_probs))
    fmqa_probs.append(fmqa_infeasible)
    
    std_probs = [res_std["state_probabilities"].get(s, 0.0) for s in feasible_strs]
    std_infeasible = max(0.0, 1.0 - sum(std_probs))
    std_probs.append(std_infeasible)
    
    xy_probs = [res_xy["state_probabilities"].get(s, 0.0) for s in feasible_strs]
    xy_infeasible = max(0.0, 1.0 - sum(xy_probs))
    xy_probs.append(xy_infeasible)
    
    x_indices = np.arange(len(labels))
    width = 0.28
    
    rects1 = ax.bar(x_indices - width, fmqa_probs, width, label='FMQA (SA, penalty)', color='#4C72B0', alpha=0.9)
    rects2 = ax.bar(x_indices, std_probs, width, label='Standard FM-QAOA (X-mixer, penalty)', color='#DD8452', alpha=0.9)
    rects3 = ax.bar(x_indices + width, xy_probs, width, label='FM-XY-QAOA (Proposed, penalty-free)', color='#55A868', alpha=0.9)
    
    # 最適解のバーを枠線でハイライト
    opt_idx = feasible_strs.index(best_x_str)
    rects3[opt_idx].set_edgecolor('black')
    rects3[opt_idx].set_linewidth(2.0)
    
    ax.set_ylabel('Sampling Probability', fontsize=12)
    ax.set_title(f'Sampling Probability Distribution Comparison (N={sum(block_sizes)}, blocks={block_sizes}, $\lambda$={lambda_val})', fontsize=13, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    png_path = f"{output_prefix}_distribution.png"
    pdf_path = f"{output_prefix}_distribution.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"プロット保存: {png_path} および {pdf_path}")
    
    return {
        "fmqa": {
            "feasibility_rate": float(res_fmqa["feasibility_rate"]),
            "opt_prob": float(p_opt_fmqa),
            "time": float(t_fmqa)
        },
        "standard_qaoa": {
            "feasibility_rate": float(res_std["feasibility_rate"]),
            "opt_prob": float(p_opt_std),
            "time": float(t_std)
        },
        "xy_qaoa": {
            "feasibility_rate": float(res_xy["feasibility_rate"]),
            "opt_prob": float(p_opt_xy),
            "time": float(t_xy)
        }
    }

def run_lambda_sweep_comparison(
    qubo_dict: Dict[tuple, float],
    block_sizes: List[int],
    lambda_list: List[float],
    exact_best_x: np.ndarray,
    exact_min_val: float,
    output_prefix: str,
    qaoa_reps: int = 1,
    qaoa_maxiter: int = 50
) -> Dict[str, Any]:
    print(f"\n==========================================")
    print(f" 実験2: ペナルティ係数 lambda スイープ感度分析")
    print(f" lambda values: {lambda_list}")
    print(f"==========================================")
    
    best_x_str = "".join(str(int(b)) for b in exact_best_x)
    
    # 提案手法 (FM-XY-QAOA) は lambda に非依存なので 1 回だけ実行
    print("\n--- Running Proposed FM-XY-QAOA (lambda-free) ---")
    res_xy = solve_xy_qaoa(qubo_dict, block_sizes, reps=qaoa_reps, maxiter=qaoa_maxiter)
    xy_feas = float(res_xy["feasibility_rate"])
    xy_opt = float(res_xy["state_probabilities"].get(best_x_str, 0.0))
    print(f"FM-XY-QAOA: Feasibility = {xy_feas*100:.1f}%, Opt Prob = {xy_opt*100:.2f}%")
    
    fmqa_feas_list = []
    fmqa_opt_list = []
    
    std_feas_list = []
    std_opt_list = []
    
    for lam in lambda_list:
        print(f"\n--- Testing lambda = {lam} ---")
        # FMQA
        res_fmqa = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=lam, num_reads=1000, seed=42)
        f_feas = float(res_fmqa["feasibility_rate"])
        f_opt = float(res_fmqa["state_probabilities"].get(best_x_str, 0.0))
        fmqa_feas_list.append(f_feas)
        fmqa_opt_list.append(f_opt)
        print(f"  [FMQA] Feasibility: {f_feas*100:.1f}%, Opt Prob: {f_opt*100:.2f}%")
        
        # Standard QAOA
        res_std = solve_standard_qaoa(qubo_dict, block_sizes, lambda_penalty=lam, reps=qaoa_reps, maxiter=qaoa_maxiter)
        s_feas = float(res_std["feasibility_rate"])
        s_opt = float(res_std["state_probabilities"].get(best_x_str, 0.0))
        std_feas_list.append(s_feas)
        std_opt_list.append(s_opt)
        print(f"  [Std QAOA] Feasibility: {s_feas*100:.1f}%, Opt Prob: {s_opt*100:.2f}%")
        
    # プロットの作成 (2軸または2サブプロット)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 制約充足率
    ax1.plot(lambda_list, [f * 100 for f in fmqa_feas_list], marker='o', label='FMQA (SA)', color='#4C72B0', linewidth=2)
    ax1.plot(lambda_list, [s * 100 for s in std_feas_list], marker='s', label='Standard FM-QAOA', color='#DD8452', linewidth=2)
    ax1.axhline(y=xy_feas * 100, color='#55A868', linestyle='--', linewidth=2.5, label=f'FM-XY-QAOA (Proposed: {xy_feas*100:.0f}%)')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontsize=11)
    ax1.set_ylabel('Feasibility Rate (%)', fontsize=11)
    ax1.set_title('Constraint Feasibility vs Penalty $\lambda$', fontsize=12, fontweight='bold')
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 最適解到達確率
    ax2.plot(lambda_list, [f * 100 for f in fmqa_opt_list], marker='o', label='FMQA (SA)', color='#4C72B0', linewidth=2)
    ax2.plot(lambda_list, [s * 100 for s in std_opt_list], marker='s', label='Standard FM-QAOA', color='#DD8452', linewidth=2)
    ax2.axhline(y=xy_opt * 100, color='#55A868', linestyle='--', linewidth=2.5, label=f'FM-XY-QAOA (Proposed: {xy_opt*100:.1f}%)')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontsize=11)
    ax2.set_ylabel('Optimal Solution Probability $P(x^*)$ (%)', fontsize=11)
    ax2.set_title('Success Probability $P(x^*)$ vs Penalty $\lambda$', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    png_path = f"{output_prefix}_lambda_sensitivity.png"
    pdf_path = f"{output_prefix}_lambda_sensitivity.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"プロット保存: {png_path} および {pdf_path}")
    
    return {
        "lambda_list": lambda_list,
        "xy_qaoa": {"feasibility": xy_feas, "opt_prob": xy_opt},
        "fmqa": {"feasibility": fmqa_feas_list, "opt_prob": fmqa_opt_list},
        "standard_qaoa": {"feasibility": std_feas_list, "opt_prob": std_opt_list}
    }

def main():
    os.makedirs(os.path.join("result", "txt"), exist_ok=True)
    os.makedirs(os.path.join("result", "json"), exist_ok=True)
    os.makedirs(os.path.join("result", "png"), exist_ok=True)
    os.makedirs(os.path.join("result", "pdf"), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("result", "txt", f"comparison_{timestamp}.txt")
    sys.stdout = Logger(log_path)
    
    print(f"============================================================")
    print(f" 3手法比較実験: FMQA vs Standard FM-QAOA vs FM-XY-QAOA")
    print(f" タイムスタンプ: {timestamp}")
    print(f"============================================================")
    
    # 難易度A: 6量子ビット (2カテゴリ x 3選択肢: [3, 3], 全64状態, 有効9状態)
    # 難易度B: 8量子ビット (2カテゴリ x 4選択肢: [4, 4], 全256状態, 有効16状態)
    experiments = [
        {"name": "DifficultyA_N6", "block_sizes": [3, 3], "seed": 42, "lambda_val": 5.0, "lambda_sweep": [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]},
        {"name": "DifficultyB_N8", "block_sizes": [4, 4], "seed": 42, "lambda_val": 5.0, "lambda_sweep": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]}
    ]
    
    all_results = {}
    
    for exp in experiments:
        name = exp["name"]
        block_sizes = exp["block_sizes"]
        total_qubits = sum(block_sizes)
        seed = exp["seed"]
        lambda_val = exp["lambda_val"]
        lambda_sweep = exp["lambda_sweep"]
        
        print(f"\n############################################################")
        print(f" 実験開始: {name} (量子ビット数: {total_qubits}, ブロック: {block_sizes})")
        print(f" 全状態空間: 2^{total_qubits} = {2**total_qubits}, 有効解数: {np.prod(block_sizes)} ({np.prod(block_sizes)/2**total_qubits*100:.2f}%)")
        print(f"############################################################")
        
        # ランダムQUBO BB関数の生成
        Q = create_random_qubo_bb(total_qubits, seed=seed)
        qubo_dict = matrix_to_qubo_dict(Q)
        
        # 厳密解の計算
        exact_best_x, exact_min_val = get_exact_minimum_categorical_bb(Q, block_sizes)
        print(f"[厳密最適解 x*] {''.join(str(int(b)) for b in exact_best_x)} | 最小値: {exact_min_val:.4f}")
        
        prefix = os.path.join("result", "png", f"comp_{name}_{timestamp}")
        
        # 1. 確率分布比較
        dist_res = run_distribution_comparison(
            qubo_dict=qubo_dict,
            block_sizes=block_sizes,
            lambda_val=lambda_val,
            exact_best_x=exact_best_x,
            exact_min_val=exact_min_val,
            output_prefix=prefix,
            qaoa_reps=1,
            qaoa_maxiter=50
        )
        
        # 2. lambda感度分析
        sweep_res = run_lambda_sweep_comparison(
            qubo_dict=qubo_dict,
            block_sizes=block_sizes,
            lambda_list=lambda_sweep,
            exact_best_x=exact_best_x,
            exact_min_val=exact_min_val,
            output_prefix=prefix,
            qaoa_reps=1,
            qaoa_maxiter=50
        )
        
        all_results[name] = {
            "block_sizes": block_sizes,
            "total_qubits": total_qubits,
            "exact_min_val": float(exact_min_val),
            "exact_best_x": [int(b) for b in exact_best_x],
            "distribution_experiment": dist_res,
            "lambda_sweep_experiment": sweep_res
        }
        
    json_path = os.path.join("result", "json", f"comparison_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n============================================================")
    print(f" 全実験が完了しました！ 結果JSON: {json_path}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
