import os
import sys
import json
import time
import copy
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any

from bb_function import (
    create_random_qubo_bb,
    get_exact_minimum_categorical_bb,
    generate_categorical_dataset,
    is_feasible_one_hot,
    evaluate_bb
)
from fm import train_factorization_machine
from fm_to_qubo import fm_to_qubo
from fmqa_solver import solve_fmqa
from qaoa_solver import solve_standard_qaoa, solve_xy_qaoa
from utils import Logger

def run_single_bbo_pipeline(
    solver_type: str,
    Q_bb: np.ndarray,
    block_sizes: List[int],
    initial_X: torch.Tensor,
    initial_y: torch.Tensor,
    num_cycles: int = 10,
    lambda_penalty: float = 5.0,
    k: int = 2,
    epochs: int = 100,
    lr: float = 0.1
) -> Dict[str, Any]:
    total_vars = sum(block_sizes)
    
    X_list = [initial_X.clone()]
    y_list = [initial_y.clone()]
    
    current_best = torch.min(initial_y).item()
    history_best = [current_best]
    
    infeasible_proposals = 0
    proposals_info = []
    
    print(f"\n>>> Starting BBO Loop for [{solver_type}] <<<")
    
    for cycle in range(1, num_cycles + 1):
        X_curr = torch.cat(X_list, dim=0)
        y_curr = torch.cat(y_list, dim=0)
        
        # FM学習
        model = train_factorization_machine(
            X_curr, y_curr, d=total_vars, k=k, epochs=epochs, learning_rate=lr
        )
        qubo_dict, offset = fm_to_qubo(model)
        
        # ソルバーによる最適化 (次の一手の提案)
        if solver_type == "FMQA":
            res = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=lambda_penalty, num_reads=500, seed=cycle*100)
            best_sample = res["best_overall_sample"]
            proposed_x = np.array(best_sample["x"], dtype=np.float32).reshape(1, -1)
        elif solver_type == "Standard QAOA":
            res = solve_standard_qaoa(qubo_dict, block_sizes, lambda_penalty=lambda_penalty, reps=1, maxiter=40)
            best_sample = res["best_overall_sample"]
            proposed_x = np.array(best_sample["x"], dtype=np.float32).reshape(1, -1)
        elif solver_type == "FM-XY-QAOA":
            res = solve_xy_qaoa(qubo_dict, block_sizes, reps=1, maxiter=40)
            best_sample = res["best_overall_sample"]
            proposed_x = np.array(best_sample["x"], dtype=np.float32).reshape(1, -1)
        else:
            raise ValueError(f"Unknown solver: {solver_type}")
            
        is_feas = is_feasible_one_hot(proposed_x[0], block_sizes)
        
        if is_feas:
            y_eval = float(evaluate_bb(proposed_x, Q_bb)[0])
            feas_str = "Feasible"
            if y_eval < current_best:
                current_best = y_eval
        else:
            # 無効解（制約違反）の場合はペナルティ評価（最悪値）となり探索失敗
            infeasible_proposals += 1
            y_eval = float(np.max(evaluate_bb(np.array(X_curr), Q_bb)) + 10.0) # ペナルティスコア
            feas_str = "★ INFEASIBLE (Penalty)"
            
        history_best.append(current_best)
        
        X_list.append(torch.tensor(proposed_x))
        y_list.append(torch.tensor([y_eval], dtype=torch.float32))
        
        proposals_info.append({
            "cycle": cycle,
            "x": [int(xi) for xi in proposed_x[0]],
            "is_feasible": is_feas,
            "y_eval": y_eval,
            "current_best": current_best
        })
        
        print(f"Cycle {cycle:2d} | x={proposed_x[0].astype(int)} | {feas_str} | y={y_eval:.4f} | Best={current_best:.4f}")
        
    return {
        "solver": solver_type,
        "history_best": history_best,
        "infeasible_proposals": infeasible_proposals,
        "infeasible_ratio": infeasible_proposals / num_cycles,
        "final_best": current_best,
        "proposals": proposals_info
    }

def main():
    os.makedirs(os.path.join("result", "txt"), exist_ok=True)
    os.makedirs(os.path.join("result", "json"), exist_ok=True)
    os.makedirs(os.path.join("result", "png"), exist_ok=True)
    os.makedirs(os.path.join("result", "pdf"), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("result", "txt", f"bbo_comp_{timestamp}.txt")
    sys.stdout = Logger(log_path)
    
    print(f"============================================================")
    print(f" BBO探索ループ比較実験: FMQA vs Standard QAOA vs FM-XY-QAOA")
    print(f" タイムスタンプ: {timestamp}")
    print(f"============================================================")
    
    # 8量子ビット (2カテゴリ x 4選択肢: [4, 4], 全256状態, 有効16状態)
    block_sizes = [4, 4]
    total_qubits = sum(block_sizes)
    seed = 42
    init_seed = 4 # 初期データセットに最適解が含まれないシード
    num_initial_samples = 6
    num_cycles = 10
    lambda_penalty = 5.0
    
    Q_bb = create_random_qubo_bb(total_qubits, seed=seed)
    exact_best_x, exact_min_val = get_exact_minimum_categorical_bb(Q_bb, block_sizes)
    print(f"[真の大域最適解] {''.join(str(int(b)) for b in exact_best_x)} | 目的値: {exact_min_val:.4f}")
    
    # 公平な初期データセット（全手法で共通の初期点を使用）
    init_X, init_y = generate_categorical_dataset(Q_bb, num_initial_samples, block_sizes, seed=init_seed)
    print(f"初期サンプル数: {num_initial_samples} | 初期最良値: {torch.min(init_y).item():.4f}")
    
    # 3手法の実行
    results = {}
    for solver in ["FMQA", "Standard QAOA", "FM-XY-QAOA"]:
        res = run_single_bbo_pipeline(
            solver_type=solver,
            Q_bb=Q_bb,
            block_sizes=block_sizes,
            initial_X=init_X,
            initial_y=init_y,
            num_cycles=num_cycles,
            lambda_penalty=lambda_penalty
        )
        results[solver] = res
        
    print("\n============================================================")
    print(" BBO探索結果サマリー")
    print("============================================================")
    for solver, res in results.items():
        print(f"[{solver:15s}] 最終到達最良値: {res['final_best']:.4f} | 制約違反提案数: {res['infeasible_proposals']}/{num_cycles} ({res['infeasible_ratio']*100:.1f}%)")
    print(f"[厳密解 (Theoretical Min)] {exact_min_val:.4f}")
    
    # プロット作成
    plt.figure(figsize=(9, 5.5))
    cycles = range(num_cycles + 1)
    
    plt.plot(cycles, results["FMQA"]["history_best"], marker='o', label=f'FMQA (SA, penalty) [Infeas: {results["FMQA"]["infeasible_proposals"]}]', color='#4C72B0', linewidth=2)
    plt.plot(cycles, results["Standard QAOA"]["history_best"], marker='s', label=f'Standard FM-QAOA [Infeas: {results["Standard QAOA"]["infeasible_proposals"]}]', color='#DD8452', linewidth=2)
    plt.plot(cycles, results["FM-XY-QAOA"]["history_best"], marker='^', label=f'FM-XY-QAOA (Proposed) [Infeas: {results["FM-XY-QAOA"]["infeasible_proposals"]}]', color='#55A868', linewidth=2.5)
    
    plt.axhline(y=exact_min_val, color='r', linestyle='--', label=f'Exact Minimum ({exact_min_val:.4f})')
    
    plt.xlabel('BBO Cycle (Evaluations)', fontsize=12)
    plt.ylabel('Best Found Objective Value', fontsize=12)
    plt.title(r'BBO Optimization Trajectory (N=' + str(total_qubits) + r', blocks=' + str(block_sizes) + r', $\lambda$=' + str(lambda_penalty) + r')', fontsize=13, fontweight='bold')
    plt.xticks(cycles)
    plt.legend(fontsize=10.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    png_path = os.path.join("result", "png", f"bbo_trajectory_comparison_N8_{timestamp}.png")
    pdf_path = os.path.join("result", "pdf", f"bbo_trajectory_comparison_N8_{timestamp}.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"\nプロット保存: {png_path} および {pdf_path}")
    
    json_path = os.path.join("result", "json", f"bbo_comparison_N8_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "block_sizes": block_sizes,
            "exact_min": float(exact_min_val),
            "exact_best_x": [int(b) for b in exact_best_x],
            "results": results
        }, f, indent=4)
    print(f"JSON保存: {json_path}")


if __name__ == "__main__":
    main()
