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

def run_scaling_benchmark(
    scales: List[Dict[str, Any]],
    seeds: List[int] = [42, 101, 2024],
    lambda_fixed: float = 5.0,
    qaoa_reps: int = 1,
    qaoa_maxiter: int = 40
) -> Dict[str, Any]:
    """
    問題サイズ（量子ビット数 N）をスケールさせ、3手法の性能を複数シードで平均評価する。
    """
    benchmark_results = []
    
    print(f"\n============================================================")
    print(f" スケーリング・ベンチマーク開始 (シード数: {len(seeds)}, 固定 lambda: {lambda_fixed})")
    print(f"============================================================")
    
    for item in scales:
        scale_name = item["name"]
        block_sizes = item["block_sizes"]
        total_qubits = sum(block_sizes)
        num_feasible = int(np.prod(block_sizes))
        total_states = 2**total_qubits
        feas_ratio = num_feasible / total_states * 100.0
        
        print(f"\n>>> 測定中: {scale_name} (N={total_qubits}, blocks={block_sizes}) <<<")
        print(f"    全空間: 2^{total_qubits} = {total_states} | 有効解: {num_feasible} ({feas_ratio:.3f}%)")
        
        fmqa_feas_list, fmqa_opt_list, fmqa_time_list = [], [], []
        std_feas_list, std_opt_list, std_time_list = [], [], []
        xy_feas_list, xy_opt_list, xy_time_list = [], [], []
        
        for seed in seeds:
            Q = create_random_qubo_bb(total_qubits, seed=seed)
            qubo_dict = matrix_to_qubo_dict(Q)
            exact_best_x, exact_min_val = get_exact_minimum_categorical_bb(Q, block_sizes)
            best_x_str = "".join(str(int(b)) for b in exact_best_x)
            
            # 1. FMQA
            t0 = time.time()
            res_fmqa = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=lambda_fixed, num_reads=1000, seed=seed)
            t_fmqa = time.time() - t0
            fmqa_feas_list.append(res_fmqa["feasibility_rate"])
            fmqa_opt_list.append(res_fmqa["state_probabilities"].get(best_x_str, 0.0))
            fmqa_time_list.append(t_fmqa)
            
            # 2. Standard QAOA
            t0 = time.time()
            res_std = solve_standard_qaoa(qubo_dict, block_sizes, lambda_penalty=lambda_fixed, reps=qaoa_reps, maxiter=qaoa_maxiter)
            t_std = time.time() - t0
            std_feas_list.append(res_std["feasibility_rate"])
            std_opt_list.append(res_std["state_probabilities"].get(best_x_str, 0.0))
            std_time_list.append(t_std)
            
            # 3. FM-XY-QAOA
            t0 = time.time()
            res_xy = solve_xy_qaoa(qubo_dict, block_sizes, reps=qaoa_reps, maxiter=qaoa_maxiter)
            t_xy = time.time() - t0
            xy_feas_list.append(res_xy["feasibility_rate"])
            xy_opt_list.append(res_xy["state_probabilities"].get(best_x_str, 0.0))
            xy_time_list.append(t_xy)
            
        record = {
            "name": scale_name,
            "total_qubits": total_qubits,
            "block_sizes": block_sizes,
            "total_states": total_states,
            "num_feasible": num_feasible,
            "feasible_volume_ratio": feas_ratio / 100.0,
            "random_guess_prob": 1.0 / total_states,
            "uniform_feasible_prob": 1.0 / num_feasible,
            "fmqa": {
                "feasibility_mean": float(np.mean(fmqa_feas_list)),
                "feasibility_std": float(np.std(fmqa_feas_list)),
                "opt_prob_mean": float(np.mean(fmqa_opt_list)),
                "opt_prob_std": float(np.std(fmqa_opt_list)),
                "time_mean": float(np.mean(fmqa_time_list))
            },
            "standard_qaoa": {
                "feasibility_mean": float(np.mean(std_feas_list)),
                "feasibility_std": float(np.std(std_feas_list)),
                "opt_prob_mean": float(np.mean(std_opt_list)),
                "opt_prob_std": float(np.std(std_opt_list)),
                "time_mean": float(np.mean(std_time_list))
            },
            "xy_qaoa": {
                "feasibility_mean": float(np.mean(xy_feas_list)),
                "feasibility_std": float(np.std(xy_feas_list)),
                "opt_prob_mean": float(np.mean(xy_opt_list)),
                "opt_prob_std": float(np.std(xy_opt_list)),
                "time_mean": float(np.mean(xy_time_list))
            }
        }
        benchmark_results.append(record)
        
        print(f"    [FMQA]     Feasibility: {record['fmqa']['feasibility_mean']*100:.1f}%, Opt Prob: {record['fmqa']['opt_prob_mean']*100:.2f}%")
        print(f"    [Std QAOA] Feasibility: {record['standard_qaoa']['feasibility_mean']*100:.1f}%, Opt Prob: {record['standard_qaoa']['opt_prob_mean']*100:.2f}%")
        print(f"    [XY-QAOA]  Feasibility: {record['xy_qaoa']['feasibility_mean']*100:.1f}%, Opt Prob: {record['xy_qaoa']['opt_prob_mean']*100:.2f}%")
        
    return {"benchmark_results": benchmark_results}

def plot_scaling_curves(benchmark_data: List[Dict[str, Any]], output_prefix: str):
    """
    スケーリング曲線をプロットする（論文・レポート用高精細スタイル）。
    """
    labels = [f"{d['name']}\n(N={d['total_qubits']})" for d in benchmark_data]
    qubits = [d['total_qubits'] for d in benchmark_data]
    
    # 1. 制約充足率のスケーリング推移
    plt.figure(figsize=(10, 5))
    x_pos = np.arange(len(labels))
    width = 0.25
    
    fmqa_feas = [d['fmqa']['feasibility_mean'] * 100 for d in benchmark_data]
    std_feas = [d['standard_qaoa']['feasibility_mean'] * 100 for d in benchmark_data]
    xy_feas = [d['xy_qaoa']['feasibility_mean'] * 100 for d in benchmark_data]
    volume_ratio = [d['feasible_volume_ratio'] * 100 for d in benchmark_data]
    
    plt.bar(x_pos - width, fmqa_feas, width, label='FMQA (SA, penalty)', color='#4C72B0', alpha=0.9)
    plt.bar(x_pos, std_feas, width, label='Standard FM-QAOA (penalty)', color='#DD8452', alpha=0.9)
    plt.bar(x_pos + width, xy_feas, width, label='FM-XY-QAOA (Proposed, penalty-free)', color='#55A868', alpha=0.9)
    plt.plot(x_pos, volume_ratio, color='gray', linestyle=':', marker='x', label='Feasible Volume Ratio (%)', linewidth=1.5)
    
    plt.ylabel('Constraint Feasibility Rate (%)', fontsize=12)
    plt.title('Constraint Feasibility Scaling across Problem Sizes', fontsize=13, fontweight='bold')
    plt.xticks(x_pos, labels, fontsize=10)
    plt.ylim(-2, 108)
    plt.legend(fontsize=10.5, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_feasibility_scaling.png", dpi=300)
    plt.savefig(f"{output_prefix}_feasibility_scaling.pdf")
    plt.close()
    
    # 2. 最適解到達確率のスケーリング推移 (対数スケール)
    plt.figure(figsize=(10, 5.5))
    
    fmqa_opt = [d['fmqa']['opt_prob_mean'] * 100 for d in benchmark_data]
    std_opt = [d['standard_qaoa']['opt_prob_mean'] * 100 for d in benchmark_data]
    xy_opt = [d['xy_qaoa']['opt_prob_mean'] * 100 for d in benchmark_data]
    uniform_feas = [d['uniform_feasible_prob'] * 100 for d in benchmark_data]
    random_guess = [d['random_guess_prob'] * 100 for d in benchmark_data]
    
    plt.plot(x_pos, fmqa_opt, marker='o', label='FMQA (SA)', color='#4C72B0', linewidth=2.2, markersize=7)
    plt.plot(x_pos, std_opt, marker='s', label='Standard FM-QAOA', color='#DD8452', linewidth=2.2, markersize=7)
    plt.plot(x_pos, xy_opt, marker='^', label='FM-XY-QAOA (Proposed)', color='#55A868', linewidth=2.5, markersize=8)
    plt.plot(x_pos, uniform_feas, linestyle='--', color='purple', alpha=0.6, label='Uniform over Feasible ($1/|\mathcal{H}_{\mathrm{feas}}|$)')
    plt.plot(x_pos, random_guess, linestyle=':', color='gray', alpha=0.6, label='Random Guess ($1/2^N$)')
    
    plt.yscale('log')
    plt.ylabel(r'Success Probability $P(x^*)$ (%) [log scale]', fontsize=12)
    plt.title(r'Optimal Solution Success Probability Scaling ($P(x^*)$)', fontsize=13, fontweight='bold')
    plt.xticks(x_pos, labels, fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_success_probability_scaling.png", dpi=300)
    plt.savefig(f"{output_prefix}_success_probability_scaling.pdf")
    plt.close()
    
    # 3. 濃縮度倍率（ランダム有効解に対するブースト率）
    plt.figure(figsize=(10, 5))
    xy_boost = [xy / uf if uf > 0 else 0 for xy, uf in zip(xy_opt, uniform_feas)]
    std_boost = [std / uf if uf > 0 else 0 for std, uf in zip(std_opt, uniform_feas)]
    fmqa_boost = [fm / uf if uf > 0 else 0 for fm, uf in zip(fmqa_opt, uniform_feas)]
    
    plt.plot(x_pos, fmqa_boost, marker='o', label='FMQA (SA)', color='#4C72B0', linewidth=2)
    plt.plot(x_pos, std_boost, marker='s', label='Standard FM-QAOA', color='#DD8452', linewidth=2)
    plt.plot(x_pos, xy_boost, marker='^', label='FM-XY-QAOA (Proposed)', color='#55A868', linewidth=2.5)
    plt.axhline(y=1.0, color='purple', linestyle='--', label='Uniform Baseline (1.0x)')
    
    plt.ylabel('Concentration Factor ($P(x^*) / P_{\mathrm{uniform}}$)', fontsize=12)
    plt.title('Quantum Optimization Concentration Advantage', fontsize=13, fontweight='bold')
    plt.xticks(x_pos, labels, fontsize=10)
    plt.legend(fontsize=10.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_concentration_boost.png", dpi=300)
    plt.savefig(f"{output_prefix}_concentration_boost.pdf")
    plt.close()
    
    print(f"プロット保存完了: {output_prefix}_*.png / pdf")

def main():
    os.makedirs(os.path.join("result", "txt"), exist_ok=True)
    os.makedirs(os.path.join("result", "json"), exist_ok=True)
    os.makedirs(os.path.join("result", "png"), exist_ok=True)
    os.makedirs(os.path.join("result", "pdf"), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("result", "txt", f"scaling_{timestamp}.txt")
    sys.stdout = Logger(log_path)
    
    print(f"============================================================")
    print(f" 3手法スケーリング包括的実験")
    print(f" タイムスタンプ: {timestamp}")
    print(f"============================================================")
    
    # スケーリングのテストスイート (7スケール)
    scales = [
        {"name": "N4_2x2",   "block_sizes": [2, 2]},          # N=4,  feas=4 / 16  (25.0%)
        {"name": "N6_3x3",   "block_sizes": [3, 3]},          # N=6,  feas=9 / 64  (14.06%)
        {"name": "N8_4x4",   "block_sizes": [4, 4]},          # N=8,  feas=16 / 256 (6.25%)
        {"name": "N9_3x3x3", "block_sizes": [3, 3, 3]},       # N=9,  feas=27 / 512 (5.27%)
        {"name": "N10_5x5",  "block_sizes": [5, 5]},          # N=10, feas=25 / 1024 (2.44%)
        {"name": "N12_3x4",  "block_sizes": [3, 3, 3, 3]},    # N=12, feas=81 / 4096 (1.98%)
        {"name": "N12_4x3",  "block_sizes": [4, 4, 4]}        # N=12, feas=64 / 4096 (1.56%)
    ]
    
    seeds = [42, 101, 2024]
    
    res = run_scaling_benchmark(
        scales=scales,
        seeds=seeds,
        lambda_fixed=5.0,
        qaoa_reps=1,
        qaoa_maxiter=40
    )
    
    json_path = os.path.join("result", "json", f"scaling_benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4)
    print(f"\n生データJSON保存: {json_path}")
    
    prefix = os.path.join("result", "png", f"scaling_{timestamp}")
    plot_scaling_curves(res["benchmark_results"], prefix)

if __name__ == "__main__":
    main()
