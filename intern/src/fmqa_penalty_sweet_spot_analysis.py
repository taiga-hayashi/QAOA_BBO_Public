import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bb_function import evaluate_bb, get_feasible_patterns, is_feasible_one_hot
from fmqa_solver import solve_fmqa
from materials_scaling_benchmark import create_scalable_materials_ground_truth, get_exact_minimum_materials

def compute_theoretical_adaptive_lambda(Q: np.ndarray, block_sizes: List[int], safety_factor: float = 1.25) -> float:
    """
    強相互作用行列 Q と One-Hot ブロック構造から、
    制約違反が目的関数の利得を上回らない理論的ペナルティ下限値 λ_adaptive を計算する。
    """
    n = sum(block_sizes)
    M = len(block_sizes)
    
    block_indices = []
    idx = 0
    for sz in block_sizes:
        block_indices.append(list(range(idx, idx + sz)))
        idx += sz
        
    # 各ブロック m において、追加で 1 ビット立てたときに獲得しうる最大引力エネルギー (負の結合)
    max_gains = []
    for m in range(M):
        inds_m = block_indices[m]
        gain_m_max = 0.0
        for i in inds_m:
            gain_i = 0.0
            # 他ブロックとの相互作用
            for other_m in range(M):
                if other_m == m:
                    continue
                other_inds = block_indices[other_m]
                # other_m 内で最も強く引き合う (最も負の大きな) 結合
                best_attraction = 0.0
                for j in other_inds:
                    q_val = Q[min(i, j), max(i, j)]
                    if q_val < 0:
                        best_attraction = max(best_attraction, -q_val)
                gain_i += best_attraction
            gain_m_max = max(gain_m_max, gain_i)
        max_gains.append(gain_m_max)
        
    crit_lambda = max(max_gains) if max_gains else 5.0
    return float(np.round(crit_lambda * safety_factor, 2))

def run_penalty_sweet_spot_study():
    print("================================================================================")
    print("  FMQA Penalty Adaptive Tuning & Sweet Spot Sweep Benchmark")
    print("================================================================================")
    
    scales = [
        {"N": 16, "M": 4, "d": 4},
        {"N": 20, "M": 5, "d": 4},
        {"N": 24, "M": 6, "d": 4},
        {"N": 28, "M": 7, "d": 4},
        {"N": 32, "M": 8, "d": 4},
    ]
    
    lambda_list = [1.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 30.0, 50.0]
    seeds = [42, 101, 2024, 777, 999]
    
    results = {}
    
    for scale in scales:
        N = scale["N"]
        M = scale["M"]
        d = scale["d"]
        block_sizes = [d] * M
        
        print(f"\n>>> Analyzing Scale N={N} (M={M} sites, d={d} elements)")
        
        scale_results = {
            "N": N, "M": M, "d": d,
            "lambda_sweep": [],
            "adaptive_eval": {
                "lambdas": [],
                "feas": [],
                "gaps": [],
                "succ_rates": []
            }
        }
        
        # 1. Lambda Sweep Analysis
        for lam in lambda_list:
            feas_runs = []
            gap_runs = []
            succ_runs = []
            unique_runs = []
            
            for seed in seeds:
                Q = create_scalable_materials_ground_truth(block_sizes, seed=seed)
                qubo_dict = {(i, j): float(Q[i, j]) for i in range(N) for j in range(i, N)}
                opt_x, opt_val = get_exact_minimum_materials(Q, block_sizes)
                
                res = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=lam, num_reads=500, seed=seed)
                feas = res["feasibility_rate"] * 100.0
                feas_runs.append(feas)
                
                best_sample = res["best_feasible_sample"]
                if best_sample is not None:
                    gap = best_sample["fval"] - opt_val
                    gap_runs.append(gap)
                    succ_runs.append(1.0 if abs(gap) < 1e-4 else 0.0)
                else:
                    gap_runs.append(np.nan)
                    succ_runs.append(0.0)
                    
                unique_feas = sum(
                    1 for bit_str in res["state_probabilities"].keys()
                    if is_feasible_one_hot([int(b) for b in bit_str], block_sizes)
                )
                unique_runs.append(unique_feas)
                
            mean_feas = float(np.mean(feas_runs))
            valid_gaps = [g for g in gap_runs if not np.isnan(g)]
            mean_gap = float(np.mean(valid_gaps)) if valid_gaps else None
            succ_rate = float(np.mean(succ_runs) * 100.0)
            mean_unique = float(np.mean(unique_runs))
            
            scale_results["lambda_sweep"].append({
                "lambda": lam,
                "feas_mean": mean_feas,
                "gap_mean": mean_gap,
                "succ_rate": succ_rate,
                "unique_mean": mean_unique
            })
            print(f"  λ={lam:4.1f} | Feas: {mean_feas:5.1f}% | Gap: {f'{mean_gap:.3f}' if mean_gap is not None else 'N/A':>7} | Succ: {succ_rate:5.1f}% | Unique: {mean_unique:4.1f}")
            
        # 2. Adaptive Lambda Validation
        print(f"  [Evaluating Adaptive Penalty λ_adaptive]")
        for seed in seeds:
            Q = create_scalable_materials_ground_truth(block_sizes, seed=seed)
            qubo_dict = {(i, j): float(Q[i, j]) for i in range(N) for j in range(i, N)}
            opt_x, opt_val = get_exact_minimum_materials(Q, block_sizes)
            
            lam_adapt = compute_theoretical_adaptive_lambda(Q, block_sizes, safety_factor=1.25)
            res_adapt = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=lam_adapt, num_reads=500, seed=seed)
            
            feas_a = res_adapt["feasibility_rate"] * 100.0
            best_sample = res_adapt["best_feasible_sample"]
            gap_a = best_sample["fval"] - opt_val if best_sample else np.nan
            succ_a = 1.0 if (best_sample and abs(gap_a) < 1e-4) else 0.0
            
            scale_results["adaptive_eval"]["lambdas"].append(lam_adapt)
            scale_results["adaptive_eval"]["feas"].append(feas_a)
            scale_results["adaptive_eval"]["gaps"].append(gap_a)
            scale_results["adaptive_eval"]["succ_rates"].append(succ_a)
            
        mean_lam_adapt = float(np.mean(scale_results["adaptive_eval"]["lambdas"]))
        mean_feas_adapt = float(np.mean(scale_results["adaptive_eval"]["feas"]))
        mean_gap_adapt = float(np.mean(scale_results["adaptive_eval"]["gaps"]))
        mean_succ_adapt = float(np.mean(scale_results["adaptive_eval"]["succ_rates"]) * 100.0)
        print(f"  >> Adaptive λ_mean={mean_lam_adapt:.1f} | Feas: {mean_feas_adapt:5.1f}% | Gap: {mean_gap_adapt:.3f} | Succ: {mean_succ_adapt:5.1f}%")
        
        results[str(N)] = scale_results
        
    os.makedirs("report", exist_ok=True)
    out_path = "report/materials_penalty_sweet_spot_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Sweet spot results saved to {out_path}")
    
    return results

if __name__ == "__main__":
    run_penalty_sweet_spot_study()
