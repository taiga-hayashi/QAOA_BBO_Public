import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from bb_function import (
    create_random_qubo_bb,
    get_exact_minimum_categorical_bb,
    is_feasible_one_hot
)
from fmqa_solver import solve_fmqa
from qaoa_solver import solve_xy_qaoa
from qubo import matrix_to_qubo_dict

def run_fmqa_penalty_sweep():
    print("============================================================")
    print(" FMQA ペナルティ係数 lambda 網羅的スイープ実験")
    print("============================================================")
    
    # lambda values: 0.1 to 100.0 (10 points log scale)
    lambda_list = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    # Test scales
    scales = [
        {"name": "N8_4x2",   "block_sizes": [4, 4],             "N": 8},
        {"name": "N12_4x3",  "block_sizes": [4, 4, 4],          "N": 12},
        {"name": "N16_4x4",  "block_sizes": [4, 4, 4, 4],       "N": 16},
        {"name": "N20_4x5",  "block_sizes": [4, 4, 4, 4, 4],    "N": 20},
    ]
    
    seeds = [42, 101, 2024]
    
    all_results = {}
    
    for sc in scales:
        sname = sc["name"]
        blocks = sc["block_sizes"]
        N = sc["N"]
        num_valid = int(np.prod(blocks))
        print(f"\n>>> 測定中: {sname} (N={N}, 有効解数={num_valid}) <<<")
        
        # FM-XY-QAOA の基準値 (lambda不要) を複数シードで測定 (OpenQARP)
        xy_feas_seeds = []
        xy_opt_seeds = []
        xy_unique_seeds = []
        for s in seeds:
            Q = create_random_qubo_bb(N, seed=s)
            q_dict = matrix_to_qubo_dict(Q)
            best_x, _ = get_exact_minimum_categorical_bb(Q, blocks)
            best_x_str = "".join(str(int(b)) for b in best_x)
            
            res_xy = solve_xy_qaoa(q_dict, blocks, reps=1, maxiter=10)
            xy_feas_seeds.append(res_xy["feasibility_rate"])
            xy_opt_seeds.append(res_xy["state_probabilities"].get(best_x_str, 0.0))
            # ユニーク有効解数
            u_valid = sum(1 for bstr, p in res_xy["state_probabilities"].items() 
                          if p > 1e-4 and is_feasible_one_hot([int(c) for c in bstr], blocks))
            xy_unique_seeds.append(u_valid)
            
        xy_baseline = {
            "feasibility_mean": float(np.mean(xy_feas_seeds)),
            "opt_prob_mean": float(np.mean(xy_opt_seeds)),
            "unique_valid_mean": float(np.mean(xy_unique_seeds))
        }
        print(f"  [FM-XY-QAOA Baseline] Feas={xy_baseline['feasibility_mean']*100:.1f}%, "
              f"OptProb={xy_baseline['opt_prob_mean']*100:.2f}%, "
              f"UniqueValid={xy_baseline['unique_valid_mean']:.1f}")
        
        # FMQA の lambda スイープ
        fmqa_by_lambda = []
        for lam in lambda_list:
            feas_list, opt_list, unique_list, trap_rate_list = [], [], [], []
            for s in seeds:
                Q = create_random_qubo_bb(N, seed=s)
                q_dict = matrix_to_qubo_dict(Q)
                best_x, _ = get_exact_minimum_categorical_bb(Q, blocks)
                best_x_str = "".join(str(int(b)) for b in best_x)
                
                # FMQA SA (num_reads=1000)
                res_fmqa = solve_fmqa(q_dict, blocks, lambda_penalty=lam, num_reads=1000, seed=s)
                feas_list.append(res_fmqa["feasibility_rate"])
                opt_list.append(res_fmqa["state_probabilities"].get(best_x_str, 0.0))
                
                # 有効解のユニーク数 & 最多出現サンプルの集中度 (局所解トラップ率)
                valid_counts = {}
                for bstr, p in res_fmqa["state_probabilities"].items():
                    if is_feasible_one_hot([int(c) for c in bstr], blocks):
                        valid_counts[bstr] = p
                unique_list.append(len(valid_counts))
                if valid_counts:
                    max_p = max(valid_counts.values())
                    trap_rate_list.append(max_p)
                else:
                    trap_rate_list.append(0.0)
                    
            record = {
                "lambda": lam,
                "feasibility_mean": float(np.mean(feas_list)),
                "feasibility_std": float(np.std(feas_list)),
                "opt_prob_mean": float(np.mean(opt_list)),
                "opt_prob_std": float(np.std(opt_list)),
                "unique_valid_mean": float(np.mean(unique_list)),
                "trap_rate_mean": float(np.mean(trap_rate_list))
            }
            fmqa_by_lambda.append(record)
            print(f"  lambda={lam:5.1f} | Feas={record['feasibility_mean']*100:5.1f}% | "
                  f"OptProb={record['opt_prob_mean']*100:5.2f}% | "
                  f"UniqueValid={record['unique_valid_mean']:4.1f} | "
                  f"MaxTrapProb={record['trap_rate_mean']*100:5.1f}%")
            
        all_results[sname] = {
            "scale": sc,
            "xy_baseline": xy_baseline,
            "fmqa_sweep": fmqa_by_lambda
        }
        
    # 保存
    os.makedirs("result/json", exist_ok=True)
    os.makedirs("report/figures", exist_ok=True)
    json_path = "result/json/fmqa_penalty_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n結果JSONを保存: {json_path}")
    
    return all_results

if __name__ == "__main__":
    run_fmqa_penalty_sweep()
