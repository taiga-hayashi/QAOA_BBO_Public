import time
import json
import numpy as np
from typing import List, Dict, Any

from bb_function import create_random_qubo_bb, get_exact_minimum_categorical_bb
from fmqa_solver import solve_fmqa
from subspace_xy_qaoa_solver import solve_subspace_xy_qaoa
from qaoa_solver import solve_standard_qaoa

def run_multi_instance_benchmark(num_instances: int = 10):
    print("================================================================================")
    print(f"  Multi-Instance Large-Scale Benchmark (N=20, 24, 28, 32 | {num_instances} Instances)")
    print("================================================================================")
    
    scales = [
        {"N": 20, "M": 5, "d": 4},
        {"N": 24, "M": 6, "d": 4},
        {"N": 28, "M": 7, "d": 4},
        {"N": 32, "M": 8, "d": 4},
    ]
    
    seeds = [42, 101, 2024, 777, 999, 123, 456, 789, 314, 555][:num_instances]
    
    results = {}
    
    for scale in scales:
        N = scale["N"]
        M = scale["M"]
        d = scale["d"]
        block_sizes = [d] * M
        total_states = 2**N
        subspace_dim = d**M
        
        print(f"\n>>> Running Scale N={N} (M={M}, d={d}) | Total: {total_states:,} | Feasible: {subspace_dim:,}")
        
        scale_data = {
            "N": N, "M": M, "d": d, "total_states": total_states, "subspace_dim": subspace_dim,
            "exact_mins": [],
            "fmqa": {"best_vals": [], "feas_rates": [], "times": [], "diffs": []},
            "xy_qaoa": {"best_vals": [], "feas_rates": [], "times": [], "diffs": []},
            "std_qaoa": {"best_vals": [], "feas_rates": [], "times": [], "feas_count": 0}
        }
        
        for idx, seed in enumerate(seeds):
            print(f"  [Instance {idx+1}/{num_instances} (Seed {seed})]")
            Q = create_random_qubo_bb(N, seed=seed)
            qubo_dict = {(i, j): float(Q[i, j]) for i in range(N) for j in range(i, N)}
            
            # Exact Minimum
            opt_x, opt_val = get_exact_minimum_categorical_bb(Q, block_sizes)
            scale_data["exact_mins"].append(opt_val)
            
            # 1. FMQA
            t0 = time.time()
            res_fmqa = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=5.0, num_reads=500, seed=seed)
            t_fmqa = time.time() - t0
            best_fmqa = res_fmqa["best_feasible_sample"]["fval"] if res_fmqa["best_feasible_sample"] else None
            feas_fmqa = res_fmqa["feasibility_rate"] * 100
            diff_fmqa = (best_fmqa - opt_val) if best_fmqa is not None else np.nan
            
            scale_data["fmqa"]["best_vals"].append(best_fmqa)
            scale_data["fmqa"]["feas_rates"].append(feas_fmqa)
            scale_data["fmqa"]["times"].append(t_fmqa)
            scale_data["fmqa"]["diffs"].append(diff_fmqa)
            
            # 2. FM-XY-QAOA (Subspace)
            t0 = time.time()
            res_xy = solve_subspace_xy_qaoa(Q, block_sizes, reps=1, maxiter=20)
            t_xy = time.time() - t0
            best_xy = res_xy["best_sampled_energy"]
            feas_xy = res_xy["feasibility_rate"] * 100
            diff_xy = best_xy - opt_val
            
            scale_data["xy_qaoa"]["best_vals"].append(best_xy)
            scale_data["xy_qaoa"]["feas_rates"].append(feas_xy)
            scale_data["xy_qaoa"]["times"].append(t_xy)
            scale_data["xy_qaoa"]["diffs"].append(diff_xy)
            
            print(f"    FMQA: Feas={feas_fmqa:5.1f}% | Diff={diff_fmqa:+.4f} | Time={t_fmqa:.2f}s")
            print(f"    XY  : Feas={feas_xy:5.1f}% | Diff={diff_xy:+.4f} | Time={t_xy:.2f}s")
            
            # 3. Standard QAOA (only run for first 3 seeds on N=20 and 1 on N=24 to avoid extreme runtime)
            if N == 20 and idx < 3:
                t0 = time.time()
                res_std = solve_standard_qaoa(qubo_dict, block_sizes, lambda_penalty=5.0, reps=1, maxiter=10)
                t_std = time.time() - t0
                best_std = res_std["best_feasible_sample"]["fval"] if res_std["best_feasible_sample"] else None
                feas_std = res_std["feasibility_rate"] * 100
                scale_data["std_qaoa"]["best_vals"].append(best_std)
                scale_data["std_qaoa"]["feas_rates"].append(feas_std)
                scale_data["std_qaoa"]["times"].append(t_std)
                print(f"    Std : Feas={feas_std:5.2f}% | Best={best_std} | Time={t_std:.2f}s")
            elif N == 24 and idx == 0:
                t0 = time.time()
                res_std = solve_standard_qaoa(qubo_dict, block_sizes, lambda_penalty=5.0, reps=1, maxiter=5)
                t_std = time.time() - t0
                best_std = res_std["best_feasible_sample"]["fval"] if res_std["best_feasible_sample"] else None
                feas_std = res_std["feasibility_rate"] * 100
                scale_data["std_qaoa"]["best_vals"].append(best_std)
                scale_data["std_qaoa"]["feas_rates"].append(feas_std)
                scale_data["std_qaoa"]["times"].append(t_std)
                print(f"    Std : Feas={feas_std:5.2f}% | Best={best_std} | Time={t_std:.2f}s")
                
        results[N] = scale_data
        
    # Save results to json
    with open("report/multi_instance_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n================================================================================")
    print("  Benchmark Completed! Summary of Results (Mean +/- Std)")
    print("================================================================================")
    for N, data in results.items():
        opt_mean = np.mean(data["exact_mins"])
        opt_std = np.std(data["exact_mins"])
        
        fmqa_val_m = np.mean(data["fmqa"]["best_vals"])
        fmqa_val_s = np.std(data["fmqa"]["best_vals"])
        fmqa_feas_m = np.mean(data["fmqa"]["feas_rates"])
        fmqa_feas_s = np.std(data["fmqa"]["feas_rates"])
        fmqa_t_m = np.mean(data["fmqa"]["times"])
        
        xy_val_m = np.mean(data["xy_qaoa"]["best_vals"])
        xy_val_s = np.std(data["xy_qaoa"]["best_vals"])
        xy_feas_m = np.mean(data["xy_qaoa"]["feas_rates"])
        xy_feas_s = np.std(data["xy_qaoa"]["feas_rates"])
        xy_t_m = np.mean(data["xy_qaoa"]["times"])
        
        print(f"\n[N={N} (10 instances)]")
        print(f"  Exact Optimum    : {opt_mean:.4f} +/- {opt_std:.4f}")
        print(f"  FMQA (SA)        : Val = {fmqa_val_m:.4f} +/- {fmqa_val_s:.4f} | Feas = {fmqa_feas_m:.2f}% +/- {fmqa_feas_s:.2f}% | Time = {fmqa_t_m:.2f}s")
        print(f"  FM-XY-QAOA (Sub) : Val = {xy_val_m:.4f} +/- {xy_val_s:.4f} | Feas = {xy_feas_m:.2f}% +/- {xy_feas_s:.2f}% | Time = {xy_t_m:.2f}s")

if __name__ == "__main__":
    run_multi_instance_benchmark(num_instances=10)
