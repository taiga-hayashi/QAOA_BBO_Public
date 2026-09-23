import time
import json
import numpy as np
from typing import List, Dict, Any

from bb_function import create_random_qubo_bb, get_exact_minimum_categorical_bb
from fmqa_solver import solve_fmqa
from enhanced_subspace_xy_qaoa_solver import (
    build_subspace_hamiltonians,
    solve_enhanced_subspace_xy_qaoa,
    run_one_hot_local_search
)

def run_accuracy_benchmark(num_instances: int = 10):
    print("================================================================================")
    print(f"  QAOA Accuracy Enhancement Benchmark (N=20, 24, 28, 32 | {num_instances} Instances)")
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
            "fmqa": {"vals": [], "feas": [], "times": []},
            "p1_base": {"vals": [], "feas": [], "times": [], "success_probs": []},
            "p2_tqa": {"vals": [], "feas": [], "times": [], "success_probs": []},
            "cvar": {"vals": [], "feas": [], "times": [], "success_probs": []},
            "all_to_all": {"vals": [], "feas": [], "times": [], "success_probs": []},
            "hybrid_ls": {"vals": [], "feas": [], "times": []},
        }
        
        for idx, seed in enumerate(seeds):
            print(f"  [Instance {idx+1}/{num_instances} (Seed {seed})]")
            Q = create_random_qubo_bb(N, seed=seed)
            qubo_dict = {(i, j): float(Q[i, j]) for i in range(N) for j in range(i, N)}
            
            # Exact Minimum
            opt_x, opt_val = get_exact_minimum_categorical_bb(Q, block_sizes)
            scale_data["exact_mins"].append(opt_val)
            
            # 1. FMQA (Classical SA)
            t0 = time.time()
            res_fmqa = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=5.0, num_reads=500, seed=seed)
            t_fmqa = time.time() - t0
            best_fmqa = res_fmqa["best_feasible_sample"]["fval"] if res_fmqa["best_feasible_sample"] else None
            feas_fmqa = res_fmqa["feasibility_rate"] * 100
            scale_data["fmqa"]["vals"].append(best_fmqa)
            scale_data["fmqa"]["feas"].append(feas_fmqa)
            scale_data["fmqa"]["times"].append(t_fmqa)
            
            # 2. FM-XY-QAOA (p=1 Baseline, Ring, Expectation)
            t0 = time.time()
            res_p1 = solve_enhanced_subspace_xy_qaoa(
                Q, block_sizes, reps=1, mixer_type="ring", objective_type="expectation", init_strategy="tqa", maxiter=20, seed=seed
            )
            t_p1 = time.time() - t0
            scale_data["p1_base"]["vals"].append(res_p1["final_best_energy"])
            scale_data["p1_base"]["feas"].append(100.0)
            scale_data["p1_base"]["times"].append(t_p1)
            scale_data["p1_base"]["success_probs"].append(res_p1["success_probability"])
            
            # 3. FM-XY-QAOA (p=2, TQA Initialized, Ring)
            t0 = time.time()
            res_p2 = solve_enhanced_subspace_xy_qaoa(
                Q, block_sizes, reps=2, mixer_type="ring", objective_type="expectation", init_strategy="tqa", maxiter=25, seed=seed
            )
            t_p2 = time.time() - t0
            scale_data["p2_tqa"]["vals"].append(res_p2["final_best_energy"])
            scale_data["p2_tqa"]["feas"].append(100.0)
            scale_data["p2_tqa"]["times"].append(t_p2)
            scale_data["p2_tqa"]["success_probs"].append(res_p2["success_probability"])
            
            # 4. FM-XY-QAOA (CVaR alpha=0.25, p=1, Ring)
            t0 = time.time()
            res_cvar = solve_enhanced_subspace_xy_qaoa(
                Q, block_sizes, reps=1, mixer_type="ring", objective_type="cvar", cvar_alpha=0.25, init_strategy="tqa", maxiter=20, seed=seed
            )
            t_cvar = time.time() - t0
            scale_data["cvar"]["vals"].append(res_cvar["final_best_energy"])
            scale_data["cvar"]["feas"].append(100.0)
            scale_data["cvar"]["times"].append(t_cvar)
            scale_data["cvar"]["success_probs"].append(res_cvar["success_probability"])
            
            # 5. FM-XY-QAOA (All-to-All Mixer, p=1)
            t0 = time.time()
            res_a2a = solve_enhanced_subspace_xy_qaoa(
                Q, block_sizes, reps=1, mixer_type="all_to_all", objective_type="expectation", init_strategy="tqa", maxiter=20, seed=seed
            )
            t_a2a = time.time() - t0
            scale_data["all_to_all"]["vals"].append(res_a2a["final_best_energy"])
            scale_data["all_to_all"]["feas"].append(100.0)
            scale_data["all_to_all"]["times"].append(t_a2a)
            scale_data["all_to_all"]["success_probs"].append(res_a2a["success_probability"])
            
            # 6. Hybrid QAOA + 1-opt Local Search (p=1, Ring)
            t0 = time.time()
            res_hyb = solve_enhanced_subspace_xy_qaoa(
                Q, block_sizes, reps=1, mixer_type="ring", objective_type="expectation", init_strategy="tqa", use_local_search=True, maxiter=20, seed=seed
            )
            t_hyb = time.time() - t0
            scale_data["hybrid_ls"]["vals"].append(res_hyb["final_best_energy"])
            scale_data["hybrid_ls"]["feas"].append(100.0)
            scale_data["hybrid_ls"]["times"].append(t_hyb)
            
            print(f"    Exact={opt_val:.4f} | p=1={res_p1['final_best_energy']:.4f} | p=2={res_p2['final_best_energy']:.4f} | CVaR={res_cvar['final_best_energy']:.4f} | A2A={res_a2a['final_best_energy']:.4f} | Hyb={res_hyb['final_best_energy']:.4f}")
            
        results[N] = scale_data
        
    with open("report/accuracy_enhancement_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n================================================================================")
    print("  Accuracy Benchmark Completed! Summary of Results (Mean +/- Std)")
    print("================================================================================")
    for N, data in results.items():
        opt_m, opt_s = np.mean(data["exact_mins"]), np.std(data["exact_mins"])
        p1_m, p1_s = np.mean(data["p1_base"]["vals"]), np.std(data["p1_base"]["vals"])
        p2_m, p2_s = np.mean(data["p2_tqa"]["vals"]), np.std(data["p2_tqa"]["vals"])
        cvar_m, cvar_s = np.mean(data["cvar"]["vals"]), np.std(data["cvar"]["vals"])
        a2a_m, a2a_s = np.mean(data["all_to_all"]["vals"]), np.std(data["all_to_all"]["vals"])
        hyb_m, hyb_s = np.mean(data["hybrid_ls"]["vals"]), np.std(data["hybrid_ls"]["vals"])
        
        print(f"\n[Scale N={N}]")
        print(f"  Exact Minimum       : {opt_m:.4f} +/- {opt_s:.4f}")
        print(f"  1. p=1 Baseline     : {p1_m:.4f} +/- {p1_s:.4f} (Gap: {p1_m - opt_m:+.4f})")
        print(f"  2. p=2 (TQA Init)   : {p2_m:.4f} +/- {p2_s:.4f} (Gap: {p2_m - opt_m:+.4f})")
        print(f"  3. CVaR (alpha=0.25): {cvar_m:.4f} +/- {cvar_s:.4f} (Gap: {cvar_m - opt_m:+.4f})")
        print(f"  4. All-to-All Mixer : {a2a_m:.4f} +/- {a2a_s:.4f} (Gap: {a2a_m - opt_m:+.4f})")
        print(f"  5. Hybrid + 1-opt LS: {hyb_m:.4f} +/- {hyb_s:.4f} (Gap: {hyb_m - opt_m:+.4f})")

if __name__ == "__main__":
    run_accuracy_benchmark(num_instances=10)
