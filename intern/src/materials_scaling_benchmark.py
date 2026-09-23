import os
import sys
import time
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

# Project imports
from bb_function import evaluate_bb, get_feasible_patterns
from fmqa_solver import solve_fmqa
from enhanced_subspace_xy_qaoa_solver import (
    solve_enhanced_subspace_xy_qaoa,
    run_one_hot_local_search
)

def create_scalable_materials_ground_truth(block_sizes: List[int], seed: int = 42) -> np.ndarray:
    """
    任意スケール (M サイト × d 元素) の多元触媒・材料組成探索用エネルギー行列。
    特定サイト間 (隣接ペア (2k, 2k+1) 等) に協同触媒効果・強シナジー相互作用 (通常結合の5〜8倍) を配置。
    """
    np.random.seed(seed)
    n = sum(block_sizes)
    M = len(block_sizes)
    Q = np.zeros((n, n), dtype=np.float32)
    
    # 各ビットがどのブロック(サイト)に属するかマッピング
    block_map = []
    for b_idx, sz in enumerate(block_sizes):
        block_map.extend([b_idx] * sz)
    block_map = np.array(block_map)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # 元素単体の化学ポテンシャル
                Q[i, i] = np.random.uniform(-2.0, 1.0)
            else:
                b_i = block_map[i]
                b_j = block_map[j]
                if b_i != b_j:
                    # 協同触媒活性サイトペア (例えば (0, 1), (2, 3), (4, 5), (6, 7) ...)
                    is_synergy_pair = (min(b_i, b_j) % 2 == 0) and (abs(b_i - b_j) == 1)
                    if is_synergy_pair:
                        # 強相関・触媒シナジー相互作用 (-6.0 〜 2.0)
                        Q[i, j] = np.random.uniform(-6.0, 2.0)
                    else:
                        # 通常のサイト間相互作用 (-1.0 〜 1.0)
                        Q[i, j] = np.random.uniform(-1.0, 1.0)
    return Q

def get_exact_minimum_materials(Q: np.ndarray, block_sizes: List[int]) -> Tuple[np.ndarray, float]:
    """全有効解 (直積) を評価して真の最小値と解を厳密取得"""
    feasible = get_feasible_patterns(block_sizes)
    vals = evaluate_bb(feasible, Q)
    min_idx = np.argmin(vals)
    return feasible[min_idx], float(vals[min_idx])

def run_materials_scaling_benchmark(num_instances: int = 5) -> Dict[str, Any]:
    print("================================================================================")
    print(f"  Scalable Materials Design BBO Benchmark (N=16, 20, 24, 28, 32 | {num_instances} Instances)")
    print("================================================================================")
    
    scales = [
        {"N": 16, "M": 4, "d": 4},
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
        
        print(f"\n>>> Running Materials Scale N={N} (M={M} sites, d={d} elements)")
        print(f"    Total Space: 2^{N} = {total_states:,} | Valid Materials: 4^{M} = {subspace_dim:,}")
        
        scale_data = {
            "N": N, "M": M, "d": d, "total_states": total_states, "subspace_dim": subspace_dim,
            "exact_mins": [],
            "fmqa_l5": {"vals": [], "feas": [], "times": [], "success_counts": 0},
            "fmqa_l15": {"vals": [], "feas": [], "times": [], "success_counts": 0},
            "p1_base": {"vals": [], "feas": [], "times": [], "success_probs": [], "success_counts": 0},
            "all_to_all": {"vals": [], "feas": [], "times": [], "success_probs": [], "success_counts": 0},
            "hybrid_ls": {"vals": [], "feas": [], "times": [], "success_counts": 0},
        }
        
        for idx, seed in enumerate(seeds):
            print(f"  --- [Instance {idx+1}/{num_instances} (Seed {seed})] ---")
            Q = create_scalable_materials_ground_truth(block_sizes, seed=seed)
            qubo_dict = {(i, j): float(Q[i, j]) for i in range(N) for j in range(i, N)}
            
            # 真の大域最適解
            opt_x, opt_val = get_exact_minimum_materials(Q, block_sizes)
            scale_data["exact_mins"].append(opt_val)
            print(f"      Exact Optimal Energy: {opt_val:.4f}")
            
            # 1. FMQA with standard penalty (lambda = 5.0)
            t0 = time.time()
            res_fmqa5 = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=5.0, num_reads=500, seed=seed)
            t_fmqa5 = time.time() - t0
            best_val5 = res_fmqa5["best_feasible_sample"]["fval"] if res_fmqa5["best_feasible_sample"] else None
            feas_rate5 = res_fmqa5["feasibility_rate"] * 100
            scale_data["fmqa_l5"]["vals"].append(best_val5)
            scale_data["fmqa_l5"]["feas"].append(feas_rate5)
            scale_data["fmqa_l5"]["times"].append(t_fmqa5)
            if best_val5 is not None and abs(best_val5 - opt_val) < 1e-4:
                scale_data["fmqa_l5"]["success_counts"] += 1
            val5_str = f"{best_val5:.3f}" if best_val5 is not None else "None"
            print(f"      FMQA (λ=5.0)  : Feas={feas_rate5:5.1f}%, Best={val5_str}, Time={t_fmqa5:.2f}s")
            
            # 2. FMQA with high penalty (lambda = 15.0) - ペナルティジレンマの検証
            t0 = time.time()
            res_fmqa15 = solve_fmqa(qubo_dict, block_sizes, lambda_penalty=15.0, num_reads=500, seed=seed)
            t_fmqa15 = time.time() - t0
            best_val15 = res_fmqa15["best_feasible_sample"]["fval"] if res_fmqa15["best_feasible_sample"] else None
            feas_rate15 = res_fmqa15["feasibility_rate"] * 100
            scale_data["fmqa_l15"]["vals"].append(best_val15)
            scale_data["fmqa_l15"]["feas"].append(feas_rate15)
            scale_data["fmqa_l15"]["times"].append(t_fmqa15)
            if best_val15 is not None and abs(best_val15 - opt_val) < 1e-4:
                scale_data["fmqa_l15"]["success_counts"] += 1
            val15_str = f"{best_val15:.3f}" if best_val15 is not None else "None"
            print(f"      FMQA (λ=15.0) : Feas={feas_rate15:5.1f}%, Best={val15_str}, Time={t_fmqa15:.2f}s")
            
            # 3. FM-XY-QAOA (p=1 Baseline, Ring Mixer)
            t0 = time.time()
            res_p1 = solve_enhanced_subspace_xy_qaoa(
                Q, block_sizes, reps=1, mixer_type="ring", objective_type="expectation", init_strategy="tqa", maxiter=20, seed=seed
            )
            t_p1 = time.time() - t0
            scale_data["p1_base"]["vals"].append(res_p1["final_best_energy"])
            scale_data["p1_base"]["feas"].append(100.0)
            scale_data["p1_base"]["times"].append(t_p1)
            scale_data["p1_base"]["success_probs"].append(res_p1["success_probability"])
            if abs(res_p1["final_best_energy"] - opt_val) < 1e-4:
                scale_data["p1_base"]["success_counts"] += 1
            print(f"      XY-QAOA (p=1) : Feas=100.0%, Best={res_p1['final_best_energy']:.3f}, P_succ={res_p1['success_probability']*100:.2f}%, Time={t_p1:.2f}s")
            
            # 4. All-to-All Mixer XY-QAOA
            t0 = time.time()
            res_a2a = solve_enhanced_subspace_xy_qaoa(
                Q, block_sizes, reps=1, mixer_type="all_to_all", objective_type="cvar", cvar_alpha=0.25, init_strategy="tqa", maxiter=20, seed=seed
            )
            t_a2a = time.time() - t0
            scale_data["all_to_all"]["vals"].append(res_a2a["final_best_energy"])
            scale_data["all_to_all"]["feas"].append(100.0)
            scale_data["all_to_all"]["times"].append(t_a2a)
            scale_data["all_to_all"]["success_probs"].append(res_a2a["success_probability"])
            if abs(res_a2a["final_best_energy"] - opt_val) < 1e-4:
                scale_data["all_to_all"]["success_counts"] += 1
            print(f"      All-to-All XY : Feas=100.0%, Best={res_a2a['final_best_energy']:.3f}, P_succ={res_a2a['success_probability']*100:.2f}%, Time={t_a2a:.2f}s")
            
            # 5. Hybrid QAOA (QAOA + 1-opt Local Search)
            t0 = time.time()
            res_hyb = solve_enhanced_subspace_xy_qaoa(
                Q, block_sizes, reps=1, mixer_type="ring", objective_type="expectation",
                init_strategy="tqa", use_local_search=True, maxiter=20, seed=seed
            )
            t_hyb = time.time() - t0
            scale_data["hybrid_ls"]["vals"].append(res_hyb["final_best_energy"])
            scale_data["hybrid_ls"]["feas"].append(100.0)
            scale_data["hybrid_ls"]["times"].append(t_hyb)
            if abs(res_hyb["final_best_energy"] - opt_val) < 1e-4:
                scale_data["hybrid_ls"]["success_counts"] += 1
            print(f"      Hybrid QAOA   : Feas=100.0%, Best={res_hyb['final_best_energy']:.3f}, Time={t_hyb:.2f}s")
            
        results[str(N)] = scale_data
        
    return results

def plot_materials_scaling_results(results: Dict[str, Any], output_path: str = "report/figures/practical_materials_bbo/materials_scaling_feasibility_and_gap.pdf"):
    """ベンチマーク結果から制約充足率と残差ギャップの推移グラフ（2パネル）を生成"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    Ns = [int(k) for k in results.keys()]
    
    # 充足率の集計
    feas_fmqa5 = [np.mean(results[str(n)]["fmqa_l5"]["feas"]) for n in Ns]
    feas_fmqa15 = [np.mean(results[str(n)]["fmqa_l15"]["feas"]) for n in Ns]
    feas_qaoa = [100.0 for _ in Ns]
    
    # 残差ギャップ（厳密解との差：平均）
    gap_fmqa5 = []
    gap_fmqa15 = []
    gap_p1 = []
    gap_a2a = []
    gap_hyb = []
    
    for n in Ns:
        opts = np.array(results[str(n)]["exact_mins"])
        
        # FMQA l5
        v5 = results[str(n)]["fmqa_l5"]["vals"]
        valid_g5 = [v - opt for v, opt in zip(v5, opts) if v is not None]
        gap_fmqa5.append(np.mean(valid_g5) if valid_g5 else np.nan)
        
        # FMQA l15
        v15 = results[str(n)]["fmqa_l15"]["vals"]
        valid_g15 = [v - opt for v, opt in zip(v15, opts) if v is not None]
        gap_fmqa15.append(np.mean(valid_g15) if valid_g15 else np.nan)
        
        # p1
        v_p1 = np.array(results[str(n)]["p1_base"]["vals"])
        gap_p1.append(np.mean(v_p1 - opts))
        
        # a2a
        v_a2a = np.array(results[str(n)]["all_to_all"]["vals"])
        gap_a2a.append(np.mean(v_a2a - opts))
        
        # hyb
        v_hyb = np.array(results[str(n)]["hybrid_ls"]["vals"])
        gap_hyb.append(np.mean(v_hyb - opts))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    
    # Left: Feasibility Rate
    ax1.plot(Ns, feas_qaoa, marker='s', markersize=8, linewidth=2.5, color='#2ca02c', label='Enhanced / Baseline QAOA (100% Guaranteed)')
    ax1.plot(Ns, feas_fmqa15, marker='^', markersize=7, linewidth=2.0, color='#1f77b4', linestyle='--', label='FMQA (λ = 15.0: High Penalty)')
    ax1.plot(Ns, feas_fmqa5, marker='o', markersize=7, linewidth=2.0, color='#d62728', linestyle='-', label='FMQA (λ = 5.0: Standard Penalty)')
    
    ax1.set_xlabel('Number of Binary Variables $N$ (Number of Sites $M = N / 4$)', fontweight='bold')
    ax1.set_ylabel('One-Hot Feasibility Rate (%)', fontweight='bold')
    ax1.set_title('(a) Constraint Feasibility in Multi-Scale Catalyst Model', fontweight='bold', pad=10)
    ax1.set_xticks(Ns)
    ax1.set_xticklabels([f"$N={n}$\n($M={n//4}$)" for n in Ns])
    ax1.set_ylim(-5, 108)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', framealpha=0.9)
    
    # Right: Energy Residual Gap
    ax2.plot(Ns, gap_p1, marker='o', markersize=7, linewidth=1.8, color='#ff7f0e', label='FM-XY-QAOA (p=1 Baseline, Ring)')
    ax2.plot(Ns, gap_a2a, marker='^', markersize=7, linewidth=1.8, color='#9467bd', label='All-to-All XY-QAOA + CVaR')
    ax2.plot(Ns, gap_fmqa15, marker='d', markersize=7, linewidth=1.8, color='#1f77b4', linestyle='--', label='FMQA (λ = 15.0: Trapped in Local Min)')
    ax2.plot(Ns, gap_hyb, marker='*', markersize=10, linewidth=2.5, color='#2ca02c', label='Hybrid QAOA (QAOA + 1-opt LS)')
    
    ax2.set_xlabel('Number of Binary Variables $N$ (Number of Sites $M = N / 4$)', fontweight='bold')
    ax2.set_ylabel('Energy Residual Gap ($E_{\\mathrm{best}} - E_{\\mathrm{exact}}$)', fontweight='bold')
    ax2.set_title('(b) Optimization Residual Gap from True Ground State', fontweight='bold', pad=10)
    ax2.set_xticks(Ns)
    ax2.set_xticklabels([f"$N={n}$\n($M={n//4}$)" for n in Ns])
    ax2.set_ylim(-0.2, max(max(gap_p1), max(gap_fmqa15)) * 1.15)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Plot saved to {output_path}")

def main():
    num_instances = 5
    results = run_materials_scaling_benchmark(num_instances=num_instances)
    
    os.makedirs("report", exist_ok=True)
    out_json = "report/materials_scaling_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Benchmark results saved to {out_json}")
    
    plot_materials_scaling_results(results)

if __name__ == "__main__":
    main()
