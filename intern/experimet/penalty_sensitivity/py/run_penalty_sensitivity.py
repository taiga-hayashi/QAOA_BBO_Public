#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_penalty_sensitivity.py
FMQA (Simulated Annealing + Penalty) における
横軸: ペナルティ係数 λ vs 縦軸: 最適化性能 の感度解析実証実験スクリプト。
前回の run_optimization_experiment.py と完全に同一の BB 関数・FMQASolver 実装を使用。
"""

import os
import sys
import json
import time
from typing import List, Tuple, Dict, Any
import numpy as np

# ==============================================================================
# 1. 2つのBB関数 (Ground Truth) 定義 (前実験と完全合致)
# ==============================================================================

def create_materials_design_ground_truth(block_sizes: List[int], seed: int = 123) -> np.ndarray:
    """BB関数1: 多元触媒・機能性材料設計モデル (Q_materials, N=16)"""
    np.random.seed(seed)
    n = sum(block_sizes)
    Q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                Q[i, i] = np.random.uniform(-2.0, 1.0)
            else:
                block_i = i // 4
                block_j = j // 4
                if block_i != block_j:
                    if (block_i == 0 and block_j == 1) or (block_i == 2 and block_j == 3):
                        Q[i, j] = np.random.uniform(-6.0, 2.0)
                    else:
                        Q[i, j] = np.random.uniform(-1.0, 1.0)
    return Q

def create_random_qubo_bb(d: int, seed: int = 42) -> np.ndarray:
    """BB関数2: 全結合ランダム相互作用モデル (Q_random, N=16)"""
    np.random.seed(seed)
    Q = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i, d):
            if i == j:
                Q[i, i] = np.random.uniform(-1.5, 1.5)
            else:
                Q[i, j] = np.random.uniform(-1.0, 1.0)
    return Q

def evaluate_bb(x: np.ndarray, Q: np.ndarray) -> float:
    return float(x @ Q @ x)

def is_feasible_one_hot(x: np.ndarray, block_sizes: List[int]) -> bool:
    idx = 0
    for sz in block_sizes:
        if np.sum(x[idx:idx+sz]) != 1.0:
            return False
        idx += sz
    return True

def get_all_feasible_patterns(block_sizes: List[int]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    patterns = []
    indices = []
    sz0, sz1, sz2, sz3 = block_sizes
    for i0 in range(sz0):
        for i1 in range(sz1):
            for i2 in range(sz2):
                for i3 in range(sz3):
                    vec = np.zeros(sum(block_sizes), dtype=np.float64)
                    vec[i0] = 1.0
                    vec[sz0 + i1] = 1.0
                    vec[sz0 + sz1 + i2] = 1.0
                    vec[sz0 + sz1 + sz2 + i3] = 1.0
                    patterns.append(vec)
                    indices.append((i0, i1, i2, i3))
    return np.array(patterns), indices

def get_exact_minimum(Q: np.ndarray, feasible_patterns: np.ndarray) -> Tuple[np.ndarray, float]:
    vals = [evaluate_bb(p, Q) for p in feasible_patterns]
    min_idx = int(np.argmin(vals))
    return feasible_patterns[min_idx], float(vals[min_idx])

def compute_adaptive_penalty_lambda(Q: np.ndarray, block_sizes: List[int], safety_factor: float = 1.25) -> Tuple[float, float]:
    M = len(block_sizes)
    block_indices = []
    idx = 0
    for sz in block_sizes:
        block_indices.append(list(range(idx, idx + sz)))
        idx += sz

    max_gains = []
    for m in range(M):
        inds_m = block_indices[m]
        gain_m_max = 0.0
        for i in inds_m:
            gain_i = 0.0
            for other_m in range(M):
                if other_m == m:
                    continue
                other_inds = block_indices[other_m]
                best_attraction = 0.0
                for j in other_inds:
                    q_val = Q[min(i, j), max(i, j)]
                    if q_val < 0:
                        best_attraction = max(best_attraction, -q_val)
                gain_i += best_attraction
            gain_m_max = max(gain_m_max, gain_i)
        max_gains.append(gain_m_max)

    crit_lambda = max(max_gains) if max_gains else 5.0
    return float(np.round(crit_lambda * safety_factor, 2)), float(np.round(crit_lambda, 2))

# ==============================================================================
# 2. 前実験と完全同一の FMQASolver (SA + Penalty)
# ==============================================================================

class FMQASolver:
    def __init__(self, block_sizes: List[int], lambda_penalty: float = 5.0):
        self.block_sizes = block_sizes
        self.lambda_penalty = lambda_penalty
        self.total_vars = sum(block_sizes)

    def energy_with_penalty(self, x: np.ndarray, Q: np.ndarray) -> float:
        e_orig = float(x @ Q @ x)
        pen = 0.0
        idx = 0
        for sz in self.block_sizes:
            pen += (np.sum(x[idx:idx+sz]) - 1.0) ** 2
            idx += sz
        return e_orig + self.lambda_penalty * pen

    def solve(self, Q: np.ndarray, num_reads: int = 1500, seed: int = 42) -> Dict[str, Any]:
        # Keep the random-number stream identical across lambda values so that
        # the sweep isolates the penalty coefficient rather than seed changes.
        rng = np.random.default_rng(seed)
        samples = []
        energies_pen = []
        energies_orig = []
        feasibilities = []

        T_start = 8.0
        T_end = 0.01
        steps = 200
        gamma = (T_end / T_start) ** (1.0 / steps)

        for read_idx in range(num_reads):
            x = rng.integers(0, 2, self.total_vars).astype(np.float64)
            current_e = self.energy_with_penalty(x, Q)
            T = T_start

            for _ in range(steps):
                flip_idx = rng.integers(0, self.total_vars)
                x_new = x.copy()
                x_new[flip_idx] = 1.0 - x_new[flip_idx]
                new_e = self.energy_with_penalty(x_new, Q)
                delta_e = new_e - current_e

                if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                    x = x_new
                    current_e = new_e
                T *= gamma

            is_feas = is_feasible_one_hot(x, self.block_sizes)
            samples.append(x)
            energies_pen.append(current_e)
            energies_orig.append(evaluate_bb(x, Q))
            feasibilities.append(is_feas)

        samples = np.array(samples)
        feas_ratio = float(np.mean(feasibilities))
        feas_indices = [i for i, f in enumerate(feasibilities) if f]

        if feas_indices:
            feas_energies = [energies_orig[i] for i in feas_indices]
            best_feas_idx = feas_indices[int(np.argmin(feas_energies))]
            best_sample = samples[best_feas_idx]
            best_feasible_energy = float(energies_orig[best_feas_idx])
            mean_feasible_energy = float(np.mean(feas_energies))
        else:
            # An infeasible energy is not an objective value on the constrained
            # problem.  Preserve it only as a diagnostic, never as a feasible
            # optimum or an optimality-gap input.
            best_idx = int(np.argmin(energies_pen))
            best_sample = None
            best_feasible_energy = None
            mean_feasible_energy = None

        return {
            "samples": samples,
            "feasibility_rate": feas_ratio,
            "best_sample": best_sample,
            "feasible_sample_count": len(feas_indices),
            "best_feasible_energy": best_feasible_energy,
            "mean_feasible_energy": mean_feasible_energy,
            "best_infeasible_penalized_energy": float(energies_pen[best_idx]) if not feas_indices else None,
            "best_infeasible_objective": float(energies_orig[best_idx]) if not feas_indices else None,
            "lambda_used": self.lambda_penalty
        }

# ==============================================================================
# 3. メイン実験ループ
# ==============================================================================

def main():
    print("=" * 75)
    print(" FMQA ペナルティ係数感度解析 実証実験 (N=16, 4サイト×4元素)")
    print(" 横軸: ペナルティ係数 λ  vs  縦軸: 最適化性能")
    print("=" * 75)

    block_sizes = [4, 4, 4, 4]
    feasible_patterns, _ = get_all_feasible_patterns(block_sizes)

    # 掃引グリッド
    lambda_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 8.17, 10.0, 12.0, 15.0, 20.0, 25.0]
    num_reads = 1500

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "N": 16,
            "block_sizes": block_sizes,
            "lambda_grid": lambda_grid,
            "num_reads_per_lambda": num_reads
        }
    }

    # --------------------------------------------------------------------------
    # 実験 1: BB-1 多元触媒モデル
    # --------------------------------------------------------------------------
    print("\n>>> [BB-1] 多元触媒モデル (Q_materials) 解析開始...")
    Q1 = create_materials_design_ground_truth(block_sizes, seed=123)
    exact_x1, exact_min1 = get_exact_minimum(Q1, feasible_patterns)
    lam1_adp, lam1_crit = compute_adaptive_penalty_lambda(Q1, block_sizes, safety_factor=1.25)
    print(f"  真の大域最適値 f*: {exact_min1:.4f}")
    print(f"  利得ベースのλ目安: {lam1_crit:.2f} | 安全係数付き目安: {lam1_adp:.2f}")

    fm_xy_p_bb1 = 0.074887  # 7.49%
    fm_xy_feas_bb1 = 1.0    # 100.0%

    bb1_sweep = []
    t0 = time.time()
    for lam in lambda_grid:
        solver = FMQASolver(block_sizes, lambda_penalty=lam)
        res = solver.solve(Q1, num_reads=num_reads, seed=123)

        # 最適解ヒット判定
        exact_hits = np.sum(np.all(res["samples"] == exact_x1, axis=1))
        p_opt = float(exact_hits / num_reads)
        gap = (float(res["best_feasible_energy"] - exact_min1)
               if res["best_feasible_energy"] is not None else None)

        point_data = {
            "lambda": lam,
            "feasibility_rate": res["feasibility_rate"],
            "success_probability": p_opt,
            "feasible_sample_count": res["feasible_sample_count"],
            "best_feasible_energy": res["best_feasible_energy"],
            "mean_feasible_energy": res["mean_feasible_energy"],
            "best_infeasible_penalized_energy": res["best_infeasible_penalized_energy"],
            "best_infeasible_objective": res["best_infeasible_objective"],
            "optimality_gap": gap
        }
        bb1_sweep.append(point_data)
        best_text = f"{res['best_feasible_energy']:8.4f}" if res["best_feasible_energy"] is not None else "N/A (実行可能解なし)"
        gap_text = f"{gap:6.4f}" if gap is not None else "N/A"
        print(f"  [λ={lam:5.2f}] 充足率: {res['feasibility_rate']*100:5.1f}% | 最適解到達率: {p_opt*100:5.2f}% | 有効最良値: {best_text} (Gap: {gap_text})")

    results["BB1_materials"] = {
        "name": "Materials Catalyst Multi-Scale Model",
        "exact_min": exact_min1,
        "exact_x": exact_x1.astype(int).tolist(),
        "lambda_gain_heuristic": lam1_crit,
        "lambda_safety_heuristic": lam1_adp,
        "lambda_heuristic_note": "利得ベースの経験的目安。制約を満たすことの理論保証や厳密な臨界値ではない。",
        "fm_xy_qaoa_baseline": {
            "feasibility_rate": fm_xy_feas_bb1,
            "success_probability": fm_xy_p_bb1,
            "best_energy": exact_min1,
            "optimality_gap": 0.0
        },
        "sweep_data": bb1_sweep,
        "runtime_sec": time.time() - t0
    }

    # --------------------------------------------------------------------------
    # 実験 2: BB-2 全結合ランダムモデル
    # --------------------------------------------------------------------------
    print("\n>>> [BB-2] 全結合ランダムモデル (Q_random) 解析開始...")
    Q2 = create_random_qubo_bb(16, seed=42)
    exact_x2, exact_min2 = get_exact_minimum(Q2, feasible_patterns)
    lam2_adp, lam2_crit = compute_adaptive_penalty_lambda(Q2, block_sizes, safety_factor=1.25)
    print(f"  真の大域最適値 f*: {exact_min2:.4f}")
    print(f"  利得ベースのλ目安: {lam2_crit:.2f} | 安全係数付き目安: {lam2_adp:.2f}")

    fm_xy_p_bb2 = 0.079612  # 7.96%
    fm_xy_feas_bb2 = 1.0    # 100.0%

    bb2_sweep = []
    t0 = time.time()
    for lam in lambda_grid:
        solver = FMQASolver(block_sizes, lambda_penalty=lam)
        res = solver.solve(Q2, num_reads=num_reads, seed=42)

        exact_hits = np.sum(np.all(res["samples"] == exact_x2, axis=1))
        p_opt = float(exact_hits / num_reads)
        gap = (float(res["best_feasible_energy"] - exact_min2)
               if res["best_feasible_energy"] is not None else None)

        point_data = {
            "lambda": lam,
            "feasibility_rate": res["feasibility_rate"],
            "success_probability": p_opt,
            "feasible_sample_count": res["feasible_sample_count"],
            "best_feasible_energy": res["best_feasible_energy"],
            "mean_feasible_energy": res["mean_feasible_energy"],
            "best_infeasible_penalized_energy": res["best_infeasible_penalized_energy"],
            "best_infeasible_objective": res["best_infeasible_objective"],
            "optimality_gap": gap
        }
        bb2_sweep.append(point_data)
        best_text = f"{res['best_feasible_energy']:8.4f}" if res["best_feasible_energy"] is not None else "N/A (実行可能解なし)"
        gap_text = f"{gap:6.4f}" if gap is not None else "N/A"
        print(f"  [λ={lam:5.2f}] 充足率: {res['feasibility_rate']*100:5.1f}% | 最適解到達率: {p_opt*100:5.2f}% | 有効最良値: {best_text} (Gap: {gap_text})")

    results["BB2_random"] = {
        "name": "Fully-Connected Random Interaction Model",
        "exact_min": exact_min2,
        "exact_x": exact_x2.astype(int).tolist(),
        "lambda_gain_heuristic": lam2_crit,
        "lambda_safety_heuristic": lam2_adp,
        "lambda_heuristic_note": "利得ベースの経験的目安。制約を満たすことの理論保証や厳密な臨界値ではない。",
        "fm_xy_qaoa_baseline": {
            "feasibility_rate": fm_xy_feas_bb2,
            "success_probability": fm_xy_p_bb2,
            "best_energy": exact_min2,
            "optimality_gap": 0.0
        },
        "sweep_data": bb2_sweep,
        "runtime_sec": time.time() - t0
    }

    # 保存
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json")
    out_file = os.path.join(out_dir, "results_penalty_sensitivity.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n★ 全感度解析結果が正常に保存されました: {out_file}")

if __name__ == "__main__":
    main()
