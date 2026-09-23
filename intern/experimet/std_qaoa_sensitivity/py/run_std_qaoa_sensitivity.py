#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_std_qaoa_sensitivity.py
Standard QAOA (X-Mixer + Penalty, p=1, N=16) における
ペナルティ係数 λ の感度解析 (Sensitivity Analysis) 実証実験スクリプト。

全ヒルベルト空間 2^16 = 65,536 次元の状態ベクトル発展を忠実にシミュレートし、
2つのBB関数 (BB-1: 多元触媒材料モデル, BB-2: 全結合ランダムモデル) に対して
λ ∈ [0.5, 25.0] (18点) をスキャンして制約充足率、真の大域最適解到達率、
およびエネルギー特性の変化を定量計測する。
"""

import os
import sys
import json
import time
from typing import List, Tuple, Dict, Any
import numpy as np

EXPERIMET_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "py"))
if EXPERIMET_PY not in sys.path:
    sys.path.insert(0, EXPERIMET_PY)
from openqarp_qaoa import OpenQARPStandardP1, all_bitstrings_lsb, one_hot_penalty

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

def get_all_feasible_patterns(block_sizes: List[int]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """One-Hot 制約を満たす全 256 個の有効ビット列パターンとインデックス組を生成"""
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

def find_exact_ground_truth(Q: np.ndarray, block_sizes: List[int]) -> Tuple[np.ndarray, float]:
    """全 256 パターンの厳密全探索により真の大域最適解と最小値を特定"""
    patterns, _ = get_all_feasible_patterns(block_sizes)
    energies = np.sum((patterns @ Q) * patterns, axis=1)
    min_idx = np.argmin(energies)
    return patterns[min_idx], float(energies[min_idx])

# ==============================================================================
# 2. Standard QAOA シミュレータ (OpenQARP, 全ヒルベルト空間 2^16 = 65,536 次元)
# ==============================================================================

class StandardQAOASimulator:
    def __init__(self, block_sizes: List[int], Q: np.ndarray):
        self.block_sizes = block_sizes
        self.Q = Q
        self.n = sum(block_sizes)
        self.dim = 2 ** self.n

        # OpenQARP/qarpx は LSB-first の計算基底インデックスを用いる。
        self.all_bits = all_bitstrings_lsb(self.n)

        # 元の目的関数エネルギー
        self.E_orig = np.sum((self.all_bits @ self.Q) * self.all_bits, axis=1)

        # One-Hot 制約違反ペナルティ量: sum_m (sum_{i in B_m} x_i - 1)^2
        self.pen = one_hot_penalty(self.all_bits, self.block_sizes)

        # 有効解マスク (256個が True)
        self.feasible_mask = (self.pen == 0.0)

    def simulate_state(
        self, gamma: float, beta: float, backend: OpenQARPStandardP1
    ) -> np.ndarray:
        """OpenQARP の p=1 Standard QAOA 回路から確率分布を取得する。"""
        return backend.probabilities(gamma, beta)

    def optimize_and_evaluate(self, lambda_val: float, opt_x: np.ndarray, num_shots: int = 1500, seed: int = 42) -> Dict[str, Any]:
        """指定された λ における QAOA 変分パラメータ最適化と統計量計測"""
        np.random.seed(seed)
        backend = OpenQARPStandardP1(self.Q, self.block_sizes, lambda_val)
        E_pen = backend.penalized_objective

        # グリッドサーチによる変分パラメータ探索 (11 x 11 = 121点)
        gammas = np.linspace(0.05, 1.2, 11)
        betas = np.linspace(0.05, 1.2, 11)

        best_expected_e = float("inf")
        opt_gamma, opt_beta = 0.5, 0.5

        for g in gammas:
            for b in betas:
                probs = self.simulate_state(g, b, backend)
                exp_e = float(np.sum(probs * E_pen))
                if exp_e < best_expected_e:
                    best_expected_e = exp_e
                    opt_gamma, opt_beta = g, b

        # 最適パラメータでの理論確率分布
        prob_dist = self.simulate_state(opt_gamma, opt_beta, backend)

        # 1. 理論制約充足率 (有効解部分空間の確率質量の和)
        theoretical_feasibility = float(np.sum(prob_dist[self.feasible_mask]))

        # 2. 真の大域最適解のインデックスと理論出現確率
        opt_idx = np.where(np.all(self.all_bits == opt_x, axis=1))[0][0]
        theoretical_p_opt = float(prob_dist[opt_idx])

        # 3. 有限ショット測定 (num_shots = 1500)
        sampled_indices = np.random.choice(self.dim, size=num_shots, p=prob_dist)
        sampled_feas = self.feasible_mask[sampled_indices]
        sampled_orig_e = self.E_orig[sampled_indices]

        empirical_feasibility = float(np.mean(sampled_feas))
        hit_count = int(np.sum(sampled_indices == opt_idx))
        empirical_p_opt = float(hit_count / num_shots)

        if np.any(sampled_feas):
            feas_energies = sampled_orig_e[sampled_feas]
            best_feas_energy = float(np.min(feas_energies))
            mean_feas_energy = float(np.mean(feas_energies))
        else:
            # No feasible sample was observed.  Objective values of infeasible
            # strings are diagnostics, not feasible-solution statistics.
            best_feas_energy = None
            mean_feas_energy = None

        return {
            "lambda": lambda_val,
            "opt_gamma": opt_gamma,
            "opt_beta": opt_beta,
            "expected_pen_energy": best_expected_e,
            "theoretical_feasibility": theoretical_feasibility,
            "empirical_feasibility": empirical_feasibility,
            "theoretical_p_opt": theoretical_p_opt,
            "empirical_p_opt": empirical_p_opt,
            "hit_count": hit_count,
            "feasible_sample_count": int(np.sum(sampled_feas)),
            "best_feas_energy": best_feas_energy,
            "mean_feas_energy": mean_feas_energy,
            "best_infeasible_objective": (float(np.min(sampled_orig_e))
                                            if not np.any(sampled_feas) else None),
            "num_shots": num_shots
        }

# ==============================================================================
# 3. 実験実行メインルーチン
# ==============================================================================

def main():
    print("=" * 75)
    print(" Standard QAOA ペナルティ係数 λ 感度解析 実証実験 (N=16, 4グループ×4元素)")
    print(" 全ヒルベルト空間 2^16 = 65,536 次元の状態ベクトル発展シミュレーション")
    print("=" * 75)

    block_sizes = [4, 4, 4, 4]
    lambdas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0, 25.0]

    # BB関数の構築
    Q_mat = create_materials_design_ground_truth(block_sizes, seed=123)
    opt_x_mat, opt_val_mat = find_exact_ground_truth(Q_mat, block_sizes)

    Q_rand = create_random_qubo_bb(d=16, seed=42)
    opt_x_rand, opt_val_rand = find_exact_ground_truth(Q_rand, block_sizes)

    results = {
        "metadata": {
            "description": "Standard QAOA (X-Mixer, p=1, N=16) penalty sensitivity; 11x11 parameter grid, not a continuous global variational optimum",
            "date": "2026-09-20",
            "qaoa_backend": "OpenQARP 0.1.0 (native QAOA / qarpx state-vector)",
            "n_qubits": 16,
            "total_states": 65536,
            "feasible_states": 256,
            "feasible_ratio_percent": 256 / 65536 * 100,  # 0.390625%
            "lambda_scan_values": lambdas,
            "num_shots": 1500
        },
        "BB1_materials": {
            "name": "Materials Catalyst (Strong Coupling)",
            "exact_min": opt_val_mat,
            "lambda_crit": 6.54,
            "lambda_adaptive": 8.17,
            "fm_xy_qaoa_reference": {
                "feasibility_rate": 1.0,
                "success_probability": 0.0749,  # 7.49%
                "runtime_sec": 0.073
            },
            "fmqa_adaptive_reference": {
                "lambda": 8.17,
                "feasibility_rate": 1.0,
                "success_probability": 0.0470  # 4.70%
            },
            "scan_data": []
        },
        "BB2_random": {
            "name": "Fully-Connected Random Interactions",
            "exact_min": opt_val_rand,
            "lambda_crit": 2.52,
            "lambda_adaptive": 3.15,
            "fm_xy_qaoa_reference": {
                "feasibility_rate": 1.0,
                "success_probability": 0.0796,  # 7.96%
                "runtime_sec": 0.016
            },
            "fmqa_adaptive_reference": {
                "lambda": 3.15,
                "feasibility_rate": 0.997,
                "success_probability": 0.0460  # 4.60%
            },
            "scan_data": []
        }
    }

    # The full scan is intentionally resumable because each point executes a
    # 121-circuit OpenQARP grid.  With no environment variables this remains
    # the original all-at-once execution.
    start_index = int(os.environ.get("OPENQARP_LAMBDA_START", "0"))
    stop_index = int(os.environ.get("OPENQARP_LAMBDA_STOP", str(len(lambdas))))
    if not 0 <= start_index < stop_index <= len(lambdas):
        raise ValueError("OPENQARP_LAMBDA_START/STOP must select a non-empty lambda slice")
    partial_run = start_index != 0 or stop_index != len(lambdas)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "json")
    out_json = os.path.join(out_dir, "results_std_qaoa_sensitivity.json")
    if partial_run and os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        results.setdefault("metadata", {}).update({
            "qaoa_backend": "OpenQARP 0.1.0 (native QAOA / qarpx state-vector)",
            "openqarp_lambda_scan_progress": f"recomputing indices {start_index}:{stop_index}",
        })

    # 実験実行
    for bb_key, Q_bb, opt_x, opt_val in [
        ("BB1_materials", Q_mat, opt_x_mat, opt_val_mat),
        ("BB2_random", Q_rand, opt_x_rand, opt_val_rand)
    ]:
        print(f"\n>>> 実験開始: {results[bb_key]['name']} (真の大域最適値: {opt_val:.4f})")
        sim = StandardQAOASimulator(block_sizes, Q_bb)
        t_start = time.time()

        for lam_index in range(start_index, stop_index):
            lam = lambdas[lam_index]
            res = sim.optimize_and_evaluate(lam, opt_x, num_shots=1500, seed=42)
            scan_data = results[bb_key]["scan_data"]
            if len(scan_data) > lam_index:
                scan_data[lam_index] = res
            else:
                if len(scan_data) != lam_index:
                    raise RuntimeError("Cannot resume: scan_data indices are not contiguous")
                scan_data.append(res)
            best_text = (f"{res['best_feas_energy']:7.4f}"
                         if res["best_feas_energy"] is not None else "N/A (有効解なし)")
            print(f"  λ = {lam:5.1f} | 理論充足率: {res['theoretical_feasibility']*100:5.2f}% (実測: {res['empirical_feasibility']*100:5.2f}%) | "
                  f"最適解確率 P(x*): {res['theoretical_p_opt']*100:6.4f}% (Hits: {res['hit_count']:2d}/1500) | "
                  f"有効最良E: {best_text}")

        t_elapsed = time.time() - t_start
        print(f"--- {results[bb_key]['name']} 完了 (所要時間: {t_elapsed:.2f}s) ---")

    # 結果保存
    if partial_run:
        if stop_index == len(lambdas):
            results["metadata"].pop("openqarp_lambda_scan_progress", None)
            results["metadata"]["openqarp_lambda_scan_status"] = "complete"
        else:
            results["metadata"]["openqarp_lambda_scan_progress"] = f"completed indices {start_index}:{stop_index}"
    else:
        results["metadata"].pop("openqarp_lambda_scan_progress", None)
        results["metadata"]["openqarp_lambda_scan_status"] = "complete"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n★ 全実験データが正常に保存されました: {out_json}")

if __name__ == "__main__":
    main()
