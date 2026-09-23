#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_qubit_scaling.py
量子ビット数 N のスケーリング特性と最適化性能・制約充足率の多角的多次元実証実験スクリプト。

対象スケール: N ∈ [8, 12, 16, 20, 24, 32, 40, 48, 64]
比較対象:
  1. Standard QAOA (全ヒルベルト空間 2^N, ペナルティ法)
  2. FMQA (Simulated Annealing + 適応ペナルティ)
  3. FM-XY-QAOA (有効部分空間 2^(N/2), XYミキサー, λ=0)

評価指標:
  - One-Hot 制約充足率 (Feasibility Rate % vs N)
  - 探索空間次元と有効空間比率 (Dimension & Ratio vs N)
  - 真の大域最適解到達率 (Ground State Hit Probability P(x*) vs N)
  - 到達最良エネルギーと最適値ギャップ (Best Energy & Gap vs N)
  - 計算実行時間 (Runtime vs N)
  - 状態ベクトル理論 RAM 使用量と古典限界境界 (Memory Scaling vs N)
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
from openqarp_qaoa import (
    OpenQARPFMXYQAOAP1,
    OpenQARPStandardP1,
    probabilities_for_patterns,
)

# ==============================================================================
# 1. 任意規模 N の BB 関数生成 & 理論値算定
# ==============================================================================

def create_scalable_materials_ground_truth(M: int, K: int = 4, seed: int = 123) -> np.ndarray:
    """M 個のブロック (各 K 選択肢, 総ビット数 N = M * K) の多元触媒材料モデル"""
    np.random.seed(seed + M)
    n = M * K
    Q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                Q[i, i] = np.random.uniform(-2.0, 1.0)
            else:
                bi = i // K
                bj = j // K
                if bi != bj:
                    # 特定の隣接・協同ブロック間に強結合シナジー
                    if (bi % 2 == 0 and bj == bi + 1):
                        Q[i, j] = np.random.uniform(-5.0, 1.5)
                    else:
                        Q[i, j] = np.random.uniform(-1.0, 1.0)
    return Q

def compute_adaptive_lambda(Q: np.ndarray, block_sizes: List[int], safety_factor: float = 1.25) -> float:
    M = len(block_sizes)
    idx = 0
    block_indices = []
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
    return float(np.round(crit_lambda * safety_factor, 2))

# ==============================================================================
# 2. FM-XY-QAOA (有効部分空間 2^(N/2) 次元シミュレータ)
# ==============================================================================

class ScalableFMXYQAOASimulator:
    """OpenQARP の Dicke+完全グラフ XY-QAOA による厳密シミュレーション。"""
    def __init__(self, M: int, K: int, Q: np.ndarray):
        self.M = M
        self.K = K
        self.n = M * K
        self.sub_dim = K ** M
        self.Q = Q

        # 部分空間内の全有効パターン生成
        self.patterns = self._build_subspace_patterns()
        # 各パターンの目的関数エネルギー
        self.energies = np.sum((self.patterns @ self.Q) * self.patterns, axis=1)
        self.opt_idx = int(np.argmin(self.energies))
        self.opt_energy = float(self.energies[self.opt_idx])
        self.opt_pattern = self.patterns[self.opt_idx]

    def _build_subspace_patterns(self) -> np.ndarray:
        patterns = np.zeros((self.sub_dim, self.n), dtype=np.float64)
        for idx in range(self.sub_dim):
            cur = idx
            for m in range(self.M - 1, -1, -1):
                choice = cur % self.K
                cur //= self.K
                patterns[idx, m * self.K + choice] = 1.0
        return patterns

    def simulate(self, num_shots: int = 1000, seed: int = 42) -> Dict[str, Any]:
        np.random.seed(seed)
        t0 = time.time()
        backend = OpenQARPFMXYQAOAP1(self.Q, [self.K] * self.M)

        # パラメータ探索 (gamma, beta)
        gammas = np.linspace(0.05, 0.8, 6)
        betas = np.linspace(0.05, 0.8, 6)

        best_exp = float("inf")
        opt_gamma, opt_beta = 0.3, 0.3

        # XY ミキサー作用: 各ブロック内で均等に遷移
        # ブロック独立の XY 遷移効果を行列ベクトル積または近似ブロックミキサーで高速計算
        # 部分空間内での XY 拡散:
        # block m の選択肢 k -> k' への振幅ホッピング
        for g in gammas:
            for b in betas:
                probs = probabilities_for_patterns(backend.probabilities(g, b), self.patterns)
                exp_e = float(np.sum(probs * self.energies))
                if exp_e < best_exp:
                    best_exp = exp_e
                    opt_gamma, opt_beta = g, b

        prob_dist = probabilities_for_patterns(
            backend.probabilities(opt_gamma, opt_beta), self.patterns
        )

        t_sim = time.time() - t0

        p_opt = float(prob_dist[self.opt_idx])
        sampled_indices = np.random.choice(self.sub_dim, size=num_shots, p=prob_dist)
        sampled_e = self.energies[sampled_indices]
        best_e = float(np.min(sampled_e))
        mean_e = float(np.mean(sampled_e))

        return {
            "feasibility_rate": 1.0,  # 厳密に 100%
            "p_opt": p_opt,
            "best_energy": best_e,
            "mean_energy": mean_e,
            "opt_gap": float(best_e - self.opt_energy),
            "runtime_sec": t_sim,
            "subspace_dim": self.sub_dim,
            "exact_min": self.opt_energy
        }

# ==============================================================================
# 3. Standard QAOA シミュレータ (全ヒルベルト空間 2^N 次元)
# ==============================================================================

class ScalableStandardQAOASimulator:
    """OpenQARP のネイティブ Standard QAOA による全空間シミュレーション。"""
    def __init__(self, M: int, K: int, Q: np.ndarray, lambda_penalty: float):
        self.M = M
        self.K = K
        self.n = M * K
        self.dim = 2 ** self.n
        self.Q = Q
        self.lambda_penalty = lambda_penalty

    def simulate(self, opt_pattern: np.ndarray, opt_energy: float, num_shots: int = 1000, seed: int = 42) -> Dict[str, Any]:
        np.random.seed(seed)
        t0 = time.time()
        backend = OpenQARPStandardP1(self.Q, [self.K] * self.M, self.lambda_penalty)
        dim = self.dim
        all_bits = backend.bits
        E_orig = backend.objective
        E_pen = backend.penalized_objective
        feasible_mask = backend.penalty == 0.0

        gammas = np.linspace(0.05, 0.8, 5)
        betas = np.linspace(0.05, 0.8, 5)

        best_exp = float("inf")
        opt_gamma, opt_beta = 0.3, 0.3

        def simulate_unitary(gamma: float, beta: float) -> np.ndarray:
            return backend.probabilities(gamma, beta)

        for g in gammas:
            for b in betas:
                probs = simulate_unitary(g, b)
                exp_e = float(np.sum(probs * E_pen))
                if exp_e < best_exp:
                    best_exp = exp_e
                    opt_gamma, opt_beta = g, b

        prob_dist = simulate_unitary(opt_gamma, opt_beta)
        t_sim = time.time() - t0

        feasibility_rate = float(np.sum(prob_dist[feasible_mask]))
        opt_idx = np.where(np.all(all_bits == opt_pattern, axis=1))[0][0]
        p_opt = float(prob_dist[opt_idx])

        sampled_indices = np.random.choice(dim, size=num_shots, p=prob_dist)
        sampled_feas = feasible_mask[sampled_indices]
        sampled_orig_e = E_orig[sampled_indices]

        if np.any(sampled_feas):
            best_e = float(np.min(sampled_orig_e[sampled_feas]))
            mean_e = float(np.mean(sampled_orig_e[sampled_feas]))
        else:
            best_e = None
            mean_e = None

        return {
            "feasibility_rate": feasibility_rate,
            "p_opt": p_opt,
            "best_energy": best_e,
            "mean_energy": mean_e,
            "opt_gap": float(best_e - opt_energy) if best_e is not None else None,
            "runtime_sec": t_sim,
            "hilbert_dim": dim
        }

# ==============================================================================
# 4. FMQA (Simulated Annealing + Penalty) スケーラブルソルバー
# ==============================================================================

class ScalableFMQASimulator:
    def __init__(self, M: int, K: int, Q: np.ndarray, lambda_penalty: float):
        self.M = M
        self.K = K
        self.n = M * K
        self.Q = Q
        self.lambda_penalty = lambda_penalty

    def solve(self, opt_pattern: np.ndarray | None, opt_energy: float | None,
              num_reads: int = 1000, seed: int = 42) -> Dict[str, Any]:
        np.random.seed(seed)
        t0 = time.time()
        n = self.n

        def energy_fn(x: np.ndarray) -> float:
            e_orig = float(x @ self.Q @ x)
            pen = 0.0
            for m in range(self.M):
                pen += (np.sum(x[m*self.K:(m+1)*self.K]) - 1.0) ** 2
            return e_orig + self.lambda_penalty * pen

        samples = []
        energies_orig = []
        feas_list = []

        # 高速化された SA サンプリング (バッチ処理)
        for _ in range(num_reads):
            # 初期ランダム解
            x = (np.random.rand(n) > 0.5).astype(np.float64)
            cur_e = energy_fn(x)
            T_sched = np.logspace(1.0, -1.5, 25)
            for T in T_sched:
                flip_idx = np.random.randint(0, n)
                x_new = x.copy()
                x_new[flip_idx] = 1.0 - x_new[flip_idx]
                new_e = energy_fn(x_new)
                dE = new_e - cur_e
                if dE < 0 or np.random.rand() < np.exp(-dE / T):
                    x = x_new
                    cur_e = new_e

            samples.append(x)
            e_orig = float(x @ self.Q @ x)
            energies_orig.append(e_orig)

            # 制約充足判定
            is_feas = True
            for m in range(self.M):
                if np.sum(x[m*self.K:(m+1)*self.K]) != 1.0:
                    is_feas = False
                    break
            feas_list.append(is_feas)

        t_sim = time.time() - t0
        samples = np.array(samples)
        energies_orig = np.array(energies_orig)
        feas_list = np.array(feas_list)

        feasibility_rate = float(np.mean(feas_list))
        p_opt = (float(np.sum(np.all(samples == opt_pattern, axis=1)) / num_reads)
                 if opt_pattern is not None else None)

        if np.any(feas_list):
            best_e = float(np.min(energies_orig[feas_list]))
            mean_e = float(np.mean(energies_orig[feas_list]))
        else:
            best_e = None
            mean_e = None

        return {
            "feasibility_rate": feasibility_rate,
            "p_opt": p_opt,
            "best_energy": best_e,
            "mean_energy": mean_e,
            "opt_gap": (float(best_e - opt_energy)
                        if best_e is not None and opt_energy is not None else None),
            "runtime_sec": t_sim
        }

# ==============================================================================
# 5. 実験実行メインルーチン
# ==============================================================================

def main():
    print("=" * 80)
    print(" 量子ビット数 N スケーリング特性・多角ベンチマーク実験 (N=8 〜 64)")
    print(" 比較: Standard QAOA (2^N) vs FMQA (SA) vs FM-XY-QAOA (2^(N/2))")
    print("=" * 80)

    scale_configs = [
        {"M": 2, "N": 8},
        {"M": 3, "N": 12},
        {"M": 4, "N": 16},
        {"M": 5, "N": 20},
        {"M": 6, "N": 24},
        {"M": 8, "N": 32},
        {"M": 10, "N": 40},
        {"M": 12, "N": 48},
        {"M": 16, "N": 64},
    ]

    results = {
        "metadata": {
            "description": "Multi-scale benchmark. Exact ground-state metrics are reported only where their state-space simulation was actually executed.",
            "date": "2026-09-20",
            "qaoa_backend": "OpenQARP 0.1.0 (qarp/qarpx)",
            "xy_mixer": "Dicke |W> initialization plus complete-graph RXX/RYY mixer",
            "K_per_block": 4,
            "scale_configs": scale_configs
        },
        "data": []
    }

    requested_n = os.environ.get("OPENQARP_SCALE_N")
    selected_n = ({int(value) for value in requested_n.split(",") if value}
                  if requested_n else {cfg["N"] for cfg in scale_configs})
    known_n = {cfg["N"] for cfg in scale_configs}
    if not selected_n <= known_n:
        raise ValueError(f"OPENQARP_SCALE_N must be drawn from {sorted(known_n)}")
    partial_run = selected_n != known_n
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_json = os.path.join(base_dir, "json", "results_qubit_scaling.json")
    if partial_run and os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        results.setdefault("metadata", {}).update({
            "qaoa_backend": "OpenQARP 0.1.0 (qarp/qarpx)",
            "openqarp_scaling_progress": f"recomputing N={sorted(selected_n)}",
        })

    for cfg in scale_configs:
        M = cfg["M"]
        N = cfg["N"]
        if N not in selected_n:
            continue
        K = 4
        block_sizes = [K] * M

        hilbert_dim = 2 ** N
        subspace_dim = K ** M
        subspace_ratio = (subspace_dim / hilbert_dim) * 100.0
        ram_bytes_full = hilbert_dim * 16.0  # complex128
        ram_bytes_sub = subspace_dim * 16.0

        print(f"\n>>> スケール M={M:2d} (N={N:2d} Qubits): 全状態 {hilbert_dim:.2e} | 有効状態 {subspace_dim:.2e} (比率: {subspace_ratio:.4e}%)")

        Q = create_scalable_materials_ground_truth(M, K, seed=123)
        lam_adapt = compute_adaptive_lambda(Q, block_sizes)

        # 1. FM-XY-QAOA の実行。OpenQARP は全 2^N 状態を保持するため、
        #    この実行環境で完走・検証する厳密 state-vector 範囲は N<=20。
        if N <= 20:
            xy_sim = ScalableFMXYQAOASimulator(M, K, Q)
            opt_pattern = xy_sim.opt_pattern
            opt_energy = xy_sim.opt_energy
            res_xy = xy_sim.simulate(num_shots=1000, seed=42)
            print(f"  [FM-XY-QAOA] 充足率: {res_xy['feasibility_rate']*100:5.1f}% | P(x*): {res_xy['p_opt']*100:6.3f}% | 時間: {res_xy['runtime_sec']:.3f}s")
        else:
            # Do not manufacture an optimum, probability, or runtime outside
            # the simulated range.  The analytical state-space size remains in
            # this record, but performance quantities are explicitly missing.
            opt_pattern = None
            opt_energy = None
            res_xy = {
                "available": False,
                "feasibility_rate": None,
                "p_opt": None,
                "best_energy": None,
                "mean_energy": None,
                "opt_gap": None,
                "runtime_sec": None,
                "subspace_dim": subspace_dim,
                "note": "Not simulated: OpenQARP's exact full-state simulator exceeds the configured numerical limit."
            }
            print("  [FM-XY-QAOA] 未計算 (OpenQARP 全状態シミュレーション上限を超過)")

        # 2. Standard QAOA の実行 (N <= 20 まで厳密全ヒルベルト発展)
        if N <= 20:
            std_sim = ScalableStandardQAOASimulator(M, K, Q, lambda_penalty=lam_adapt)
            res_std = std_sim.simulate(opt_pattern, opt_energy, num_shots=1000, seed=42)
            print(f"  [Std QAOA  ] 充足率: {res_std['feasibility_rate']*100:5.2f}% | P(x*): {res_std['p_opt']*100:6.4f}% | 時間: {res_std['runtime_sec']:.3f}s")
        else:
            res_std = {
                "available": False,
                "feasibility_rate": None,
                "p_opt": None,
                "best_energy": None,
                "mean_energy": None,
                "opt_gap": None,
                "runtime_sec": None,
                "hilbert_dim": hilbert_dim,
                "note": "Not simulated: full 2^N state-vector simulation exceeds the configured numerical limit."
            }
            print("  [Std QAOA  ] 未計算 (全ヒルベルト空間が数値シミュレーション上限を超過)")

        # 3. FMQA (Simulated Annealing, N=8〜64 全領域実行)
        fmqa_sim = ScalableFMQASimulator(M, K, Q, lambda_penalty=lam_adapt)
        res_fmqa = fmqa_sim.solve(opt_pattern, opt_energy, num_reads=600, seed=42)
        p_text = f"{res_fmqa['p_opt']*100:6.3f}%" if res_fmqa["p_opt"] is not None else "N/A (最適値未確定)"
        print(f"  [FMQA (SA) ] 充足率: {res_fmqa['feasibility_rate']*100:5.1f}% | P(x*): {p_text} | 時間: {res_fmqa['runtime_sec']:.3f}s")

        scale_record = {
            "M": M,
            "N": N,
            "hilbert_dim": hilbert_dim,
            "subspace_dim": subspace_dim,
            "subspace_ratio_percent": subspace_ratio,
            "ram_bytes_full": ram_bytes_full,
            "ram_bytes_sub": ram_bytes_sub,
            "lambda_adaptive": lam_adapt,
            "exact_min": opt_energy,
            "exact_min_available": opt_energy is not None,
            "FM_XY_QAOA": res_xy,
            "Standard_QAOA": res_std,
            "FMQA": res_fmqa
        }
        existing_index = next(
            (index for index, record in enumerate(results["data"]) if record.get("N") == N), None
        )
        if existing_index is None:
            results["data"].append(scale_record)
        else:
            results["data"][existing_index] = scale_record

    results["data"].sort(key=lambda record: record["N"])
    if partial_run:
        results["metadata"]["openqarp_scaling_progress"] = f"completed N={sorted(selected_n)}"
    else:
        results["metadata"].pop("openqarp_scaling_progress", None)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"\n★ スケーリング結果を保存しました（未計算範囲は null）: {out_json}")

if __name__ == "__main__":
    main()
