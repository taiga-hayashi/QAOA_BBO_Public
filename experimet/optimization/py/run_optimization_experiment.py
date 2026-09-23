#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_optimization_experiment.py
experimet/optimization フォルダ内での実証実験スクリプト:
2つのBB関数 (多元触媒材料モデル Q_materials, 全結合ランダムモデル Q_random) に対する
3つの最適化手法 (FMQA, Standard QAOA, FM-XY-QAOA) の最適化実行および性能比較。

★ FMQA は目的関数の強相互作用スケールに応じた利得ベースの適応ペナルティ目安 λ_adaptive を
  自動算定（スイートスポット設定）した上で最適化を実行。

Part 1: 直接QUBO最適化・サンプリング性能比較 (適切な λ_adaptive の適用)
Part 2: 100サイクルのBBO (Black-Box Optimization) 自律探索ループ比較
"""

import os
import sys
import json
import time
from collections import OrderedDict
from typing import List, Dict, Tuple, Any
import numpy as np

EXPERIMET_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "py"))
if EXPERIMET_PY not in sys.path:
    sys.path.insert(0, EXPERIMET_PY)
from openqarp_qaoa import OpenQARPFMXYQAOAP1, OpenQARPStandardP1, probabilities_for_patterns


# Exact circuit grids are reused only when the surrogate QUBO and penalty are
# bit-identical.  Sampling remains seed-specific, so this changes runtime but
# not the proposal distribution or any reported statistic.
_STANDARD_QAOA_GRID_CACHE: OrderedDict[tuple[bytes, tuple[int, ...], float], tuple[np.ndarray, float, float]] = OrderedDict()
_XY_QAOA_GRID_CACHE: OrderedDict[tuple[bytes, tuple[int, ...]], tuple[np.ndarray, float, float]] = OrderedDict()
_QAOA_CACHE_LIMIT = 32


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _QAOA_CACHE_LIMIT:
        cache.popitem(last=False)

# ==============================================================================
# 1. 適応ペナルティ係数 λ_adaptive の利得ベース目安
# ==============================================================================

def compute_theoretical_adaptive_lambda(Q: np.ndarray, block_sizes: List[int], safety_factor: float = 1.25) -> float:
    """
    強相互作用行列 Q と One-Hot ブロック構造から、
    目的関数の負の相互作用から利得ベースのペナルティ目安を計算する。

    これは経験的な安全係数付きヒューリスティックであり、制約充足や
    大域最適性を保証する理論下限値ではない。
    各ブロック m において、追加で 1 ビット立てたときに獲得しうる最大引力エネルギー (負の結合) を相殺する。
    """
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
    return float(np.round(crit_lambda * safety_factor, 2))

# ==============================================================================
# 2. 2つのBB関数 (Ground Truth) 定義
# ==============================================================================

def create_materials_design_ground_truth(block_sizes: List[int], seed: int = 123) -> np.ndarray:
    """
    BB関数1: 多元触媒・機能性材料設計モデル (Q_materials)
    4サイト×4元素 (N=16) の協同触媒シナジー相互作用 (サイト0-1, 2-3間に-6.0〜2.0の強相関結合)
    """
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
                        # 強相関協同触媒効果 (スケール5〜8倍)
                        Q[i, j] = np.random.uniform(-6.0, 2.0)
                    else:
                        Q[i, j] = np.random.uniform(-1.0, 1.0)
    return Q

def create_random_qubo_bb(d: int, seed: int = 42) -> np.ndarray:
    """
    BB関数2: 全結合ランダム相互作用モデル (Q_random)
    全変数間に均一な独立ランダム結合を持つフラストレーション・スピングラス系
    """
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
    """二値ベクトル x に対する BB関数の値 x^T Q x"""
    return float(x @ Q @ x)

def is_feasible_one_hot(x: np.ndarray, block_sizes: List[int]) -> bool:
    """One-Hot制約を満たすか判定"""
    idx = 0
    for sz in block_sizes:
        if np.sum(x[idx:idx+sz]) != 1.0:
            return False
        idx += sz
    return True

def get_all_feasible_patterns(block_sizes: List[int]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """すべての有効解 (4^4 = 256 通り) を生成"""
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
    """全探索による真の大域最適解と最小値"""
    vals = [evaluate_bb(p, Q) for p in feasible_patterns]
    min_idx = int(np.argmin(vals))
    return feasible_patterns[min_idx], float(vals[min_idx])

# ==============================================================================
# 3. 3つの最適化手法の実装
# ==============================================================================

class FMQASolver:
    """
    手法1: FMQA (Simulated Annealing + One-Hot Penalty)
    ペナルティ係数 lambda_penalty を適切に設定可能
    """
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

    def solve(self, Q: np.ndarray, num_reads: int = 1000, seed: int = 42) -> Dict[str, Any]:
        np.random.seed(seed)
        samples = []
        energies_pen = []
        energies_orig = []
        feasibilities = []

        T_start = 8.0
        T_end = 0.01
        steps = 180
        gamma = (T_end / T_start) ** (1.0 / steps)

        for read_idx in range(num_reads):
            x = np.random.randint(0, 2, self.total_vars).astype(np.float64)
            current_e = self.energy_with_penalty(x, Q)
            T = T_start

            for _ in range(steps):
                flip_idx = np.random.randint(0, self.total_vars)
                x_new = x.copy()
                x_new[flip_idx] = 1.0 - x_new[flip_idx]
                new_e = self.energy_with_penalty(x_new, Q)
                delta_e = new_e - current_e

                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
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
            best_energy = float(energies_orig[best_feas_idx])
            mean_feas_energy = float(np.mean(feas_energies))
        else:
            best_idx = int(np.argmin(energies_pen))
            best_sample = samples[best_idx]
            best_energy = float(energies_orig[best_idx])
            mean_feas_energy = float(best_energy)

        unique_samples = len(set(tuple(s) for s in samples))

        return {
            "samples": samples,
            "feasibility_rate": feas_ratio,
            "best_sample": best_sample,
            "best_energy": best_energy,
            "mean_energy": mean_feas_energy,
            "unique_samples": unique_samples,
            "unique_ratio": unique_samples / num_reads,
            "lambda_used": self.lambda_penalty
        }


class StandardQAOASolver:
    """
    手法2: Standard QAOA (X-Mixer + One-Hot Penalty, p=1)
    OpenQARP のネイティブ QAOA 回路による全空間状態ベクトル進化
    """
    def __init__(self, block_sizes: List[int], lambda_penalty: float = 5.0):
        self.block_sizes = block_sizes
        self.lambda_penalty = lambda_penalty
        self.n_qubits = sum(block_sizes)
        self.dim = 2 ** self.n_qubits

    def solve(self, Q: np.ndarray, num_shots: int = 1000, seed: int = 42) -> Dict[str, Any]:
        np.random.seed(seed)
        dim = self.dim
        all_bits = np.zeros((dim, self.n_qubits), dtype=np.float64)
        values = np.arange(dim, dtype=np.uint64)[:, None]
        all_bits[:] = ((values >> np.arange(self.n_qubits, dtype=np.uint64)) & 1)
        E_orig = np.sum((all_bits @ Q) * all_bits, axis=1)
        pen = np.zeros(dim, dtype=np.float64)
        start = 0
        for size in self.block_sizes:
            pen += (all_bits[:, start:start + size].sum(axis=1) - 1.0) ** 2
            start += size
        E_pen = E_orig + self.lambda_penalty * pen
        feasible_mask = pen == 0.0

        best_expected_e = float("inf")
        opt_gamma, opt_beta = 0.5, 0.5

        gammas = np.linspace(0.05, 1.2, 10)
        betas = np.linspace(0.05, 1.2, 10)

        def simulate(gamma: float, beta: float) -> np.ndarray:
            return backend.probabilities(gamma, beta)

        cache_key = (np.ascontiguousarray(Q, dtype=np.float64).tobytes(), tuple(self.block_sizes), self.lambda_penalty)
        cached = _cache_get(_STANDARD_QAOA_GRID_CACHE, cache_key)
        if cached is None:
            backend = OpenQARPStandardP1(Q, self.block_sizes, self.lambda_penalty)
            for g in gammas:
                for b in betas:
                    probs = simulate(g, b)
                    exp_val = np.sum(probs * E_pen)
                    if exp_val < best_expected_e:
                        best_expected_e = exp_val
                        opt_gamma, opt_beta = g, b
            prob_dist = simulate(opt_gamma, opt_beta)
            _cache_put(_STANDARD_QAOA_GRID_CACHE, cache_key, (prob_dist, opt_gamma, opt_beta))
        else:
            prob_dist, opt_gamma, opt_beta = cached
        feas_ratio = float(np.sum(prob_dist[feasible_mask]))

        sampled_indices = np.random.choice(dim, size=num_shots, p=prob_dist)
        sampled_bits = all_bits[sampled_indices]
        sampled_feas = feasible_mask[sampled_indices]
        sampled_orig_e = E_orig[sampled_indices]

        if np.any(sampled_feas):
            feas_sample_e = sampled_orig_e[sampled_feas]
            best_energy = float(np.min(feas_sample_e))
            best_sample = sampled_bits[sampled_feas][int(np.argmin(feas_sample_e))]
            mean_energy = float(np.mean(feas_sample_e))
        else:
            best_idx = int(np.argmin(sampled_orig_e))
            best_sample = sampled_bits[best_idx]
            best_energy = float(sampled_orig_e[best_idx])
            mean_energy = float(np.mean(sampled_orig_e))

        unique_samples = len(set(tuple(s) for s in sampled_bits))

        return {
            "probabilities": prob_dist,
            "all_bits": all_bits,
            "samples": sampled_bits,
            "feasible_mask": feasible_mask,
            "feasibility_rate": feas_ratio,
            "best_sample": best_sample,
            "best_energy": best_energy,
            "mean_energy": mean_energy,
            "unique_samples": unique_samples,
            "unique_ratio": unique_samples / num_shots,
            "opt_gamma": opt_gamma,
            "opt_beta": opt_beta,
            "lambda_used": self.lambda_penalty
        }


class FMXYQAOASolver:
    """
    手法3: FM-XY-QAOA (Subspace-Preserving XY-Mixer, Penalty-free lambda=0, p=1)
    OpenQARP の Dicke 初期状態と完全グラフ XY ミキサーによる p=1 回路
    """
    def __init__(self, block_sizes: List[int]):
        self.block_sizes = block_sizes
        self.n_blocks = len(block_sizes)
        self.block_sz = block_sizes[0]
        self.subspace_dim = int(np.prod(block_sizes)) # 256
        self.patterns, self.indices = get_all_feasible_patterns(block_sizes)

    def solve(self, Q: np.ndarray, num_shots: int = 1000, seed: int = 42) -> Dict[str, Any]:
        np.random.seed(seed)
        patterns = self.patterns
        E_orig = np.sum((patterns @ Q) * patterns, axis=1)

        gammas = np.linspace(0.05, 1.2, 10)
        betas = np.linspace(0.05, 1.2, 10)
        best_expected_e = float("inf")
        opt_gamma, opt_beta = 0.5, 0.5

        def simulate(gamma: float, beta: float) -> np.ndarray:
            return probabilities_for_patterns(backend.probabilities(gamma, beta), patterns)

        cache_key = (np.ascontiguousarray(Q, dtype=np.float64).tobytes(), tuple(self.block_sizes))
        cached = _cache_get(_XY_QAOA_GRID_CACHE, cache_key)
        if cached is None:
            backend = OpenQARPFMXYQAOAP1(Q, self.block_sizes)
            for g in gammas:
                for b in betas:
                    probs = simulate(g, b)
                    exp_val = np.sum(probs * E_orig)
                    if exp_val < best_expected_e:
                        best_expected_e = exp_val
                        opt_gamma, opt_beta = g, b
            prob_dist = simulate(opt_gamma, opt_beta)
            _cache_put(_XY_QAOA_GRID_CACHE, cache_key, (prob_dist, opt_gamma, opt_beta))
        else:
            prob_dist, opt_gamma, opt_beta = cached

        sampled_indices = np.random.choice(self.subspace_dim, size=num_shots, p=prob_dist)
        sampled_patterns = patterns[sampled_indices]
        sampled_energies = E_orig[sampled_indices]

        best_idx = int(np.argmin(sampled_energies))
        best_sample = sampled_patterns[best_idx]
        best_energy = float(sampled_energies[best_idx])
        mean_energy = float(np.mean(sampled_energies))

        unique_samples = len(set(tuple(s) for s in sampled_patterns))

        return {
            "probabilities": prob_dist,
            "patterns": patterns,
            "samples": sampled_patterns,
            "feasibility_rate": 1.0, # 厳密に100.0%
            "best_sample": best_sample,
            "best_energy": best_energy,
            "mean_energy": mean_energy,
            "unique_samples": unique_samples,
            "unique_ratio": unique_samples / num_shots,
            "opt_gamma": opt_gamma,
            "opt_beta": opt_beta,
            "lambda_used": 0.0 # ペナルティ不要
        }

# ==============================================================================
# 4. BBO サロゲートモデル & 閉ループ探索
# ==============================================================================

class SimpleSurrogateModel:
    def __init__(self, n_vars: int, alpha: float = 0.05):
        self.n_vars = n_vars
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = self.n_vars
        pair_indices = [(i, j) for i in range(n) for j in range(i, n)]
        n_features = len(pair_indices)

        M = len(X)
        Phi = np.zeros((M, n_features), dtype=np.float64)
        for m in range(M):
            x = X[m]
            for f_idx, (i, j) in enumerate(pair_indices):
                Phi[m, f_idx] = x[i] * x[j]

        reg = self.alpha * np.eye(n_features)
        try:
            w = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(Phi.T @ Phi + reg) @ (Phi.T @ y)

        Q_surr = np.zeros((n, n), dtype=np.float64)
        for f_idx, (i, j) in enumerate(pair_indices):
            Q_surr[i, j] = w[f_idx]
        return Q_surr

def run_bbo_closed_loop(
    solver_name: str,
    Q_bb: np.ndarray,
    block_sizes: List[int],
    opt_x: np.ndarray,
    opt_val: float,
    num_cycles: int = 10,
    seed: int = 42,
    checkpoint: Dict[str, Any] | None = None,
    end_cycle: int | None = None,
) -> Dict[str, Any]:
    n = sum(block_sizes)
    if checkpoint is None:
        np.random.seed(seed)
        feasible_patterns, _ = get_all_feasible_patterns(block_sizes)

        # 初期データ: ランダムな正当候補 5 点 (大域最適解は除外)
        init_indices = np.random.choice(len(feasible_patterns), size=10, replace=False)
        X_init = []
        y_init = []
        for idx in init_indices:
            p = feasible_patterns[idx]
            if not np.array_equal(p, opt_x):
                X_init.append(p)
                y_init.append(evaluate_bb(p, Q_bb))
            if len(X_init) == 5:
                break

        X_history = list(X_init)
        y_history = list(y_init)
        current_best = float(np.min(y_history))
        best_history = [current_best]
        infeasible_count = 0
        proposals = []
        start_cycle = 1
    else:
        X_history = [np.asarray(x, dtype=np.float64) for x in checkpoint["X_history"]]
        y_history = [float(y) for y in checkpoint["y_history"]]
        current_best = float(checkpoint["current_best"])
        best_history = [float(value) for value in checkpoint["best_history"]]
        infeasible_count = int(checkpoint["infeasible_count"])
        proposals = list(checkpoint["proposals"])
        start_cycle = int(checkpoint["next_cycle"])

    if end_cycle is None:
        end_cycle = num_cycles
    if not start_cycle <= end_cycle <= num_cycles:
        raise ValueError("end_cycle must satisfy next_cycle <= end_cycle <= num_cycles")

    surrogate = SimpleSurrogateModel(n_vars=n, alpha=0.05)

    for cycle in range(start_cycle, end_cycle + 1):
        X_arr = np.array(X_history)
        y_arr = np.array(y_history)
        Q_surr = surrogate.fit(X_arr, y_arr)

        # サロゲートのスケールに応じた動的ペナルティ目安を計算
        lambda_adapt = compute_theoretical_adaptive_lambda(Q_surr, block_sizes, safety_factor=1.25)

        if solver_name == "FMQA":
            # 適切なペナルティ係数を定めて実行
            solver = FMQASolver(block_sizes, lambda_penalty=lambda_adapt)
            res = solver.solve(Q_surr, num_reads=500, seed=seed * 1000 + cycle)
        elif solver_name == "Standard QAOA":
            # Standard QAOA も適応ペナルティを適用
            solver = StandardQAOASolver(block_sizes, lambda_penalty=lambda_adapt)
            res = solver.solve(Q_surr, num_shots=500, seed=seed * 1000 + cycle)
        elif solver_name == "FM-XY-QAOA":
            # 提案手法はペナルティ完全不要 (lambda = 0)
            solver = FMXYQAOASolver(block_sizes)
            res = solver.solve(Q_surr, num_shots=500, seed=seed * 1000 + cycle)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        # 全手法に共通の候補選択規則を適用する。
        # 未評価かつ実行可能なサンプルだけを候補とし、サロゲート値が最良のものを選ぶ。
        seen = {tuple(np.asarray(x, dtype=np.float64)) for x in X_history}
        unseen_feasible = {}
        for sample in res["samples"]:
            sample = np.asarray(sample, dtype=np.float64)
            key = tuple(sample)
            if key not in seen and is_feasible_one_hot(sample, block_sizes):
                unseen_feasible[key] = sample

        if unseen_feasible:
            candidate_pool = list(unseen_feasible.values())
        else:
            # 有効な未評価サンプルがなければBB関数を評価しない。
            # 全実行可能解の探索で補うと、低い制約充足率を持つ手法だけを
            # 外部の全探索で救済することになり、比較が不公平になる。
            infeasible_count += 1
            best_history.append(current_best)
            proposals.append({
                "cycle": cycle,
                "is_feasible": False,
                "candidate_found": False,
                "y_eval": None,
                "current_best": float(current_best),
                "lambda_used": float(lambda_adapt) if solver_name != "FM-XY-QAOA" else 0.0,
                "x": None,
                "num_unseen_feasible_samples": 0
            })
            continue

        surrogate_values = [evaluate_bb(p, Q_surr) for p in candidate_pool]
        prop_x = candidate_pool[int(np.argmin(surrogate_values))]

        is_feas = is_feasible_one_hot(prop_x, block_sizes)
        if is_feas:
            y_eval = evaluate_bb(prop_x, Q_bb)
            if y_eval < current_best:
                current_best = y_eval
        else:
            infeasible_count += 1
            y_eval = float(np.max(y_history) + 10.0)

        X_history.append(prop_x)
        y_history.append(y_eval)
        best_history.append(current_best)

        proposals.append({
            "cycle": cycle,
            "is_feasible": bool(is_feas),
            "candidate_found": True,
            "y_eval": float(y_eval),
            "current_best": float(current_best),
            "lambda_used": float(lambda_adapt) if solver_name != "FM-XY-QAOA" else 0.0,
            "x": [int(v) for v in prop_x],
            "num_unseen_feasible_samples": len(unseen_feasible)
        })

    output = {
        "solver": solver_name,
        "best_history": best_history,
        "infeasible_count": infeasible_count,
        "final_best": current_best,
        "optimality_gap": float(current_best - opt_val),
        "proposals": proposals
    }
    output["checkpoint"] = {
        "X_history": [[int(value) for value in x] for x in X_history],
        "y_history": [float(value) for value in y_history],
        "current_best": float(current_best),
        "best_history": [float(value) for value in best_history],
        "infeasible_count": int(infeasible_count),
        "proposals": proposals,
        "next_cycle": end_cycle + 1,
    }
    return output

# ==============================================================================
# 5. メイン実行ルーチン
# ==============================================================================

def main():
    print("==========================================================================")
    print(" 2つのBB関数に対する3手法最適化 実証実験 (N=16, 4サイト×4元素)")
    print(" ★ FMQA は利得ベースのペナルティ目安 λ_adaptive を用いて実行")
    print(" 実行ディレクトリ: experimet/optimization/")
    print("==========================================================================")

    block_sizes = [4, 4, 4, 4]
    feasible_patterns, _ = get_all_feasible_patterns(block_sizes)

    # 1. BB-1 多元触媒モデル
    print("\n>>> [BB-1] 多元触媒・機能性材料設計モデル (Q_materials) 構築中...")
    Q_mat = create_materials_design_ground_truth(block_sizes, seed=123)
    opt_x_mat, opt_val_mat = get_exact_minimum(Q_mat, feasible_patterns)
    lambda_mat_adapt = compute_theoretical_adaptive_lambda(Q_mat, block_sizes, safety_factor=1.25)
    print(f"真の大域最適値 f*: {opt_val_mat:.4f}")
    print(f"利得ベースのペナルティ目安 λ_adaptive: {lambda_mat_adapt:.2f}（保証値ではない）")

    # 2. BB-2 全結合ランダムモデル
    print("\n>>> [BB-2] 全結合ランダム相互作用モデル (Q_random) 構築中...")
    Q_rand = create_random_qubo_bb(sum(block_sizes), seed=42)
    opt_x_rand, opt_val_rand = get_exact_minimum(Q_rand, feasible_patterns)
    lambda_rand_adapt = compute_theoretical_adaptive_lambda(Q_rand, block_sizes, safety_factor=1.25)
    print(f"真の大域最適値 f*: {opt_val_rand:.4f}")
    print(f"利得ベースのペナルティ目安 λ_adaptive: {lambda_rand_adapt:.2f}（保証値ではない）")

    results: Dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "N": sum(block_sizes),
            "block_sizes": block_sizes,
            "total_states": 2 ** sum(block_sizes),
            "valid_states": len(feasible_patterns),
            "qaoa_backend": "OpenQARP 0.1.0 (qarp/qarpx)",
            "qaoa_circuit_convention": "p=1; Standard uses native QAOA, FM-XY uses Dicke-state preparation plus complete-graph RXX/RYY mixer"
        },
        "BB1_materials": {
            "name": "Materials Catalyst Multi-Scale Model",
            "exact_min": opt_val_mat,
            "exact_x": [int(x) for x in opt_x_mat],
            "lambda_adaptive": lambda_mat_adapt,
            "lambda_default": 5.0,
            "part1_direct": {},
            "part2_bbo": {}
        },
        "BB2_random": {
            "name": "Fully-Connected Random QUBO Model",
            "exact_min": opt_val_rand,
            "exact_x": [int(x) for x in opt_x_rand],
            "lambda_adaptive": lambda_rand_adapt,
            "lambda_default": 5.0,
            "part1_direct": {},
            "part2_bbo": {}
        }
    }

    # ==========================================================================
    # PART 1: 直接QUBO最適化・サンプリング比較
    # ==========================================================================
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, "json", "results_two_bb_optimization.json")
    reuse_part1 = False
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            if (
                old_data.get("metadata", {}).get("qaoa_backend")
                == "OpenQARP 0.1.0 (qarp/qarpx)"
                and "BB1_materials" in old_data
                and "part1_direct" in old_data["BB1_materials"]
            ):
                results["BB1_materials"]["part1_direct"] = old_data["BB1_materials"]["part1_direct"]
                results["BB2_random"]["part1_direct"] = old_data["BB2_random"]["part1_direct"]
                reuse_part1 = True
                print("\n★ OpenQARP による既存の PART 1 実測データを再利用します。")
            else:
                print("\n★ 既存 PART 1 は OpenQARP 実行前のため再計算します。")
        except Exception as e:
            print(f"Note: 既存データの読み込みをスキップしました: {e}")

    if not reuse_part1:
        print("\n" + "="*70)
        print(" PART 1: 直接QUBO最適化・サンプリング性能比較 (1,000 Reads/Shots)")
        print("="*70)

        configs = [
            ("BB1_materials", Q_mat, opt_x_mat, opt_val_mat, lambda_mat_adapt, results["BB1_materials"]["part1_direct"]),
            ("BB2_random", Q_rand, opt_x_rand, opt_val_rand, lambda_rand_adapt, results["BB2_random"]["part1_direct"])
        ]

        for bb_name, Q_target, opt_x, opt_val, l_adapt, res_dict in configs:
            print(f"\n--- 実行中: {bb_name} (適切ペナルティ λ_adaptive = {l_adapt}) ---")

            # 1-A. FMQA (デフォルト固定ペナルティ λ=5.0)
            fmqa_def = FMQASolver(block_sizes, lambda_penalty=5.0)
            res_fmqa_def = fmqa_def.solve(Q_target, num_reads=1000, seed=42)
            p_opt_fmqa_def = sum(np.array_equal(s, opt_x) for s in res_fmqa_def["samples"]) / 1000.0
            res_dict["FMQA_default"] = {
                "lambda": 5.0,
                "feasibility_rate": res_fmqa_def["feasibility_rate"],
                "success_probability": p_opt_fmqa_def,
                "best_energy": res_fmqa_def["best_energy"],
                "optimality_gap": float(res_fmqa_def["best_energy"] - opt_val),
                "mean_energy": res_fmqa_def["mean_energy"],
                "unique_ratio": res_fmqa_def["unique_ratio"]
            }

            # 1-B. FMQA (適切なペナルティ係数 λ_adaptive)
            t0 = time.time()
            fmqa_adapt = FMQASolver(block_sizes, lambda_penalty=l_adapt)
            res_fmqa_adapt = fmqa_adapt.solve(Q_target, num_reads=1000, seed=42)
            t_fmqa_adapt = time.time() - t0
            p_opt_fmqa_adapt = sum(np.array_equal(s, opt_x) for s in res_fmqa_adapt["samples"]) / 1000.0
            res_dict["FMQA_adaptive"] = {
                "lambda": l_adapt,
                "feasibility_rate": res_fmqa_adapt["feasibility_rate"],
                "success_probability": p_opt_fmqa_adapt,
                "best_energy": res_fmqa_adapt["best_energy"],
                "optimality_gap": float(res_fmqa_adapt["best_energy"] - opt_val),
                "mean_energy": res_fmqa_adapt["mean_energy"],
                "unique_ratio": res_fmqa_adapt["unique_ratio"],
                "runtime_sec": t_fmqa_adapt
            }

            # 2. Standard QAOA
            t0 = time.time()
            std_qaoa = StandardQAOASolver(block_sizes, lambda_penalty=l_adapt)
            res_std = std_qaoa.solve(Q_target, num_shots=1000, seed=42)
            t_std = time.time() - t0
            opt_idx_all = np.where(np.all(res_std["all_bits"] == opt_x, axis=1))[0][0]
            p_opt_std = float(res_std["probabilities"][opt_idx_all])
            res_dict["Standard_QAOA"] = {
                "lambda": l_adapt,
                "feasibility_rate": res_std["feasibility_rate"],
                "success_probability": p_opt_std,
                "best_energy": res_std["best_energy"],
                "optimality_gap": float(res_std["best_energy"] - opt_val),
                "mean_energy": res_std["mean_energy"],
                "unique_ratio": res_std["unique_ratio"],
                "runtime_sec": t_std
            }

            # 3. FM-XY-QAOA
            t0 = time.time()
            xy_qaoa = FMXYQAOASolver(block_sizes)
            res_xy = xy_qaoa.solve(Q_target, num_shots=1000, seed=42)
            t_xy = time.time() - t0
            opt_idx_sub = np.where(np.all(res_xy["patterns"] == opt_x, axis=1))[0][0]
            p_opt_xy = float(res_xy["probabilities"][opt_idx_sub])
            res_dict["FM_XY_QAOA"] = {
                "lambda": 0.0,
                "feasibility_rate": res_xy["feasibility_rate"],
                "success_probability": p_opt_xy,
                "best_energy": res_xy["best_energy"],
                "optimality_gap": float(res_xy["best_energy"] - opt_val),
                "mean_energy": res_xy["mean_energy"],
                "unique_ratio": res_xy["unique_ratio"],
                "runtime_sec": t_xy
            }

    # ==========================================================================
    # PART 2: 100サイクルの BBO 閉ループ自律探索比較（利得ベースの動的λ目安）
    # ==========================================================================
    bbo_cycles = 100
    print("\n" + "="*70)
    print(f" PART 2: {bbo_cycles}サイクルの BBO (ブラックボックス最適化) 自律探索ループ実証")
    print("="*70)

    for bb_name, Q_target, opt_x, opt_val, l_adapt, res_dict in [
        ("BB1_materials", Q_mat, opt_x_mat, opt_val_mat, lambda_mat_adapt, results["BB1_materials"]["part2_bbo"]),
        ("BB2_random", Q_rand, opt_x_rand, opt_val_rand, lambda_rand_adapt, results["BB2_random"]["part2_bbo"])
    ]:
        print(f"\n--- BBO実行中: {bb_name} ({bbo_cycles} サイクル) ---")

        for solver_name in ["FMQA", "Standard QAOA", "FM-XY-QAOA"]:
            t0 = time.time()
            bbo_res = run_bbo_closed_loop(
                solver_name=solver_name,
                Q_bb=Q_target,
                block_sizes=block_sizes,
                opt_x=opt_x,
                opt_val=opt_val,
                num_cycles=bbo_cycles,
                seed=42
            )
            t_bbo = time.time() - t0
            bbo_res["runtime_sec"] = t_bbo
            key = solver_name.replace(" ", "_").replace("-", "_")
            res_dict[key] = bbo_res
            print(f"  [{solver_name:14s}] 最良値: {bbo_res['final_best']:.4f} (Gap: {bbo_res['optimality_gap']:.4f}) | 無効提案: {bbo_res['infeasible_count']:2d}/{bbo_cycles}回 | 所要時間: {t_bbo:.2f}s")

    # 結果保存
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n★ 全実験結果が正常に保存されました: {json_path}")

if __name__ == "__main__":
    main()
