#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_xy_qaoa_bbo_scaling.py
100-cycle BBO スケーリング実証実験スクリプト (N=8, 12, 16, 20)

比較対象:
  1. FM-XY-QAOA (最高性能設定: p=2, 49 candidates, Subspace-Preserving, λ=0)
  2. Standard QAOA (全ヒルベルト空間 2^N, p=1, 動的適応ペナルティ λ_adaptive)
  3. FMQA (Simulated Annealing + 動的適応ペナルティ λ_adaptive)

対象問題:
  - BB-1: 多元触媒・機能性材料設計モデル (materials_qubo, 強相互作用シナジー結合)
  - BB-2: 全結合ランダムモデル (random_qubo)

規約遵守:
  - OpenQARP / qarpx ネイティブ厳密シミュレーション
  - N <= 20 の厳密実行限界
  - 100 サイクル BBO (初期 5 点, 提案 100 点, best_history 101 点)
  - チェックポイントによる途中保存 & 再開対応
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# intern/ およびプロジェクトルートのパスを通す
CURRENT_FILE = Path(__file__).resolve()
if CURRENT_FILE.parents[3].name == "intern":
    INTERN_DIR = CURRENT_FILE.parents[3]
    ROOT = CURRENT_FILE.parents[4]
else:
    ROOT = CURRENT_FILE.parents[3]
    INTERN_DIR = ROOT / "intern"

for p in [ROOT, INTERN_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.qarp_backend import (
    OpenQARPStandardQAOA,
    OpenQARPXYQAOA,
    all_bitstrings_lsb,
    one_hot_patterns,
    one_hot_penalty,
)


K_CHOICES = 4  # 各ブロックの選択肢数 (One-Hot k=4)


# ==============================================================================
# 1. スケーラブル BB 関数 (Ground Truth) 生成
# ==============================================================================

def materials_qubo(n_blocks: int, seed: int = 123) -> np.ndarray:
    """Scalable BB-1: 多元触媒・機能性材料設計モデル (N = n_blocks * 4)"""
    rng = np.random.RandomState(seed)
    n_qubits = n_blocks * K_CHOICES
    qubo = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for left in range(n_qubits):
        for right in range(left, n_qubits):
            if left == right:
                qubo[left, right] = rng.uniform(-2.0, 1.0)
                continue
            left_block, right_block = left // K_CHOICES, right // K_CHOICES
            if (left_block, right_block) in {(0, 1), (2, 3)}:
                qubo[left, right] = rng.uniform(-6.0, 2.0)
            elif left_block != right_block:
                qubo[left, right] = rng.uniform(-1.0, 1.0)
    return qubo


def random_qubo(n_qubits: int, seed: int = 42) -> np.ndarray:
    """Scalable BB-2: 全結合ランダム QUBO モデル"""
    rng = np.random.RandomState(seed)
    qubo = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    for left in range(n_qubits):
        for right in range(left, n_qubits):
            qubo[left, right] = rng.uniform(-1.5, 1.5) if left == right else rng.uniform(-1.0, 1.0)
    return qubo


def compute_adaptive_lambda(Q: np.ndarray, block_sizes: List[int], safety_factor: float = 1.25) -> float:
    """目的関数の負の結合スケールに応じた動的適応ペナルティ目安"""
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


def evaluate_bb(x: np.ndarray, Q: np.ndarray) -> float:
    return float(x @ Q @ x)


def is_feasible_one_hot(x: np.ndarray, block_sizes: List[int]) -> bool:
    start = 0
    for sz in block_sizes:
        if np.sum(x[start : start + sz]) != 1.0:
            return False
        start += sz
    return True


# ==============================================================================
# 2. サロゲートモデル (Ridge 回帰 二次多項式モデル)
# ==============================================================================

class SimpleSurrogateModel:
    def __init__(self, n_vars: int, alpha: float = 0.05):
        self.n_vars = n_vars
        self.alpha = alpha
        self.pair_indices = [(i, j) for i in range(n_vars) for j in range(i, n_vars)]
        self.n_features = len(self.pair_indices)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        M = len(X)
        Phi = np.zeros((M, self.n_features), dtype=np.float64)
        for m in range(M):
            x = X[m]
            for f_idx, (i, j) in enumerate(self.pair_indices):
                Phi[m, f_idx] = x[i] * x[j]

        reg = self.alpha * np.eye(self.n_features)
        try:
            w = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(Phi.T @ Phi + reg) @ (Phi.T @ y)

        Q_surr = np.zeros((self.n_vars, self.n_vars), dtype=np.float64)
        for f_idx, (i, j) in enumerate(self.pair_indices):
            Q_surr[i, j] = w[f_idx]
        return Q_surr


# ==============================================================================
# 3. 各ソルバーの実装
# ==============================================================================

# キャッシュ用変数
_PARAM_CANDIDATES_P2_CACHE: List[Tuple[List[float], List[float]]] | None = None
_FULL_SPACE_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def get_p2_parameter_candidates() -> List[Tuple[List[float], List[float]]]:
    """先行実験 xy_qaoa_accuracy_scaling で実証された最高性能 p=2, 49 候補"""
    global _PARAM_CANDIDATES_P2_CACHE
    if _PARAM_CANDIDATES_P2_CACHE is None:
        rng = np.random.default_rng(42)
        candidates = [([0.3, 0.3], [0.3, 0.3])]
        for _ in range(48):
            params = rng.uniform(0.05, 0.8, size=4)
            candidates.append(([float(params[0]), float(params[1])], [float(params[2]), float(params[3])]))
        _PARAM_CANDIDATES_P2_CACHE = candidates
    return _PARAM_CANDIDATES_P2_CACHE


class FMXYQAOABestSolver:
    """
    手法1: FM-XY-QAOA (最高性能設定: p=2, 49 candidates)
    OpenQARP / qarpx ネイティブ回路, One-Hot 初期状態, 完全グラフ RXX/RYY ミキサー, λ=0
    """
    def __init__(self, block_sizes: List[int]):
        self.block_sizes = tuple(block_sizes)
        self.n_qubits = sum(block_sizes)
        self.patterns = one_hot_patterns(self.block_sizes)
        self.pattern_indices = np.asarray(self.patterns, dtype=np.uint64) @ (
            1 << np.arange(self.n_qubits, dtype=np.uint64)
        )
        self.subspace_dim = len(self.patterns)
        self.candidates = get_p2_parameter_candidates()

    def solve(self, Q_surr: np.ndarray, num_shots: int = 500, seed: int = 42) -> Dict[str, Any]:
        backend = OpenQARPXYQAOA(Q_surr, self.block_sizes)
        surr_energies = np.sum((self.patterns @ Q_surr) * self.patterns, axis=1)

        best_exp = float("inf")
        best_sub_probs = None
        best_cand = self.candidates[0]

        for gammas, betas in self.candidates:
            probs_full = backend.probabilities(gammas, betas)
            probs_sub = probs_full[self.pattern_indices]
            exp_e = float(np.dot(probs_sub, surr_energies))
            if exp_e < best_exp:
                best_exp = exp_e
                best_sub_probs = probs_sub
                best_cand = (gammas, betas)

        # サンプリング
        assert best_sub_probs is not None
        p_norm = best_sub_probs / best_sub_probs.sum()
        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(self.subspace_dim, size=num_shots, p=p_norm)
        samples = self.patterns[sampled_indices]

        return {
            "samples": samples,
            "feasibility_rate": 1.0,
            "best_parameters": best_cand,
            "expected_surrogate_energy": best_exp,
        }


class StandardQAOABestSolver:
    """
    手法2: Standard QAOA (全空間 2^N, p=1, 5x5 グリッド探索, 動的適応ペナルティ λ_adaptive)
    """
    def __init__(self, block_sizes: List[int], lambda_penalty: float):
        self.block_sizes = tuple(block_sizes)
        self.n_qubits = sum(block_sizes)
        self.lambda_penalty = lambda_penalty

        global _FULL_SPACE_CACHE
        if self.n_qubits not in _FULL_SPACE_CACHE:
            bits = all_bitstrings_lsb(self.n_qubits)
            pen = one_hot_penalty(bits, self.block_sizes)
            _FULL_SPACE_CACHE[self.n_qubits] = (bits, pen)
        self.bits, self.penalty = _FULL_SPACE_CACHE[self.n_qubits]
        self.feasible_mask = self.penalty == 0.0

        # p=1 グリッド (5x5 = 25 点)
        side = 5
        self.candidates = [
            ([float(g)], [float(b)])
            for g in np.linspace(0.05, 0.8, side)
            for b in np.linspace(0.05, 0.8, side)
        ]

    def solve(self, Q_surr: np.ndarray, num_shots: int = 500, seed: int = 42) -> Dict[str, Any]:
        backend = OpenQARPStandardQAOA(Q_surr, self.block_sizes, self.lambda_penalty)
        obj = np.einsum("ni,ij,nj->n", self.bits, Q_surr, self.bits)
        penalized_obj = obj + self.lambda_penalty * self.penalty

        best_exp = float("inf")
        best_probs = None
        best_cand = self.candidates[0]

        for gammas, betas in self.candidates:
            probs = backend.probabilities(gammas, betas)
            exp_e = float(np.dot(probs, penalized_obj))
            if exp_e < best_exp:
                best_exp = exp_e
                best_probs = probs
                best_cand = (gammas, betas)

        assert best_probs is not None
        p_norm = best_probs / best_probs.sum()
        rng = np.random.default_rng(seed)
        dim = len(self.bits)
        sampled_indices = rng.choice(dim, size=num_shots, p=p_norm)
        samples = self.bits[sampled_indices]
        feas_rate = float(np.mean(self.feasible_mask[sampled_indices]))

        return {
            "samples": samples,
            "feasibility_rate": feas_rate,
            "best_parameters": best_cand,
            "expected_surrogate_energy": best_exp,
        }


class FMQASolver:
    """
    手法3: FMQA (Simulated Annealing + 動的適応ペナルティ λ_adaptive)
    """
    def __init__(self, block_sizes: List[int], lambda_penalty: float):
        self.block_sizes = tuple(block_sizes)
        self.n_qubits = sum(block_sizes)
        self.lambda_penalty = lambda_penalty

    def solve(self, Q_surr: np.ndarray, num_reads: int = 500, seed: int = 42) -> Dict[str, Any]:
        np.random.seed(seed)
        n = self.n_qubits
        M = len(self.block_sizes)

        def energy_fn(x: np.ndarray) -> float:
            e_orig = float(x @ Q_surr @ x)
            pen = 0.0
            start = 0
            for sz in self.block_sizes:
                pen += (np.sum(x[start : start + sz]) - 1.0) ** 2
                start += sz
            return e_orig + self.lambda_penalty * pen

        samples = []
        feas_count = 0
        T_sched = np.logspace(1.0, -1.5, 25)

        for _ in range(num_reads):
            x = (np.random.rand(n) > 0.5).astype(np.float64)
            cur_e = energy_fn(x)
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
            if is_feasible_one_hot(x, list(self.block_sizes)):
                feas_count += 1

        return {
            "samples": np.array(samples),
            "feasibility_rate": float(feas_count / num_reads),
        }


# ==============================================================================
# 4. 100 サイクル BBO 閉ループ探索ルーチン
# ==============================================================================

def run_bbo_trajectory(
    problem_name: str,
    method_name: str,
    N: int,
    Q_bb: np.ndarray,
    block_sizes: List[int],
    opt_x: np.ndarray,
    opt_val: float,
    checkpoint_path: Path,
    num_cycles: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """100 サイクル BBO を実行し、チェックポイントを管理する。"""
    n = sum(block_sizes)
    patterns = one_hot_patterns(block_sizes)

    # 既存チェックポイントの確認
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("next_cycle") == num_cycles + 1:
            print(f"  [{problem_name} | {method_name} | N={N}] 既に完走済み (100 cycles) -> キャッシュをロード")
            return saved["result"]
        # 途中再開
        checkpoint = saved["checkpoint"]
        X_history = [np.asarray(x, dtype=np.float64) for x in checkpoint["X_history"]]
        y_history = [float(y) for y in checkpoint["y_history"]]
        current_best = float(checkpoint["current_best"])
        best_history = [float(v) for v in checkpoint["best_history"]]
        infeasible_count = int(checkpoint["infeasible_count"])
        proposals = list(checkpoint["proposals"])
        start_cycle = int(checkpoint["next_cycle"])
        cycle_times = list(checkpoint.get("cycle_times", []))
    else:
        # 初期化: 正当候補 5 点 (大域最適解は除外)
        np.random.seed(seed)
        perm = np.random.permutation(len(patterns))
        X_init = []
        y_init = []
        for idx in perm:
            p = patterns[idx]
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
        cycle_times = []

    surrogate = SimpleSurrogateModel(n_vars=n, alpha=0.05)

    print(f"  [{problem_name} | {method_name} | N={N}] サイクル {start_cycle} 〜 {num_cycles} 開始...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for cycle in range(start_cycle, num_cycles + 1):
        t0_cycle = time.time()
        X_arr = np.array(X_history)
        y_arr = np.array(y_history)
        Q_surr = surrogate.fit(X_arr, y_arr)

        lam_adapt = compute_adaptive_lambda(Q_surr, block_sizes)

        # ソルバーの実行
        cycle_seed = seed * 1000 + cycle
        if method_name == "FM-XY-QAOA":
            solver = FMXYQAOABestSolver(block_sizes)
            res = solver.solve(Q_surr, num_shots=500, seed=cycle_seed)
        elif method_name == "Standard QAOA":
            solver = StandardQAOABestSolver(block_sizes, lambda_penalty=lam_adapt)
            res = solver.solve(Q_surr, num_shots=500, seed=cycle_seed)
        elif method_name == "FMQA":
            solver = FMQASolver(block_sizes, lambda_penalty=lam_adapt)
            res = solver.solve(Q_surr, num_reads=500, seed=cycle_seed)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # 共通の提案候補選択規則
        seen = {tuple(np.asarray(x, dtype=np.float64)) for x in X_history}
        unseen_feasible = {}
        for sample in res["samples"]:
            sample = np.asarray(sample, dtype=np.float64)
            key = tuple(sample)
            if key not in seen and is_feasible_one_hot(sample, block_sizes):
                unseen_feasible[key] = sample

        t_elapsed = time.time() - t0_cycle
        cycle_times.append(t_elapsed)

        if unseen_feasible:
            candidate_pool = list(unseen_feasible.values())
            surr_vals = [evaluate_bb(p, Q_surr) for p in candidate_pool]
            best_cand_idx = int(np.argmin(surr_vals))
            next_x = candidate_pool[best_cand_idx]
            y_eval = evaluate_bb(next_x, Q_bb)

            X_history.append(next_x)
            y_history.append(y_eval)

            if y_eval < current_best:
                current_best = y_eval

            best_history.append(float(current_best))
            proposals.append({
                "cycle": cycle,
                "is_feasible": True,
                "candidate_found": True,
                "y_eval": float(y_eval),
                "current_best": float(current_best),
                "lambda_used": 0.0 if method_name == "FM-XY-QAOA" else float(lam_adapt),
                "num_unseen_feasible_samples": len(candidate_pool),
                "cycle_time_sec": float(t_elapsed),
            })
        else:
            infeasible_count += 1
            best_history.append(float(current_best))
            proposals.append({
                "cycle": cycle,
                "is_feasible": False,
                "candidate_found": False,
                "y_eval": None,
                "current_best": float(current_best),
                "lambda_used": 0.0 if method_name == "FM-XY-QAOA" else float(lam_adapt),
                "num_unseen_feasible_samples": 0,
                "cycle_time_sec": float(t_elapsed),
            })

        # チェックポイント保存 (10 サイクルごと、または完走時)
        if cycle % 10 == 0 or cycle == num_cycles:
            result_data = {
                "method": method_name,
                "problem": problem_name,
                "N": N,
                "final_best": float(current_best),
                "exact_min": float(opt_val),
                "optimality_gap": float(current_best - opt_val),
                "infeasible_count": int(infeasible_count),
                "feasibility_success_rate": float((num_cycles - infeasible_count) / num_cycles),
                "best_history": best_history,
                "proposals": proposals,
                "mean_cycle_time": float(np.mean(cycle_times)),
                "total_runtime_sec": float(np.sum(cycle_times)),
            }
            chk_data = {
                "next_cycle": cycle + 1,
                "checkpoint": {
                    "X_history": [x.tolist() for x in X_history],
                    "y_history": y_history,
                    "current_best": current_best,
                    "best_history": best_history,
                    "infeasible_count": infeasible_count,
                    "proposals": proposals,
                    "next_cycle": cycle + 1,
                    "cycle_times": cycle_times,
                },
                "result": result_data,
            }
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(chk_data, f, indent=2)
            print(f"    -> サイクル {cycle}/{num_cycles} 完了 (best: {current_best:.4f}, gap: {current_best - opt_val:.4f}, failures: {infeasible_count})")

    return result_data


# ==============================================================================
# 5. メイン実行ルーチン
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="100-cycle BBO Scaling Experiment (N=8..20)")
    parser.add_argument("--problems", nargs="+", choices=["BB1_materials", "BB2_random"], default=["BB1_materials", "BB2_random"])
    parser.add_argument("--methods", nargs="+", choices=["FM-XY-QAOA", "Standard QAOA", "FMQA"], default=["FM-XY-QAOA", "Standard QAOA", "FMQA"])
    parser.add_argument("--scales", nargs="+", type=int, default=[8, 12, 16, 20])
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 80)
    print(" 100-cycle BBO スケーリング実証実験 (N=8 〜 20)")
    print(f" 手法: {args.methods}")
    print(f" 問題: {args.problems}")
    print(f" スケール: N = {args.scales}")
    print(f" サイクル数: {args.cycles}")
    print("=" * 80)

    exp_dir = INTERN_DIR / "experimet" / "xy_qaoa_bbo_scaling"
    chk_dir = exp_dir / "json" / "checkpoints"
    out_json = exp_dir / "json" / "results_xy_qaoa_bbo_scaling.json"

    all_results: Dict[str, Any] = {
        "metadata": {
            "description": "100-cycle BBO scaling benchmark comparing FM-XY-QAOA (p=2, 49 candidates) with Standard QAOA (p=1, adaptive lambda) and FMQA across N=8,12,16,20.",
            "backend": "OpenQARP 0.1.0 (qarpx)",
            "xy_qaoa_setting": "p=2, 49 candidates (seed=42 uniform + [0.3]^4 candidate), lambda=0",
            "std_qaoa_setting": "p=1, 5x5 grid (25 points), adaptive penalty lambda",
            "fmqa_setting": "Simulated Annealing (500 reads), adaptive penalty lambda",
            "bbo_protocol": "Initial 5 feasible points, Ridge quadratic surrogate, 100 proposals (best_history length 101)",
            "scales": args.scales,
            "problems": args.problems,
            "methods": args.methods,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "problems": {}
    }

    # 既存の集約 JSON があればロード
    if out_json.exists():
        with out_json.open("r", encoding="utf-8") as f:
            stored = json.load(f)
            all_results["problems"] = stored.get("problems", {})

    for prob in args.problems:
        print(f"\n######################################################################")
        print(f" 問題: {prob}")
        print(f"######################################################################")
        if prob not in all_results["problems"]:
            all_results["problems"][prob] = {"scales": {}}

        for N in args.scales:
            M = N // K_CHOICES
            block_sizes = [K_CHOICES] * M
            patterns = one_hot_patterns(block_sizes)

            if prob == "BB1_materials":
                Q_bb = materials_qubo(M, seed=123)
            else:
                Q_bb = random_qubo(N, seed=42)

            energies = np.sum((patterns @ Q_bb) * patterns, axis=1)
            opt_idx = int(np.argmin(energies))
            opt_val = float(energies[opt_idx])
            opt_x = patterns[opt_idx]

            print(f"\n--- スケール N={N} (M={M} blocks, 実行可能空間 {len(patterns)}, 真の最適値 {opt_val:.4f}) ---")
            scale_key = str(N)
            if scale_key not in all_results["problems"][prob]["scales"]:
                all_results["problems"][prob]["scales"][scale_key] = {
                    "N": N,
                    "M": M,
                    "exact_min": opt_val,
                    "methods": {},
                }

            for method in args.methods:
                method_slug = method.lower().replace("-", "_").replace(" ", "_")
                chk_file = chk_dir / f"{prob.lower()}_{method_slug}_N{N}_seed{args.seed}.json"

                res = run_bbo_trajectory(
                    problem_name=prob,
                    method_name=method,
                    N=N,
                    Q_bb=Q_bb,
                    block_sizes=block_sizes,
                    opt_x=opt_x,
                    opt_val=opt_val,
                    checkpoint_path=chk_file,
                    num_cycles=args.cycles,
                    seed=args.seed,
                )
                all_results["problems"][prob]["scales"][scale_key]["methods"][method] = res

        # 問題ごとに随時集約 JSON を更新
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # 正規ツリー側の experimet/ にもコピー
    canonical_json = ROOT / "experimet" / "xy_qaoa_bbo_scaling" / "json" / "results_xy_qaoa_bbo_scaling.json"
    canonical_json.parent.mkdir(parents=True, exist_ok=True)
    with canonical_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f" 全 100-cycle BBO スケーリング実験完了！")
    print(f" 結果出力: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()
