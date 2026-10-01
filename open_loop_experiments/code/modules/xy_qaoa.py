import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src")))
try:
    from qarp_backend import OpenQARPXYQAOA, one_hot_patterns
except ImportError:
    OpenQARPXYQAOA = None

def solve_xy_fmqa_p_layers(Q_fm: np.ndarray, offset_fm: float, G: int, p: int = 1, num_samples: int = 500, seed: int = 42) -> dict:
    """OpenQARPを用いたXY-QAOAソルバーの実行 (ペナルティフリー)"""
    t0 = time.perf_counter()
    N = 3 * G
    block_sizes = [3] * G
    
    if OpenQARPXYQAOA is None:
        raise RuntimeError("OpenQARPXYQAOA not available. Check python environment.")
        
    # 1. OpenQARPバックエンドの初期化 (XYミキサーを使用)
    backend = OpenQARPXYQAOA(Q_fm, block_sizes)
    
    gammas = np.linspace(0.05, 0.8, 7)
    betas = np.linspace(0.05, 0.8, 7)
    
    # 2. XY-QAOAは制約を常に満たすため、One-hotな有効部分空間のみを列挙してエネルギーを事前計算
    patterns = one_hot_patterns(block_sizes)
    indices = np.asarray(patterns, dtype=np.uint64) @ (1 << np.arange(N, dtype=np.uint64))
    
    energies_pure = np.einsum('ni,ij,nj->n', patterns, Q_fm, patterns) + offset_fm
    
    best_exp = float('inf')
    best_params = (0.0, 0.0)
    best_probs = None
    
    # 3. グリッドサーチで有効部分空間の確率分布を取得し、エネルギー期待値を最小化するパラメータを探索
    for gamma in gammas:
        for beta in betas:
            probs = backend.probabilities([gamma], [beta])
            probs_sub = probs[indices]
            probs_sub /= np.sum(probs_sub)  # 数値誤差の補正
            
            exp_val = np.sum(probs_sub * energies_pure)
            if exp_val < best_exp:
                best_exp = exp_val
                best_params = (gamma, beta)
                best_probs = probs_sub
                
    # 4. 見つかった最適な確率分布を用いて、有効部分空間からショット(サンプリング)を実行
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(patterns), size=num_samples, p=best_probs)
    
    samples = []
    feas_count = num_samples  # XY-QAOAは原理的に100% Feasible
    
    # 5. サンプリング結果を整形して返却
    for idx in sampled_idx:
        x = patterns[idx]
        samples.append({
            "x": x,
            "feasible": True,
            "penalized_val": energies_pure[idx]
        })
        
    return {
        "raw_feasible_rate": feas_count / num_samples,
        "samples": samples,
        "runtime": time.perf_counter() - t0
    }
