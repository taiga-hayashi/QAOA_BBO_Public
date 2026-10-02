import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))
try:
    from qarp_backend import OpenQARPStandardQAOA, all_bitstrings_lsb
except ImportError:
    OpenQARPStandardQAOA = None

def solve_penalty_qaoa(Q_fm: np.ndarray, offset_fm: float, G: int, lambda_val: float, num_samples: int = 500, seed: int = 42) -> dict:
    """OpenQARPを用いたStandard Penalty-QAOAソルバーの実行"""
    t0 = time.perf_counter()
    N = 3 * G
    block_sizes = [3] * G
    
    if OpenQARPStandardQAOA is None:
        raise RuntimeError("OpenQARPStandardQAOA not available. Check python environment.")
        
    # 1. OpenQARPバックエンドの初期化 (ここでQUBOとペナルティが結合される)
    backend = OpenQARPStandardQAOA(Q_fm, block_sizes, lambda_val)
    
    # グリッドサーチ用のパラメータ空間
    gammas = np.linspace(0.05, 0.8, 7)
    betas = np.linspace(0.05, 0.8, 7)
    
    # 2. 全ビット列とそのペナルティ込みのエネルギーを事前計算
    bits = all_bitstrings_lsb(N)
    from qarp_backend import one_hot_penalty
    penalties = one_hot_penalty(bits, block_sizes)
    
    energies_pure = np.einsum('ni,ij,nj->n', bits, Q_fm, bits) + offset_fm
    energies_penalized = energies_pure + lambda_val * penalties
    
    best_exp = float('inf')
    best_params = (0.0, 0.0)
    best_probs = None
    
    # 3. 全ての(gamma, beta)について確率分布を取得し、エネルギー期待値を最小化するパラメータを探索
    for gamma in gammas:
        for beta in betas:
            probs = backend.probabilities([gamma], [beta])
            exp_val = np.sum(probs * energies_penalized)
            if exp_val < best_exp:
                best_exp = exp_val
                best_params = (gamma, beta)
                best_probs = probs
                
    # 4. 見つかった最適な確率分布を用いてショット(サンプリング)を実行
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(bits), size=num_samples, p=best_probs)
    
    samples = []
    feas_count = 0
    
    # 5. サンプリング結果を検証し、制約を満たしているか(feasible)を判定して返却
    for idx in sampled_indices:
        x = bits[idx]
        feasible = penalties[idx] == 0.0
        if feasible:
            feas_count += 1
            
        samples.append({
            "x": x,
            "feasible": feasible,
            "penalized_val": energies_penalized[idx]
        })
        
    return {
        "raw_feasible_rate": feas_count / num_samples,
        "samples": samples,
        "runtime": time.perf_counter() - t0
    }
