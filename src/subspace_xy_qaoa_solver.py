import itertools
import time
from typing import List, Tuple, Dict, Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize

def build_subspace_basis(block_sizes: List[int]) -> np.ndarray:
    """各ブロックのOne-Hot状態のみを並べた部分空間基底 (|X| x N) を生成"""
    block_eyes = [np.eye(sz, dtype=np.int8) for sz in block_sizes]
    basis = []
    for combo in itertools.product(*block_eyes):
        basis.append(np.concatenate(combo))
    return np.array(basis, dtype=np.int8)

def build_subspace_hamiltonians(qubo_matrix: np.ndarray, block_sizes: List[int]):
    """
    有効部分空間内でのコストハミルトニアン H_C（対角ベクトル）と
    リング型XYミキサー H_M（スパースCSR行列）を構築する。
    """
    dim_subspace = int(np.prod(block_sizes))
    basis = build_subspace_basis(block_sizes) # (|X|, N)
    
    # 1. Cost Hamiltonian H_C (対角成分のみ: 各状態のエネルギー x^T Q x)
    # y = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j
    energies = np.einsum('ni,ij,nj->n', basis.astype(np.float64), qubo_matrix, basis.astype(np.float64))
    
    # 2. Ring XY Mixer
    # 各ブロック内で隣接スピンを交換 (01 <-> 10)
    # 状態のインデックスを引くためのマップ
    # ブロックごとの選択肢インデックス
    block_indices = []
    idx = 0
    for sz in block_sizes:
        block_indices.append(list(range(idx, idx + sz)))
        idx += sz
        
    # 各基底行をブロックごとの選択肢タプル (c_1, c_2, ..., c_M) に変換
    tuple_to_idx = {}
    basis_tuples = []
    for i, row in enumerate(basis):
        tup = []
        for sz, b_inds in zip(block_sizes, block_indices):
            # 1が立っている位置
            one_pos = np.where(row[b_inds] == 1)[0][0]
            tup.append(one_pos)
        tup = tuple(tup)
        tuple_to_idx[tup] = i
        basis_tuples.append(tup)
        
    rows = []
    cols = []
    data = []
    
    # 各基底から、各ブロックでリング隣接交換して得られる状態への遷移
    M = len(block_sizes)
    for i, tup in enumerate(basis_tuples):
        tup_list = list(tup)
        for m in range(M):
            sz = block_sizes[m]
            curr = tup_list[m]
            # リング状の隣接 (curr + 1) % sz と (curr - 1) % sz
            for nxt in [(curr + 1) % sz, (curr - 1 + sz) % sz]:
                if curr != nxt:
                    new_tup = list(tup_list)
                    new_tup[m] = nxt
                    j = tuple_to_idx[tuple(new_tup)]
                    # 遷移要素 0.5 (X_i X_j + Y_i Y_j の係数)
                    rows.append(i)
                    cols.append(j)
                    data.append(0.5)
                    
    H_M = sp.csr_matrix((data, (rows, cols)), shape=(dim_subspace, dim_subspace), dtype=np.float64)
    return energies, H_M, basis

def solve_subspace_xy_qaoa(qubo_matrix: np.ndarray, block_sizes: List[int], reps: int = 1, maxiter: int = 30) -> Dict[str, Any]:
    """
    全空間 2^N を持たず、有効部分空間 |X| = prod(d_m) のみでQAOAを実行する超高速ソルバー
    """
    energies, H_M, basis = build_subspace_hamiltonians(qubo_matrix, block_sizes)
    dim_sub = len(energies)
    
    # 初期状態: W状態直積 = 部分空間内の一様重ね合わせ (1/sqrt(|X|))
    psi_0 = np.full(dim_sub, 1.0 / np.sqrt(dim_sub), dtype=np.complex128)
    
    # トロッター展開 / スパース行列指数関数による状態更新
    # 簡易・高速化のため 1次のトロッター化: exp(-i * beta * H_M) @ exp(-i * gamma * H_C) @ psi
    # H_C は対角なので位相回転: psi * exp(-1j * gamma * energies)
    # H_M はスパースなので scipy.sparse.linalg.expm_multiply
    
    def get_state(params):
        gammas = params[:reps]
        betas = params[reps:]
        psi = psi_0.copy()
        for g, b in zip(gammas, betas):
            # Phase separation
            psi = psi * np.exp(-1j * g * energies)
            # Mixer evolution
            psi = expm_multiply(-1j * b * H_M, psi)
        return psi
        
    def objective(params):
        psi = get_state(params)
        probs = np.abs(psi)**2
        return np.sum(probs * energies)
        
    # 初期パラメータ
    np.random.seed(42)
    init_params = np.random.uniform(0, np.pi, size=2 * reps)
    
    res = minimize(objective, init_params, method='COBYLA', options={'maxiter': maxiter})
    
    opt_psi = get_state(res.x)
    probs = np.abs(opt_psi)**2
    
    # 最良解の抽出
    best_idx = np.argmin(energies)
    exact_min = energies[best_idx]
    
    # サンプリング期待値・最良サンプル
    sample_best_idx = np.argmax(probs) # 最も確率が高い状態
    
    # 確率上位のサンプリング探索（有効解から必ず得られる）
    sampled_indices = np.random.choice(dim_sub, size=500, p=probs / np.sum(probs))
    best_sampled_idx = sampled_indices[np.argmin(energies[sampled_indices])]
    
    return {
        "opt_energy": res.fun,
        "best_sampled_energy": energies[best_sampled_idx],
        "exact_min_energy": exact_min,
        "success_probability": probs[best_idx],
        "feasibility_rate": 1.0, # 部分空間内なので数学的に厳密に 100%
        "dim_subspace": dim_sub,
        "params": res.x
    }
