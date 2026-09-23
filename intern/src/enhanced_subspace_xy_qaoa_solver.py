import itertools
import time
from typing import List, Tuple, Dict, Any, Optional
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

def build_subspace_hamiltonians(qubo_matrix: np.ndarray, block_sizes: List[int], mixer_type: str = "ring"):
    """
    有効部分空間内でのコストハミルトニアン H_C（対角ベクトル）と
    XYミキサー H_M（スパースCSR行列: 'ring' または 'all_to_all'）を構築する。
    """
    dim_subspace = int(np.prod(block_sizes))
    basis = build_subspace_basis(block_sizes) # (|X|, N)
    
    # 1. Cost Hamiltonian H_C (対角成分のみ: 各状態のエネルギー x^T Q x)
    energies = np.einsum('ni,ij,nj->n', basis.astype(np.float64), qubo_matrix, basis.astype(np.float64))
    
    # 2. XY Mixer
    block_indices = []
    idx = 0
    for sz in block_sizes:
        block_indices.append(list(range(idx, idx + sz)))
        idx += sz
        
    tuple_to_idx = {}
    basis_tuples = []
    for i, row in enumerate(basis):
        tup = []
        for sz, b_inds in zip(block_sizes, block_indices):
            one_pos = np.where(row[b_inds] == 1)[0][0]
            tup.append(one_pos)
        tup = tuple(tup)
        tuple_to_idx[tup] = i
        basis_tuples.append(tup)
        
    rows = []
    cols = []
    data = []
    
    M = len(block_sizes)
    for i, tup in enumerate(basis_tuples):
        tup_list = list(tup)
        for m in range(M):
            sz = block_sizes[m]
            curr = tup_list[m]
            
            if mixer_type == "ring":
                # リング隣接交換
                neighbors = [(curr + 1) % sz, (curr - 1 + sz) % sz]
                for nxt in neighbors:
                    if curr != nxt:
                        new_tup = list(tup_list)
                        new_tup[m] = nxt
                        j = tuple_to_idx[tuple(new_tup)]
                        rows.append(i)
                        cols.append(j)
                        data.append(0.5)
            elif mixer_type == "all_to_all":
                # ブロック内完全結合交換 (全ての異なるペア)
                for nxt in range(sz):
                    if curr != nxt:
                        new_tup = list(tup_list)
                        new_tup[m] = nxt
                        j = tuple_to_idx[tuple(new_tup)]
                        rows.append(i)
                        cols.append(j)
                        data.append(0.5)
                        
    H_M = sp.csr_matrix((data, (rows, cols)), shape=(dim_subspace, dim_subspace), dtype=np.float64)
    return energies, H_M, basis

def run_one_hot_local_search(best_x: np.ndarray, qubo_matrix: np.ndarray, block_sizes: List[int]) -> Tuple[np.ndarray, float]:
    """
    サンプリング解を初期値とし、One-Hot制約を厳密に保ったまま
    1サイトずつ最適元素に入れ替える高速1-opt局所探索（古典後処理）
    """
    curr_x = best_x.copy().flatten()
    N = len(curr_x)
    M = len(block_sizes)
    
    block_indices = []
    idx = 0
    for sz in block_sizes:
        block_indices.append(list(range(idx, idx + sz)))
        idx += sz
        
    def eval_x(x):
        return float(x @ qubo_matrix @ x)
        
    curr_val = eval_x(curr_x)
    
    improved = True
    max_steps = 100
    step = 0
    while improved and step < max_steps:
        step += 1
        improved = False
        best_cand_x = curr_x.copy()
        best_cand_val = curr_val
        
        for m in range(M):
            b_inds = block_indices[m]
            curr_pos = np.where(curr_x[b_inds] == 1)[0][0]
            
            for alt_pos in range(len(b_inds)):
                if alt_pos == curr_pos:
                    continue
                cand_x = curr_x.copy()
                cand_x[b_inds[curr_pos]] = 0
                cand_x[b_inds[alt_pos]] = 1
                cand_val = eval_x(cand_x)
                
                if cand_val < best_cand_val - 1e-7:
                    best_cand_val = cand_val
                    best_cand_x = cand_x
                    improved = True
                    
        if improved:
            curr_x = best_cand_x
            curr_val = best_cand_val
            
    return curr_x, curr_val

def solve_enhanced_subspace_xy_qaoa(
    qubo_matrix: np.ndarray,
    block_sizes: List[int],
    reps: int = 1,
    mixer_type: str = "ring",
    objective_type: str = "expectation",
    cvar_alpha: float = 0.25,
    init_strategy: str = "tqa",
    use_local_search: bool = False,
    maxiter: int = 30,
    seed: int = 42
) -> Dict[str, Any]:
    """
    精度向上技術を統合した高機能部分空間QAOAソルバー:
    - reps: p (1, 2, 3, ...)
    - mixer_type: 'ring' or 'all_to_all'
    - objective_type: 'expectation' or 'cvar'
    - init_strategy: 'tqa' (Trotterized Quantum Annealing) or 'random'
    - use_local_search: QAOAサンプリング解を古典1-optで後処理
    """
    energies, H_M, basis = build_subspace_hamiltonians(qubo_matrix, block_sizes, mixer_type=mixer_type)
    dim_sub = len(energies)
    
    psi_0 = np.full(dim_sub, 1.0 / np.sqrt(dim_sub), dtype=np.complex128)
    
    def get_state(params):
        gammas = params[:reps]
        betas = params[reps:]
        psi = psi_0.copy()
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * energies)
            psi = expm_multiply(-1j * b * H_M, psi)
        return psi
        
    def objective(params):
        psi = get_state(params)
        probs = np.abs(psi)**2
        probs = probs / np.sum(probs)
        
        if objective_type == "expectation":
            return np.sum(probs * energies)
        elif objective_type == "cvar":
            # CVaR: エネルギー昇順にソートし、累積確率が alpha に達するまでの条件付き期待値
            sort_idx = np.argsort(energies)
            sorted_energies = energies[sort_idx]
            sorted_probs = probs[sort_idx]
            
            cum_probs = np.cumsum(sorted_probs)
            cutoff_mask = cum_probs <= cvar_alpha
            if not np.any(cutoff_mask):
                cutoff_mask[0] = True
                
            selected_energies = sorted_energies[cutoff_mask]
            selected_probs = sorted_probs[cutoff_mask]
            denom = np.sum(selected_probs)
            if denom < 1e-12:
                return float(sorted_energies[0])
            return float(np.sum(selected_energies * selected_probs) / denom)
        else:
            raise ValueError(f"Unknown objective_type: {objective_type}")
            
    # 初期パラメータ設計
    if init_strategy == "tqa":
        # Trotterized Quantum Annealing 線形スケジュール
        # gamma: 0 -> dt, beta: dt -> 0
        total_time = 1.0
        dt = total_time / reps
        gammas_0 = [dt * (k + 0.5) / reps for k in range(reps)]
        betas_0 = [dt * (1.0 - (k + 0.5) / reps) for k in range(reps)]
        init_params = np.array(gammas_0 + betas_0, dtype=np.float64)
    else:
        np.random.seed(seed)
        init_params = np.random.uniform(0, np.pi, size=2 * reps)
        
    res = minimize(objective, init_params, method='COBYLA', options={'maxiter': maxiter})
    
    opt_psi = get_state(res.x)
    probs = np.abs(opt_psi)**2
    probs = probs / np.sum(probs)
    
    best_idx = np.argmin(energies)
    exact_min = energies[best_idx]
    
    np.random.seed(seed)
    sampled_indices = np.random.choice(dim_sub, size=500, p=probs)
    best_sampled_idx = sampled_indices[np.argmin(energies[sampled_indices])]
    qaoa_best_energy = energies[best_sampled_idx]
    qaoa_best_x = basis[best_sampled_idx]
    
    final_best_energy = qaoa_best_energy
    final_best_x = qaoa_best_x
    
    if use_local_search:
        ls_x, ls_energy = run_one_hot_local_search(qaoa_best_x, qubo_matrix, block_sizes)
        final_best_energy = ls_energy
        final_best_x = ls_x
        
    return {
        "opt_objective": res.fun,
        "qaoa_best_energy": qaoa_best_energy,
        "final_best_energy": final_best_energy,
        "final_best_x": final_best_x,
        "exact_min_energy": exact_min,
        "success_probability": probs[best_idx],
        "feasibility_rate": 1.0,
        "dim_subspace": dim_sub,
        "params": res.x
    }
