from typing import Dict, Tuple, List, Any
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from bb_function import is_feasible_one_hot

def create_block_w_state(block_sizes: List[int]) -> QuantumCircuit:
    """各ブロックで1つのビットが1であるW状態の直積回路を構築する"""
    total_qubits = sum(block_sizes)
    qc = QuantumCircuit(total_qubits)
    idx = 0
    for sz in block_sizes:
        qubits = list(range(idx, idx + sz))
        vec = np.zeros(2**sz)
        for k in range(sz):
            vec[1 << k] = 1.0 / np.sqrt(sz)
        qc.prepare_state(vec, qubits)
        idx += sz
    return qc

def create_block_xy_mixer(block_sizes: List[int]) -> SparsePauliOp:
    """各ブロックの内部でのみ励起（1）を交換するブロック直和型XYミキサーを構築する"""
    total_qubits = sum(block_sizes)
    pauli_list = []
    idx = 0
    for sz in block_sizes:
        for j in range(sz):
            for k in range(j + 1, sz):
                q1 = idx + j
                q2 = idx + k
                # Qiskitのエンディアン: インデックス0が右端
                # XX term
                chars_x = ['I'] * total_qubits
                chars_x[total_qubits - 1 - q1] = 'X'
                chars_x[total_qubits - 1 - q2] = 'X'
                pauli_list.append((''.join(chars_x), 0.5))
                # YY term
                chars_y = ['I'] * total_qubits
                chars_y[total_qubits - 1 - q1] = 'Y'
                chars_y[total_qubits - 1 - q2] = 'Y'
                pauli_list.append((''.join(chars_y), 0.5))
        idx += sz
    return SparsePauliOp.from_list(pauli_list)

def _build_base_qp(qubo_dict: Dict[Tuple[int, int], float], num_vars: int, offset: float = 0.0) -> QuadraticProgram:
    qp = QuadraticProgram()
    for i in range(num_vars):
        qp.binary_var(name=f"x_{i}")
    linear = {}
    quadratic = {}
    for (i, j), weight in qubo_dict.items():
        if i == j:
            linear[f"x_{i}"] = weight
        else:
            quadratic[(f"x_{i}", f"x_{j}")] = weight
    qp.minimize(constant=offset, linear=linear, quadratic=quadratic)
    return qp

def _extract_qaoa_summary(
    result: Any,
    block_sizes: List[int],
    qubo_dict: Dict[Tuple[int, int], float],
    offset: float,
    solver_name: str,
    lambda_penalty: float = 0.0
) -> Dict[str, Any]:
    state_probs = {}
    feasible_prob_sum = 0.0
    best_feasible = None
    min_feasible_val = float("inf")
    
    total_vars = sum(block_sizes)
    
    # Qiskit QAOAの結果から固有状態分布を取得
    eigen_res = getattr(result, "min_eigen_solver_result", None)
    if eigen_res is not None and hasattr(eigen_res, "eigenstate") and isinstance(eigen_res.eigenstate, dict):
        # eigenstate は {'bitstring': probability} の形式 (Qiskit endian: 右端が q0)
        for raw_bstr, prob in eigen_res.eigenstate.items():
            prob = float(prob)
            # 反転して (x0, x1, ...) の順序にする
            inv_bstr = raw_bstr[::-1]
            bit_tuple = tuple(int(c) for c in inv_bstr)
            state_probs[inv_bstr] = prob
            
            is_feas = is_feasible_one_hot(list(bit_tuple), block_sizes)
            if is_feas:
                feasible_prob_sum += prob
                pure_fval = offset
                for (i, j), w in qubo_dict.items():
                    if bit_tuple[i] == 1 and bit_tuple[j] == 1:
                        pure_fval += w
                if pure_fval < min_feasible_val:
                    min_feasible_val = pure_fval
                    best_feasible = {
                        "bitstring": inv_bstr,
                        "x": bit_tuple,
                        "fval": pure_fval,
                        "probability": prob
                    }
    else:
        for sample in result.samples:
            bit_tuple = tuple(int(xi) for xi in sample.x)
            bit_str = "".join(str(b) for b in bit_tuple)
            prob = float(sample.probability)
            state_probs[bit_str] = state_probs.get(bit_str, 0.0) + prob
            
            is_feas = is_feasible_one_hot(list(bit_tuple), block_sizes)
            if is_feas:
                feasible_prob_sum += prob
                pure_fval = offset
                for (i, j), w in qubo_dict.items():
                    if bit_tuple[i] == 1 and bit_tuple[j] == 1:
                        pure_fval += w
                if pure_fval < min_feasible_val:
                    min_feasible_val = pure_fval
                    best_feasible = {
                        "bitstring": bit_str,
                        "x": bit_tuple,
                        "fval": pure_fval,
                        "raw_fval": float(sample.fval),
                        "probability": prob
                    }
                
    best_overall_sample = result.samples[0]
    best_overall_tuple = tuple(int(xi) for xi in best_overall_sample.x)
    best_overall = {
        "bitstring": "".join(str(b) for b in best_overall_tuple),
        "x": best_overall_tuple,
        "raw_fval": float(best_overall_sample.fval),
        "feasible": is_feasible_one_hot(list(best_overall_tuple), block_sizes),
        "probability": float(best_overall_sample.probability)
    }
    
    return {
        "solver": solver_name,
        "lambda_penalty": lambda_penalty,
        "state_probabilities": state_probs,
        "feasibility_rate": min(1.0, feasible_prob_sum),
        "best_feasible_sample": best_feasible,
        "best_overall_sample": best_overall,
        "raw_result": result
    }


def solve_xy_qaoa(
    qubo_dict: Dict[Tuple[int, int], float],
    block_sizes: List[int],
    offset: float = 0.0,
    reps: int = 1,
    maxiter: int = 50,
    optimizer_name: str = "COBYLA"
) -> Dict[str, Any]:
    """
    ブロックXYミキサーとW状態直積を用いたペナルティフリーQAOAで制約付きQUBOを解く。
    """
    total_vars = sum(block_sizes)
    qp = _build_base_qp(qubo_dict, total_vars, offset)
    
    init_state = create_block_w_state(block_sizes)
    mixer = create_block_xy_mixer(block_sizes)
    
    opt_name = optimizer_name.upper()
    if opt_name == "SPSA":
        optimizer = SPSA(maxiter=maxiter)
    elif opt_name == "SLSQP":
        optimizer = SLSQP(maxiter=maxiter)
    else:
        optimizer = COBYLA(maxiter=maxiter)
        
    sampler = AerSampler()
    pm = generate_preset_pass_manager(optimization_level=1, target=AerSimulator().target)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
        initial_state=init_state,
        mixer=mixer,
        transpiler=pm
    )
    
    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)
    
    return _extract_qaoa_summary(
        result=result,
        block_sizes=block_sizes,
        qubo_dict=qubo_dict,
        offset=offset,
        solver_name="FM-XY-QAOA (Proposed)",
        lambda_penalty=0.0
    )

def solve_standard_qaoa(
    qubo_dict: Dict[Tuple[int, int], float],
    block_sizes: List[int],
    lambda_penalty: float = 5.0,
    offset: float = 0.0,
    reps: int = 1,
    maxiter: int = 50,
    optimizer_name: str = "COBYLA"
) -> Dict[str, Any]:
    """
    ペナルティ項付きStandard QAOA（標準Xミキサー）で解く。
    """
    total_vars = sum(block_sizes)
    qp = QuadraticProgram()
    for i in range(total_vars):
        qp.binary_var(name=f"x_{i}")
        
    # ペナルティ項付きの係数を集計
    # H = sum w_ij x_i x_j + lambda * sum_m (sum_{j in C_m} x_j - 1)^2
    # (sum x_j - 1)^2 = 1 - sum x_j + 2 sum_{j < k} x_j x_k
    total_offset = offset
    linear = {f"x_{i}": 0.0 for i in range(total_vars)}
    quadratic = {}
    
    # 目的関数
    for (i, j), weight in qubo_dict.items():
        if i == j:
            linear[f"x_{i}"] += weight
        else:
            p = (f"x_{min(i, j)}", f"x_{max(i, j)}")
            quadratic[p] = quadratic.get(p, 0.0) + weight
            
    # ペナルティ項の追加
    idx = 0
    for sz in block_sizes:
        total_offset += lambda_penalty * 1.0
        for j in range(sz):
            linear[f"x_{idx + j}"] -= lambda_penalty * 1.0
        for j in range(sz):
            for k in range(j + 1, sz):
                p = (f"x_{idx + j}", f"x_{idx + k}")
                quadratic[p] = quadratic.get(p, 0.0) + 2.0 * lambda_penalty
        idx += sz
        
    qp.minimize(constant=total_offset, linear=linear, quadratic=quadratic)
    
    opt_name = optimizer_name.upper()
    if opt_name == "SPSA":
        optimizer = SPSA(maxiter=maxiter)
    elif opt_name == "SLSQP":
        optimizer = SLSQP(maxiter=maxiter)
    else:
        optimizer = COBYLA(maxiter=maxiter)
        
    sampler = AerSampler()
    pm = generate_preset_pass_manager(optimization_level=1, target=AerSimulator().target)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, transpiler=pm)
    
    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)
    
    return _extract_qaoa_summary(
        result=result,
        block_sizes=block_sizes,
        qubo_dict=qubo_dict,
        offset=offset,
        solver_name="Standard FM-QAOA",
        lambda_penalty=lambda_penalty
    )

def solve_qubo_qaoa(qubo_dict: Dict[Tuple[int, int], float], offset: float = 0.0, reps: int = 1, maxiter: int = 100, optimizer_name: str = "COBYLA"):
    """後方互換用: 非制約QUBOを標準QAOAで解く"""
    if not qubo_dict:
        raise ValueError("QUBO dictionary is empty.")
    max_idx = max(max(i, j) for i, j in qubo_dict.keys())
    num_vars = max_idx + 1
    qp = _build_base_qp(qubo_dict, num_vars, offset)
    
    opt_name = optimizer_name.upper()
    if opt_name == "SPSA":
        optimizer = SPSA(maxiter=maxiter)
    elif opt_name == "SLSQP":
        optimizer = SLSQP(maxiter=maxiter)
    else:
        optimizer = COBYLA(maxiter=maxiter)
        
    sampler = AerSampler()
    pm = generate_preset_pass_manager(optimization_level=1, target=AerSimulator().target)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, transpiler=pm)
    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    return min_eigen_optimizer.solve(qp)

