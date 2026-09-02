from typing import Dict, Tuple
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

def solve_qubo_qaoa(qubo_dict: Dict[Tuple[int, int], float], offset: float = 0.0, reps: int = 1, maxiter: int = 100):
    """
    QUBO辞書を受け取り、QAOAを用いて最適化（最小化）する。
    
    Args:
        qubo_dict: {(i, j): weight} の形式のQUBO辞書
        offset: バイアス項
        reps: QAOAのレイヤー数 (p)
        maxiter: 古典オプティマイザの最大イテレーション数
        
    Returns:
        result: QiskitのOptimizationResultオブジェクト
    """
    # 変数の数を特定する
    if not qubo_dict:
        raise ValueError("QUBO dictionary is empty.")
        
    max_idx = max(max(i, j) for i, j in qubo_dict.keys())
    num_vars = max_idx + 1

    # QuadraticProgram の構築
    qp = QuadraticProgram()
    
    # バイナリ変数の追加
    for i in range(num_vars):
        qp.binary_var(name=f"x_{i}")
    
    linear = {}
    quadratic = {}
    
    for (i, j), weight in qubo_dict.items():
        if i == j:
            linear[f"x_{i}"] = weight
        else:
            quadratic[(f"x_{i}", f"x_{j}")] = weight

    # 最小化問題としてセット
    qp.minimize(constant=offset, linear=linear, quadratic=quadratic)
    
    # QAOAとソルバーのセットアップ
    optimizer = COBYLA(maxiter=maxiter)
    sampler = AerSampler()  # 高速なシミュレータ (qiskit-aer)
    pm = generate_preset_pass_manager(optimization_level=1, target=AerSimulator().target)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, transpiler=pm)
    
    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    
    # 最適化の実行
    print(f"Solving QUBO with {num_vars} variables using QAOA (reps={reps})...")
    result = min_eigen_optimizer.solve(qp)
    
    return result
