"""
qaoa_tutorial/06_hands_on_qiskit_implementation.py

【QAOA学習教材 ハンズオン演習コード】
Standard QAOA vs XY-QAOA（W状態初期化 + XYミキサー）の最小トイ問題実装

対象問題:
- 2つのカテゴリ変数 C1, C2（各2選択肢、総ビット数 N=4）
- One-Hot制約: x0 + x1 = 1,  x2 + x3 = 1
- 有効解: |1010>, |1001>, |0110>, |0101> の4通りのみ (全16状態中 25%)
- 目的関数: f(x) = -2*x0 - x1 - x2 - 3*x3 + 4*x0*x2 - 2*x1*x3
  大域的最適解: x* = (0, 1, 0, 1)  (f = -6.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from scipy.optimize import minimize

# -------------------------------------------------------------
# 1. 目的関数の定義と古典全探索
# -------------------------------------------------------------
def cost_function(x):
    """x: 長さ4の二値ベクトル [x0, x1, x2, x3]"""
    return -2.0*x[0] - 1.0*x[1] - 1.0*x[2] - 3.0*x[3] + 4.0*x[0]*x[2] - 2.0*x[1]*x[3]

def is_feasible(x):
    """One-Hot制約を満たすかどうか判定"""
    return (x[0] + x[1] == 1) and (x[2] + x[3] == 1)

print("=== 1. 古典全探索による全状態 (16通り) の評価 ===")
feasible_states = []
exact_min_val = float("inf")
exact_best_x = None

for idx in range(16):
    bitstring = f"{idx:04b}" # 例: "0101"
    x = np.array([int(b) for b in bitstring])
    feas = is_feasible(x)
    val = cost_function(x)
    if feas:
        feasible_states.append((bitstring, val))
        if val < exact_min_val:
            exact_min_val = val
            exact_best_x = bitstring
    print(f"|{bitstring}>: Cost = {val:+5.1f} | Feasible = {feas}")

print("\n--- 実行可能解 (4通り) ---")
for bs, val in feasible_states:
    marker = "  <-- 大域的最適解 (Global Optima)" if bs == exact_best_x else ""
    print(f"  |{bs}> : f(x) = {val:+5.1f}{marker}")


from qiskit.circuit.library import DiagonalGate

# -------------------------------------------------------------
# 2. 手法A: Standard QAOA 回路構築 (ペナルティ法)
# -------------------------------------------------------------
def build_standard_qaoa_circuit(gamma, beta, lambda_penalty=5.0):
    r"""
    Standard QAOA:
    - 初期状態: |+>^{\otimes 4}
    - コスト層: H_total = H_cost + \lambda * H_penalty
    - ミキサー層: \sum X_i
    """
    qc = QuantumCircuit(4)
    # 初期化: アダマールゲート
    for i in range(4):
        qc.h(i)
    
    # 目的関数の位相回転
    diag_phases = []
    for idx in range(16):
        bitstring = f"{idx:04b}"
        x = np.array([int(b) for b in bitstring])
        penalty = ((x[0] + x[1] - 1)**2 + (x[2] + x[3] - 1)**2)
        total_energy = cost_function(x) + lambda_penalty * penalty
        diag_phases.append(np.exp(-1j * gamma * total_energy))
    qc.append(DiagonalGate(diag_phases), [0, 1, 2, 3])
    
    # 標準Xミキサー: exp(-i * beta * X_i) = Rx(2*beta)
    for i in range(4):
        qc.rx(2 * beta, i)
        
    qc.measure_all()
    return qc


# -------------------------------------------------------------
# 3. 手法B: 提案手法 FM-XY-QAOA 回路構築 (ペナルティフリー)
# -------------------------------------------------------------
def build_xy_qaoa_circuit(gamma, beta):
    r"""
    FM-XY-QAOA:
    - 初期状態: |W_2> \otimes |W_2> = (|10>+|01>)/sqrt(2) \otimes (|10>+|01>)/sqrt(2)
    - コスト層: H_obj のみ (ペナルティ項ゼロ \lambda=0)
    - ミキサー層: XYミキサー 1/2(X0 X1 + Y0 Y1) + 1/2(X2 X3 + Y2 Y3)
    """
    qc = QuantumCircuit(4)
    
    # 初期化: 各ブロックで W_2 状態を生成
    # ブロック1 (qubit 0, 1):
    qc.x(0)
    qc.h(1)
    qc.cx(1, 0)
    # ブロック2 (qubit 2, 3):
    qc.x(2)
    qc.h(3)
    qc.cx(3, 2)
    
    # コスト層: ペナルティ項なし (\lambda=0) の目的関数のみ
    diag_phases = []
    for idx in range(16):
        bitstring = f"{idx:04b}"
        x = np.array([int(b) for b in bitstring])
        val = cost_function(x)
        diag_phases.append(np.exp(-1j * gamma * val))
    qc.append(DiagonalGate(diag_phases), [0, 1, 2, 3])
    
    # XYミキサー層: exp(-i * beta * 1/2(XX + YY))
    # Qiskit では XX+YY 相互作用は RXX + RYY で等価に実現可能
    # ブロック1 (0, 1)
    qc.rxx(beta, 0, 1)
    qc.ryy(beta, 0, 1)
    # ブロック2 (2, 3)
    qc.rxx(beta, 2, 3)
    qc.ryy(beta, 2, 3)
    
    qc.measure_all()
    return qc


# -------------------------------------------------------------
# 4. サンプリングシミュレーションと性能比較
# -------------------------------------------------------------
sim = AerSimulator()
shots = 5000

# 適切なパラメータでサンプリング比較
gamma_test = 0.5
beta_test = 0.4

print("\n=== 2. 量子回路シミュレーション実行 ===")

from qiskit import transpile

# Standard QAOA 実行
qc_std = build_standard_qaoa_circuit(gamma_test, beta_test, lambda_penalty=5.0)
qc_std_t = transpile(qc_std, sim)
counts_std = sim.run(qc_std_t, shots=shots).result().get_counts()

# XY-QAOA 実行
qc_xy = build_xy_qaoa_circuit(gamma_test, beta_test)
qc_xy_t = transpile(qc_xy, sim)
counts_xy = sim.run(qc_xy_t, shots=shots).result().get_counts()

def analyze_counts(counts, name):
    feasible_count = 0
    optimal_count = 0
    for bs, cnt in counts.items():
        # Qiskitのビット順序はリトルエンディアン (q3 q2 q1 q0)
        # 上で定義した順序 (q0 q1 q2 q3) に直す
        natural_bs = bs[::-1]
        x = np.array([int(b) for b in natural_bs])
        if is_feasible(x):
            feasible_count += cnt
            if natural_bs == exact_best_x:
                optimal_count += cnt
                
    feas_rate = (feasible_count / shots) * 100.0
    opt_rate = (optimal_count / shots) * 100.0
    print(f"\n[{name}]")
    print(f"  総測定ショット数: {shots}")
    print(f"  制約充足率 (Feasible Rate) : {feas_rate:6.2f}%")
    print(f"  大域的最適解到達率 P(x*)    : {opt_rate:6.2f}%")
    return feas_rate, opt_rate

analyze_counts(counts_std, "手法1: Standard QAOA (ペナルティ法 lambda=5.0)")
analyze_counts(counts_xy,  "手法2: 提案手法 FM-XY-QAOA (ペナルティフリー lambda=0)")

print("\n=======================================================")
print("【結論】")
print("Standard QAOA はペナルティ項を設けても制約違反解を多数測定するのに対し、")
print("FM-XY-QAOA は理論通り 100.0% の制約充足率を達成し、無効解を完全排除します。")
print("=======================================================")
