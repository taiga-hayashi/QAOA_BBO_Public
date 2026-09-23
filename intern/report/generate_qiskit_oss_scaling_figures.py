import matplotlib.pyplot as plt
import numpy as np

# 統一フォント・スタイル設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Qiskit 公式実測データ (N = 4, 8, 12, 16, 20, 24)
scales = ['N=4\n(4x1)', 'N=8\n(4x2)', 'N=12\n(4x3)', 'N=16\n(4x4)', 'N=20\n(4x5)', 'N=24\n(4x6)']
N_qubits = [4, 8, 12, 16, 20, 24]

# 制約充足率 (%)
xy_feas = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
std_feas = [11.3, 4.4, 1.9, 0.3, 0.1, 0.0]
fmqa_feas = [99.5, 100.0, 100.0, 100.0, 100.0, 99.9]
valid_ratio = [25.0, 6.25, 1.56, 0.39, 0.098, 0.024]

# Qiskit 実行時間 (秒)
xy_time = [0.04, 0.07, 0.13, 0.36, 2.93, 34.29]
std_time = [0.09, 0.24, 0.79, 1.14, 4.62, 58.1]
fmqa_time = [0.05, 0.10, 0.18, 0.24, 0.27, 0.32]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2))

# Subplot 1: 制約充足率推移
ax1.plot(N_qubits, xy_feas, marker='^', color='#2ca02c', linewidth=2.8, markersize=9, label='FM-XY-QAOA (Qiskit OSS)')
ax1.plot(N_qubits, fmqa_feas, marker='o', color='#1f77b4', linewidth=2.2, markersize=7, label='FMQA (SA, neal)')
ax1.plot(N_qubits, std_feas, marker='s', color='#d62728', linewidth=2.2, markersize=7, label='Standard QAOA (Qiskit OSS)')
ax1.plot(N_qubits, valid_ratio, linestyle='--', color='#9467bd', linewidth=1.8, label=r'Random Guess ($|\mathcal{X}|/2^N$)')

ax1.set_xlabel('Number of Qubits $N$', fontsize=11.5, fontweight='bold')
ax1.set_ylabel('Feasibility Rate (%)', fontsize=11.5, fontweight='bold')
ax1.set_title('(a) Constraint Satisfaction vs Qubit Scale', fontsize=12.5, fontweight='bold', pad=10)
ax1.set_ylim(-3, 106)
ax1.set_xticks(N_qubits)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='center right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

ax1.annotate('100% Feasible\nGuaranteed', xy=(20, 100), xytext=(17, 82),
             arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#2ca02c', fontsize=10, ha='center')

ax1.annotate('Standard QAOA\nCollapses to ~0%', xy=(20, 0.1), xytext=(17.5, 20),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#d62728', fontsize=10, ha='center')

# Subplot 2: Qiskit シミュレーション実行時間 (対数スケール)
ax2.plot(N_qubits, xy_time, marker='^', color='#2ca02c', linewidth=2.8, markersize=9, label='FM-XY-QAOA (Qiskit)')
ax2.plot(N_qubits, std_time, marker='s', color='#d62728', linewidth=2.2, markersize=7, label='Standard QAOA (Qiskit)')
ax2.plot(N_qubits, fmqa_time, marker='o', color='#1f77b4', linewidth=2.2, markersize=7, label='FMQA (SA, neal)')

ax2.axvline(x=24, color='#7f7f7f', linestyle='--', linewidth=1.5)
ax2.set_yscale('log')
ax2.set_xlabel('Number of Qubits $N$', fontsize=11.5, fontweight='bold')
ax2.set_ylabel('Execution Time per Sample (s) [log]', fontsize=11.5, fontweight='bold')
ax2.set_title('(b) Qiskit (OSS) Execution Time Scaling', fontsize=12.5, fontweight='bold', pad=10)
ax2.set_xticks(N_qubits)
ax2.set_ylim(0.015, 600)
ax2.grid(True, which='both', linestyle=':', alpha=0.6)
ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

ax2.annotate('Simulation Barrier\n(at $N=24$: 34.3s)', xy=(24, 34.29), xytext=(17.8, 120),
             arrowprops=dict(facecolor='#7f7f7f', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#555555', fontsize=9.5, ha='center')

# Subplot 3: メモリ使用量 (RAM) のスケーリング推移と物理限界
N_extended = np.array([4, 8, 12, 16, 20, 24, 28, 32, 40, 60])
# 全状態ベクトルRAM (bytes): 2^N * 16 bytes (complex128)
ram_full_bytes = (2.0 ** N_extended) * 16.0
ram_full_gb = ram_full_bytes / (1024.0 ** 3)

# One-Hot 有効部分空間のRAM (bytes): 4^(N/4) * 16 bytes
ram_subspace_bytes = (4.0 ** (N_extended / 4.0)) * 16.0
ram_subspace_gb = ram_subspace_bytes / (1024.0 ** 3)

ax3.plot(N_extended, ram_full_gb, marker='D', color='#d62728', linewidth=2.5, markersize=7, label=r'Full Statevector $\mathbb{C}^{2^N}$ ($2^N \times 16$B)')
ax3.plot(N_extended, ram_subspace_gb, marker='s', color='#2ca02c', linewidth=2.2, linestyle='--', markersize=6, label=r'One-Hot Subspace $\mathcal{X}$ ($4^{N/4} \times 16$B)')

ax3.set_yscale('log')
ax3.set_xlabel('Number of Qubits $N$', fontsize=11.5, fontweight='bold')
ax3.set_ylabel('Memory Requirement (GB) [log]', fontsize=11.5, fontweight='bold')
ax3.set_title('(c) RAM Scaling & Classical Simulation Limits', fontsize=12.5, fontweight='bold', pad=10)
ax3.set_xticks([4, 12, 20, 24, 28, 32, 40, 60])
ax3.set_xlim(2, 62)
ax3.set_ylim(1e-7, 1e11)
ax3.grid(True, which='both', linestyle=':', alpha=0.6)

# 物理限界の基準線
ax3.axhline(y=0.268, color='#333333', linestyle=':', alpha=0.7)
ax3.text(4, 0.35, 'Local PC Real-time Limit (N=24: 268MB)', fontsize=8.5, color='#333333', fontweight='bold')

ax3.axhline(y=16.0, color='#e6550d', linestyle=':', alpha=0.7)
ax3.text(4, 20.0, 'Typical PC Limit (16-32GB, N<=28)', fontsize=8.5, color='#e6550d', fontweight='bold')

ax3.axhline(y=128.0, color='#1f77b4', linestyle='--', linewidth=1.6, alpha=0.85)
ax3.text(4, 160.0, 'Dedicated Server Limit (128GB RAM: Solvable up to N=32)', fontsize=9.0, color='#1f77b4', fontweight='bold')

ax3.annotate('Server Limit Boundary\n(N=32: 68.7GB <= 128GB)\n[N>=33: Server OOM]', xy=(32, 68.7), xytext=(28, 4000),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.5, headwidth=5),
             fontweight='bold', color='#1f77b4', fontsize=8.5, ha='center')

ax3.annotate('FMQA Scale (N=60)\n18.4 EB (Impossible!)', xy=(60, 1.84e10), xytext=(52, 2e7),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=5),
             fontweight='bold', color='#d62728', fontsize=8.5, ha='center')

ax3.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.0)

plt.tight_layout()
os.makedirs('report/figures/large_scale_and_enhancement', exist_ok=True)
plt.savefig('report/figures/large_scale_and_enhancement/extreme_scale_limit.pdf', dpi=300)
plt.savefig('report/figures/large_scale_and_enhancement/extreme_scale_limit.png', dpi=300)
print("Successfully generated 3-panel Qiskit OSS scaling and memory figures!")
