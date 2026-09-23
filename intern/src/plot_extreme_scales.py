import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

qubits = np.array([12, 14, 16, 18, 20, 22, 24])
total_states = 2**qubits
valid_states = np.array([81, 144, 256, 576, 1024, 2304, 4096])
valid_ratio = (valid_states / total_states) * 100.0

# Feasibility
fmqa_feas = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
xy_feas = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
std_feas = np.array([1.86, 1.46, 0.49, 0.10, 0.10, np.nan, np.nan])

# Optimal Prob
fmqa_opt = np.array([13.80, 9.20, 8.30, 7.70, 5.40, 9.00, 6.40])
xy_opt = np.array([1.0742, 2.1484, 0.3906, 0.1953, 0.05, 0.02, 0.0977])
std_opt = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, np.nan, np.nan])
uniform_random = (1.0 / valid_states) * 100.0

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# Subplot 1: Feasible Ratio vs Feasibility Rate
ax1 = axes[0]
ax1.plot(qubits, fmqa_feas, 'o-', color='#1f77b4', linewidth=2.0, label=r'FMQA (SA, $\lambda=5.0$)')
ax1.plot(qubits, xy_feas, 's--', color='#2ca02c', linewidth=2.2, label=r'FM-XY-QAOA (Proposed, $\lambda=0$)')
ax1.plot(qubits[:5], std_feas[:5], '^-.', color='#ff7f0e', linewidth=2.0, label=r'Standard QAOA ($\lambda=5.0$)')
ax1.axhline(0.0, color='gray', linestyle=':', alpha=0.6)

# Fill breakdown zone
ax1.axvspan(12, 24, alpha=0.08, color='red', label=r'Std QAOA Failure Regime ($P_{\rm feas} < 2\%$)')
ax1.set_xlabel(r'Number of Qubits $N$')
ax1.set_ylabel(r'Feasibility Rate (%)')
ax1.set_title(r'(a) Feasibility Collapse vs Perfect Preservation')
ax1.set_ylim(-5, 108)
ax1.set_xticks(qubits)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='center right', framealpha=0.9)

# Subplot 2: State Space Sparsity & Optimal Prob
ax2 = axes[1]
ax2.semilogy(qubits, total_states, 'k--', alpha=0.4, label=r'Total Hilbert Space $2^N$')
ax2.semilogy(qubits, valid_states, 'k:', linewidth=1.5, label=r'Valid Manifold $\prod d_m$')
ax2.semilogy(qubits, fmqa_opt, 'o-', color='#1f77b4', linewidth=2.0, label=r'FMQA $P(x^*)$ (%)')
ax2.semilogy(qubits, xy_opt, 's-', color='#2ca02c', linewidth=2.2, label=r'FM-XY $P(x^*)$ (%)')
ax2.semilogy(qubits, uniform_random, 'x--', color='#9467bd', linewidth=1.5, label=r'Uniform Random $1/\prod d_m$ (%)')

# Annotation for N=24
ax2.annotate('4.0x Quantum Boost\nat $N=24$ (16.7M states)',
             xy=(24, 0.0977), xytext=(17.5, 0.005),
             arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9, fontweight='bold', color='#2ca02c',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', edgecolor='#2ca02c', alpha=0.9))

ax2.set_xlabel(r'Number of Qubits $N$')
ax2.set_ylabel(r'Dimension / Probability (Log scale)')
ax2.set_title(r'(b) State Sparsity & Extreme Scaling ($N \leq 24$)')
ax2.set_xticks(qubits)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.legend(loc='lower left', framealpha=0.9, fontsize=8)

plt.tight_layout()
os.makedirs('report/figures/large_scale_and_enhancement', exist_ok=True)
plt.savefig('report/figures/large_scale_and_enhancement/extreme_scale_limit.pdf', bbox_inches='tight')
plt.savefig('result/pdf/extreme_scale_limit.pdf', bbox_inches='tight')
plt.savefig('result/png/extreme_scale_limit.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hayashitaiga/.gemini/antigravity-ide/brain/72da51dd-fe34-4f5b-83cd-2b7aa88bade0/extreme_scale_limit.png', dpi=300, bbox_inches='tight')
print("Extreme scale plot saved successfully.")
