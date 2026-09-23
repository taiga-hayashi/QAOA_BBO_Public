import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/report/figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11.5
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
plt.subplots_adjust(wspace=0.25, bottom=0.15, top=0.88, left=0.08, right=0.96)

# ----------------------------------------------------------------------
# Panel 1: Best Feasible Energy vs Exact Optimum
# ----------------------------------------------------------------------
ax1 = axes[0]
scales = [20, 24, 28, 32]
labels = ['N=20\n(1.05M)', 'N=24\n(16.78M)', 'N=28\n(268.4M)', 'N=32\n(4.29B)']
x_pos = np.arange(len(scales))

exact_mins = [-5.4792, -8.2339, -10.8034, -12.9681]
fmqa_bests = [-5.4792, -8.2339, -10.8034, -12.9681]
xy_bests   = [-5.4792, -6.6964, -9.6549, -11.4104]
std_bests  = [-1.6956, 1.3051, np.nan, np.nan]

# Exact line
ax1.plot(x_pos, exact_mins, 'k--', linewidth=2.0, label='Exact Global Minimum $f(\mathbf{x}^*)$')
ax1.plot(x_pos, fmqa_bests, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='FMQA (SA, penalty $\lambda=5$)')
ax1.plot(x_pos, xy_bests, '^-', color='#2ca02c', linewidth=2.8, markersize=9, label='FM-XY-QAOA (Subspace, 100% feas)')
ax1.plot(x_pos[:2], std_bests[:2], 's:', color='#ff7f0e', linewidth=2.0, markersize=7, label='Standard QAOA (penalty)')

# Failed mark for Standard QAOA at N=24
ax1.annotate('Failed\n(0.1% feas, f>0)', (x_pos[1], std_bests[1]), textcoords='offset points', xytext=(-30, -25),
             ha='center', fontsize=9, fontweight='bold', color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels)
ax1.set_xlabel('Problem Scale $N$ (Total Search Space $2^N$)', fontweight='bold')
ax1.set_ylabel('Objective Value (Lower is Better)', fontweight='bold')
ax1.set_title('(a) Best Feasible Objective Value ($N=20 \sim 32$)', fontweight='bold', y=1.02)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left', framealpha=0.95)

# ----------------------------------------------------------------------
# Panel 2: Simulation Time Comparison (Qiskit Full vs Subspace)
# ----------------------------------------------------------------------
ax2 = axes[1]
# Execution times in seconds
times_std = [4.79, 29.08]
times_xy_qiskit = [7.55, 71.14]
times_xy_subspace = [0.106, 0.309, 1.19, 4.77]
times_fmqa = [0.12, 0.16, 0.14, 0.18]

ax2.plot(x_pos[:2], times_xy_qiskit, 'r-o', linewidth=2.5, markersize=8, label='FM-XY-QAOA in Qiskit Full ($2^N$)')
ax2.plot(x_pos, times_xy_subspace, 'g-^', linewidth=2.8, markersize=9, label='FM-XY-QAOA Subspace ($\prod d_m$)')
ax2.plot(x_pos, times_fmqa, 'b-s', linewidth=2.0, linestyle='--', markersize=7, label='FMQA (Classical SA)')

# Highlight speedup at N=24
ax2.annotate('230x Speedup!\n71.1s -> 0.31s', (x_pos[1], times_xy_subspace[1]), textcoords='offset points', xytext=(35, 18),
             ha='center', fontsize=9.5, fontweight='bold', color='green',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.2", fc="#eaffea", ec="green", lw=1))

# Highlight N=32 feasibility
ax2.annotate('N=32 (4.29B states)\nOnly 4.77s on PC!', (x_pos[3], times_xy_subspace[3]), textcoords='offset points', xytext=(-35, 30),
             ha='center', fontsize=9.5, fontweight='bold', color='purple',
             arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
             bbox=dict(boxstyle="round,pad=0.2", fc="#f3eaff", ec="purple", lw=1))

ax2.set_yscale('log')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.set_xlabel('Problem Scale $N$ (Total Search Space $2^N$)', fontweight='bold')
ax2.set_ylabel('Execution Time (Seconds, Log Scale)', fontweight='bold')
ax2.set_title('(b) Execution Time: Full Space vs Subspace Solver', fontweight='bold')
ax2.grid(True, which="both", linestyle='--', alpha=0.5)
ax2.legend(loc='upper left', framealpha=0.95)

plt.tight_layout()

pdf_path = os.path.join(OUT_DIR, "large_scale_benchmark_n20_to_n32.pdf")
png_path = os.path.join(OUT_DIR, "large_scale_benchmark_n20_to_n32.png")
plt.savefig(pdf_path, dpi=300)
plt.savefig(png_path, dpi=300)
plt.close()

print("Generated:")
print(f"  {pdf_path}")
print(f"  {png_path}")
