import os
import numpy as np
import matplotlib.pyplot as plt

# 出力先
OUT_DIR = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/report/figures/large_scale_and_enhancement"
os.makedirs(OUT_DIR, exist_ok=True)

# フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
plt.subplots_adjust(wspace=0.28, bottom=0.15, top=0.88, left=0.06, right=0.97)

# ----------------------------------------------------------------------
# Panel 1: RAM Memory Scaling (Log Scale)
# ----------------------------------------------------------------------
ax1 = axes[0]
N_full = np.arange(4, 65, 2)
# Full Statevector RAM in Bytes: 2^N * 16
ram_full_bytes = 2.0**N_full * 16.0
ram_full_gb = ram_full_bytes / (1024**3)

# Subspace d=4 (N = 4M, M = N/4): 4^M * 16 = 2^(2M) * 16 = 2^(N/2) * 16
N_sub4 = np.arange(4, 65, 4)
ram_sub4_bytes = 4.0**(N_sub4 / 4.0) * 16.0
ram_sub4_gb = ram_sub4_bytes / (1024**3)

# Subspace d=8 (N = 8M, M = N/8): 8^M * 16 = 2^(3M) * 16 = 2^(3N/8) * 16
N_sub8 = np.arange(8, 65, 8)
ram_sub8_bytes = 8.0**(N_sub8 / 8.0) * 16.0
ram_sub8_gb = ram_sub8_bytes / (1024**3)

ax1.plot(N_full, ram_full_gb, 'r-o', linewidth=2.5, markersize=5, label=r'Full Statevector ($2^N \times 16\mathrm{B}$)')
ax1.plot(N_sub4, ram_sub4_gb, 'g-s', linewidth=2.5, markersize=6, label=r'FM-XY Subspace ($d=4$: $4^M \times 16\mathrm{B}$)')
ax1.plot(N_sub8, ram_sub8_gb, 'b-^', linewidth=2.0, linestyle='--', markersize=6, label=r'FM-XY Subspace ($d=8$: $8^M \times 16\mathrm{B}$)')

# Hardware reference horizontal lines
ax1.axhline(32.0, color='gray', linestyle=':', linewidth=1.5)
ax1.text(6, 38.0, 'Local PC Limit (32 GB RAM)', color='#444444', fontsize=9.5, fontweight='bold')

ax1.axhline(128.0, color='purple', linestyle='--', linewidth=1.5)
ax1.text(6, 160.0, 'Server Upper Bound (128 GB RAM)', color='purple', fontsize=9.5, fontweight='bold')

ax1.axhline(2048.0, color='darkred', linestyle='-.', linewidth=1.5)
ax1.text(6, 2600.0, 'Huge Server Node (2 TB RAM)', color='darkred', fontsize=9.5, fontweight='bold')

# Kitai N=60 highlight
ax1.scatter([60], [17.2], color='green', s=120, zorder=5)
ax1.annotate('Kitai et al. N=60\nOnly 17.2 GB!\n(Fits in 128GB Server)', 
             xy=(60, 17.2), xytext=(38, 0.05),
             arrowprops=dict(arrowstyle="->", color='green', lw=1.5),
             fontsize=9.5, fontweight='bold', color='green',
             bbox=dict(boxstyle="round,pad=0.3", fc="#eaffea", ec="green", lw=1))

ax1.set_yscale('log')
ax1.set_xlim(2, 66)
ax1.set_ylim(1e-6, 1e12)
ax1.set_xlabel('Total Problem Variables / Qubits ($N$)')
ax1.set_ylabel('Required RAM (GB, Log Scale)')
ax1.set_title('(a) Simulation Memory: Full Space vs Subspace', fontweight='bold')
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.legend(loc='lower right', framealpha=0.95)

# ----------------------------------------------------------------------
# Panel 2: Effective State Space Compression Ratio
# ----------------------------------------------------------------------
ax2 = axes[1]
# Valid Ratio: (d / 2^d)^M
# For d=4: (4/16)^M = (1/4)^M = (1/4)^(N/4) = 2^(-N/2)
N_vals = np.arange(4, 25, 4)
ratio_percent_d4 = (1.0 / 4.0)**(N_vals / 4.0) * 100.0

ax2.plot(N_vals, ratio_percent_d4, 'r-o', linewidth=2.5, markersize=6, label='Valid Subspace Ratio ($|\mathcal{X}| / 2^N$)')
ax2.plot(N_vals, [100.0]*len(N_vals), 'g-s', linewidth=2.5, markersize=6, label='FM-XY-QAOA Feasibility (100% Locked)')

# Bar comparison at N=16 and N=24
for x, r in zip(N_vals, ratio_percent_d4):
    ax2.text(x, r * 1.5, f"{r:.2e}%", ha='center', va='bottom', fontsize=8.5, color='darkred', fontweight='bold')

ax2.set_yscale('log')
ax2.set_xlim(2, 26)
ax2.set_ylim(1e-4, 200.0)
ax2.set_xlabel('Total Qubits ($N$)')
ax2.set_ylabel('Ratio / Feasibility (%, Log Scale)')
ax2.set_title('(b) Feasible Space Ratio vs Constraint Feasibility', fontweight='bold')
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.legend(loc='center right', framealpha=0.95)

# ----------------------------------------------------------------------
# Panel 3: Hardware Physical Qubits (Minor Embedding Overhead)
# ----------------------------------------------------------------------
ax3 = axes[2]
N_log = np.arange(4, 65, 4)
# FM-XY-QAOA (Gate-based): 1-to-1 -> Physical Qubits = N
phys_gate = N_log

# FMQA (Annealing on Pegasus / Chimera): Complete graph K_N minor embedding
# On Pegasus, embedding K_N requires roughly N^2 / 4 to N^2 / 3 physical qubits
phys_anneal_min = (N_log**2) / 4.0
phys_anneal_typ = (N_log**2) / 3.0

ax3.plot(N_log, phys_gate, 'g-s', linewidth=2.5, markersize=6, label='FM-XY-QAOA (1-to-1 Mapping: $Q_{\mathrm{phys}} = N$)')
ax3.plot(N_log, phys_anneal_typ, 'r--o', linewidth=2.2, markersize=5, label='FMQA on D-Wave Pegasus (Minor Embedding: $O(N^2)$)')
ax3.fill_between(N_log, phys_anneal_min, phys_anneal_typ * 1.2, color='red', alpha=0.15, label='Embedding Chain Overhead Range')

# Hardware limits
ax3.axhline(64, color='teal', linestyle=':', linewidth=1.5)
ax3.text(5, 70, 'Fujitsu 64Q QPU', color='teal', fontsize=9.5, fontweight='bold')

ax3.axhline(127, color='navy', linestyle=':', linewidth=1.5)
ax3.text(5, 137, 'IBM Heron/Eagle (127Q)', color='navy', fontsize=9.5, fontweight='bold')

ax3.scatter([60], [60], color='green', s=100, zorder=5)
ax3.annotate('FM-XY at N=60:\nOnly 60 Qubits!', xy=(60, 60), xytext=(35, 180),
             arrowprops=dict(arrowstyle="->", color='green', lw=1.5),
             fontsize=9.5, fontweight='bold', color='green',
             bbox=dict(boxstyle="round,pad=0.3", fc="#eaffea", ec="green", lw=1))

ax3.scatter([60], [(60**2)/3.0], color='red', s=100, zorder=5)
ax3.annotate('FMQA at N=60:\n~1,200 Physical Spins!\n(High Chain Break Risk)', xy=(60, 1200), xytext=(22, 1050),
             arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
             fontsize=9.5, fontweight='bold', color='red',
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffeaea", ec="red", lw=1))

ax3.set_xlim(2, 66)
ax3.set_ylim(0, 1500)
ax3.set_xlabel('Logical Variables ($N$)')
ax3.set_ylabel('Required Physical Qubits / Spins')
ax3.set_title('(c) Physical Hardware Qubit Overhead', fontweight='bold')
ax3.grid(True, ls="--", alpha=0.4)
ax3.legend(loc='upper left', framealpha=0.95)

plt.tight_layout()

# 保存
pdf_path = os.path.join(OUT_DIR, "symmetry_scaling_explanation.pdf")
png_path = os.path.join(OUT_DIR, "symmetry_scaling_explanation.png")
plt.savefig(pdf_path, dpi=300)
plt.savefig(png_path, dpi=300)
plt.close()

print(f"Successfully generated:")
print(f"  {pdf_path}")
print(f"  {png_path}")
