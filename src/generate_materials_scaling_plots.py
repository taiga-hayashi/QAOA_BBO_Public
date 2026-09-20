import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# 1. Load benchmark results
with open("report/materials_scaling_benchmark_results.json") as f:
    bench_data = json.load(f)

with open("report/materials_penalty_sweet_spot_results.json") as f:
    sweet_data = json.load(f)

Ns = [16, 20, 24, 28, 32]

# -----------------------------------------------------------------------------
# FIGURE 1: Updated Scaling Comparison (Feasibility & Residual Gap)
# -----------------------------------------------------------------------------
feas_fmqa_fixed5 = [np.mean(bench_data[str(n)]["fmqa_l5"]["feas"]) for n in Ns]
feas_fmqa_adapt = [np.mean(sweet_data[str(n)]["adaptive_eval"]["feas"]) for n in Ns]
feas_qaoa = [100.0 for _ in Ns]

gap_p1 = [np.mean(np.array(bench_data[str(n)]["p1_base"]["vals"]) - np.array(bench_data[str(n)]["exact_mins"])) for n in Ns]
gap_a2a = [np.mean(np.array(bench_data[str(n)]["all_to_all"]["vals"]) - np.array(bench_data[str(n)]["exact_mins"])) for n in Ns]
gap_hyb = [np.mean(np.array(bench_data[str(n)]["hybrid_ls"]["vals"]) - np.array(bench_data[str(n)]["exact_mins"])) for n in Ns]
gap_fmqa_adapt = [np.mean(sweet_data[str(n)]["adaptive_eval"]["gaps"]) for n in Ns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.4))

# (a) Feasibility Rate - Using slight horizontal offset so markers do not overlap
Ns_arr = np.array(Ns, dtype=float)
offset = 0.35

# 1. FMQA Adaptive (Tuned to satisfy constraints: 100%)
ax1.plot(Ns_arr - offset, feas_fmqa_adapt, marker='o', markersize=9, linewidth=2.4, color='#1f77b4', linestyle='-', 
         label='FMQA (Tuned $\\lambda_{\\mathrm{adaptive}}$: 100% Feasible)')

# 2. QAOA (Guaranteed 100%)
ax1.plot(Ns_arr + offset, feas_qaoa, marker='s', markersize=8, linewidth=2.4, color='#2ca02c', linestyle='-', 
         label='QAOA (FM-XY / Hybrid: 100% Guaranteed, $\\lambda$-free)')

# 3. FMQA Fixed 5.0 (Un-tuned Reference)
ax1.plot(Ns_arr, feas_fmqa_fixed5, marker='x', markersize=7, linewidth=1.5, color='#d62728', linestyle=':', alpha=0.6,
         label='FMQA (Un-tuned $\\lambda=5.0$: Infeasible Collapse 0%)')

# Annotations on ax1
ax1.annotate('Tuned FMQA & QAOA\n100% Constraint Satisfaction',
             xy=(24, 100), xytext=(24, 76),
             ha='center', fontsize=9.5, fontweight='bold', color='#0b4a72',
             bbox=dict(boxstyle='round,pad=0.45', facecolor='#e6f2fa', edgecolor='#1f77b4', lw=1.5),
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.8))

ax1.annotate(r'Un-tuned Baseline ($\lambda=5.0$)' + '\n' + r'Collapse due to strong synergies (0%)',
             xy=(24, 0), xytext=(24, 25),
             ha='center', fontsize=8.5, color='#9c1414',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdeded', edgecolor='#d62728', lw=1.2, alpha=0.9),
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.3))

ax1.set_xlabel('Number of Binary Variables $N$ (Sites $M = N / 4$)', fontweight='bold')
ax1.set_ylabel('One-Hot Feasibility Rate (%)', fontweight='bold')
ax1.set_title('(a) Constraint Feasibility in Multi-Scale Catalyst Model', fontweight='bold', pad=10)
ax1.set_xticks(Ns)
ax1.set_xticklabels([f"$N={n}$\n($M={n//4}$)" for n in Ns])
ax1.set_ylim(-10, 115)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='center left', framealpha=0.95, fontsize=9.2)

# (b) Residual Gap
ax2.plot(Ns, gap_p1, marker='o', markersize=7, linewidth=1.8, color='#ff7f0e', label='FM-XY-QAOA (p=1 Baseline, Ring)')
ax2.plot(Ns, gap_a2a, marker='^', markersize=7, linewidth=1.8, color='#9467bd', label='All-to-All XY-QAOA + CVaR')
ax2.plot(Ns, gap_hyb, marker='*', markersize=10, linewidth=2.5, color='#2ca02c', label='Hybrid QAOA (QAOA + 1-opt LS)')
ax2.plot(Ns, gap_fmqa_adapt, marker='d', markersize=7, linewidth=1.8, color='#1f77b4', linestyle='--', label='FMQA (Tuned $\\lambda_{\\mathrm{adaptive}}$: SA Optimal)')

ax2.set_xlabel('Number of Binary Variables $N$ (Sites $M = N / 4$)', fontweight='bold')
ax2.set_ylabel('Energy Residual Gap ($E_{\\mathrm{best}} - E_{\\mathrm{exact}}$)', fontweight='bold')
ax2.set_title('(b) Optimization Residual Gap from Ground State', fontweight='bold', pad=10)
ax2.set_xticks(Ns)
ax2.set_xticklabels([f"$N={n}$\n($M={n//4}$)" for n in Ns])
ax2.set_ylim(-0.2, max(gap_p1) * 1.15)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left', framealpha=0.92, fontsize=10)

plt.tight_layout()
fig1_path = "report/figures/practical_materials_bbo/materials_scaling_feasibility_and_gap.pdf"
os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"[SUCCESS] Updated Figure 1 saved to {fig1_path}")

# -----------------------------------------------------------------------------
# FIGURE 2: FMQA Penalty Sweet Spot Analysis (Feasibility & Success Rate vs Lambda)
# -----------------------------------------------------------------------------
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14.0, 5.2))

colors = {16: '#1f77b4', 20: '#ff7f0e', 24: '#2ca02c', 28: '#d62728', 32: '#9467bd'}
markers = {16: 'o', 20: 's', 24: '^', 28: 'd', 32: 'v'}

for n in Ns:
    sw = sweet_data[str(n)]["lambda_sweep"]
    lams = [item["lambda"] for item in sw]
    feas_list = [item["feas_mean"] for item in sw]
    succ_list = [item["succ_rate"] for item in sw]
    
    ax3.plot(lams, feas_list, marker=markers[n], markersize=6, linewidth=1.8, color=colors[n], label=f'$N={n}$ ($M={n//4}$)')
    ax4.plot(lams, succ_list, marker=markers[n], markersize=6, linewidth=1.8, color=colors[n], label=f'$N={n}$ ($M={n//4}$)')

# Shading Zones on ax3
ax3.axvspan(1.0, 6.0, color='red', alpha=0.12, label='Collapse Zone (Violation Favored)')
ax3.axvspan(10.0, 15.0, color='green', alpha=0.12, label='Sweet Spot Zone (100% Feasible)')

ax3.set_xscale('log')
ax3.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontweight='bold')
ax3.set_ylabel('One-Hot Feasibility Rate (%)', fontweight='bold')
ax3.set_title('(a) Constraint Feasibility vs Penalty $\lambda$', fontweight='bold', pad=10)
ax3.set_ylim(-5, 108)
ax3.grid(True, which='both', linestyle=':', alpha=0.6)
ax3.legend(loc='lower right', framealpha=0.9, fontsize=9.5)

# Shading Zones on ax4
ax4.axvspan(10.0, 15.0, color='green', alpha=0.12, label='Sweet Spot Zone (Max Ground State)')
ax4.axvspan(25.0, 50.0, color='gray', alpha=0.15, label='Penalty Trap Zone (Local Trapping)')

ax4.set_xscale('log')
ax4.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontweight='bold')
ax4.set_ylabel('Ground State Success Rate (%)', fontweight='bold')
ax4.set_title('(b) Ground State Success Rate vs Penalty $\lambda$', fontweight='bold', pad=10)
ax4.set_ylim(-5, 108)
ax4.grid(True, which='both', linestyle=':', alpha=0.6)
ax4.legend(loc='lower left', framealpha=0.9, fontsize=9.5)

plt.tight_layout()
fig2_path = "report/figures/practical_materials_bbo/materials_fmqa_penalty_sweet_spot.pdf"
os.makedirs(os.path.dirname(fig2_path), exist_ok=True)
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"[SUCCESS] Figure 2 (Sweet Spot) saved to {fig2_path}")
