import json
import matplotlib.pyplot as plt
import numpy as np

# Load JSON results
with open("result/json/fmqa_penalty_sweep_results.json") as f:
    data = json.load(f)

# Styles
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

colors = {'N8_4x2': '#1f77b4', 'N12_4x3': '#ff7f0e', 'N16_4x4': '#2ca02c', 'N20_4x5': '#d62728'}
labels = {'N8_4x2': 'N=8 (4x2)', 'N12_4x3': 'N=12 (4x3)', 'N16_4x4': 'N=16 (4x4)', 'N20_4x5': 'N=20 (4x5)'}

for sname, sdata in data.items():
    sweep = sdata["fmqa_sweep"]
    lams = [d["lambda"] for d in sweep]
    feas = [d["feasibility_mean"] * 100 for d in sweep]
    opts = [d["opt_prob_mean"] * 100 for d in sweep]
    uniques = [d["unique_valid_mean"] for d in sweep]
    traps = [d["trap_rate_mean"] * 100 for d in sweep]
    
    col = colors[sname]
    lab = labels[sname]
    
    # 1. Feasibility
    ax1.plot(lams, feas, marker='o', color=col, linewidth=2.2, label=lab)
    
    # 2. Optimal Success Prob
    ax2.plot(lams, opts, marker='s', color=col, linewidth=2.2, label=lab)
    
    # 3. Unique Valid Solutions
    ax3.plot(lams, uniques, marker='^', color=col, linewidth=2.2, label=lab)
    
    # 4. Max Trap Prob
    ax4.plot(lams, traps, marker='d', color=col, linewidth=2.2, label=lab)

# FM-XY-QAOA Baseline (常に100%充足)
ax1.axhline(y=100.0, color='#2ca02c', linestyle='--', linewidth=2.5, label='FM-XY-QAOA (100% Guaranteed, $\\lambda$-free)')
ax1.axvspan(0.1, 1.0, color='red', alpha=0.12, label='Feasibility Collapse Zone ($\\lambda < 1.0$)')

ax1.set_xscale('log')
ax1.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Feasibility Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Constraint Feasibility Rate vs Penalty $\lambda$', fontsize=12, fontweight='bold')
ax1.set_ylim(-5, 108)
ax1.grid(True, which='both', linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', fontsize=9.5, framealpha=0.9)

# Subplot 2: Opt Prob
ax2.axvspan(20.0, 100.0, color='gray', alpha=0.15, label='Random Degeneration Zone ($\\lambda \geq 20$)')
ax2.set_xscale('log')
ax2.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Optimal Success Prob $P(x^*)$ (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Optimal Solution Probability $P(x^*)$ vs Penalty $\lambda$', fontsize=12, fontweight='bold')
ax2.grid(True, which='both', linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.9)

# Subplot 3: Unique Valid
ax3.set_xscale('log')
ax3.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of Unique Valid Solutions', fontsize=11, fontweight='bold')
ax3.set_title('(c) Sample Diversity (Unique Feasible States) vs Penalty $\lambda$', fontsize=12, fontweight='bold')
ax3.grid(True, which='both', linestyle=':', alpha=0.6)
ax3.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

# Subplot 4: Trap Rate
ax4.set_xscale('log')
ax4.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Most Frequent Solution Ratio (%)', fontsize=11, fontweight='bold')
ax4.set_title('(d) Solution Concentration (Local Trap Risk) vs Penalty $\lambda$', fontsize=12, fontweight='bold')
ax4.grid(True, which='both', linestyle=':', alpha=0.6)
ax4.legend(loc='upper right', fontsize=9.5, framealpha=0.9)

plt.tight_layout()
os.makedirs("report/figures/penalty_sensitivity", exist_ok=True)
plt.savefig("report/figures/penalty_sensitivity/fmqa_penalty_sweep_4panel.pdf", dpi=300)
plt.savefig("report/figures/penalty_sensitivity/fmqa_penalty_sweep_4panel.png", dpi=300)
print("Successfully generated report/figures/penalty_sensitivity/fmqa_penalty_sweep_4panel.pdf!")
