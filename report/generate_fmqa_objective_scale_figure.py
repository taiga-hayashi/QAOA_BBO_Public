import json
import matplotlib.pyplot as plt
import numpy as np

with open("result/json/fmqa_lambda_vs_obj_scale.json") as f:
    data = json.load(f)

# Styles
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

colors = {'0.5': '#1f77b4', '1.0': '#2ca02c', '3.0': '#ff7f0e', '10.0': '#d62728'}
markers = {'0.5': 'o', '1.0': 's', '3.0': '^', '10.0': 'd'}

for s_fac, records in data.items():
    lams = [r["lambda"] for r in records]
    feas = [r["feas_mean"] * 100 for r in records]
    opts = [r["opt_mean"] * 100 for r in records]
    
    col = colors[s_fac]
    m = markers[s_fac]
    lab = f"Objective Scale {s_fac}x"
    
    ax1.plot(lams, feas, marker=m, color=col, linewidth=2.2, markersize=7, label=lab)
    ax2.plot(lams, opts, marker=m, color=col, linewidth=2.2, markersize=7, label=lab)

# FM-XY-QAOA Baseline (常に100%充足)
ax1.axhline(y=100.0, color='purple', linestyle='--', linewidth=2.5, label='FM-XY-QAOA (Any Scale: 100%)')
ax1.axvline(x=5.0, color='gray', linestyle=':', linewidth=1.8, label='Fixed $\\lambda=5.0$ Baseline')

ax1.set_xscale('log')
ax1.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Constraint Feasibility Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Constraint Feasibility vs $\lambda$ across Objective Scales ($N=12$)', fontsize=12, fontweight='bold')
ax1.set_ylim(-5, 108)
ax1.grid(True, which='both', linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', fontsize=9.5, framealpha=0.9)

ax1.annotate('Scale 10x:\n$\lambda=5$ causes 0% Feas!\n(BBO Collapse)', xy=(5.0, 0), xytext=(2.2, 35),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#d62728', fontsize=9.5, ha='center')

# Subplot 2: Opt Prob
ax2.axvline(x=5.0, color='gray', linestyle=':', linewidth=1.8, label='Fixed $\\lambda=5.0$ Baseline')
ax2.set_xscale('log')
ax2.set_xlabel('Penalty Coefficient $\lambda$ (log scale)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Optimal Success Probability $P(x^*)$ (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Optimal Solution Probability $P(x^*)$ Sweet Spot Shift', fontsize=12, fontweight='bold')
ax2.grid(True, which='both', linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', fontsize=9.5, framealpha=0.9)

ax2.annotate('Sweet Spot Shifts Right\nwith Objective Scale', xy=(20.0, 87.5), xytext=(5.0, 80),
             arrowprops=dict(facecolor='#333333', shrink=0.08, width=1.5, headwidth=6),
             fontweight='bold', color='#333333', fontsize=10, ha='center')

plt.tight_layout()
os.makedirs("report/figures/penalty_sensitivity", exist_ok=True)
plt.savefig("report/figures/penalty_sensitivity/fmqa_lambda_vs_obj_scale.pdf", dpi=300)
plt.savefig("report/figures/penalty_sensitivity/fmqa_lambda_vs_obj_scale.png", dpi=300)
print("Saved report/figures/penalty_sensitivity/fmqa_lambda_vs_obj_scale.pdf!")
