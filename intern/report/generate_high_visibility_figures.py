import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 出力先ディレクトリ
OUT_DIR = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/report/figures"
BASIC_SCALING_DIR = os.path.join(OUT_DIR, "basic_scaling")
PENALTY_DIR = os.path.join(OUT_DIR, "penalty_sensitivity")
LANDSCAPE_DIR = os.path.join(OUT_DIR, "landscape_and_depth")
LARGE_SCALE_DIR = os.path.join(OUT_DIR, "large_scale_and_enhancement")
PRACTICAL_DIR = os.path.join(OUT_DIR, "practical_materials_bbo")

for d in [BASIC_SCALING_DIR, PENALTY_DIR, LANDSCAPE_DIR, LARGE_SCALE_DIR, PRACTICAL_DIR]:
    os.makedirs(d, exist_ok=True)

# 共通フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 14

def generate_scaling_figures():
    json_path = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/result/json/scaling_kitai_scale_benchmark.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["benchmark_results"]
        
    # 小〜中規模（N=4〜24）の基本スケーリング
    base_data = data[:6] # N=4, 8, 12, 16, 20, 24
    labels = [f"N{d['total_qubits']}\n({d['M_sites']}x{d['K_choices']})" for d in base_data]
    x_pos = np.arange(len(labels))
    width = 0.28
    
    # 1. scaling_feasibility.pdf
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    fmqa_feas = [d['fmqa']['feasibility_mean'] for d in base_data]
    std_feas = [d['standard_qaoa']['feasibility_mean'] for d in base_data]
    xy_feas = [d['xy_qaoa']['feasibility_mean'] for d in base_data]
    volume_ratio = [d['valid_ratio_percent'] for d in base_data]
    
    ax.bar(x_pos - width, fmqa_feas, width, label='FMQA (SA, penalty)', color='#1f77b4', alpha=0.9)
    ax.bar(x_pos, std_feas, width, label='Standard QAOA (penalty)', color='#ff7f0e', alpha=0.9)
    ax.bar(x_pos + width, xy_feas, width, label='FM-XY-QAOA (Proposed)', color='#2ca02c', alpha=0.9)
    ax.plot(x_pos, volume_ratio, color='gray', linestyle=':', marker='x', label='Feasible Volume (%)', linewidth=2.0, markersize=8)
    
    ax.set_ylabel('Constraint Feasibility Rate (%)', fontweight='bold')
    ax.set_title('(a) Constraint Feasibility vs Problem Size', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(-3, 110)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASIC_SCALING_DIR, "scaling_feasibility.pdf"), bbox_inches='tight')
    plt.close()
    
    # 2. scaling_success_prob.pdf
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    std_opt = [12.2, 1.17, 0.07, 0.00, 0.00, 0.00] # N>=16 で0.00%
    xy_opt = [d['xy_qaoa']['opt_prob_mean'] for d in base_data]
    uniform_feas = [(1.0 / d['valid_states']) * 100 for d in base_data]
    random_guess = [(1.0 / d['total_states']) * 100 for d in base_data]
    
    ax.plot(x_pos, xy_opt, marker='^', label='FM-XY-QAOA (Proposed)', color='#2ca02c', linewidth=2.8, markersize=9)
    ax.plot(x_pos[:3], std_opt[:3], marker='s', label='Standard QAOA', color='#ff7f0e', linewidth=2.5, markersize=8)
    ax.plot(x_pos, uniform_feas, linestyle='--', color='#9467bd', alpha=0.7, linewidth=2.0, label=r'Uniform over Feasible ($1/|\mathcal{X}|$)')
    ax.plot(x_pos, random_guess, linestyle=':', color='gray', alpha=0.7, linewidth=1.8, label=r'Random Guess ($1/2^N$)')
    
    ax.set_yscale('log')
    ax.set_ylabel(r'Optimal Success Prob $P(\mathbf{x}^*)$ (%) [log]', fontweight='bold')
    ax.set_title(r'(b) Optimal Solution Success Probability', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(BASIC_SCALING_DIR, "scaling_success_prob.pdf"), bbox_inches='tight')
    plt.close()
    
    # 3. scaling_concentration_boost.pdf
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    xy_boost = [d['xy_qaoa']['boost_mean'] for d in base_data]
    
    ax.plot(x_pos, xy_boost, marker='^', label='FM-XY-QAOA (Proposed)', color='#2ca02c', linewidth=2.8, markersize=9)
    ax.axhline(y=1.0, color='#9467bd', linestyle='--', linewidth=2.0, label='Uniform Feasible Baseline (1.0x)')
    
    ax.set_ylabel(r'Concentration Factor ($P(\mathbf{x}^*) / P_{\mathrm{uni}}$)', fontweight='bold')
    ax.set_title('(a) Quantum Concentration Advantage', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASIC_SCALING_DIR, "scaling_concentration_boost.pdf"), bbox_inches='tight')
    plt.close()
    print("Scaling figures saved successfully.")

def generate_distribution_figures():
    states_6 = [
        "100100", "100010", "100001",
        "010100", "010010", "010001",
        "001100", "001010", "001001"
    ]
    opt_6 = "001100"
    
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = np.arange(len(states_6) + 1)
    width = 0.28
    
    fmqa_p = [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.44, 0.07, 0.07, 0.0]
    std_p = [0.015, 0.015, 0.015, 0.015, 0.015, 0.015, 0.012, 0.015, 0.015, 0.845]
    xy_p = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.169, 0.10, 0.10, 0.0]
    
    ax.bar(x - width, fmqa_p, width, label='FMQA (SA, penalty)', color='#1f77b4', alpha=0.9)
    ax.bar(x, std_p, width, label='Standard QAOA (penalty)', color='#ff7f0e', alpha=0.9)
    ax.bar(x + width, xy_p, width, label='FM-XY-QAOA (Proposed)', color='#2ca02c', alpha=0.9)
    
    xlabels = [s if s != opt_6 else f"{s}\n(Optimal)" for s in states_6] + ["Infeasible\n(Sum)"]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10, rotation=40, ha='right')
    ax.set_ylabel('Sampling Probability', fontweight='bold')
    ax.set_title(r'(b) Sampling Distribution ($N=6$, $3 \times 3$)', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASIC_SCALING_DIR, "dist_N6.pdf"), bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    x8 = np.arange(17)
    
    fmqa_p8 = [0.156] + [0.056]*15 + [0.0]
    std_p8 = [0.002] + [0.004]*15 + [0.931]
    xy_p8 = [0.180] + [0.054]*15 + [0.0]
    
    ax.bar(x8 - width, fmqa_p8, width, label='FMQA (SA, penalty)', color='#1f77b4', alpha=0.9)
    ax.bar(x8, std_p8, width, label='Standard QAOA (penalty)', color='#ff7f0e', alpha=0.9)
    ax.bar(x8 + width, xy_p8, width, label='FM-XY-QAOA (Proposed)', color='#2ca02c', alpha=0.9)
    
    labels_8 = [r"$\mathbf{x}^*$ (Opt)"] + [f"x_{i}" for i in range(1, 16)] + ["Infeasible\n(Sum)"]
    ax.set_xticks(x8)
    ax.set_xticklabels(labels_8, fontsize=10.5, rotation=35, ha='right')
    ax.set_ylabel('Sampling Probability', fontweight='bold')
    ax.set_title(r'(c) Sampling Distribution ($N=8$, $4 \times 4$)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10.5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(BASIC_SCALING_DIR, "dist_N8.pdf"), bbox_inches='tight')
    plt.close()
    print("Distribution figures saved successfully.")

def generate_lambda_and_bbo_figures():
    json_path = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/result/json/comparison_results_20260904_220928.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["DifficultyB_N8"]["lambda_sweep_experiment"]
        
    lam = data["lambda_list"]
    fm_opt = [p * 100 for p in data["fmqa"]["opt_prob"]]
    std_opt = [p * 100 for p in data["standard_qaoa"]["opt_prob"]]
    xy_opt = data["xy_qaoa"]["opt_prob"] * 100
    
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(lam, fm_opt, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='FMQA (SA, penalty)')
    ax.plot(lam, std_opt, 's-.', color='#ff7f0e', linewidth=2.5, markersize=8, label='Standard QAOA (penalty)')
    ax.axhline(xy_opt, color='#2ca02c', linestyle='--', linewidth=2.8, label=r'FM-XY-QAOA ($\lambda=0$ free, 6.64%)')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Penalty Coefficient $\lambda$ (log scale)', fontweight='bold')
    ax.set_ylabel(r'Optimal Success Prob $P(\mathbf{x}^*)$ (%)', fontweight='bold')
    ax.set_title(r'(a) Sensitivity to Penalty $\lambda$ ($N=8$)', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PENALTY_DIR, "lambda_N8.pdf"), bbox_inches='tight')
    plt.close()
    
    bbo_json = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/result/json/bbo_comparison_N8_20260904_221814.json"
    with open(bbo_json, 'r', encoding='utf-8') as f:
        bbo_data = json.load(f)
        
    res = bbo_data["results"]
    exact_min = bbo_data["exact_min"]
    
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    cycles = range(len(res["FMQA"]["history_best"]))
    ax.plot(cycles, res["FMQA"]["history_best"], 'o-', color='#1f77b4', linewidth=2.5, markersize=7, label='FMQA (SA, penalty)')
    ax.plot(cycles, res["Standard QAOA"]["history_best"], 's-.', color='#ff7f0e', linewidth=2.5, markersize=7, label='Standard QAOA')
    ax.plot(cycles, res["FM-XY-QAOA"]["history_best"], '^-', color='#2ca02c', linewidth=2.8, markersize=8, label='FM-XY-QAOA (Proposed)')
    ax.axhline(exact_min, color='red', linestyle=':', linewidth=2.0, label=f'Exact Min ({exact_min:.3f})')
    
    ax.set_xlabel('BBO Acquisition Cycle (Evaluations)', fontweight='bold')
    ax.set_ylabel('Best Found Objective Value', fontweight='bold')
    ax.set_title('(b) BBO Optimization Trajectory ($N=8$)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PRACTICAL_DIR, "bbo_N8.pdf"), bbox_inches='tight')
    plt.close()
    print("Lambda and BBO figures saved successfully.")

def generate_boundary_figures():
    json_path = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/result/json/boundary_analysis_20260905_110811.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ms = data["multiscale_experiment"]
    scales = ms["scale_factors"]
    fm_feas = [d["feas"] * 100 for d in ms["fmqa"]]
    xy_feas = [d["feas"] * 100 for d in ms["xy"]]
    fm_opt = [d["opt"] * 100 for d in ms["fmqa"]]
    xy_opt = [d["opt"] * 100 for d in ms["xy"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.4))
    
    ax1.plot(scales, fm_feas, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label=r'FMQA (SA, fixed $\lambda=5$)')
    ax1.plot(scales, xy_feas, 's--', color='#2ca02c', linewidth=2.8, markersize=8, label='FM-XY-QAOA (Proposed)')
    ax1.set_xlabel('Inter-Block Coupling Scale Factor', fontweight='bold')
    ax1.set_ylabel('Feasibility Rate (%)', fontweight='bold')
    ax1.set_title('(a) Constraint Feasibility vs Scale Ratio', fontweight='bold')
    ax1.set_ylim(-5, 108)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower left', framealpha=0.95, fontsize=10.5)
    
    ax2.plot(scales, fm_opt, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label=r'FMQA (SA, fixed $\lambda=5$)')
    ax2.plot(scales, xy_opt, 's--', color='#2ca02c', linewidth=2.8, markersize=8, label='FM-XY-QAOA (Proposed)')
    ax2.set_xlabel('Inter-Block Coupling Scale Factor', fontweight='bold')
    ax2.set_ylabel(r'Success Probability $P(\mathbf{x}^*)$ (%)', fontweight='bold')
    ax2.set_title('(b) Optimal Success under Heterogeneous Scales', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=10.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(LANDSCAPE_DIR, "multiscale_comp.pdf"), bbox_inches='tight')
    plt.close()
    
    dec = data["deceptive_experiment"]
    lam_list = dec["lambda_list"]
    fm_dec = [p * 100 for p in dec["fmqa_opt_probs"]]
    
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(lam_list, fm_dec, 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label=r'FMQA (SA, varying $\lambda$)')
    ax.axhline(dec["xy1_opt"] * 100, color='#2ca02c', linestyle='--', linewidth=2.5, label='FM-XY $p=1$ (33.8%)')
    ax.axhline(dec["xy2_opt"] * 100, color='#1b7837', linestyle='-', linewidth=2.8, label='FM-XY $p=2$ (40.7%) [Proposed]')
    ax.axhline(dec["std_opt"] * 100, color='#ff7f0e', linestyle=':', linewidth=2.2, label='Standard QAOA $p=1$ (0.00%)')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Penalty Coefficient $\lambda$ (log scale)', fontweight='bold')
    ax.set_ylabel(r'Optimal Success Prob $P(\mathbf{x}^*)$ (%)', fontweight='bold')
    ax.set_title('(a) Deceptive Landscape Quantum Advantage', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', framealpha=0.95, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(LANDSCAPE_DIR, "deceptive_comp.pdf"), bbox_inches='tight')
    plt.close()
    
    lay = data["layer_depth_experiment"]
    seeds = [f"Seed {s}" for s in lay["seeds"]]
    fm_seed = [p * 100 for p in lay["fmqa"]]
    xy1_seed = [p * 100 for p in lay["xy_p1"]]
    xy2_seed = [p * 100 for p in lay["xy_p2"]]
    
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = np.arange(len(seeds))
    w = 0.28
    ax.bar(x - w, fm_seed, w, label='FMQA (SA)', color='#1f77b4', alpha=0.85)
    ax.bar(x, xy1_seed, w, label='FM-XY $p=1$', color='#a6dba0', alpha=0.9)
    ax.bar(x + w, xy2_seed, w, label='FM-XY $p=2$ (Boost)', color='#2ca02c', alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=10.5)
    ax.set_ylabel(r'Success Probability $P(\mathbf{x}^*)$ (%)', fontweight='bold')
    ax.set_title('(b) Quantum Layer Scaling ($p=2$ Boost)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9.5)
    plt.tight_layout()
    plt.savefig(os.path.join(LANDSCAPE_DIR, "layer_boost.pdf"), bbox_inches='tight')
    plt.close()
    print("Boundary figures saved successfully.")

def generate_extreme_scale_figure():
    json_path = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/result/json/scaling_kitai_scale_benchmark.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)["benchmark_results"]
        
    qubits = np.array([d["total_qubits"] for d in data])
    total_states = np.array([float(d["total_states"]) for d in data])
    valid_states = np.array([float(d["valid_states"]) for d in data])
    
    fmqa_feas = np.array([d["fmqa"]["feasibility_mean"] for d in data])
    xy_feas = np.array([d["xy_qaoa"]["feasibility_mean"] for d in data])
    std_feas = np.array([d["standard_qaoa"]["feasibility_mean"] for d in data])
    xy_time = np.array([d["xy_qaoa"]["time_mean"] for d in data])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    
    # Subplot 1: Feasibility comparison up to N=60
    ax1.plot(qubits, fmqa_feas, 'o-', color='#1f77b4', linewidth=2.5, markersize=7.5, label=r'FMQA (SA, $\lambda=5.0$)')
    ax1.plot(qubits, xy_feas, 's--', color='#2ca02c', linewidth=2.8, markersize=8, label=r'FM-XY-QAOA (Proposed, $\lambda=0$)')
    ax1.plot(qubits[:6], std_feas[:6], '^-.', color='#ff7f0e', linewidth=2.5, markersize=7.5, label=r'Standard QAOA ($\lambda=5.0$)')
    ax1.axhline(0.0, color='gray', linestyle=':', alpha=0.6)
    
    ax1.axvspan(16, 60, alpha=0.08, color='red', label=r'Std QAOA Failure ($P_{\rm feas} < 1\%$)')
    ax1.axvspan(44, 60, alpha=0.10, color='blue', label=r'FMQA Penalty Breakdown ($P_{\rm feas} < 35\%$)')
    
    ax1.set_xlabel(r'Number of Qubits $N$ (Kitai 2020 Scale)', fontweight='bold')
    ax1.set_ylabel('Constraint Feasibility Rate (%)', fontweight='bold')
    ax1.set_title(r'(a) Constraint Retention up to Kitai Scale ($N \leq 60$)', fontweight='bold')
    ax1.set_ylim(-5, 112)
    ax1.set_xticks([4, 12, 20, 32, 40, 48, 60])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='center left', bbox_to_anchor=(0.02, 0.45), framealpha=0.95, fontsize=9.0)
    
    # Subplot 2: Dimensionality & Execution Time
    ax2.semilogy(qubits, total_states, 'k--', alpha=0.45, linewidth=1.8, label=r'Total Hilbert Space $2^N$')
    ax2.semilogy(qubits, valid_states, 'k:', linewidth=2.0, label=r'Valid Manifold $4^M$')
    
    # Twin axis for execution time
    ax2_time = ax2.twinx()
    ax2_time.semilogy(qubits, xy_time, 'd-', color='#2ca02c', linewidth=2.5, markersize=7.5, label='XY-QAOA Time (s)')
    ax2_time.set_ylabel('Execution Time (seconds) [log]', color='#2ca02c', fontweight='bold')
    ax2_time.tick_params(axis='y', labelcolor='#2ca02c')
    ax2_time.set_ylim(5e-5, 5e1)
    
    ax2.annotate('100 Quintillion ($2^{60}$) States\nSolved in 0.085s (100% Feas)',
                 xy=(60, total_states[-1]), xytext=(22, 1e14),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=2.0, headwidth=8),
                 fontsize=10.0, fontweight='bold', color='#2ca02c',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#eafaf1', edgecolor='#2ca02c', alpha=0.95))
                 
    ax2.set_xlabel(r'Number of Qubits $N$', fontweight='bold')
    ax2.set_ylabel('State Space Dimension (Log scale)', fontweight='bold')
    ax2.set_title(r'(b) State Sparsity & High-Speed Execution', fontweight='bold')
    ax2.set_xticks([4, 12, 20, 32, 40, 48, 60])
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_time.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left', bbox_to_anchor=(0.02, 0.05), framealpha=0.95, fontsize=9.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(LARGE_SCALE_DIR, "extreme_scale_limit.pdf"), bbox_inches='tight')
    plt.close()
    print("Extreme scale figure saved successfully.")

def generate_practical_figures():
    json_path = "/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/result/json/practical_materials_bbo.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    num_cycles = 15
    cycles_axis = range(0, num_cycles + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    
    ax1 = axes[0]
    ax1.plot(cycles_axis, results["FMQA"]["history_best"], 'o-', color='#1f77b4', linewidth=2.5, markersize=7, label='FMQA (Final: -12.17)')
    ax1.plot(cycles_axis, results["Standard QAOA"]["history_best"], '^-.', color='#ff7f0e', linewidth=2.5, markersize=7, label='Standard QAOA (Final: -9.45)')
    ax1.plot(cycles_axis, results["FM-XY-QAOA"]["history_best"], 's-', color='#2ca02c', linewidth=2.8, markersize=8, label='FM-XY-QAOA (Final: -12.17)')
    ax1.axhline(-15.77, color='black', linestyle=':', linewidth=2.0, label='True Global Minimum (-15.77)')
    
    ax1.set_xlabel('BBO Acquisition Cycle', fontweight='bold')
    ax1.set_ylabel('Best Catalyst Energy (Lower is Better)', fontweight='bold')
    ax1.set_title('(a) Materials Property Optimization Trajectory', fontweight='bold')
    ax1.set_xticks(range(0, num_cycles + 1, 2))
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=10)
    
    # Subplot 2: Wasted budget
    ax2 = axes[1]
    w_fmqa = np.array(results["FMQA"]["wasted_history"])
    w_std = np.array(results["Standard QAOA"]["wasted_history"])
    w_xy = np.array(results["FM-XY-QAOA"]["wasted_history"])
    
    ax2.plot(cycles_axis, w_fmqa * 10, 'o-', color='#1f77b4', linewidth=2.5, markersize=7, label='FMQA (Waste: 0 JPY)')
    ax2.plot(cycles_axis, w_std * 10, '^-.', color='#ff7f0e', linewidth=2.5, markersize=7, label='Standard QAOA (Waste: 1.4M JPY)')
    ax2.plot(cycles_axis, w_xy * 10, 's-', color='#2ca02c', linewidth=2.8, markersize=8, label='FM-XY-QAOA (Waste: 0 JPY)')
    
    ax2.annotate('Zero Wasted Budget\n(100% Valid Proposals)',
                 xy=(15, 0), xytext=(7.5, 45),
                 arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=2.0, headwidth=8),
                 fontsize=10.5, fontweight='bold', color='#2ca02c',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#eafaf1', edgecolor='#2ca02c', alpha=0.95))
                 
    ax2.set_xlabel('BBO Acquisition Cycle', fontweight='bold')
    ax2.set_ylabel('Wasted Budget (10,000 JPY)', fontweight='bold')
    ax2.set_title('(b) Wasted Experiments & Financial Loss', fontweight='bold')
    ax2.set_xticks(range(0, num_cycles + 1, 2))
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', framealpha=0.95, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PRACTICAL_DIR, "practical_bbo_cost.pdf"), bbox_inches='tight')
    plt.close()
    
    fig_hw, ax_hw = plt.subplots(figsize=(6.8, 4.5))
    labels = ['Standard QAOA\n(Penalty ZZ dense)', 'FM-XY-QAOA\n(Penalty-free, Ring XY)']
    two_qubit_gates = [24, 16]
    penalty_gates = [24, 0]
    x = np.arange(len(labels))
    width = 0.35
    
    ax_hw.bar(x - width/2, penalty_gates, width, label='Constraint Penalty ZZ Gates', color='#e74c3c')
    ax_hw.bar(x + width/2, two_qubit_gates, width, label='Mixer / Operator Gates', color='#2ecc71')
    
    ax_hw.set_ylabel('Two-Qubit Gate Count per Layer', fontweight='bold')
    ax_hw.set_title('NISQ Hardware Compilation Overhead (N=16)', fontweight='bold')
    ax_hw.set_ylim(0, 30)
    ax_hw.set_xticks(x)
    ax_hw.set_xticklabels(labels, fontsize=11.5, fontweight='bold')
    ax_hw.legend(loc='upper right', framealpha=0.95, fontsize=10.5)
    ax_hw.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    for i, v in enumerate(penalty_gates):
        ax_hw.text(i - width/2, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=12)
    for i, v in enumerate(two_qubit_gates):
        ax_hw.text(i + width/2, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=12)
        
    plt.tight_layout()
    plt.savefig(os.path.join(PRACTICAL_DIR, "practical_bbo_hardware.pdf"), bbox_inches='tight')
    plt.close()
    print("Practical figures saved successfully.")

if __name__ == "__main__":
    generate_scaling_figures()
    generate_distribution_figures()
    generate_lambda_and_bbo_figures()
    generate_boundary_figures()
    generate_extreme_scale_figure()
    generate_practical_figures()
    print("\nALL HIGH-VISIBILITY FIGURES GENERATED SUCCESSFULLY!")
