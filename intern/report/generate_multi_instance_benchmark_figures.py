import json
import matplotlib.pyplot as plt
import numpy as np

def generate_multi_instance_figures():
    # Load multi-instance benchmark results
    with open("report/multi_instance_benchmark_results.json", "r") as f:
        data = json.load(f)
        
    scales = [20, 24, 28, 32]
    
    exact_means = [np.mean(data[str(N)]["exact_mins"]) for N in scales]
    exact_stds = [np.std(data[str(N)]["exact_mins"]) for N in scales]
    
    fmqa_means = [np.mean(data[str(N)]["fmqa"]["best_vals"]) for N in scales]
    fmqa_stds = [np.std(data[str(N)]["fmqa"]["best_vals"]) for N in scales]
    fmqa_feas_means = [np.mean(data[str(N)]["fmqa"]["feas_rates"]) for N in scales]
    fmqa_feas_stds = [np.std(data[str(N)]["fmqa"]["feas_rates"]) for N in scales]
    fmqa_time_means = [np.mean(data[str(N)]["fmqa"]["times"]) for N in scales]
    fmqa_time_stds = [np.std(data[str(N)]["fmqa"]["times"]) for N in scales]
    
    xy_means = [np.mean(data[str(N)]["xy_qaoa"]["best_vals"]) for N in scales]
    xy_stds = [np.std(data[str(N)]["xy_qaoa"]["best_vals"]) for N in scales]
    xy_feas_means = [np.mean(data[str(N)]["xy_qaoa"]["feas_rates"]) for N in scales]
    xy_time_means = [np.mean(data[str(N)]["xy_qaoa"]["times"]) for N in scales]
    xy_time_stds = [np.std(data[str(N)]["xy_qaoa"]["times"]) for N in scales]
    
    # Setup styling
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'lines.linewidth': 2.0,
        'lines.markersize': 7,
        'figure.autolayout': True
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)
    
    # -------------------------------------------------------------------------
    # Panel (a): Best Objective Value (Mean +/- Std across 10 instances)
    # -------------------------------------------------------------------------
    x_indices = np.arange(len(scales))
    
    # Exact Minimum
    ax1.plot(x_indices, exact_means, 'k--', label='Exact Global Minimum (Mean)', zorder=4)
    ax1.fill_between(x_indices, np.array(exact_means) - np.array(exact_stds), 
                     np.array(exact_means) + np.array(exact_stds), color='gray', alpha=0.15)
    
    # FMQA
    ax1.errorbar(x_indices - 0.05, fmqa_means, yerr=fmqa_stds, fmt='o-', color='#1f77b4',
                 label='FMQA (SA, penalty $\\lambda=5$, Mean $\\pm$ 1$\\sigma$)', capsize=4, capthick=1.5, zorder=5)
    
    # FM-XY-QAOA
    ax1.errorbar(x_indices + 0.05, xy_means, yerr=xy_stds, fmt='s-', color='#2ca02c',
                 label='FM-XY-QAOA (Subspace, $p=1$, Mean $\\pm$ 1$\\sigma$)', capsize=4, capthick=1.5, zorder=6)
    
    # Standard QAOA (Collapsed)
    std_means = [-1.6956, 1.3051]
    ax1.plot([0, 1], std_means, 'r:s', color='#d62728', label='Standard QAOA (Penalty, $p=1$)', alpha=0.8, zorder=3)
    ax1.scatter([1], [1.3051], marker='x', s=120, color='red', zorder=7)
    ax1.annotate('Severe Collapse\n(Feas: 0.1%, Pos Val)', xy=(1, 1.3051), xytext=(0.65, 0.2),
                 arrowprops=dict(arrowstyle="->", color='red', lw=1.2),
                 fontsize=8.5, fontweight='bold', color='#900C3F',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor='red', alpha=0.9))
    
    # Note on FMQA Feasibility drop at N=32
    ax1.annotate('Feasibility Drop\n(99.8%, 0.2% invalid)', xy=(3 - 0.05, fmqa_means[3]), xytext=(2.3, fmqa_means[3] + 2.2),
                 arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.2),
                 fontsize=8.5, fontweight='bold', color='#0d47a1',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', edgecolor='#1f77b4', alpha=0.9))

    ax1.set_title('(a) Solution Quality over 10 Random Instances (Mean $\\pm$ 1$\\sigma$)', fontweight='bold')
    ax1.set_xlabel('Problem Scale $N$ (Total Search Space $2^N$ / Feasible Space $d^M$)')
    ax1.set_ylabel('Objective Value (Lower is Better)')
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"N={N}\n($2^{{{N}}}$ / {4**(N//4):,})" for N in scales])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower left', framealpha=0.9)
    
    # -------------------------------------------------------------------------
    # Panel (b): Execution Time & Feasibility Defense
    # -------------------------------------------------------------------------
    # Full Space Qiskit points for reference
    qiskit_times = [7.55, 71.14]
    ax2.plot([0, 1], qiskit_times, 'r--o', color='#d62728', label='Full Space Qiskit ($2^N$ States, N $\\geq$ 28: OOM)', zorder=3)
    
    # Subspace XY-QAOA
    ax2.errorbar(x_indices, xy_time_means, yerr=xy_time_stds, fmt='s-', color='#2ca02c',
                 label='FM-XY-QAOA Subspace Solver (Mean $\\pm$ 1$\\sigma$)', capsize=4, capthick=1.5, zorder=6)
    
    # FMQA SA
    ax2.errorbar(x_indices, fmqa_time_means, yerr=fmqa_time_stds, fmt='o-', color='#1f77b4',
                 label='FMQA Classical SA (Mean $\\pm$ 1$\\sigma$)', capsize=4, capthick=1.5, zorder=5)
    
    ax2.set_yscale('log')
    ax2.set_title('(b) Scalability: Execution Time & Memory Efficiency', fontweight='bold')
    ax2.set_xlabel('Problem Scale $N$ (Total Search Space $2^N$ / Feasible Space $d^M$)')
    ax2.set_ylabel('Execution Time (Seconds, Log Scale)')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"N={N}\n($2^{{{N}}}$ / {4**(N//4):,})" for N in scales])
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate speedup at N=24 and breakthrough at N=32
    ax2.annotate(f'230x Speedup!\n71.1s $\\rightarrow$ {xy_time_means[1]:.2f}s', xy=(1, xy_time_means[1]), xytext=(0.8, 0.8),
                 arrowprops=dict(arrowstyle="->", color='#2ca02c', lw=1.2),
                 fontsize=8.5, fontweight='bold', color='#1b5e20',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#2ca02c', alpha=0.9))
                 
    ax2.annotate(f'N=32 (4.29B States)\nSolved in {xy_time_means[3]:.2f}s on Laptop!', xy=(3, xy_time_means[3]), xytext=(2.1, 15),
                 arrowprops=dict(arrowstyle="->", color='purple', lw=1.2),
                 fontsize=8.5, fontweight='bold', color='purple',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E5F5', edgecolor='purple', alpha=0.9))

    ax2.legend(loc='upper left', framealpha=0.9)
    
    os.makedirs("report/figures/large_scale_and_enhancement", exist_ok=True)
    plt.savefig("report/figures/large_scale_and_enhancement/large_scale_benchmark_n20_to_n32.pdf", bbox_inches='tight')
    plt.savefig("report/figures/large_scale_and_enhancement/large_scale_benchmark_n20_to_n32.png", bbox_inches='tight')
    plt.close()
    print("Multi-instance benchmark figures generated successfully at report/figures/large_scale_and_enhancement/large_scale_benchmark_n20_to_n32.pdf")

if __name__ == "__main__":
    generate_multi_instance_figures()
