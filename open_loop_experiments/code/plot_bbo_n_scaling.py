import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_bbo_scaling(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs(f"{out_dir}/png", exist_ok=True)
    os.makedirs(f"{out_dir}/pdf", exist_ok=True)
    
    solvers = df["Solver"].unique()
    colors = {
        "LargePenalty-FMQA": "red",
        "LargePenalty-FMQA-variant": "orange",
        "XY-FMQAOA": "blue",
        "XY-FMQAOA-variant": "cyan"
    }
    markers = {
        "LargePenalty-FMQA": "o",
        "LargePenalty-FMQA-variant": "s",
        "XY-FMQAOA": "^",
        "XY-FMQAOA-variant": "D"
    }
    
    for prob in ["bb1", "bb2"]:
        sub_df = df[df["Problem"] == prob].sort_values("N")
        if sub_df.empty:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        Ns = sorted(sub_df["N"].unique())
        
        for s in solvers:
            s_data = sub_df[sub_df["Solver"] == s]
            if not s_data.empty:
                c = colors.get(s, "black")
                m = markers.get(s, "o")
                
                # Plot Median Regret
                ax1.plot(s_data["N"], s_data["MedianRegret"], marker=m, color=c, label=s)
                # Plot Target Rate
                ax2.plot(s_data["N"], s_data["TargetRate"], marker=m, color=c, label=s)
                
        ax1.set_title(f"BBO Final Median Regret vs N ({prob.upper()})")
        ax1.set_xlabel("Problem Size (N)")
        ax1.set_ylabel("Median Regret (100 cycles)")
        ax1.set_xticks(Ns)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title(f"BBO Final Target Rate vs N ({prob.upper()})")
        ax2.set_xlabel("Problem Size (N)")
        ax2.set_ylabel("Target Hit Rate")
        ax2.set_xticks(Ns)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{out_dir}/png/bbo_scaling_{prob}.png", dpi=150)
        plt.savefig(f"{out_dir}/pdf/bbo_scaling_{prob}.pdf")
        plt.close()
        
    print("BBO scaling plots generated successfully.")

if __name__ == "__main__":
    csv_path = "../open_loop_experiments/bbo/full_comparison/csv/summary_stats.csv"
    out_dir = "../open_loop_experiments/bbo/full_comparison"
    plot_bbo_scaling(csv_path, out_dir)
