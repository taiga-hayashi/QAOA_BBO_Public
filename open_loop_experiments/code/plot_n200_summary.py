import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_n200_summary(out_dir="../open_loop_experiments/open_loop/5_n200_test"):
    csv_files = glob.glob(f"{out_dir}/csv/n200_test_*.csv")
    if not csv_files:
        print("No CSV files found.")
        return
        
    all_data = []
    for f in csv_files:
        basename = os.path.basename(f)
        # e.g. n200_test_bb1_N18.csv
        parts = basename.replace(".csv", "").split("_")
        prob = parts[2]
        N = int(parts[3][1:])
        
        df = pd.read_csv(f)
        df["Problem"] = prob
        df["N"] = N
        all_data.append(df)
        
    full_df = pd.concat(all_data, ignore_index=True)
    
    os.makedirs(f"{out_dir}/png", exist_ok=True)
    os.makedirs(f"{out_dir}/pdf", exist_ok=True)
    
    for prob in ["bb1", "bb2"]:
        sub_df = full_df[full_df["Problem"] == prob].sort_values("N")
        if sub_df.empty:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        Ns = sorted(sub_df["N"].unique())
        solvers = ["Classical-SA", "Penalty-QAOA", "XY-QAOA"]
        colors = {"Classical-SA": "blue", "Penalty-QAOA": "red", "XY-QAOA": "green"}
        markers = {"Classical-SA": "o", "Penalty-QAOA": "s", "XY-QAOA": "^"}
        
        for s in solvers:
            s_data = sub_df[sub_df["Solver"] == s]
            if not s_data.empty:
                ax1.plot(s_data["N"], s_data["Feasible"], marker=markers[s], color=colors[s], label=s)
                ax2.plot(s_data["N"], s_data["Regret"], marker=markers[s], color=colors[s], label=s)
                
        ax1.set_title(f"Feasible Rate vs N ({prob.upper()}, Train N=200)")
        ax1.set_xlabel("Problem Size (N)")
        ax1.set_ylabel("Feasible Rate (%)")
        ax1.set_xticks(Ns)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title(f"Normalized Regret vs N ({prob.upper()}, Train N=200)")
        ax2.set_xlabel("Problem Size (N)")
        ax2.set_ylabel("Regret (0.0=Optimal, 1.0=Worst)")
        ax2.set_xticks(Ns)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{out_dir}/png/n200_test_summary_{prob}.png", dpi=150)
        plt.savefig(f"{out_dir}/pdf/n200_test_summary_{prob}.pdf")
        plt.close()
        
    print("N200 Summary plots generated.")

if __name__ == "__main__":
    plot_n200_summary()
