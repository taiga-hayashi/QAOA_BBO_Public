import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../intern_0924")))
from src.problem import get_bb_qubo, generate_initial_dataset, evaluate_bb, get_exact_ground_truth
from src.fm import train_fm
from src.solvers import solve_classical_sa, solve_penalty_qaoa, solve_xy_fmqa_p_layers

def run_tuning_for_problem(N, problem_type="bb1", num_train=500, seeds=[42]):
    lambdas = [0.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2500.0, 5000.0]
    G = N // 3
    
    results = {
        "Classical-SA": {l: {"feas": [], "regret": []} for l in lambdas},
        "Penalty-QAOA": {l: {"feas": [], "regret": []} for l in lambdas},
        "XY-QAOA": {"feas": [], "regret": []}
    }
    
    for seed in seeds:
        print(f"Running {problem_type.upper()} N={N} Seed={seed}...")
        Q_bb = get_bb_qubo(problem_type, N, seed=seed)
        f_opt, f_worst, _, _ = get_exact_ground_truth(Q_bb, G)
        
        X_train, y_train = generate_initial_dataset(Q_bb, G, num_samples=num_train, seed=seed)
        fm_model = train_fm(X_train, y_train, d=N, k=2, epochs=200, lr=0.1, seed=seed)
        
        W = fm_model.V.detach().numpy()
        w_0 = fm_model.lin.bias.detach().numpy()[0]
        w_i = fm_model.lin.weight.detach().numpy()[0]
        Q_fm = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            Q_fm[i, i] = w_i[i]
            for j in range(i + 1, N):
                Q_fm[i, j] = np.dot(W[i], W[j])
        offset_fm = w_0
        
        res_xy = solve_xy_fmqa_p_layers(Q_fm, offset_fm, G, p=3, num_samples=500, seed=seed)
        feas_xy = res_xy["raw_feasible_rate"]
        best_xy = float('inf')
        for sample in res_xy["samples"]:
            if sample["feasible"]:
                best_xy = min(best_xy, evaluate_bb(Q_bb, np.array(sample["x"])))
        reg_xy = (best_xy - f_opt) / (f_worst - f_opt) if feas_xy > 0 else 1.0
        results["XY-QAOA"]["feas"].append(feas_xy)
        results["XY-QAOA"]["regret"].append(reg_xy)
        
        for l_val in lambdas:
            res_sa = solve_classical_sa(Q_fm, offset_fm, G, lambda_val=l_val, num_reads=500, seed=seed)
            f_sa = res_sa["raw_feasible_rate"]
            b_sa = float('inf')
            for sample in res_sa["samples"]:
                if sample["feasible"]:
                    b_sa = min(b_sa, evaluate_bb(Q_bb, np.array(sample["x"])))
            r_sa = (b_sa - f_opt) / (f_worst - f_opt) if f_sa > 0 else 1.0
            results["Classical-SA"][l_val]["feas"].append(f_sa)
            results["Classical-SA"][l_val]["regret"].append(r_sa)
            
            try:
                res_qa = solve_penalty_qaoa(Q_fm, offset_fm, G, lambda_val=l_val, num_samples=500, seed=seed)
                f_qa = res_qa["raw_feasible_rate"]
                b_qa = float('inf')
                for sample in res_qa["samples"]:
                    if sample["feasible"]:
                        b_qa = min(b_qa, evaluate_bb(Q_bb, np.array(sample["x"])))
                r_qa = (b_qa - f_opt) / (f_worst - f_opt) if f_qa > 0 else 1.0
            except:
                f_qa, r_qa = 0.0, 1.0
            results["Penalty-QAOA"][l_val]["feas"].append(f_qa)
            results["Penalty-QAOA"][l_val]["regret"].append(r_qa)
            
    xy_feas = np.mean(results["XY-QAOA"]["feas"])
    xy_regret = np.mean(results["XY-QAOA"]["regret"])
    
    agg = {"Lambda": lambdas, "SA_Feas": [], "SA_Regret": [], "QAOA_Feas": [], "QAOA_Regret": []}
    for l_val in lambdas:
        agg["SA_Feas"].append(np.mean(results["Classical-SA"][l_val]["feas"]))
        agg["SA_Regret"].append(np.mean(results["Classical-SA"][l_val]["regret"]))
        agg["QAOA_Feas"].append(np.mean(results["Penalty-QAOA"][l_val]["feas"]))
        agg["QAOA_Regret"].append(np.mean(results["Penalty-QAOA"][l_val]["regret"]))
        
    return agg, xy_feas, xy_regret

def plot_results(agg_data, xy_feas, xy_regret, title, out_prefix):
    lambdas = agg_data["Lambda"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Feasible Rate Plot (Linear Y-axis)
    ax1.plot(lambdas, np.array(agg_data["SA_Feas"]) * 100, marker='o', label='Classical-SA (Penalty)', color='blue')
    ax1.plot(lambdas, np.array(agg_data["QAOA_Feas"]) * 100, marker='s', label='Penalty-QAOA', color='red')
    ax1.axhline(y=xy_feas * 100, color='green', linestyle='--', label='XY-QAOA (No Penalty)')
    ax1.set_title(f'Feasible Rate vs Lambda ({title})')
    ax1.set_xlabel('Penalty Lambda (Log Scale)')
    ax1.set_ylabel('Feasible Rate (%)')
    ax1.set_xscale('symlog', linthresh=10.0) 
    ax1.set_ylim(-5, 105) # Linear y-axis
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Regret Plot (Linear Y-axis instead of Log)
    ax2.plot(lambdas, agg_data["SA_Regret"], marker='o', label='Classical-SA (Penalty)', color='blue')
    ax2.plot(lambdas, agg_data["QAOA_Regret"], marker='s', label='Penalty-QAOA', color='red')
    ax2.axhline(y=xy_regret, color='green', linestyle='--', label='XY-QAOA (No Penalty)')
    ax2.set_title(f'Best Regret vs Lambda ({title})')
    ax2.set_xlabel('Penalty Lambda (Log Scale)')
    ax2.set_ylabel('Normalized Regret (0.0 = True Opt, 1.0 = Worst)')
    ax2.set_xscale('symlog', linthresh=10.0)
    ax2.set_ylim(-0.05, 1.05) # Linear y-axis
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=150)
    plt.savefig(f"{out_prefix}.pdf".replace("/png/", "/pdf/"))
    plt.close()

if __name__ == "__main__":
    # We use 3 seeds for robust mean
    seeds_to_run = [42, 101, 2024]
    
    for n in [18, 21, 24]:
        num_train = 500 if n == 18 else (1000 if n == 21 else 2000)
        
        # BB1
        agg1, xy_feas1, xy_regret1 = run_tuning_for_problem(n, "bb1", num_train=num_train, seeds=seeds_to_run)
        plot_results(agg1, xy_feas1, xy_regret1, f"BB1 N={n} (3 seeds mean, Up to L=5000)", f"../results_1001/open_loop/png/lambda_tuning_large_bb1_N{n}")
        
        # BB2
        agg2, xy_feas2, xy_regret2 = run_tuning_for_problem(n, "bb2", num_train=num_train, seeds=seeds_to_run)
        plot_results(agg2, xy_feas2, xy_regret2, f"BB2 N={n} (3 seeds mean, Up to L=5000)", f"../results_1001/open_loop/png/lambda_tuning_large_bb2_N{n}")
        
    print("All N sizes and large scale lambda plots generated successfully.")
