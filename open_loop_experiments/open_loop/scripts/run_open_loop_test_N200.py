import sys
import numpy as np
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../intern_0924")))
from src.problem import get_bb_qubo, generate_initial_dataset, evaluate_bb, get_exact_ground_truth
from src.fm import train_fm
from src.solvers import solve_classical_sa, solve_penalty_qaoa, solve_xy_fmqa_p_layers

def run_test_n200(N, problem_type, seeds=[42, 101, 2024]):
    G = N // 3
    num_train = 200
    
    results = []
    
    for seed in seeds:
        print(f"Running {problem_type.upper()} N={N} Seed={seed} (Train=200)...")
        Q_bb = get_bb_qubo(problem_type, N, seed=seed)
        f_opt, f_worst, _, _ = get_exact_ground_truth(Q_bb, G)
        
        # 1. Train FM (Variant: Huber+L2 for fairness and stability at N=200)
        # Note: We use the variant settings because 200 is small for N=24,27
        # We will use the original train_fm signature but we can't pass weight_decay directly, 
        # so we will use the custom train_fm_custom we wrote earlier.
        from plot_fm_learning_curve import train_fm_custom
        X_train, y_train = generate_initial_dataset(Q_bb, G, num_samples=num_train, seed=seed)
        fm_model = train_fm_custom(X_train, y_train, d=N, k=2, epochs=200, lr=0.1, weight_decay=1e-2, use_huber=True, seed=seed)
        
        # Extract QUBO
        W = fm_model.V.detach().numpy()
        w_0 = fm_model.lin.bias.detach().numpy()[0]
        w_i = fm_model.lin.weight.detach().numpy()[0]
        Q_fm = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            Q_fm[i, i] = w_i[i]
            for j in range(i + 1, N):
                Q_fm[i, j] = np.dot(W[i], W[j])
        offset_fm = w_0
        
        solvers = {
            "Classical-SA": lambda: solve_classical_sa(Q_fm, offset_fm, G, lambda_val=5.0, num_reads=500, seed=seed),
            "Penalty-QAOA": lambda: solve_penalty_qaoa(Q_fm, offset_fm, G, lambda_val=5.0, num_samples=500, seed=seed),
            "XY-QAOA": lambda: solve_xy_fmqa_p_layers(Q_fm, offset_fm, G, p=3, num_samples=500, seed=seed)
        }
        
        for s_name, s_func in solvers.items():
            try:
                res = s_func()
                feas_rate = res["raw_feasible_rate"]
                best_y = float('inf')
                for sample in res["samples"]:
                    if sample["feasible"]:
                        best_y = min(best_y, evaluate_bb(Q_bb, np.array(sample["x"])))
                regret = (best_y - f_opt) / (f_worst - f_opt) if feas_rate > 0 else 1.0
            except Exception as e:
                feas_rate = 0.0
                regret = 1.0
                
            results.append({
                "Problem": problem_type.upper(),
                "N": N,
                "Seed": seed,
                "Solver": s_name,
                "Feasible_Rate": feas_rate * 100,
                "Regret": regret
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    all_df = []
    seeds = [42, 101, 2024, 777, 999] # 5 seeds for robustness
    
    for n in [18, 21, 24, 27]:
        for prob in ["bb1", "bb2"]:
            df = run_test_n200(n, prob, seeds=seeds)
            all_df.append(df)
            
    final_df = pd.concat(all_df, ignore_index=True)
    
    # Calculate means
    summary = final_df.groupby(["Problem", "N", "Solver"]).mean(numeric_only=True).reset_index()
    summary = summary.drop(columns=["Seed"])
    
    out_dir = "../results_1001/open_loop/md"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(f"{out_dir}/open_loop_N200_summary.md", "w") as f:
        f.write("# Open Loop Solver Performance (Train N=200)\n\n")
        f.write("Averaged over 5 seeds.\n\n")
        f.write(summary.to_markdown(index=False))
        
    final_df.to_csv(f"{out_dir}/open_loop_N200_raw.csv", index=False)
    print("N=200 experiments completed.")
