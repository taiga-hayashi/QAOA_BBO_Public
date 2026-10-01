import sys
import numpy as np
import os
from src.problem import get_bb_qubo, generate_initial_dataset, evaluate_bb, get_exact_ground_truth
from src.fm import train_fm
from src.solvers import solve_classical_sa, solve_penalty_qaoa, solve_xy_fmqa_p_layers

def run_lambda_tuning(N, problem_type="bb1", num_train=1000, seed=42, file=None):
    def p(text):
        print(text, file=file)
        print(text)
        
    p(f"\n=======================================================")
    p(f"Lambda Tuning Test: {problem_type.upper()} N={N} Seed={seed}")
    p(f"=======================================================")
    
    G = N // 3
    Q_bb = get_bb_qubo(problem_type, N, seed=seed)
    f_opt, f_worst, _, _ = get_exact_ground_truth(Q_bb, G)
    p(f"True Optimal: {f_opt:.4f}, True Worst: {f_worst:.4f}")
    
    # Generate perfect FM
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
    
    lambdas = [1.0, 5.0, 10.0, 15.0, 20.0, 50.0]
    
    # 1. XY-QAOA (Lambda-free baseline)
    p("\n--- Baseline: XY-QAOA (p=3, lambda=0) ---")
    res_xy = solve_xy_fmqa_p_layers(Q_fm, offset_fm, G, p=3, num_samples=500, seed=seed)
    feas_xy = res_xy["raw_feasible_rate"]
    best_xy = float('inf')
    for sample in res_xy["samples"]:
        if sample["feasible"]:
            best_xy = min(best_xy, evaluate_bb(Q_bb, np.array(sample["x"])))
    reg_xy = (best_xy - f_opt) / (f_worst - f_opt) if feas_xy > 0 else 1.0
    p(f"Feasible Rate: {feas_xy*100:.1f}% | Best Regret: {reg_xy:.4f}")
    
    # 2. Penalty-QAOA and Classical-SA tuning
    p("\n--- Penalty Solvers Tuning ---")
    p(f"{'Lambda':>6} | {'Classical-SA Feas%':>18} | {'SA Regret':>10} | {'Penalty-QAOA Feas%':>18} | {'QAOA Regret':>11}")
    p("-" * 75)
    
    for l_val in lambdas:
        # SA
        res_sa = solve_classical_sa(Q_fm, offset_fm, G, lambda_val=l_val, num_reads=500, seed=seed)
        f_sa = res_sa["raw_feasible_rate"]
        b_sa = float('inf')
        for sample in res_sa["samples"]:
            if sample["feasible"]:
                b_sa = min(b_sa, evaluate_bb(Q_bb, np.array(sample["x"])))
        r_sa = (b_sa - f_opt) / (f_worst - f_opt) if f_sa > 0 else 1.0
        
        # QAOA
        try:
            res_qa = solve_penalty_qaoa(Q_fm, offset_fm, G, lambda_val=l_val, num_samples=500, seed=seed)
            f_qa = res_qa["raw_feasible_rate"]
            b_qa = float('inf')
            for sample in res_qa["samples"]:
                if sample["feasible"]:
                    b_qa = min(b_qa, evaluate_bb(Q_bb, np.array(sample["x"])))
            r_qa = (b_qa - f_opt) / (f_worst - f_opt) if f_qa > 0 else 1.0
        except Exception as e:
            f_qa, r_qa = 0.0, 1.0 # Failed
            
        p(f"{l_val:>6.1f} | {f_sa*100:>17.1f}% | {r_sa:>10.4f} | {f_qa*100:>17.1f}% | {r_qa:>11.4f}")

if __name__ == "__main__":
    out_file = "experiments/OPEN_LOOP_PERFECT_FM_TEST/lambda_tuning_results.md"
    with open(out_file, "w") as f:
        run_lambda_tuning(N=18, problem_type="bb1", num_train=500, seed=42, file=f)
        run_lambda_tuning(N=18, problem_type="bb2", num_train=500, seed=42, file=f)
        run_lambda_tuning(N=21, problem_type="bb1", num_train=1000, seed=42, file=f)
