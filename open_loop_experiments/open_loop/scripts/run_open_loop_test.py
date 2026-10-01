import sys
import numpy as np
import torch
import json
import os
from src.problem import get_bb_qubo, generate_initial_dataset, evaluate_bb, get_exact_ground_truth
from src.fm import train_fm, evaluate_fm_accuracy
from src.solvers import solve_classical_sa, solve_penalty_qaoa, solve_xy_fmqa_p_layers

def run_open_loop_test(N, problem_type="bb1", num_train=1000, seed=42, file=None):
    def p(text):
        print(text, file=file)
        print(text)
        
    p(f"\n--- Open-Loop Solver Test: {problem_type.upper()} N={N} Seed={seed} ---")
    G = N // 3
    Q_bb = get_bb_qubo(problem_type, N, seed=seed)
    
    f_opt, f_worst, _, _ = get_exact_ground_truth(Q_bb, G)
    p(f"True Optimal: {f_opt:.4f}, True Worst: {f_worst:.4f}")
    
    X_train, y_train = generate_initial_dataset(Q_bb, G, num_samples=num_train, seed=seed)
    
    p(f"Training FM with {num_train} samples...")
    # Use baseline training settings for FM to isolate solver performance
    fm_model = train_fm(X_train, y_train, d=N, k=2, epochs=200, lr=0.1, seed=seed)
    
    X_test, y_test = generate_initial_dataset(Q_bb, G, num_samples=1000, seed=seed+1)
    acc = evaluate_fm_accuracy(fm_model, X_test, y_test)
    p(f"FM Accuracy -> Spearman: {acc['spearman_rho']:.4f}, Top-5% Recall: {acc['top5_recall']:.4f}")
    
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
        "Classical-SA (Penalty, lambda=5)": lambda: solve_classical_sa(Q_fm, offset_fm, G, lambda_val=5.0, num_reads=500, seed=seed),
        "Penalty-QAOA (lambda=5)": lambda: solve_penalty_qaoa(Q_fm, offset_fm, G, lambda_val=5.0, num_samples=500, seed=seed),
        "XY-QAOA (p=3)": lambda: solve_xy_fmqa_p_layers(Q_fm, offset_fm, G, p=3, num_samples=500, seed=seed)
    }
    
    for name, solver_func in solvers.items():
        p(f"  Running {name}...")
        try:
            res = solver_func()
            raw_feas_rate = res["raw_feasible_rate"]
            best_true_energy = float('inf')
            feasible_samples_found = 0
            for sample in res["samples"]:
                if sample["feasible"]:
                    feasible_samples_found += 1
                    x_arr = np.array(sample["x"], dtype=np.float32)
                    true_y = evaluate_bb(Q_bb, x_arr)
                    if true_y < best_true_energy:
                        best_true_energy = true_y
            
            if feasible_samples_found > 0:
                regret = (best_true_energy - f_opt) / (f_worst - f_opt) if f_worst > f_opt else 0.0
            else:
                regret = 1.0
                
            p(f"    -> Feasible Rate: {raw_feas_rate*100:.1f}%, Best Regret: {regret:.4f}")
        except Exception as e:
            p(f"    -> Failed: {e}")

if __name__ == "__main__":
    out_file = "experiments/OPEN_LOOP_PERFECT_FM_TEST/results.md"
    with open(out_file, "w") as f:
        run_open_loop_test(N=21, problem_type="bb1", num_train=1000, seed=42, file=f)
        run_open_loop_test(N=21, problem_type="bb2", num_train=1000, seed=42, file=f)
        run_open_loop_test(N=24, problem_type="bb1", num_train=2000, seed=42, file=f)
        run_open_loop_test(N=24, problem_type="bb2", num_train=2000, seed=42, file=f)
