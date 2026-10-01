import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../intern_0924")))
from src.problem import get_bb_qubo, generate_initial_dataset, evaluate_bb, get_exact_ground_truth
from plot_fm_learning_curve import train_fm_custom
from src.solvers import solve_xy_fmqa_p_layers

def run_xy_p_layers(N, problem_type="bb1", num_train=1000, seeds=[42, 101, 2024]):
    layers = [1, 2, 3]
    G = N // 3
    
    results = {
        "p": layers,
        "Feasible_Rate": {1: [], 2: [], 3: []},
        "Regret": {1: [], 2: [], 3: []}
    }
    
    for seed in seeds:
        print(f"Running {problem_type.upper()} N={N} Seed={seed}...")
        Q_bb = get_bb_qubo(problem_type, N, seed=seed)
        f_opt, f_worst, _, _ = get_exact_ground_truth(Q_bb, G)
        
        # Train a good FM to isolate solver performance
        X_train, y_train = generate_initial_dataset(Q_bb, G, num_samples=num_train, seed=seed)
        fm_model = train_fm_custom(X_train, y_train, d=N, k=2, epochs=200, lr=0.1, weight_decay=1e-2, use_huber=True, seed=seed)
        
        W = fm_model.V.detach().numpy()
        w_0 = fm_model.lin.bias.detach().numpy()[0]
        w_i = fm_model.lin.weight.detach().numpy()[0]
        Q_fm = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            Q_fm[i, i] = w_i[i]
            for j in range(i + 1, N):
                Q_fm[i, j] = np.dot(W[i], W[j])
        offset_fm = w_0
        
        for p in layers:
            print(f"  Evaluating XY-QAOA p={p}...")
            try:
                res = solve_xy_fmqa_p_layers(Q_fm, offset_fm, G, p=p, num_samples=500, seed=seed)
                feas = res["raw_feasible_rate"]
                best_y = float('inf')
                for sample in res["samples"]:
                    if sample["feasible"]:
                        best_y = min(best_y, evaluate_bb(Q_bb, np.array(sample["x"])))
                regret = (best_y - f_opt) / (f_worst - f_opt) if feas > 0 else 1.0
            except Exception as e:
                feas = 0.0
                regret = 1.0
                
            results["Feasible_Rate"][p].append(feas)
            results["Regret"][p].append(regret)
            
    # Calculate means
    mean_feas = [np.mean(results["Feasible_Rate"][p]) * 100 for p in layers]
    mean_regret = [np.mean(results["Regret"][p]) for p in layers]
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Feasible Rate (Should be 100%)
    ax1.bar(layers, mean_feas, color='green', alpha=0.7)
    ax1.set_title(f"Feasible Rate vs Depth p ({problem_type.upper()} N={N})")
    ax1.set_xlabel("QAOA Depth (p)")
    ax1.set_ylabel("Feasible Rate (%)")
    ax1.set_xticks(layers)
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Regret
    ax2.plot(layers, mean_regret, marker='o', color='purple', linewidth=2, markersize=8)
    ax2.set_title(f"Best Regret vs Depth p ({problem_type.upper()} N={N})")
    ax2.set_xlabel("QAOA Depth (p)")
    ax2.set_ylabel("Normalized Regret (Lower is better)")
    ax2.set_xticks(layers)
    ax2.set_ylim(-0.02, max(0.1, max(mean_regret) * 1.2))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_dir = "../results_1001/open_loop/png"
    pdf_dir = "../results_1001/open_loop/pdf"
    plt.savefig(f"{out_dir}/xy_p_layers_{problem_type}_N{N}.png", dpi=150)
    plt.savefig(f"{pdf_dir}/xy_p_layers_{problem_type}_N{N}.pdf")
    plt.close()

if __name__ == "__main__":
    for prob in ["bb1", "bb2"]:
        run_xy_p_layers(18, prob, num_train=500)
        run_xy_p_layers(24, prob, num_train=2000)
    print("Done plotting p-layers.")
