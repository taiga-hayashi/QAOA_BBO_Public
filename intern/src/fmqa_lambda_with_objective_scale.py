import json
import numpy as np
from bb_function import create_random_qubo_bb, get_exact_minimum_categorical_bb, is_feasible_one_hot
from fmqa_solver import solve_fmqa
from qubo import matrix_to_qubo_dict

# N=12 (4x3, 64 states)
blocks = [4, 4, 4]
N = 12
seeds = [42, 101, 2024]
lambda_list = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
scale_factors = [0.5, 1.0, 3.0, 10.0]

results = {}

for s_fac in scale_factors:
    results[str(s_fac)] = []
    print(f"\n=== Objective Scale Factor: {s_fac}x ===")
    for lam in lambda_list:
        feas_list, opt_list = [], []
        for s in seeds:
            Q = create_random_qubo_bb(N, seed=s) * s_fac
            q_dict = matrix_to_qubo_dict(Q)
            best_x, _ = get_exact_minimum_categorical_bb(Q, blocks)
            best_x_str = "".join(str(int(b)) for b in best_x)
            
            rf = solve_fmqa(q_dict, blocks, lambda_penalty=lam, num_reads=1000, seed=s)
            feas_list.append(rf["feasibility_rate"])
            opt_list.append(rf["state_probabilities"].get(best_x_str, 0.0))
            
        rec = {
            "lambda": lam,
            "feas_mean": float(np.mean(feas_list)),
            "opt_mean": float(np.mean(opt_list))
        }
        results[str(s_fac)].append(rec)
        print(f"  lam={lam:4.1f} | Feas={rec['feas_mean']*100:5.1f}% | OptProb={rec['opt_mean']*100:5.2f}%")

with open("result/json/fmqa_lambda_vs_obj_scale.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved result/json/fmqa_lambda_vs_obj_scale.json")
