import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../intern_0924")))
from plot_fm_parity import run_parity_experiment
from plot_fm_learning_curve import run_learning_curve
from plot_lambda_tuning_large import run_tuning_for_problem, plot_results

print("--- Starting N=27 Experiments ---")

# 1. Parity Plot
print("1. Parity Plot N=27")
run_parity_experiment(27, "bb1", seed=42)
run_parity_experiment(27, "bb2", seed=42)

# 2. Learning Curve
print("\n2. Learning Curve N=27")
run_learning_curve(27, "bb1", seed=42)
run_learning_curve(27, "bb2", seed=42)

# 3. Lambda Tuning
print("\n3. Lambda Tuning N=27")
seeds_to_run = [42, 101, 2024]
num_train_27 = 3000

agg1, xy_feas1, xy_regret1 = run_tuning_for_problem(27, "bb1", num_train=num_train_27, seeds=seeds_to_run)
plot_results(agg1, xy_feas1, xy_regret1, "BB1 N=27 (3 seeds mean, Up to L=5000)", "../results_1001/open_loop/png/lambda_tuning_large_bb1_N27")

agg2, xy_feas2, xy_regret2 = run_tuning_for_problem(27, "bb2", num_train=num_train_27, seeds=seeds_to_run)
plot_results(agg2, xy_feas2, xy_regret2, "BB2 N=27 (3 seeds mean, Up to L=5000)", "../results_1001/open_loop/png/lambda_tuning_large_bb2_N27")

print("All N=27 tasks completed!")
