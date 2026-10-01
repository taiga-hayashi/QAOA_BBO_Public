import os
import numpy as np
import pandas as pd

from modules.bb_functions import get_bb_qubo, get_exact_ground_truth, evaluate_bb
from modules.fm import train_fm_model, fm_to_qubo
from modules.sa import solve_classical_sa
from modules.qaoa import solve_penalty_qaoa
from modules.xy_qaoa import solve_xy_fmqa_p_layers

def generate_initial_dataset(Q_bb, G, num_samples, seed):
    """
    指定されたデータ数だけ One-hot 制約を満たすランダムなビット列を生成し、
    真のブラックボックス関数(Q_bb)でエネルギーを評価して教師データとする。
    """
    rng = np.random.default_rng(seed)
    X = []
    for _ in range(num_samples):
        x = np.zeros(3 * G)
        for g in range(G):
            x[3 * g + rng.integers(0, 3)] = 1.0
        X.append(x)
    X = np.array(X)
    
    y = np.array([float(val @ Q_bb @ val) for val in X])
    return X, y

def run_n200_test(N: int, problem_type="bb1", out_dir="plots/n200_test"):
    """
    200件の限られた初期データで学習したFMに対して各ソルバーを実行し、
    限られた近似精度の中での「制約充足率」と「最適化性能」を比較する。
    """
    print(f"Running N=200 Open Loop Test for {problem_type.upper()} N={N}")
    G = N // 3
    
    # 1. 評価対象の真の関数と最適解の取得
    Q_bb = get_bb_qubo(problem_type, N, seed=42)
    f_opt, f_worst, _, _ = get_exact_ground_truth(Q_bb, G)
    
    seeds = [42, 101, 2024]
    results = []
    
    for seed in seeds:
        # 2. サロゲートモデル(FM)の構築
        X_train, y_train = generate_initial_dataset(Q_bb, G, 200, seed)
        # N=200で過学習を防ぐため、Huber+L2正則化(Variant手法)を用いる
        fm_model = train_fm_model(X_train, y_train, d=N, epochs=200, weight_decay=1e-2, use_huber=True, seed=seed)
        Q_fm, offset_fm = fm_to_qubo(fm_model)
        
        # 3. 古典ソルバー (SA) の実行
        res_sa = solve_classical_sa(Q_fm, offset_fm, G, lambda_val=5.0, seed=seed)
        b_sa = float('inf')
        for s in res_sa["samples"]:
            if s["feasible"]: b_sa = min(b_sa, evaluate_bb(Q_bb, s["x"]))
        sa_reg = (b_sa - f_opt) / (f_worst - f_opt) if res_sa["raw_feasible_rate"] > 0 else 1.0
        results.append({"Solver": "Classical-SA", "Seed": seed, "Feasible": res_sa["raw_feasible_rate"]*100, "Regret": sa_reg})
        
        # 4. 量子ソルバー (Penalty-QAOA) の実行
        try:
            res_qa = solve_penalty_qaoa(Q_fm, offset_fm, G, lambda_val=5.0, seed=seed)
            b_qa = float('inf')
            for s in res_qa["samples"]:
                if s["feasible"]: b_qa = min(b_qa, evaluate_bb(Q_bb, s["x"]))
            qa_reg = (b_qa - f_opt) / (f_worst - f_opt) if res_qa["raw_feasible_rate"] > 0 else 1.0
            results.append({"Solver": "Penalty-QAOA", "Seed": seed, "Feasible": res_qa["raw_feasible_rate"]*100, "Regret": qa_reg})
        except Exception:
            results.append({"Solver": "Penalty-QAOA", "Seed": seed, "Feasible": 0.0, "Regret": 1.0})
            
        # 5. 量子ソルバー (XY-QAOA) の実行
        try:
            res_xy = solve_xy_fmqa_p_layers(Q_fm, offset_fm, G, p=1, seed=seed)
            b_xy = float('inf')
            for s in res_xy["samples"]:
                if s["feasible"]: b_xy = min(b_xy, evaluate_bb(Q_bb, s["x"]))
            xy_reg = (b_xy - f_opt) / (f_worst - f_opt) if res_xy["raw_feasible_rate"] > 0 else 1.0
            results.append({"Solver": "XY-QAOA", "Seed": seed, "Feasible": res_xy["raw_feasible_rate"]*100, "Regret": xy_reg})
        except Exception:
            results.append({"Solver": "XY-QAOA", "Seed": seed, "Feasible": 0.0, "Regret": 1.0})
            
    # 6. 全シードの結果を集計してCSVに出力
    df = pd.DataFrame(results)
    summary = df.groupby("Solver").mean(numeric_only=True).reset_index().drop(columns=["Seed"])
    
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(f"{out_dir}/n200_test_{problem_type}_N{N}.csv", index=False)
    print(summary)

if __name__ == "__main__":
    run_n200_test(18, "bb1")
