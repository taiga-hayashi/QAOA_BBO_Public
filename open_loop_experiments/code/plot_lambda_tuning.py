import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.bb_functions import get_bb_qubo, get_exact_ground_truth, evaluate_bb
from modules.fm import train_fm_model, fm_to_qubo
from modules.sa import solve_classical_sa
from modules.qaoa import solve_penalty_qaoa
from modules.xy_qaoa import solve_xy_fmqa_p_layers

def generate_initial_dataset(Q_bb: np.ndarray, G: int, num_samples: int, seed: int):
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

def run_lambda_tuning(N: int, problem_type="bb1", out_dir="../open_loop/2_lambda_tuning"):
    """
    ペナルティ係数(lambda)を変化させながら各ソルバーを実行し、
    制約充足率(Feasible Rate)と最適化性能(Regret)の推移をプロットする。
    """
    print(f"Generating Lambda Tuning Plot for {problem_type.upper()} N={N}")
    G = N // 3
    
    # 1. 評価対象の真のブラックボックス関数を取得し、正規化基準となる最適解/最悪解を計算
    Q_bb = get_bb_qubo(problem_type, N, seed=42)
    f_opt, f_worst, _, _ = get_exact_ground_truth(Q_bb, G)
    
    # 2. ソルバー比較のため、近似誤差がボトルネックにならないよう十分なデータ数(500件)でFMを学習
    X_train, y_train = generate_initial_dataset(Q_bb, G, 500, 42)
    fm_model = train_fm_model(X_train, y_train, d=N, epochs=200, use_huber=False)
    Q_fm, offset_fm = fm_to_qubo(fm_model)
    
    # 検証するペナルティ係数 lambda のリスト
    lambdas = [0.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    sa_feas, sa_reg = [], []
    qaoa_feas, qaoa_reg = [], []
    
    for l in lambdas:
        # 3. 古典ソルバー (SA) による最適化
        res_sa = solve_classical_sa(Q_fm, offset_fm, G, l, num_reads=100)
        sa_feas.append(res_sa["raw_feasible_rate"] * 100)
        
        b_sa = float('inf')
        for s in res_sa["samples"]:
            if s["feasible"]:
                b_sa = min(b_sa, evaluate_bb(Q_bb, s["x"]))
        # 制約を満たす解がない場合は Regret を最悪値 1.0 に設定する
        r_sa = (b_sa - f_opt) / (f_worst - f_opt) if res_sa["raw_feasible_rate"] > 0 else 1.0
        sa_reg.append(r_sa)
        
        # 4. 量子ソルバー (Penalty-QAOA) による最適化
        try:
            res_qa = solve_penalty_qaoa(Q_fm, offset_fm, G, l, num_samples=100)
            qaoa_feas.append(res_qa["raw_feasible_rate"] * 100)
            b_qa = float('inf')
            for s in res_qa["samples"]:
                if s["feasible"]:
                    b_qa = min(b_qa, evaluate_bb(Q_bb, s["x"]))
            r_qa = (b_qa - f_opt) / (f_worst - f_opt) if res_qa["raw_feasible_rate"] > 0 else 1.0
            qaoa_reg.append(r_qa)
        except Exception:
            qaoa_feas.append(0.0)
            qaoa_reg.append(1.0)
            
    # 5. ペナルティフリーの XY-QAOA (lambdaに依存しないため1度だけ計算)
    try:
        res_xy = solve_xy_fmqa_p_layers(Q_fm, offset_fm, G, p=1, num_samples=100)
        xy_f = res_xy["raw_feasible_rate"] * 100
        b_xy = float('inf')
        for s in res_xy["samples"]:
            if s["feasible"]:
                b_xy = min(b_xy, evaluate_bb(Q_bb, s["x"]))
        xy_r = (b_xy - f_opt) / (f_worst - f_opt) if res_xy["raw_feasible_rate"] > 0 else 1.0
    except Exception:
        xy_f, xy_r = 0.0, 1.0
        
    # 6. プロットの描画と保存
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左図: Feasible Rate (制約充足率)
    ax1.plot(lambdas, sa_feas, marker='o', label='Classical-SA (Penalty)')
    ax1.plot(lambdas, qaoa_feas, marker='s', label='Penalty-QAOA (OpenQARP)')
    ax1.axhline(y=xy_f, color='green', linestyle='--', label='XY-QAOA (No Penalty)')
    ax1.set_title(f'Feasible Rate vs Lambda ({problem_type.upper()} N={N})')
    ax1.set_xlabel('Penalty Coefficient $\lambda$ (Log Scale)')
    ax1.set_ylabel('Feasible Rate (%)')
    ax1.set_xscale('symlog', linthresh=10.0)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 右図: Normalized Regret (最適化性能)
    ax2.plot(lambdas, sa_reg, marker='o', label='Classical-SA (Penalty)')
    ax2.plot(lambdas, qaoa_reg, marker='s', label='Penalty-QAOA (OpenQARP)')
    ax2.axhline(y=xy_r, color='green', linestyle='--', label='XY-QAOA (No Penalty)')
    ax2.set_title(f'Normalized Regret vs Lambda ({problem_type.upper()} N={N})')
    ax2.set_xlabel('Penalty Coefficient $\lambda$ (Log Scale)')
    ax2.set_ylabel('Normalized Regret (0.0=Optimal, 1.0=Worst/Infeasible)')
    ax2.set_xscale('symlog', linthresh=10.0)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs(f"{out_dir}/png", exist_ok=True)
    os.makedirs(f"{out_dir}/pdf", exist_ok=True)
    plt.savefig(f"{out_dir}/png/lambda_tuning_{problem_type}_N{N}.png", dpi=150)
    plt.savefig(f"{out_dir}/pdf/lambda_tuning_{problem_type}_N{N}.pdf")
    plt.close()

if __name__ == "__main__":
    run_lambda_tuning(18, "bb1")
