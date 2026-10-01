import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from modules.bb_functions import get_bb_qubo
from modules.fm import train_fm_model

def generate_initial_dataset(Q_bb, G, num_samples, seed):
    """One-hot制約を満たすランダムビット列を生成し、真のBB関数で評価する"""
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

def run_learning_curve(N: int, problem_type="bb1", out_dir="plots/learning_curve"):
    """
    初期データ数を徐々に増やしながらFMを学習し、
    学習データ(Train)と未知データ(Test)に対する予測誤差(MSE)の推移をプロットする。
    """
    print(f"Running Learning Curve for {problem_type.upper()} N={N}")
    G = N // 3
    
    # 1. 真の関数の取得
    Q_bb = get_bb_qubo(problem_type, N, seed=42)
    
    # 2. 汎化性能を評価するための大規模な共通テストセット(2000件)を生成
    X_test, y_test = generate_initial_dataset(Q_bb, G, 2000, 142)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    train_sizes = [20, 50, 100, 200, 300, 500, 750, 1000]
    res_base = {"train": [], "test": []}
    res_var = {"train": [], "test": []}
    
    for n_train in train_sizes:
        # 3. 指定したデータ数で学習データを生成
        X_train, y_train = generate_initial_dataset(Q_bb, G, n_train, 42)
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        
        # 4. ベースラインFM (MSELoss) の学習と誤差評価
        fm_base = train_fm_model(X_train, y_train, d=N, epochs=120, use_huber=False)
        with torch.no_grad():
            res_base["train"].append(np.mean((fm_base(x_train_tensor).numpy() - y_train)**2))
            res_base["test"].append(np.mean((fm_base(x_test_tensor).numpy() - y_test)**2))
            
        # 5. 提案手法FM (Huber+L2) の学習と誤差評価
        fm_var = train_fm_model(X_train, y_train, d=N, epochs=200, weight_decay=1e-2, use_huber=True)
        with torch.no_grad():
            res_var["train"].append(np.mean((fm_var(x_train_tensor).numpy() - y_train)**2))
            res_var["test"].append(np.mean((fm_var(x_test_tensor).numpy() - y_test)**2))
            
    # 6. 学習曲線の描画と保存
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(train_sizes, res_base["train"], marker='o', label='Train MSE')
    ax1.plot(train_sizes, res_base["test"], marker='s', label='Test MSE')
    ax1.set_yscale('log')
    ax1.set_title(f"Baseline FM (MSE, Ep 120)")
    ax1.set_xlabel("Number of Training Samples")
    ax1.set_ylabel("Mean Squared Error (Log Scale)")
    ax1.legend()
    
    ax2.plot(train_sizes, res_var["train"], marker='o', label='Train MSE')
    ax2.plot(train_sizes, res_var["test"], marker='s', label='Test MSE')
    ax2.set_yscale('log')
    ax2.set_title(f"Variant FM (Huber+L2, Ep 200)")
    ax2.set_xlabel("Number of Training Samples")
    ax2.legend()
    
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/fm_learning_curve_{problem_type}_N{N}.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_learning_curve(18, "bb1")
