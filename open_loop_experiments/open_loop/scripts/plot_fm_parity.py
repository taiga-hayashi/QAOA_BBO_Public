import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../intern_0924")))
from src.problem import get_bb_qubo, generate_initial_dataset
from src.fm import TorchFM

def train_fm_custom(X, y, d, k=2, epochs=120, lr=0.1, weight_decay=0.0, use_huber=False, seed=42):
    torch.manual_seed(seed)
    model = TorchFM(d=d, k=k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.HuberLoss(delta=1.0) if use_huber else nn.MSELoss()

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def plot_parity(ax, y_true_train, y_pred_train, y_true_test, y_pred_test, title):
    mse_test = np.mean((y_true_test - y_pred_test)**2)
    rho, _ = stats.spearmanr(y_true_test, y_pred_test)
    
    ax.scatter(y_true_test, y_pred_test, color='gray', alpha=0.5, s=15, label='Test (Unseen)')
    ax.scatter(y_true_train, y_pred_train, color='red', alpha=0.9, s=40, marker='x', label='Train (20 pts)')
    
    min_val = min(np.min(y_true_test), np.min(y_pred_test))
    max_val = max(np.max(y_true_test), np.max(y_pred_test))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='y = x')
    
    ax.set_title(f"{title}\nMSE: {mse_test:.3f} | Spearman: {rho:.3f}")
    ax.set_xlabel('True BB Energy')
    ax.set_ylabel('Predicted FM Energy')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

def run_parity_experiment(N, problem_type="bb1", seed=42):
    print(f"Running Parity Plot Experiment for {problem_type.upper()} N={N} Seed={seed}...")
    G = N // 3
    Q_bb = get_bb_qubo(problem_type, N, seed=seed)
    
    X_train, y_train = generate_initial_dataset(Q_bb, G, num_samples=20, seed=seed)
    X_test, y_test = generate_initial_dataset(Q_bb, G, num_samples=2000, seed=seed+100)
    
    print("Training Baseline FM...")
    fm_base = train_fm_custom(X_train, y_train, d=N, k=2, epochs=120, lr=0.1, weight_decay=0.0, use_huber=False, seed=seed)
    with torch.no_grad():
        y_pred_train_base = fm_base(torch.tensor(X_train, dtype=torch.float32)).numpy()
        y_pred_test_base = fm_base(torch.tensor(X_test, dtype=torch.float32)).numpy()
        
    print("Training Variant FM...")
    fm_var = train_fm_custom(X_train, y_train, d=N, k=2, epochs=200, lr=0.1, weight_decay=1e-2, use_huber=True, seed=seed)
    with torch.no_grad():
        y_pred_train_var = fm_var(torch.tensor(X_train, dtype=torch.float32)).numpy()
        y_pred_test_var = fm_var(torch.tensor(X_test, dtype=torch.float32)).numpy()
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_parity(ax1, y_train, y_pred_train_base, y_test, y_pred_test_base, "Baseline (MSE Loss, No L2, Ep 120)")
    plot_parity(ax2, y_train, y_pred_train_var, y_test, y_pred_test_var, "Variant (Huber Loss, L2=1e-2, Ep 200)")
    
    plt.suptitle(f"FM Parity Plot: {problem_type.upper()} N={N} (Trained on 20 points)", fontsize=14, y=0.98)
    plt.tight_layout()
    
    out_dir = "../results_1001/open_loop/png"
    pdf_dir = "../results_1001/open_loop/pdf"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    plt.savefig(f"{out_dir}/fm_parity_{problem_type}_N{N}.png", dpi=150)
    plt.savefig(f"{pdf_dir}/fm_parity_{problem_type}_N{N}.pdf")
    plt.close()
    print("Done.")

if __name__ == "__main__":
    for prob in ["bb1", "bb2"]:
        for n in [18, 24]:
            run_parity_experiment(n, prob, seed=42)
