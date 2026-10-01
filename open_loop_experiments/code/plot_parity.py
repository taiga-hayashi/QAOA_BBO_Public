import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
import torch

from modules.bb_functions import get_bb_qubo
from modules.fm import train_fm_model

def generate_initial_dataset(Q_bb, G, num_samples, seed):
    """Generate one-hot encoded random samples and evaluate their BB energies."""
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

def plot_parity_ax(ax, y_true_train, y_pred_train, y_true_test, y_pred_test, title):
    """Helper to draw a parity plot on a given matplotlib axis."""
    mse_test = np.mean((y_true_test - y_pred_test)**2)
    rho, _ = stats.spearmanr(y_true_test, y_pred_test)
    
    ax.scatter(y_true_test, y_pred_test, color='gray', alpha=0.5, s=15, label='Test (Unseen)')
    ax.scatter(y_true_train, y_pred_train, color='red', alpha=0.9, s=40, marker='x', label='Train (20 pts)')
    
    min_val = min(np.min(y_true_test), np.min(y_pred_test))
    max_val = max(np.max(y_true_test), np.max(y_pred_test))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='y = x')
    
    ax.set_title(f"{title}\nMSE: {mse_test:.3f} | Spearman: {rho:.3f}")
    ax.set_xlabel('True BB Energy (Ground Truth)')
    ax.set_ylabel('Predicted FM Energy (Surrogate)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

def run_parity(N: int, problem_type="bb1", out_dir="../open_loop/3_fm_parity"):
    """Plot the parity of the FM predictions to visualize overfitting mitigation."""
    print(f"Generating Parity Plot for {problem_type.upper()} N={N}")
    G = N // 3
    Q_bb = get_bb_qubo(problem_type, N, seed=42)
    
    X_train, y_train = generate_initial_dataset(Q_bb, G, 20, 42)
    X_test, y_test = generate_initial_dataset(Q_bb, G, 2000, 142)
    
    # [A] Baseline FM
    fm_base = train_fm_model(X_train, y_train, d=N, epochs=120, use_huber=False)
    with torch.no_grad():
        y_pred_train_base = fm_base(torch.tensor(X_train, dtype=torch.float32)).numpy()
        y_pred_test_base = fm_base(torch.tensor(X_test, dtype=torch.float32)).numpy()
        
    # [B] Variant FM (Huber + L2)
    fm_var = train_fm_model(X_train, y_train, d=N, epochs=200, weight_decay=1e-2, use_huber=True)
    with torch.no_grad():
        y_pred_train_var = fm_var(torch.tensor(X_train, dtype=torch.float32)).numpy()
        y_pred_test_var = fm_var(torch.tensor(X_test, dtype=torch.float32)).numpy()
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_parity_ax(ax1, y_train, y_pred_train_base, y_test, y_pred_test_base, "Baseline (MSE, Ep 120)")
    plot_parity_ax(ax2, y_train, y_pred_train_var, y_test, y_pred_test_var, "Variant (Huber+L2, Ep 200)")
    
    plt.suptitle(f"FM Parity Plot: {problem_type.upper()} N={N}", fontsize=14)
    plt.tight_layout()
    
    os.makedirs(f"{out_dir}/png", exist_ok=True)
    os.makedirs(f"{out_dir}/pdf", exist_ok=True)
    plt.savefig(f"{out_dir}/png/fm_parity_{problem_type}_N{N}.png", dpi=150)
    plt.savefig(f"{out_dir}/pdf/fm_parity_{problem_type}_N{N}.pdf")
    plt.close()

if __name__ == "__main__":
    run_parity(18, "bb1")
