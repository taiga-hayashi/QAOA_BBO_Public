import sys
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def run_learning_curve(N, problem_type="bb1", seed=42):
    print(f"Running Learning Curve for {problem_type.upper()} N={N} Seed={seed}...")
    G = N // 3
    Q_bb = get_bb_qubo(problem_type, N, seed=seed)
    
    # Pre-generate a large fixed test set (Random MSE)
    X_test, y_test = generate_initial_dataset(Q_bb, G, num_samples=2000, seed=seed+100)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    train_sizes = [20, 50, 100, 200, 300, 500, 750, 1000]
    
    results = {
        "Baseline": {"train_mse": [], "test_mse": []},
        "Variant": {"train_mse": [], "test_mse": []}
    }
    
    for n_train in train_sizes:
        print(f"  Training with N_train = {n_train}")
        X_train, y_train = generate_initial_dataset(Q_bb, G, num_samples=n_train, seed=seed)
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        
        # 1. Baseline
        fm_base = train_fm_custom(X_train, y_train, d=N, k=2, epochs=120, lr=0.1, weight_decay=0.0, use_huber=False, seed=seed)
        with torch.no_grad():
            pred_train_base = fm_base(x_train_tensor).numpy()
            pred_test_base = fm_base(x_test_tensor).numpy()
        results["Baseline"]["train_mse"].append(np.mean((pred_train_base - y_train)**2))
        results["Baseline"]["test_mse"].append(np.mean((pred_test_base - y_test)**2))
        
        # 2. Variant (Huber + L2)
        fm_var = train_fm_custom(X_train, y_train, d=N, k=2, epochs=200, lr=0.1, weight_decay=1e-2, use_huber=True, seed=seed)
        with torch.no_grad():
            pred_train_var = fm_var(x_train_tensor).numpy()
            pred_test_var = fm_var(x_test_tensor).numpy()
        results["Variant"]["train_mse"].append(np.mean((pred_train_var - y_train)**2))
        results["Variant"]["test_mse"].append(np.mean((pred_test_var - y_test)**2))
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Baseline
    ax1.plot(train_sizes, results["Baseline"]["train_mse"], marker='o', label='Train MSE', color='blue')
    ax1.plot(train_sizes, results["Baseline"]["test_mse"], marker='s', label='Test (Random) MSE', color='red')
    ax1.set_title(f"Baseline FM (MSE, No L2, Ep 120)")
    ax1.set_xlabel('Number of Initial Data Points')
    ax1.set_ylabel('Mean Squared Error (Log Scale)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Variant
    ax2.plot(train_sizes, results["Variant"]["train_mse"], marker='o', label='Train MSE', color='blue')
    ax2.plot(train_sizes, results["Variant"]["test_mse"], marker='s', label='Test (Random) MSE', color='red')
    ax2.set_title(f"Variant FM (Huber, L2=1e-2, Ep 200)")
    ax2.set_xlabel('Number of Initial Data Points')
    ax2.set_ylabel('Mean Squared Error (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(f"FM Learning Curve: {problem_type.upper()} N={N}", fontsize=14)
    plt.tight_layout()
    
    out_dir = "../results_1001/open_loop/png"
    pdf_dir = "../results_1001/open_loop/pdf"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    plt.savefig(f"{out_dir}/fm_learning_curve_{problem_type}_N{N}.png", dpi=150)
    plt.savefig(f"{pdf_dir}/fm_learning_curve_{problem_type}_N{N}.pdf")
    plt.close()
    print("Done.")

if __name__ == "__main__":
    for prob in ["bb1", "bb2"]:
        run_learning_curve(N=18, problem_type=prob, seed=42)
        run_learning_curve(N=24, problem_type=prob, seed=42)
