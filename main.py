import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from fm import TorchFM
from fm_to_qubo import fm_to_qubo
from qaoa_solver import solve_qubo_qaoa

# 1. ブラックボックス(BB)関数としてのランダムQUBOの定義
def create_random_qubo_bb(d: int, seed: int = 42) -> np.ndarray:
    """d次元のランダムQUBO行列（上三角）を生成する"""
    np.random.seed(seed)
    Q = np.random.uniform(-1, 1, size=(d, d))
    # 上三角行列にする
    Q = np.triu(Q)
    return Q

def evaluate_bb(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """バイナリ入力 x に対してBB関数 (x^T Q x) の値を計算する"""
    # x: (N, d), Q: (d, d)
    return np.einsum('ni,ij,nj->n', x, Q, x)

# 2. データセットの生成
def generate_dataset(Q: np.ndarray, num_samples: int, d: int, seed: int = 42):
    """ランダムなバイナリ入力を生成し、BB関数でラベル付けしたデータセットを作成"""
    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(num_samples, d)).astype(np.float32)
    y = evaluate_bb(X, Q).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

# 古典的な全探索（厳密解）を求める関数
def get_exact_minimum_bb(Q: np.ndarray, d: int):
    """BB関数の真の最小値とその入力を全探索で求める"""
    all_patterns = np.array(list(itertools.product([0, 1], repeat=d)), dtype=np.float32)
    y_all = evaluate_bb(all_patterns, Q)
    min_idx = np.argmin(y_all)
    return all_patterns[min_idx], y_all[min_idx]

def get_fm_minimum(model: TorchFM, d: int):
    """FMサロゲートモデル上の最小値とその入力を全探索で求める"""
    all_patterns = list(itertools.product([0, 1], repeat=d))
    x_tensor = torch.tensor(all_patterns, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x_tensor)
    min_idx = torch.argmin(outputs).item()
    return all_patterns[min_idx], outputs[min_idx].item()

def main():
    # シードの固定
    torch.manual_seed(42)
    np.random.seed(42)
    
    # パラメータ設定
    d = 10           # 変数の数 (次元数)
    k = 4            # FMの相互作用ベクトルの次元数
    num_samples = 200 # FMの学習用データ数
    
    print("=== 1. ブラックボックス関数(ランダムQUBO)の定義とデータ生成 ===")
    Q_bb = create_random_qubo_bb(d)
    X_train, y_train = generate_dataset(Q_bb, num_samples, d)
    print(f"BB関数次元: {d}, 訓練データ数: {num_samples}")
    
    print("\n=== 2. FMモデルの学習 ===")
    model = TorchFM(d=d, k=k)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    epochs = 150
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | MSE Loss: {loss.item():.4f}")
            
    print("\n=== 3. 学習済みFMからQUBOへの変換 ===")
    qubo_dict, offset = fm_to_qubo(model)
    print(f"FM QUBO offset (bias): {offset:.4f}")
    
    print("\n=== 4. 厳密解の計算 (全探索 2^d) ===")
    best_x_bb, min_val_bb = get_exact_minimum_bb(Q_bb, d)
    print(f"[BB関数の真の最小値]  Val: {min_val_bb:.4f}, x: {tuple(int(xi) for xi in best_x_bb)}")
    
    best_x_fm, min_val_fm = get_fm_minimum(model, d)
    print(f"[FM上の全探索最小値]  Val: {min_val_fm:.4f}, x: {best_x_fm}")
    
    print("\n=== 5. QAOAによる最適化 (FMサロゲートのQUBOを解く) ===")
    # 実行時間短縮のため reps=1, maxiter=50 としています。
    # 精度を上げる場合は reps を増やしますが、実行時間がかかります。
    result = solve_qubo_qaoa(qubo_dict, offset=offset, reps=1, maxiter=50)
    
    print("\n=== 6. 結果検証 ===")
    qaoa_x_tuple = tuple(int(xi) for xi in result.x)
    print(f"QAOA Approximate Minimum (on FM): {result.fval:.4f}")
    print(f"QAOA Optimal variables        : {qaoa_x_tuple}")
    
    # QAOAが出した「FM上での最適解」を、真のBB関数に入れた時の評価値
    qaoa_x_array = np.array(qaoa_x_tuple, dtype=np.float32).reshape(1, -1)
    qaoa_bb_val = evaluate_bb(qaoa_x_array, Q_bb)[0]
    print(f"-> QAOA解を真のBB関数で評価した値 : {qaoa_bb_val:.4f}")
    
    # QAOAのサンプリング結果トップ3
    print("\nTop 3 QAOA Samples (on FM):")
    for i, sample in enumerate(result.samples[:3]):
        samp_x_tuple = tuple(int(xi) for xi in sample.x)
        samp_x_array = np.array(samp_x_tuple, dtype=np.float32).reshape(1, -1)
        bb_val = evaluate_bb(samp_x_array, Q_bb)[0]
        print(f"Rank {i+1}: x={samp_x_tuple}, FM_val={sample.fval:.4f} (BB_val={bb_val:.4f}), prob={sample.probability:.4f}")

if __name__ == "__main__":
    main()
