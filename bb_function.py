import numpy as np
import torch
import itertools

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

def generate_dataset(Q: np.ndarray, num_samples: int, d: int, seed: int = 42):
    """ランダムなバイナリ入力を生成し、BB関数でラベル付けしたデータセットを作成"""
    np.random.seed(seed)
    X = np.random.randint(0, 2, size=(num_samples, d)).astype(np.float32)
    y = evaluate_bb(X, Q).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

def get_exact_minimum_bb(Q: np.ndarray, d: int):
    """BB関数の真の最小値とその入力を全探索で求める"""
    all_patterns = np.array(list(itertools.product([0, 1], repeat=d)), dtype=np.float32)
    y_all = evaluate_bb(all_patterns, Q)
    min_idx = np.argmin(y_all)
    return all_patterns[min_idx], y_all[min_idx]
