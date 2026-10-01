import numpy as np

def apply_one_hot_penalty(Q: np.ndarray, offset: float, G: int, lambda_val: float):
    """Add one-hot penalty terms to the QUBO matrix."""
    N = 3 * G
    Q_pen = Q.copy()
    offset_pen = offset + lambda_val * G
    
    for g in range(G):
        for i in range(3):
            idx1 = 3 * g + i
            Q_pen[idx1, idx1] -= lambda_val
            for j in range(i + 1, 3):
                idx2 = 3 * g + j
                Q_pen[idx1, idx2] += 2.0 * lambda_val
                
    return Q_pen, offset_pen
