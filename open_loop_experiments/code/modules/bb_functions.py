import numpy as np
from itertools import product

def generate_bb1_matrix(N: int, seed: int = 42) -> np.ndarray:
    """Generate a random asymmetric matrix for the BB1 problem."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-10, 10, size=(N, N))

def generate_bb2_matrix(N: int, seed: int = 42) -> np.ndarray:
    """Generate a symmetric matrix for the BB2 problem."""
    rng = np.random.default_rng(seed + 1)
    matrix = rng.uniform(-10, 10, size=(N, N))
    return (matrix + matrix.T) / 2

def get_bb_qubo(problem_type: str, N: int, seed: int = 42) -> np.ndarray:
    """Get the specific black-box QUBO matrix based on type."""
    if problem_type == "bb1":
        return generate_bb1_matrix(N, seed)
    elif problem_type == "bb2":
        return generate_bb2_matrix(N, seed)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

def evaluate_bb(Q_bb: np.ndarray, x: np.ndarray) -> float:
    """Evaluate true black-box energy for a given bitstring."""
    return float(x @ Q_bb @ x)

def get_exact_ground_truth(Q_bb: np.ndarray, G: int):
    """Exhaustively search the feasible one-hot subspace for exact min/max energies."""
    best_val = float('inf')
    worst_val = float('-inf')
    best_x = None
    
    for choices in product(range(3), repeat=G):
        x = np.zeros(3 * G)
        for g, c in enumerate(choices):
            x[3*g + c] = 1.0
            
        val = evaluate_bb(Q_bb, x)
        if val < best_val:
            best_val = val
            best_x = x
        if val > worst_val:
            worst_val = val
            
    return best_val, worst_val, best_x, None
