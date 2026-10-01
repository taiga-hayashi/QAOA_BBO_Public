import numpy as np
import neal

def solve_classical_sa(Q_fm: np.ndarray, offset_fm: float, G: int, lambda_val: float, num_reads: int = 500, seed: int = 42) -> dict:
    """Solve penalized QUBO using D-Wave Neal Simulated Annealing."""
    from modules.qubo import apply_one_hot_penalty
    
    Q_pen, off_pen = apply_one_hot_penalty(Q_fm, offset_fm, G, lambda_val)
    
    N = 3 * G
    Q_dict = {(i, j): Q_pen[i, j] for i in range(N) for j in range(i, N)}
    
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q_dict, num_reads=num_reads, seed=seed)
    
    samples = []
    feas_count = 0
    
    for sample, energy, num_occ in response.data(['sample', 'energy', 'num_occurrences']):
        x = np.array([sample[i] for i in range(N)])
        
        feasible = True
        for g in range(G):
            if np.sum(x[3*g:3*g+3]) != 1.0:
                feasible = False
                break
                
        if feasible:
            feas_count += num_occ
            
        for _ in range(num_occ):
            samples.append({
                "x": x,
                "feasible": feasible,
                "penalized_val": energy + off_pen
            })
            
    return {
        "raw_feasible_rate": feas_count / num_reads,
        "samples": samples
    }
