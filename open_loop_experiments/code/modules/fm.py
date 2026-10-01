import torch
import torch.nn as nn
import numpy as np

class TorchFM(nn.Module):
    """PyTorch implementation of Factorization Machines (FM)."""
    def __init__(self, d: int, k: int = 2):
        super().__init__()
        self.d = d
        self.k = k
        self.V = nn.Parameter(torch.randn(d, k))
        self.lin = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1_inner = x @ self.V
        term1 = torch.sum(term1_inner * term1_inner, dim=1, keepdim=True)
        term2_inner = (x ** 2) @ (self.V ** 2)
        term2 = torch.sum(term2_inner, dim=1, keepdim=True)
        interaction = 0.5 * (term1 - term2)
        return (interaction + self.lin(x)).view(-1)

def train_fm_model(X: np.ndarray, y: np.ndarray, d: int, k: int = 2, epochs: int = 120, lr: float = 0.1, weight_decay: float = 0.0, use_huber: bool = False, seed: int = 42) -> TorchFM:
    """Train FM with optional Huber+L2 regularization."""
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

def fm_to_qubo(fm_model: TorchFM):
    """Extract QUBO matrix and offset from trained FM."""
    W = fm_model.V.detach().numpy()
    w_0 = fm_model.lin.bias.detach().numpy()[0]
    w_i = fm_model.lin.weight.detach().numpy()[0]
    N = fm_model.d
    
    Q_fm = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        Q_fm[i, i] = w_i[i]
        for j in range(i + 1, N):
            Q_fm[i, j] = np.dot(W[i], W[j])
            
    return Q_fm, w_0
