import torch
import torch.nn as nn
import numpy as np
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TorchFM(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(d, k, device=device))
        self.lin = nn.Linear(d, 1).to(device)
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, active_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if active_idx is not None:
            E = self.V[active_idx]  # (B,F,K)
            summed = torch.sum(E, dim=1)
            term1 = torch.sum(summed * summed, dim=1)
            term2 = torch.sum(E * E, dim=(1, 2))
            interaction = 0.5 * (term1 - term2)

            lin_w = self.lin.weight.view(-1)
            linear = lin_w[active_idx].sum(dim=1) + self.lin.bias.view(1)
            return interaction + linear.view(-1)

        term1_inner = x @ self.V
        term1 = torch.sum(term1_inner * term1_inner, dim=1, keepdim=True)
        term2_inner = x.pow(2) @ self.V.pow(2)
        term2 = torch.sum(term2_inner, dim=1, keepdim=True)
        interaction = 0.5 * (term1 - term2)
        return (interaction + self.lin(x)).view(-1)


def train_factorization_machine(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    d: int,
    k: int = 2,
    epochs: int = 120,
    learning_rate: float = 0.1,
) -> TorchFM:
    """Fit a freshly initialized FM surrogate with the shared Adam/MSE policy."""
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs must be positive and learning_rate must be greater than zero.")
    model = TorchFM(d=d, k=k)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    return model
