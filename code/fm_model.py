"""Factorization Machine (TorchFM) model and deterministic training routine."""
from __future__ import annotations

import hashlib
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn


class TorchFM(nn.Module):
    """Second-order Factorization Machine on binary variables x in {0, 1}^d."""

    def __init__(self, d: int, k: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(d, k))
        self.lin = nn.Linear(d, 1)
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term1_inner = x @ self.V
        term1 = torch.sum(term1_inner * term1_inner, dim=1, keepdim=True)
        term2_inner = x.pow(2) @ self.V.pow(2)
        term2 = torch.sum(term2_inner, dim=1, keepdim=True)
        interaction = 0.5 * (term1 - term2)
        return (interaction + self.lin(x)).view(-1)


def model_sha256(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_fm(
    x_train: np.ndarray,
    y_train: np.ndarray | list[float],
    d: int,
    fm_cfg: Dict[str, Any],
    seed: int,
) -> TorchFM:
    """Deterministically train a Factorization Machine with MSE loss."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    rank = int(fm_cfg["rank"])
    epochs = int(fm_cfg["epochs"])
    lr = float(fm_cfg["learning_rate"])
    opt_name = str(fm_cfg.get("optimizer", "AdamW"))
    wd = float(fm_cfg.get("weight_decay", 0.0))

    optimizers = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
    if opt_name not in optimizers:
        raise ValueError(f"Unsupported optimizer '{opt_name}'. Choose 'Adam' or 'AdamW'.")

    model = TorchFM(d=d, k=rank)
    optimizer = optimizers[opt_name](model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_t), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model
