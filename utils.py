import sys
import torch
import itertools
from fm import TorchFM

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_fm_minimum(model: TorchFM, d: int):
    """FMサロゲートモデル上の最小値とその入力を全探索で求める"""
    all_patterns = list(itertools.product([0, 1], repeat=d))
    x_tensor = torch.tensor(all_patterns, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x_tensor)
    min_idx = torch.argmin(outputs).item()
    return all_patterns[min_idx], outputs[min_idx].item()
