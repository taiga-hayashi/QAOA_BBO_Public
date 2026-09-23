"""Compatibility imports for experiment scripts; implementation lives in src."""
from pathlib import Path
import sys
import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from qarp_backend import OpenQARPStandardQAOA, OpenQARPXYQAOA, all_bitstrings_lsb, one_hot_penalty

class OpenQARPStandardP1(OpenQARPStandardQAOA):
    def probabilities(self, gamma: float, beta: float) -> np.ndarray:
        return super().probabilities([gamma], [beta])

class OpenQARPFMXYQAOAP1(OpenQARPXYQAOA):
    def probabilities(self, gamma: float, beta: float) -> np.ndarray:
        return super().probabilities([gamma], [beta])

def probabilities_for_patterns(full_probabilities: np.ndarray, patterns: np.ndarray) -> np.ndarray:
    indices = np.asarray(patterns, dtype=np.uint64) @ (1 << np.arange(patterns.shape[1], dtype=np.uint64))
    return np.asarray(full_probabilities[indices], dtype=np.float64)
