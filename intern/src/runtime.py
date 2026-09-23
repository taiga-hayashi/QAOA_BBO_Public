"""Shared runtime utilities for reproducible FMQA/QAOA workflows.

The project contains both small demonstration scripts and longer benchmark
workflows.  Keeping path resolution, configuration validation, output layout,
and random-number seeding here prevents those concerns from drifting apart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class MainRunConfig:
    """Validated configuration for the original FMQA/QAOA demonstration."""

    seed: int = 42
    d: int = 5
    k: int = 2
    num_initial_samples: int = 10
    num_bbo_cycles: int = 5
    epochs: int = 150
    lr: float = 0.1
    qaoa_reps: int = 1
    qaoa_maxiter: int = 50
    qaoa_optimizer: str = "COBYLA"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MainRunConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        supplied = {name: value for name, value in values.items() if name in allowed}
        config = cls(**supplied)
        if config.d < 1 or config.k < 1:
            raise ValueError("'d' and 'k' must be positive integers.")
        if config.num_initial_samples < 1 or config.num_bbo_cycles < 1:
            raise ValueError("Initial samples and BBO cycles must both be positive.")
        if config.epochs < 1 or config.qaoa_reps < 1 or config.qaoa_maxiter < 1:
            raise ValueError("epochs, qaoa_reps, and qaoa_maxiter must be positive.")
        if config.lr <= 0:
            raise ValueError("'lr' must be positive.")
        return config


def project_root(anchor_file: str | Path) -> Path:
    """Return the project root for a file located directly below ``src``."""
    return Path(anchor_file).resolve().parent.parent


def load_main_config(config_path: str | Path) -> MainRunConfig:
    """Load the YAML configuration and reject malformed top-level content."""
    with Path(config_path).open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, Mapping):
        raise ValueError("config.yaml must contain a mapping of configuration values.")
    return MainRunConfig.from_mapping(values)


def create_result_layout(root: str | Path) -> dict[str, Path]:
    """Create and return the standard result directories by artifact type."""
    result_root = Path(root) / "result"
    layout = {name: result_root / name for name in ("txt", "json", "png", "pdf", "csv")}
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible local runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
