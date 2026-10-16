"""Categorical table evaluators, One-Hot codec, selection, and diagnostics."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Set, Tuple

import numpy as np
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parent


class CategoricalTableProblem:
    """Complete categorical lookup table; true values are used only for evaluation/audit."""

    def __init__(self, data_path: str | Path, config_path: str | Path):
        p_data = Path(data_path)
        p_cfg = Path(config_path)
        if not p_data.is_absolute():
            p_data = (CODE_DIR / p_data).resolve()
        if not p_cfg.is_absolute():
            p_cfg = (CODE_DIR / p_cfg).resolve()
        self.data_path = p_data
        self.config_path = p_cfg

        schema = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.options = tuple(tuple(p["options"]) for p in schema["parameters"])
        self.group_sizes = tuple(map(len, self.options))
        self.n_bits = sum(self.group_sizes)
        if not self.options or any(not opts or len(set(opts)) != len(opts) for opts in self.options):
            raise ValueError("Each parameter must have distinct, nonempty options.")
        self.problem_id = schema.get("problem_id", "perovskites")

        self._values: Dict[Tuple[str, ...], float] = {}
        with self.data_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) != len(self.options) + 1:
                    raise ValueError("Each row must contain one label per parameter and one objective.")
                key = tuple(row[:-1])
                if key in self._values:
                    raise ValueError(f"Duplicate candidate {key}")
                val = float(row[-1])
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite objective for {key}")
                self._values[key] = val

        self.categories = [list(c) for c in itertools.product(*self.options)]
        if set(self._values) != {tuple(c) for c in self.categories}:
            raise ValueError("Dataset must cover exactly the Cartesian candidate space.")

        self.patterns = np.array([self.encode(c) for c in self.categories], dtype=np.int8)
        self.lookup = {tuple(row): idx for idx, row in enumerate(self.patterns)}
        self.valid_basis_indices = self.patterns.astype(np.uint64) @ (
            1 << np.arange(self.n_bits, dtype=np.uint64)
        )
        # Audit / metric constants (never exposed to the sampler or FM training)
        self._all_true_vals = np.array([self._values[tuple(c)] for c in self.categories], dtype=float)
        self.f_opt = float(self._all_true_vals.min())
        self.f_worst = float(self._all_true_vals.max())
        self.data_sha256 = hashlib.sha256(self.data_path.read_bytes()).hexdigest()

    def encode(self, categories: Sequence[str]) -> Tuple[int, ...]:
        if len(categories) != len(self.options):
            raise ValueError("One category per parameter required.")
        bits = []
        for label, opts in zip(categories, self.options):
            bits.extend(int(i == opts.index(label)) for i in range(len(opts)))
        return tuple(bits)

    def evaluate_id(self, candidate_id: int) -> float:
        """Evaluate the true Black-Box objective for a feasible candidate index."""
        return self._values[tuple(self.categories[candidate_id])]

    def normalized_regret(self, f_best: float) -> float:
        """Normalized Regret R = (f_best - f_opt) / (f_worst - f_opt)."""
        return float((f_best - self.f_opt) / (self.f_worst - self.f_opt))

    def initial_dataset(self, seed: int, n_initial: int) -> list[int]:
        """Generate deterministic initial candidate IDs for a given seed."""
        rng = np.random.default_rng(seed)
        return rng.choice(len(self.categories), size=n_initial, replace=False).tolist()

    def evaluate_surrogate_diagnostics(
        self,
        fm_predictions: np.ndarray,
        unseen_all: list[int],
    ) -> Dict[str, float | int | None]:
        """Compute unseen Spearman rank correlation rho_t and true rank of the FM-minimum candidate."""
        if len(unseen_all) < 2:
            return {"unseen_spearman_rho": None, "unseen_fm_min_true_rank": None}
        pred_u = fm_predictions[unseen_all]
        true_u = self._all_true_vals[unseen_all]
        if np.std(pred_u) < 1e-12 or np.std(true_u) < 1e-12:
            rho = 0.0
        else:
            res = spearmanr(pred_u, true_u)
            rho = float(res.statistic) if np.isfinite(res.statistic) else 0.0
        fm_min_local_idx = int(np.argmin(pred_u))
        fm_min_true_val = float(true_u[fm_min_local_idx])
        # 1-indexed true rank among unseen candidates (1 = best unseen candidate)
        true_rank = int(1 + np.sum(true_u < fm_min_true_val - 1e-12))
        return {
            "unseen_spearman_rho": rho,
            "unseen_fm_min_true_rank": true_rank,
        }

    def select_unseen_candidate(
        self,
        raw_bits: np.ndarray,
        fm_predictions: np.ndarray,
        seen_ids: Set[int],
    ) -> Tuple[int | None, Dict[str, Any]]:
        """Filter raw bitstrings for One-Hot feasibility and unseen status, then pick the lowest FM prediction."""
        ids = [self.lookup.get(tuple(row)) for row in raw_bits]
        n_total = len(ids)
        n_feasible = sum(i is not None for i in ids)
        n_invalid = n_total - n_feasible
        n_seen_dup = sum(i in seen_ids for i in ids if i is not None)
        unseen_pool = {i for i in ids if i is not None and i not in seen_ids}
        n_unseen_hits = sum(i is not None and i not in seen_ids for i in ids)
        n_within_dup = n_unseen_hits - len(unseen_pool)

        chosen = (
            min(unseen_pool, key=lambda i: (float(fm_predictions[i]), tuple(self.patterns[i])))
            if unseen_pool
            else None
        )
        unseen_all = [i for i in range(len(self.categories)) if i not in seen_ids]
        unseen_fm_min = float(min(fm_predictions[unseen_all])) if unseen_all else None
        selected_fm_gap = (
            float(fm_predictions[chosen] - unseen_fm_min)
            if (chosen is not None and unseen_fm_min is not None)
            else None
        )
        fm_min_hit = int(
            chosen is not None
            and unseen_fm_min is not None
            and abs(float(fm_predictions[chosen]) - unseen_fm_min) < 1e-8
        )
        surrogate_diag = self.evaluate_surrogate_diagnostics(fm_predictions, unseen_all)
        stats = {
            "raw_sample_count": n_total,
            "raw_feasible_rate": float(n_feasible / n_total) if n_total > 0 else 0.0,
            "invalid_count": n_invalid,
            "evaluated_duplicate_count": n_seen_dup,
            "unique_unseen_count": len(unseen_pool),
            "within_unseen_duplicate_count": n_within_dup,
            "unseen_success": int(chosen is not None),
            "fm_minimum_hit": fm_min_hit,
            "selected_fm_gap": selected_fm_gap,
            **surrogate_diag,
        }
        return chosen, stats


class PerovskitesProblem(CategoricalTableProblem):
    """Backward-compatible Perovskites evaluator."""

    def __init__(self, data_path: str | Path, config_path: str | Path):
        super().__init__(data_path, config_path)
        if self.group_sizes != (16, 3, 4):
            raise ValueError(f"Expected group_sizes (16, 3, 4), got {self.group_sizes}")


def load_problem(cfg: Dict[str, Any]) -> CategoricalTableProblem:
    """Select a registered problem without changing sampler or training code."""
    name = cfg.get("name", "perovskites")
    if name not in {"perovskites", "multilayer_filter"}:
        raise ValueError(f"Unknown problem: {name}")
    cls = PerovskitesProblem if name == "perovskites" else CategoricalTableProblem
    problem = cls(cfg["data_path"], cfg["config_path"])
    expected = (16, 3, 4) if name == "perovskites" else (4,) * 6
    if problem.group_sizes != expected or tuple(cfg["group_sizes"]) != expected:
        raise ValueError(f"{name} requires group_sizes {expected}")
    if name == "multilayer_filter" and problem.problem_id != "multilayer_filter_L6_K4_code50nm_v1":
        raise ValueError("Unexpected optical filter variant.")
    return problem
