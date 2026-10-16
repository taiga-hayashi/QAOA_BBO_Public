"""Configuration loader and validator with CLI override support."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Tuple

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CODE_DIR / "config.json"


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply dot-separated key=value overrides (e.g. qaoa.depth=3 penalty_sweep.enabled=true)."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be in key=value format, got: {item}")
        key_path, raw_val = item.split("=", 1)
        keys = key_path.strip().split(".")
        target = out
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                raise KeyError(f"Unknown config section '{k}' in '{key_path}'")
            target = target[k]
        last_key = keys[-1]
        if last_key not in target:
            raise KeyError(f"Unknown config key '{last_key}' in '{key_path}'")
        target[last_key] = _parse_value(raw_val.strip())
    return out


def format_alpha_method_name(alpha: float) -> str:
    """Format a canonical method name for an arbitrary SA penalty alpha, e.g. 'FMQA(alpha=1.0)'."""
    return f"FMQA(alpha={alpha:g})"


def parse_custom_sa_spec(
    method: str,
    base_sa_cfg: Dict[str, Any],
    base_penalty_cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]] | None:
    """Parse SA method strings such as:
    - 'TunedPenalty-FMQA' or 'TunedPenalty-FMQA(reads=10,sweeps=10)'
    - 'LargePenalty-FMQA' or 'LargePenalty-FMQA(reads=10,sweeps=100)'
    - 'FMQA(alpha=1.0,reads=10,sweeps=10)'
    Returns (alpha, sa_cfg_dict) or None if not an SA method.
    """
    sa_cfg = copy.deepcopy(base_sa_cfg)
    if method == "TunedPenalty-FMQA":
        return float(base_penalty_cfg["tuned_penalty_alpha"]), sa_cfg
    if method == "LargePenalty-FMQA":
        return float(base_penalty_cfg["large_penalty_alpha"]), sa_cfg

    for prefix, default_alpha in (
        ("TunedPenalty-FMQA(", float(base_penalty_cfg["tuned_penalty_alpha"])),
        ("LargePenalty-FMQA(", float(base_penalty_cfg["large_penalty_alpha"])),
        ("FMQA(", float(base_penalty_cfg["tuned_penalty_alpha"])),
        ("SA(", float(base_penalty_cfg["tuned_penalty_alpha"])),
    ):
        if method.startswith(prefix) and method.endswith(")"):
            inner = method[len(prefix):-1]
            alpha = default_alpha
            for tok in (t.strip() for t in inner.split(",") if t.strip()):
                if tok.startswith("alpha="):
                    alpha = float(tok[len("alpha="):])
                elif tok.startswith("reads="):
                    sa_cfg["num_reads"] = int(tok[len("reads="):])
                elif tok.startswith("sweeps="):
                    sa_cfg["num_sweeps"] = int(tok[len("sweeps="):])
                else:
                    raise ValueError(
                        f"Unrecognized SA token '{tok}' in '{method}'. "
                        "Use alpha=<float>, reads=<int>, or sweeps=<int>."
                    )
            if alpha < 0.0:
                raise ValueError(f"SA penalty alpha must be >= 0, got {alpha}")
            return alpha, sa_cfg
    return None


def parse_custom_sa_alpha(method: str) -> float | None:
    """Backward-compatible helper for extracting custom alpha when present."""
    for prefix in ("FMQA(alpha=", "SA(alpha="):
        if method.startswith(prefix) and method.endswith(")"):
            first_tok = method[len(prefix):-1].split(",")[0].strip()
            return float(first_tok)
    return None


def parse_custom_xy_qaoa(method: str, base_qaoa_cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """Parse parameterized XY-FMQAOA strings like 'XY-FMQAOA(p=3,cvar,powell49,shots=10)' into a qaoa_cfg dict."""
    if not (method.startswith("XY-FMQAOA(") and method.endswith(")")):
        return None
    inner = method[len("XY-FMQAOA("):-1]
    tokens = [t.strip() for t in inner.split(",") if t.strip()]
    q_cfg = copy.deepcopy(base_qaoa_cfg)
    for tok in tokens:
        if tok.startswith("p="):
            q_cfg["depth"] = int(tok[2:])
        elif tok in ("mean", "cvar"):
            q_cfg["objective"] = tok
        elif tok in ("legacy9", "wide49", "powell49", "powell1024"):
            q_cfg["angle_search_mode"] = tok
        elif tok.startswith("shots="):
            q_cfg["shots"] = int(tok[len("shots="):])
        else:
            raise ValueError(
                f"Unrecognized token '{tok}' in '{method}'. "
                "Use p=<int>, mean|cvar, legacy9|wide49|powell49|powell1024, or shots=<int>."
            )
    return q_cfg


def parse_custom_standard_qaoa(method: str, base_qaoa_cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """Reuse the common angle/depth parser for standard X-mixer variants."""
    prefix = "Penalty-FMQAOA("
    if not (method.startswith(prefix) and method.endswith(")")):
        return None
    parsed = parse_custom_xy_qaoa("XY-FMQAOA(" + method[len(prefix):], base_qaoa_cfg)
    out = copy.deepcopy(parsed)
    for key in ("depth", "objective", "angle_search_mode"):
        out["standard_" + key] = parsed[key]
    return out


def validate_config(cfg: Dict[str, Any]) -> None:
    """Enforce mathematical and fairness rules from AGENTS.md."""
    if cfg["penalty"]["xy_qaoa_lambda"] != 0.0:
        raise ValueError("Strict prohibition: XY-FMQAOA penalty lambda MUST be 0.0.")
    if cfg["sa"]["num_reads"] < 1 or cfg["sa"]["num_sweeps"] < 1:
        raise ValueError("sa.num_reads and sa.num_sweeps must be >= 1.")
    if cfg["qaoa"]["shots"] < 1 or cfg["qaoa"]["depth"] < 1:
        raise ValueError("qaoa.shots and qaoa.depth must be >= 1.")
    if cfg["qaoa"]["objective"] not in ("mean", "cvar"):
        raise ValueError("qaoa.objective must be 'mean' or 'cvar'.")
    if not (0.0 < float(cfg["qaoa"]["cvar_alpha"]) <= 1.0):
        raise ValueError("qaoa.cvar_alpha must be in (0, 1].")
    if cfg.get("penalty_sweep", {}).get("enabled", False):
        alphas = cfg["penalty_sweep"].get("alphas", [])
        if not alphas or any(float(a) < 0.0 for a in alphas):
            raise ValueError("penalty_sweep.alphas must be a non-empty list of non-negative numbers.")


def load_config(argv: list[str] | None = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Modular FMQA / FMQAOA Runner for categorical problems")
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to JSON configuration file (default: code/config.json)"
    )
    parser.add_argument(
        "-s", "--set", nargs="*", default=[], metavar="KEY=VAL",
        help="Override config values via dot notation, e.g. --set n_seeds=20 low_budget_sweep.enabled=true"
    )
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    explicit_seeds_override = any(item.split("=", 1)[0].strip() == "seeds" for item in (args.set or []))
    explicit_n_seeds_override = any(item.split("=", 1)[0].strip() == "n_seeds" for item in (args.set or []))

    if args.set:
        cfg = apply_overrides(cfg, args.set)

    # Sync seeds with n_seeds from seed_pool unless explicit seeds=[...] was passed on CLI
    if not explicit_seeds_override and "seed_pool" in cfg and "n_seeds" in cfg:
        n_s = int(cfg["n_seeds"])
        pool = list(cfg["seed_pool"])
        if n_s > len(pool):
            extra = [1000 + 37 * k for k in range(n_s - len(pool))]
            pool.extend(extra)
        if explicit_n_seeds_override or len(cfg.get("seeds", [])) != n_s:
            cfg["seeds"] = pool[:n_s]

    active_presets = sum([
        bool(cfg.get("penalty_sweep", {}).get("enabled", False)),
        bool(cfg.get("xy_ablation", {}).get("enabled", False)),
        bool(cfg.get("low_budget_sweep", {}).get("enabled", False)),
    ])
    if active_presets > 1:
        raise ValueError("Enable at most one of penalty_sweep, xy_ablation, or low_budget_sweep at a time.")

    if cfg.get("penalty_sweep", {}).get("enabled", False):
        sweep_methods = [format_alpha_method_name(float(a)) for a in cfg["penalty_sweep"]["alphas"]]
        if cfg["penalty_sweep"].get("include_xy_qaoa", True):
            sweep_methods.append("XY-FMQAOA")
        if cfg["penalty_sweep"].get("include_exact_fm", True):
            sweep_methods.append("Exact-FM")
        cfg["methods"] = sweep_methods

    if cfg.get("xy_ablation", {}).get("enabled", False):
        ablation_methods = []
        if cfg["xy_ablation"].get("include_exact_fm", True):
            ablation_methods.append("Exact-FM")
        if cfg["xy_ablation"].get("include_sa_baselines", True):
            ablation_methods.extend(["TunedPenalty-FMQA", "LargePenalty-FMQA"])
        ablation_methods.extend(cfg["xy_ablation"]["variants"])
        cfg["methods"] = ablation_methods

    if cfg.get("low_budget_sweep", {}).get("enabled", False):
        lb = cfg["low_budget_sweep"]
        lb_methods = []
        if lb.get("include_exact_fm", True):
            lb_methods.append("Exact-FM")
        for n in lb.get("sample_counts", [10, 30]):
            for sw in lb.get("sa_sweeps", [10, 100]):
                lb_methods.append(f"TunedPenalty-FMQA(reads={n},sweeps={sw})")
                lb_methods.append(f"LargePenalty-FMQA(reads={n},sweeps={sw})")
            for xv in lb.get("xy_variants", ["p=1,mean,legacy9", "p=3,cvar,powell49"]):
                lb_methods.append(f"XY-FMQAOA({xv},shots={n})")
        cfg["methods"] = lb_methods

    validate_config(cfg)
    return cfg
