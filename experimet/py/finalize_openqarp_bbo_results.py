#!/usr/bin/env python3
"""Promote complete, checkpointed OpenQARP BBO trajectories into result JSON.

The BBO jobs are deliberately checkpointed because a full 100-cycle trajectory
can outlive an interactive command slice.  This utility is the only promotion
step: it refuses incomplete trajectories and never copies a pre-migration
result into an OpenQARP-labelled result file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ("BB1_materials", "BB2_random")
METHODS = ("FMQA", "Standard QAOA", "FM-XY-QAOA")


def load_complete(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("checkpoint", {}).get("next_cycle") != 101:
        raise ValueError(f"Incomplete checkpoint: {path}")
    result = dict(stored["result"])
    if len(result.get("best_history", [])) != 101 or len(result.get("proposals", [])) != 100:
        raise ValueError(f"Malformed complete trajectory: {path}")
    return result


def finalize_optimization() -> None:
    output = ROOT / "optimization" / "json" / "results_two_bb_optimization.json"
    checkpoints = output.parent / "openqarp_checkpoints"
    data = json.loads(output.read_text(encoding="utf-8"))
    keys = {"FMQA": "FMQA", "Standard QAOA": "Standard_QAOA", "FM-XY-QAOA": "FM_XY_QAOA"}
    for problem in PROBLEMS:
        target = data[problem]["part2_bbo"]
        for method in METHODS:
            target[keys[method]] = load_complete(checkpoints / f"{problem.lower().replace('_materials', '').replace('_random', '')}_{'xy' if method == 'FM-XY-QAOA' else 'std' if method == 'Standard QAOA' else 'fmqa'}_seed42.json")
    metadata = data.setdefault("metadata", {})
    metadata.update({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "qaoa_backend": "OpenQARP 0.1.0 (qarp/qarpx)",
        "bbo_execution": "All QAOA BBO trajectories were rerun for 100 cycles with checkpointed native OpenQARP state-vector circuits.",
    })
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


def finalize_seed_robustness() -> None:
    output = ROOT / "bbo_seed_robustness" / "json" / "results_bbo_seed_robustness.json"
    checkpoints = output.parent / "openqarp_checkpoints"
    data = json.loads(output.read_text(encoding="utf-8"))
    seeds = data["metadata"]["seeds"]
    short = {"FMQA": "fmqa", "Standard QAOA": "std", "FM-XY-QAOA": "xy"}
    for problem in PROBLEMS:
        problem_short = "bb1" if problem == "BB1_materials" else "bb2"
        method_data = data["problems"][problem]["methods"]
        for method in METHODS:
            runs = []
            for seed in seeds:
                # The first Standard-QAOA batch retained the long problem
                # labels; later jobs use concise bb1/bb2 labels.  Resolve
                # both deterministically while still refusing a missing run.
                filename = f"{problem_short}_{short[method]}_seed{seed}.json"
                checkpoint_path = checkpoints / filename
                if not checkpoint_path.exists() and method == "Standard QAOA":
                    checkpoint_path = checkpoints / f"{problem}_{short[method]}_seed{seed}.json"
                run = load_complete(checkpoint_path)
                run["seed"] = seed
                runs.append(run)
            method_data[method] = runs

        summary = {}
        for method, runs in method_data.items():
            finals = np.asarray([run["final_best"] for run in runs], dtype=float)
            failures = np.asarray([run["infeasible_count"] for run in runs], dtype=float)
            summary[method] = {
                "mean_final_best": float(finals.mean()),
                "std_final_best": float(finals.std(ddof=1)),
                "mean_candidate_failures": float(failures.mean()),
                "success_count": int(np.sum(np.isclose(finals, data["problems"][problem]["exact_min"], atol=1e-10))),
            }
        data["problems"][problem]["summary"] = summary

    metadata = data.setdefault("metadata", {})
    metadata.update({
        "qaoa_backend": "OpenQARP 0.1.0 (qarp/qarpx)",
        "bbo_execution": "Every seed/problem QAOA trajectory was rerun for 100 cycles with checkpointed native OpenQARP state-vector circuits.",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("optimization", "seed_robustness"))
    args = parser.parse_args()
    if args.suite == "optimization":
        finalize_optimization()
    else:
        finalize_seed_robustness()


if __name__ == "__main__":
    main()
