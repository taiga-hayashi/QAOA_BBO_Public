#!/usr/bin/env python3
"""Reproduce the fixed-seed experiment snapshot on the internship PC."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
ENV = os.environ | {
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": "/private/tmp/mplconfig-openqarp",
    "XDG_CACHE_HOME": "/private/tmp/xdg-cache-openqarp",
}


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, env=ENV, check=True)


def bbo(problem: str, method: str, seed: int, checkpoint: Path) -> None:
    run(
        PYTHON, "experimet/py/run_openqarp_bbo_chunk.py", "--problem", problem,
        "--method", method, "--seed", str(seed), "--checkpoint", str(checkpoint),
        "--chunk", "15", "--until-complete",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="discard only local BBO checkpoints before rerunning")
    args = parser.parse_args()
    if args.fresh:
        for directory in ROOT.glob("experimet/**/json/openqarp_checkpoints"):
            shutil.rmtree(directory)

    run(PYTHON, "experimet/py/rerun_openqarp_direct_comparison.py")
    for problem, short in (("BB1_materials", "bb1"), ("BB2_random", "bb2")):
        for method, key in (("FMQA", "fmqa"), ("Standard QAOA", "std"), ("FM-XY-QAOA", "xy")):
            bbo(problem, method, 42, Path(f"experimet/optimization/json/openqarp_checkpoints/{short}_{key}_seed42.json"))
    run(PYTHON, "experimet/py/finalize_openqarp_bbo_results.py", "optimization")

    run(PYTHON, "experimet/qubit_scaling/py/run_qubit_scaling.py")
    run(PYTHON, "experimet/std_qaoa_sensitivity/py/run_std_qaoa_sensitivity.py")
    run(PYTHON, "experimet/shot_budget_sensitivity/py/run_shot_budget_sensitivity.py")
    run(PYTHON, "experimet/penalty_sensitivity/py/run_penalty_sensitivity.py")
    run(PYTHON, "experimet/adaptive_lambda_validation/py/run_adaptive_lambda_validation.py")

    for problem, short in (("BB1_materials", "bb1"), ("BB2_random", "bb2")):
        for seed in (11, 22, 33):
            for method, key in (("FMQA", "fmqa"), ("Standard QAOA", "std"), ("FM-XY-QAOA", "xy")):
                bbo(problem, method, seed, Path(f"experimet/bbo_seed_robustness/json/openqarp_checkpoints/{short}_{key}_seed{seed}.json"))
    run(PYTHON, "experimet/py/finalize_openqarp_bbo_results.py", "seed_robustness")

    for script in (
        "experimet/optimization/py/plot_optimization_results.py",
        "experimet/bbo_seed_robustness/py/plot_bbo_seed_robustness.py",
        "experimet/qubit_scaling/py/plot_qubit_scaling.py",
        "experimet/std_qaoa_sensitivity/py/plot_std_qaoa_sensitivity.py",
        "experimet/shot_budget_sensitivity/py/plot_shot_budget_sensitivity.py",
        "experimet/penalty_sensitivity/py/plot_penalty_sensitivity.py",
        "experimet/py/export_individual_panels.py",
    ):
        run(PYTHON, script)


if __name__ == "__main__":
    main()
