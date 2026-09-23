#!/usr/bin/env python3
"""Run a resumable, OpenQARP-backed slice of one N=16 BBO trajectory.

This runner exists because the exact p=1 parameter grids are deliberately
recomputed for every changed surrogate, while the execution environment limits
an individual foreground command to roughly 30 seconds.  It preserves the
complete BBO state between slices; splitting therefore produces exactly the
same trajectory as one uninterrupted call.
"""

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPT_SCRIPT = ROOT / "optimization" / "py" / "run_optimization_experiment.py"
spec = importlib.util.spec_from_file_location("optimization_experiment", OPT_SCRIPT)
exp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["BB1_materials", "BB2_random"], required=True)
    parser.add_argument("--method", choices=["FMQA", "Standard QAOA", "FM-XY-QAOA"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument(
        "--until-complete",
        action="store_true",
        help="Keep running chunks until --cycles is reached, checkpointing after every chunk.",
    )
    args = parser.parse_args()

    blocks = [4, 4, 4, 4]
    patterns, _ = exp.get_all_feasible_patterns(blocks)
    if args.problem == "BB1_materials":
        qubo = exp.create_materials_design_ground_truth(blocks, seed=123)
    else:
        qubo = exp.create_random_qubo_bb(16, seed=42)
    optimum_x, optimum = exp.get_exact_minimum(qubo, patterns)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    while True:
        checkpoint = None
        if args.checkpoint.exists():
            with args.checkpoint.open(encoding="utf-8") as f:
                stored = json.load(f)
            checkpoint = stored["checkpoint"]
        start_cycle = 1 if checkpoint is None else checkpoint["next_cycle"]
        if start_cycle > args.cycles:
            print("complete")
            return

        result = exp.run_bbo_closed_loop(
            args.method,
            qubo,
            blocks,
            optimum_x,
            optimum,
            num_cycles=args.cycles,
            seed=args.seed,
            checkpoint=checkpoint,
            end_cycle=min(args.cycles, start_cycle + args.chunk - 1),
        )
        with args.checkpoint.open("w", encoding="utf-8") as f:
            json.dump({"checkpoint": result.pop("checkpoint"), "result": result}, f, indent=2)
        complete_at = min(args.cycles, start_cycle + args.chunk - 1)
        print(f"completed cycles 1-{complete_at}", flush=True)
        if not args.until_complete or complete_at >= args.cycles:
            return


if __name__ == "__main__":
    main()
