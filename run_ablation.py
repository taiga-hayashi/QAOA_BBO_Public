"""Run closed-loop ablation CSV targets and update `results/csv/`.

Targets (each individually configurable in `config.json`):
    - closedloop_l7_alpha_sweep
    - closedloop_l7_paired_ranks
    - closedloop_l7_schedule_retune
    - closedloop_l678_explore_5seeds

Usage:
    python run_ablation.py                                    # すべて実行
    python run_ablation.py --target closedloop_l7_alpha_sweep # 特定のCSVだけ計算
    python run_ablation.py --smoke                            # 軽量テスト
"""
from __future__ import annotations

from src.csv_export import export_target
from src.runners import (
    load_effective_config,
    parse_cli_args,
    prepare_run_dir,
    prepare_truth_tables,
    run_target,
    target_layers,
    targets_in_group,
)


def main() -> None:
    default_cfg = load_effective_config(False)
    choices = ["all", *targets_in_group(default_cfg, "ablation")]
    args = parse_cli_args(__doc__, choices)
    cfg = load_effective_config(args.smoke, args.config)
    run_dir = prepare_run_dir(args.run_dir, smoke=args.smoke, cfg=cfg, workers=args.workers)

    targets = targets_in_group(cfg, "ablation") if args.target == "all" else [args.target]
    layers = sorted({L for t in targets for L in target_layers(cfg["targets"][t])})
    prepare_truth_tables(cfg, run_dir, layers_list=layers, tmm_backend=args.tmm_backend)

    for tname in targets:
        run_target(tname, cfg, run_dir, workers=args.workers)
        export_target(tname, cfg, run_dir, args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
