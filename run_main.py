"""Run the main 30-seed closed-loop optimization (`closedloop_l678_fom_curves.csv`).

Usage:
    python run_main.py                  # L=6,7,8 すべて実行して CSV 出力
    python run_main.py --layers 6 7     # L=6,7 だけ計算して CSV 出力
    python run_main.py --smoke          # 軽量テスト（outputs/smoke/ 出力）
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
)


def main() -> None:
    target = "closedloop_l678_fom_curves"
    args = parse_cli_args(__doc__, ["all", target], include_layers_filter=True)
    cfg = load_effective_config(args.smoke, args.config)
    run_dir = prepare_run_dir(args.run_dir, smoke=args.smoke, cfg=cfg, workers=args.workers)

    layers = target_layers(cfg["targets"][target], layers_filter=args.layers)
    prepare_truth_tables(cfg, run_dir, layers_list=layers, tmm_backend=args.tmm_backend)
    run_target(target, cfg, run_dir, workers=args.workers, layers_filter=args.layers)
    export_target(target, cfg, run_dir, args.out_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
