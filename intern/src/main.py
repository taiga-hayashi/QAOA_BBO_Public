"""Original FMQA/QAOA black-box-optimization demonstration entry point."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

try:  # Package execution: python -m src.main
    from .bb_function import create_random_qubo_bb, evaluate_bb, generate_dataset, get_exact_minimum_bb
    from .fm import train_factorization_machine
    from .fm_to_qubo import fm_to_qubo
    from .qaoa_solver import solve_qubo_qaoa
    from .runtime import MainRunConfig, create_result_layout, load_main_config, project_root, seed_everything
    from .utils import Logger
except ImportError:  # Legacy direct execution: python src/main.py
    from bb_function import create_random_qubo_bb, evaluate_bb, generate_dataset, get_exact_minimum_bb
    from fm import train_factorization_machine
    from fm_to_qubo import fm_to_qubo
    from qaoa_solver import solve_qubo_qaoa
    from runtime import MainRunConfig, create_result_layout, load_main_config, project_root, seed_everything
    from utils import Logger


def run(config: MainRunConfig, output: dict[str, Path]) -> dict[str, Any]:
    """Run the BBO loop and return its serializable result payload."""
    seed_everything(config.seed)
    print("=== 1. ブラックボックス関数(ランダムQUBO)の定義と初期データ生成 ===")
    qubo_bb = create_random_qubo_bb(config.d, seed=config.seed)
    best_x_bb, minimum_bb = get_exact_minimum_bb(qubo_bb, config.d)
    print(f"[BB関数の真の最小値] Val: {minimum_bb:.4f}, x: {tuple(int(v) for v in best_x_bb)}")

    x_train, y_train = generate_dataset(qubo_bb, config.num_initial_samples, config.d)
    x_train_history, y_train_history = [x_train], [y_train]
    current_best = torch.min(y_train).item()
    best_history = [current_best]
    qaoa_proposals: list[dict[str, Any]] = []
    print(f"BB関数次元: {config.d}, 初期データ数: {config.num_initial_samples}")
    print(f"初期データ中の最小値: {current_best:.4f}\n")
    print("=== 2. BBOループ開始 ===")

    for cycle in range(1, config.num_bbo_cycles + 1):
        print(f"\n--- Cycle {cycle}/{config.num_bbo_cycles} ---")
        x_current = torch.cat(x_train_history, dim=0)
        y_current = torch.cat(y_train_history, dim=0)
        model = train_factorization_machine(
            x_current, y_current, d=config.d, k=config.k,
            epochs=config.epochs, learning_rate=config.lr,
        )
        with torch.no_grad():
            final_loss = torch.mean((model(x_current) - y_current) ** 2).item()
        print(f"FM学習完了 (データ数: {len(x_current)}) | 最終 Loss: {final_loss:.4f}")

        qubo_dict, offset = fm_to_qubo(model)
        result = solve_qubo_qaoa(
            qubo_dict, offset=offset, reps=config.qaoa_reps,
            maxiter=config.qaoa_maxiter, optimizer_name=config.qaoa_optimizer,
        )
        best_sample = result.samples[0]
        proposed_x = tuple(int(value) for value in best_sample.x)
        proposed_array = np.asarray(proposed_x, dtype=np.float32).reshape(1, -1)
        evaluated_value = float(evaluate_bb(proposed_array, qubo_bb)[0])
        print(f"QAOA提案解: {proposed_x} | BB評価値: {evaluated_value:.4f} (FM予測値: {best_sample.fval:.4f})")

        x_train_history.append(torch.tensor(proposed_array))
        y_train_history.append(torch.tensor([evaluated_value], dtype=torch.float32))
        if evaluated_value < current_best:
            current_best = evaluated_value
            print(f"★ 最良値を更新しました！ -> {current_best:.4f}")
        best_history.append(current_best)
        qaoa_proposals.append({
            "cycle": cycle, "proposed_x": proposed_x, "bb_eval_val": evaluated_value,
            "fm_predict_val": float(best_sample.fval), "current_best_bb": float(current_best),
        })

    payload = {
        "BB_exact_minimum": {"val": float(minimum_bb), "x": tuple(int(v) for v in best_x_bb)},
        "history_bb_min": [float(value) for value in best_history],
        "qaoa_proposals": qaoa_proposals,
    }
    _save_artifacts(payload, output, minimum_bb)
    return payload


def _save_artifacts(payload: dict[str, Any], output: dict[str, Path], exact_minimum: float) -> None:
    """Persist machine-readable results and their matching history figure."""
    print("\n=== 3. 結果の保存とプロット ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output["json"] / f"bbo_result_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)

    history = payload["history_bb_min"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(range(len(history)), history, marker="o", label="Best Found BB Value")
    axis.axhline(y=exact_minimum, color="r", linestyle="--", label="Exact Minimum")
    axis.set(xlabel="BBO Cycle", ylabel="Objective Function Value (Min)")
    axis.set_xticks(range(len(history)))
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    png_path = output["png"] / f"bbo_history_{timestamp}.png"
    pdf_path = output["pdf"] / f"bbo_history_{timestamp}.pdf"
    figure.savefig(png_path)
    figure.savefig(pdf_path)
    plt.close(figure)

    csv_path = output["csv"] / f"bbo_history_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cycle", "best_found_bb_value"])
        writer.writerows((cycle, float(value)) for cycle, value in enumerate(history))
    print(f"プロットを {png_path} および {pdf_path} に保存しました。")
    print(f"プロット用データを {csv_path} に保存しました。")
    print(f"構造化データを {json_path} に保存しました。")


def main() -> None:
    root = project_root(__file__)
    output = create_result_layout(root)
    config = load_main_config(root / "config.yaml")
    log_path = output["txt"] / f"output_{datetime.now():%Y%m%d_%H%M%S}.txt"
    original_stdout = sys.stdout
    with Logger(log_path, terminal=original_stdout) as logger:
        sys.stdout = logger
        try:
            print(f"=== ログを {log_path} に保存します ===")
            run(config, output)
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
