import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from fm import TorchFM
from fm_to_qubo import fm_to_qubo
from qaoa_solver import solve_qubo_qaoa

from bb_function import create_random_qubo_bb, evaluate_bb, generate_dataset, get_exact_minimum_bb
from utils import Logger, get_fm_minimum
import os
import sys
import json
from datetime import datetime

def main():
    # 結果保存用ディレクトリの作成
    os.makedirs(os.path.join("result", "txt"), exist_ok=True)
    os.makedirs(os.path.join("result", "json"), exist_ok=True)
    os.makedirs(os.path.join("result", "png"), exist_ok=True)
    os.makedirs(os.path.join("result", "pdf"), exist_ok=True)
    os.makedirs(os.path.join("result", "csv"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("result", "txt", f"output_{timestamp}.txt")
    
    # 標準出力をファイルにも書き出すように設定
    original_stdout = sys.stdout
    sys.stdout = Logger(log_path)
    print(f"=== ログを {log_path} に保存します ===")

    import matplotlib.pyplot as plt
    import yaml

    # config.yaml から設定を読み込む (カレントディレクトリに依存しない絶対パス解決)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # シードの固定
    seed_val = config.get("seed", 42)
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    
    # パラメータ設定
    d = config.get("d", 5)
    k = config.get("k", 2)
    num_initial_samples = config.get("num_initial_samples", 10)
    num_bbo_cycles = config.get("num_bbo_cycles", 5)
    epochs = config.get("epochs", 150)
    lr = config.get("lr", 0.1)
    qaoa_reps = config.get("qaoa_reps", 1)
    qaoa_maxiter = config.get("qaoa_maxiter", 50)
    
    print("=== 1. ブラックボックス関数(ランダムQUBO)の定義と初期データ生成 ===")
    Q_bb = create_random_qubo_bb(d, seed=seed_val)
    
    # 厳密解の計算 (全探索 2^d) 比較用
    best_x_bb, min_val_bb = get_exact_minimum_bb(Q_bb, d)
    print(f"[BB関数の真の最小値]  Val: {min_val_bb:.4f}, x: {tuple(int(xi) for xi in best_x_bb)}")

    X_train, y_train = generate_dataset(Q_bb, num_initial_samples, d)
    
    # データセットをリストで管理 (後で追加するため)
    X_train_list = [X_train]
    y_train_list = [y_train]
    
    current_best_bb = torch.min(y_train).item()
    history_bb_min = [current_best_bb]
    
    print(f"BB関数次元: {d}, 初期データ数: {num_initial_samples}")
    print(f"初期データ中の最小値: {current_best_bb:.4f}\n")
    
    qaoa_results_json = []

    print("=== 2. BBOループ開始 ===")
    for cycle in range(1, num_bbo_cycles + 1):
        print(f"\n--- Cycle {cycle}/{num_bbo_cycles} ---")
        
        # データの結合
        X_current = torch.cat(X_train_list, dim=0)
        y_current = torch.cat(y_train_list, dim=0)
        
        # FMモデルの学習 (毎回初期化して学習)
        model = TorchFM(d=d, k=k)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model(X_current)
            loss = criterion(preds, y_current)
            loss.backward()
            optimizer.step()
            
        print(f"FM学習完了 (データ数: {len(X_current)}) | 最終 Loss: {loss.item():.4f}")
        
        # 学習済みFMからQUBOへの変換
        qubo_dict, offset = fm_to_qubo(model)
        
        # QAOAによる最適化
        result = solve_qubo_qaoa(qubo_dict, offset=offset, reps=qaoa_reps, maxiter=qaoa_maxiter)
        
        # QAOAの最良解を新しい入力として評価
        best_sample = result.samples[0]
        qaoa_x_tuple = tuple(int(xi) for xi in best_sample.x)
        qaoa_x_array = np.array(qaoa_x_tuple, dtype=np.float32).reshape(1, -1)
        
        # 真のBB関数で評価
        y_new_val = evaluate_bb(qaoa_x_array, Q_bb)[0]
        print(f"QAOA提案解: {qaoa_x_tuple} | BB評価値: {y_new_val:.4f} (FM予測値: {best_sample.fval:.4f})")
        
        # データセットへの追加
        X_train_list.append(torch.tensor(qaoa_x_array))
        y_train_list.append(torch.tensor([y_new_val], dtype=torch.float32))
        
        # 最良値の更新と記録
        if y_new_val < current_best_bb:
            current_best_bb = y_new_val
            print(f"★ 最良値を更新しました！ -> {current_best_bb:.4f}")
            
        history_bb_min.append(current_best_bb)
        
        # JSON記録用
        qaoa_results_json.append({
            "cycle": cycle,
            "proposed_x": qaoa_x_tuple,
            "bb_eval_val": float(y_new_val),
            "fm_predict_val": float(best_sample.fval),
            "current_best_bb": float(current_best_bb)
        })

    print("\n=== 3. 結果の保存とプロット ===")
    
    # 構造化データ(JSON)の保存
    result_data = {
        "BB_exact_minimum": {
            "val": float(min_val_bb),
            "x": tuple(int(xi) for xi in best_x_bb)
        },
        "history_bb_min": [float(v) for v in history_bb_min],
        "qaoa_proposals": qaoa_results_json
    }
    json_path = os.path.join("result", "json", f"bbo_result_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)
        
    # プロットの作成
    plt.figure(figsize=(8, 5))
    cycles = range(num_bbo_cycles + 1)
    plt.plot(cycles, history_bb_min, marker='o', label='Best Found BB Value')
    plt.axhline(y=min_val_bb, color='r', linestyle='--', label='Exact Minimum')
    
    plt.xlabel('BBO Cycle')
    plt.ylabel('Objective Function Value (Min)')
    plt.title('BBO Optimization History (QAOA)')
    plt.xticks(cycles)
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join("result", "png", f"bbo_history_{timestamp}.png")
    plot_pdf_path = os.path.join("result", "pdf", f"bbo_history_{timestamp}.pdf")
    plt.savefig(plot_path)
    plt.savefig(plot_pdf_path)
    print(f"プロットを {plot_path} および {plot_pdf_path} に保存しました。")
    
    # CSVの作成
    import csv
    csv_path = os.path.join("result", "csv", f"bbo_history_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle", "best_found_bb_value"])
        for c, val in zip(cycles, history_bb_min):
            writer.writerow([c, float(val)])
    print(f"プロット用データを {csv_path} に保存しました。")
    print(f"構造化データを {json_path} に保存しました。")

    # 標準出力を元に戻す
    sys.stdout = original_stdout

if __name__ == "__main__":
    main()
