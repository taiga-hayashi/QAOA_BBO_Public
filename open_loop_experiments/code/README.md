# Open Loop Experiments Reproduction Code

このディレクトリ (`results_1001/code/`) は、PestControl / FMQAOA ベンチマークにおける**オープンループ（純粋な関数近似および量子・古典ソルバーの性能評価）実験**を、富士通様の **OpenQARP (`qarpx`)** を用いて完全に再現するためのモジュール群です。

## ディレクトリ構造

```text
code/
 ├── modules/
 │    ├── bb_functions.py   # ブラックボックス関数 (BB1/BB2) の生成
 │    ├── fm.py             # TorchFMの定義と学習 (MSE版 / Huber+L2版 対応)
 │    ├── qubo.py           # QUBO行列への One-hot ペナルティ付与
 │    ├── sa.py             # Neal を用いた古典 Simulated Annealing ソルバー
 │    ├── qaoa.py           # OpenQARP (qarpx) を用いた Standard Penalty-QAOA
 │    └── xy_qaoa.py        # OpenQARP (qarpx) を用いた XY-QAOA
 │
 ├── plot_parity.py         # FMの真値 vs 予測値の散布図 (過学習評価)
 ├── plot_lambda_tuning.py  # ペナルティ係数 lambda を振ったソルバー比較
 ├── plot_learning_curve.py # 初期データ数による学習曲線
 ├── run_n200_test.py       # データ数200時のソルバー性能比較
 ├── main.py                # 全ての実験をコマンドライン引数から実行するランナー
 └── README.md              # このファイル
```

## 動作環境の前提

OpenQARP 公式モジュールである `qarpx` がインストールされた Python 仮想環境で実行する必要があります。

*   **推奨環境**: `~/.venvs/fas/`
*   もし `qarpx` が未インストールの場合は、プロジェクトルート等で `uv pip install openqarp` 等を実行して導入してください。

## 使い方 (`main.py`)

すべての実験は `main.py` を通じて単一のコマンドで実行できます。

### 基本的な構文
```bash
python main.py --task <TASK_NAME> --N <PROBLEM_SIZE> --problem <PROBLEM_TYPE>
```

### 引数の詳細
*   `--task` (必須): 実行したい実験タスク名。以下のいずれかを指定します。
    *   `parity`: パリティプロット (Train vs Test の散布図)
    *   `lambda_tuning`: lambda を振った際の制約充足率と Regret
    *   `learning_curve`: 初期データ数に応じた Train/Test MSE の推移
    *   `n200_test`: 初期データ200件時の各ソルバーの性能比較 (CSV出力)
*   `--N` (任意, デフォルト: 18): 問題のサイズ (例: 18, 21, 24, 27)
*   `--problem` (任意, デフォルト: bb1): 問題のタイプ (`bb1` または `bb2`)

### 実行例

**1. N=18, BB1 のパリティプロットを生成する**
```bash
/Users/hayashitaiga/.venvs/fas/bin/python main.py --task parity --N 18 --problem bb1
```

**2. N=24, BB2 の lambda チューニングプロットを生成する**
```bash
/Users/hayashitaiga/.venvs/fas/bin/python main.py --task lambda_tuning --N 24 --problem bb2
```

## 出力について
デフォルトでは、プロットやCSVファイルは `code/plots/` 配下の各タスク名に対応するディレクトリ（`plots/parity/` など）に自動生成されます。
