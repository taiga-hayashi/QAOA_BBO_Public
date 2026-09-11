# QAOA_BBO: Factorization Machine + QAOA for Black-Box Optimization

ブラックボックス最適化（Black-Box Optimization: BBO）において、機械学習サロゲートモデル（Factorization Machine: FM）と量子近似最適化アルゴリズム（QAOA）を融合した最適化フレームワークです。

## 主な機能
- **FMサロゲートモデリング**: 2次の特徴量相互作用を学習し、目的関数をモデル化
- **QUBO変換**: 学習済みFMからIsingハミルトニアン／QUBOパラメータを厳密に導出
- **Qiskit QAOAソルバー**: 
  - 通常のXミキサー（ペナルティ法）
  - ワンホット制約を満たすブロック直和型XYミキサー法（W状態初期化）
- **高速シミュレーション**: `qiskit-aer` によるサンプリング・期待値評価

## 環境構築と実行 (uv 推奨)

本プロジェクトは [uv](https://docs.astral.sh/uv/) を用いた高速な環境管理に対応しています。

### 1. 依存ライブラリの同期と実行
```bash
# uv run を使うと、仮想環境の作成から実行まで全自動で行われます
uv run python main.py
```

### 2. 従来の pip / venv を使用する場合
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## プロジェクト構成
- `main.py`: メインエントリポイント
- `config.yaml`: 最適化パラメータ・実験設定
- `src/`:
  - `fm.py`: Factorization Machine 実装
  - `fm_to_qubo.py`: FMパラメータからQUBOへの変換
  - `qaoa_solver.py`: Qiskitを用いたQAOAソルバー（XYミキサー・W状態生成）
  - `bb_function.py`: テスト用ブラックボックス関数
  - `scaling_experiment.py`: スケーリング比較実験スクリプト
- `qaoa_tutorial/`: QAOA徹底解説チュートリアル資料・LaTeX教科書
- `report/`: スケーリング検証実験レポート
