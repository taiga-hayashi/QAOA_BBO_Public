# QAOA_BBO: Factorization Machine + QAOA for Black-Box Optimization

ブラックボックス最適化（Black-Box Optimization: BBO）において、機械学習サロゲートモデル（Factorization Machine: FM）と量子近似最適化アルゴリズム（QAOA）を融合した最適化フレームワークのコアコードです。

## 主な機能
- **FMサロゲートモデリング**: 2次の特徴量相互作用を学習し、目的関数をモデル化
- **QUBO変換**: 学習済みFMからIsingハミルトニアン／QUBOパラメータを厳密に導出
- **Qiskit QAOAソルバー**: Qiskit Algorithms と Qiskit Aer による高速シミュレーション

## 環境構築と実行

### 1. uv を使用する場合（推奨）
```bash
# 依存関係の解決と実行が一発で行われます
uv run python main.py
```

### 2. 通常の pip / venv を使用する場合
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## ファイル構成
- `main.py`: BBO最適化ループのメインスクリプト
- `config.yaml`: 最適化パラメータ設定ファイル
- `fm.py`: Factorization Machine (PyTorch実装)
- `fm_to_qubo.py`: FMパラメータからQUBOへの変換
- `qaoa_solver.py`: Qiskitを用いたQAOAソルバー
- `bb_function.py`: テスト用ブラックボックス関数（ランダムQUBO）
- `utils.py`: ロガー等のユーティリティ
