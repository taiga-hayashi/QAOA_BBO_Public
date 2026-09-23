# QAOA_BBO: Factorization Machine + QAOA for Black-Box Optimization

ブラックボックス最適化（Black-Box Optimization: BBO）において、機械学習サロゲートモデル（Factorization Machine: FM）と量子近似最適化アルゴリズム（QAOA）を融合した最適化フレームワークのコアコードです。

コードの詳細な数理仕様や各モジュールの内部アーキテクチャについては、**[詳細解説ドキュメント (CODE_STRUCTURE.md)](./CODE_STRUCTURE.md)** をご覧ください。

---

## 主な機能
- **FMサロゲートモデリング**: 2次の特徴量相互作用を $O(kd)$ で高速学習し、目的関数をモデル化
- **QUBO変換**: 学習済みFMからIsingハミルトニアン／QUBOパラメータを厳密に導出
- **Qiskit QAOAソルバー**: Qiskit Algorithms と Qiskit Aer による高速シミュレーション（$d \ge 20$ のIndexErrorパッチ完備）
- **自動ロギング & 可視化**: 最適化推移のグラフ（PNG/PDF）、CSV、JSON、実行ログを `result/` に自動保存

---

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

---

## ファイル構成と役割一覧

| ファイル名 | 役割・機能解説 |
| :--- | :--- |
| **[`main.py`](./main.py)** | BBOクローズドループの統括エントリポイント。初期サンプリング $\to$ FM学習 $\to$ QUBO変換 $\to$ QAOAサンプリング $\to$ 評価更新を自動ループ実行 |
| **[`config.yaml`](./config.yaml)** | 実験条件の設定ファイル（問題次元 $d$、FM潜在次元 $k$、BBO反復回数、QAOAレイヤー数 $p$ 等） |
| **[`bb_function.py`](./bb_function.py)** | 未知のブラックボックス関数（ランダムQUBO）の定義、行列積評価、真の理論最小値の全探索算出 |
| **[`fm.py`](./fm.py)** | PyTorchによる Factorization Machine (FM) モデル。2次特徴量相互作用を $O(kd)$ で計算 |
| **[`fm_to_qubo.py`](./fm_to_qubo.py)** | 学習済みFMパラメータ（線形重み $w$ と潜在ベクトル内積 $v_i \cdot v_j$）からQUBO辞書形式への変換 |
| **[`qaoa_solver.py`](./qaoa_solver.py)** | Qiskit公式フレームワークを用いたQAOAソルバー。大規模問題（$d \ge 20$）でのクラッシュ回避パッチを内包 |
| **[`utils.py`](./utils.py)** | コンソールとログファイル同時書き込みを行う `Logger`、およびFM予測値全探索関数 |
| **[`CODE_STRUCTURE.md`](./CODE_STRUCTURE.md)** | **【詳細仕様書】パイプラインの数理背景、全モジュールの関数仕様、データフローの徹底解説** |

より詳細な各モジュールの内部実装・数理解説は [CODE_STRUCTURE.md](./CODE_STRUCTURE.md) をご参照ください。

---

## 富士通インターンシップ研究パッケージ (`intern/`)

本リポジトリの [`intern/`](./intern/) ディレクトリには、富士通インターンシップにて実施された **OpenQARP を用いた FM-XY-QAOA および制約付きブラックボックス最適化（BBO）の大規模ベンチマーク実験パッケージ** が収録されています。

- **主要手法**: FMQA, Standard QAOA (X-mixer + Penalty), FM-XY-QAOA (XY-mixer, 厳密制約保持)
- **実験検証**: 100サイクルBBOスケーリング（$N=8 \sim 20$）、ペナルティ係数感度分析、ビット数スケーリング
- **自己完結設計**: ソースコード、実行結果（JSON/Checkpoints）、統合図および独立パネル図（PDF/PNG）、TeXレポート一式を内包

詳細な仕様および実行方法については、[`intern/README.md`](./intern/README.md) および [`intern/agent.md`](./intern/agent.md) をご参照ください。

