# QAOA_BBO コード構造・モジュール詳細解説仕様書

本ドキュメントでは、**QAOA_BBO** プロジェクトにおける全体アーキテクチャ、制御フロー、および構成されるすべての Python スクリプト（`.py`）と設定ファイル（`.yaml`）について、**コードの行レベル・関数レベルの動作、入出力仕様、数理的背景、実装上の工夫**を極めて詳細に解説します。

---

## 目次
1. [全体アーキテクチャとBBOクローズドループ](#1-全体アーキテクチャとbboクローズドループ)
2. [ファイル一覧と役割対照表](#2-ファイル一覧と役割対照表)
3. [各モジュールのコード詳細解説](#3-各モジュールのコード詳細解説)
   - 3.1 [`main.py`（BBOクローズドループ統括制御）](#31-mainpy-bboクローズドループ統括制御)
   - 3.2 [`config.yaml`（実験パラメータ一元管理）](#32-configyaml-実験パラメータ一元管理)
   - 3.3 [`bb_function.py`（ブラックボックス真の目的関数）](#33-bb_functionpy-ブラックボックス真の目的関数)
   - 3.4 [`fm.py`（Factorization Machine サロゲートモデル）](#34-fmpy-factorization-machine-サロゲートモデル)
   - 3.5 [`fm_to_qubo.py`（FMパラメータからQUBOへの厳密変換）](#35-fm_to_qubopy-fmパラメータからquboへの厳密変換)
   - 3.6 [`qaoa_solver.py`（Qiskit公式QAOAソルバー & 大規模バグ回避パッチ）](#36-qaoa_solverpy-qiskit公式qaoaソルバー--大規模バグ回避パッチ)
   - 3.7 [`utils.py`（ロギング & デバッグ用全探索検証ツール）](#37-utilspy-ロギング--デバッグ用全探索検証ツール)
4. [データの流れとテンソル・辞書構造の推移](#4-データの流れとテンソル辞書構造の推移)
5. [環境構築と実行コマンド](#5-環境構築と実行コマンド)

---

## 1. 全体アーキテクチャとBBOクローズドループ

本システムは、目的関数の解析的な数式が未知（ブラックボックス）であり、1回の評価に多大なコスト（合成実験や大規模シミュレーション）を要する最適化問題を解くための**量子・古典ハイブリッド型BBO（Black-Box Optimization）パイプライン**です。

```mermaid
graph TD
    A["<b>1. ブラックボックス関数 (bb_function.py)</b><br/>真の目的関数 y = x^T Q_bb x"] -->|初期サンプリング (10点)| B["<b>初期データセット (X_train, y_train)</b>"]
    
    subgraph BBO_Loop ["BBO反復サイクル (main.py: 1 〜 num_bbo_cycles)"]
        B --> C["<b>2. 代理モデル学習 (fm.py)</b><br/>TorchFM: 2次相互作用を O(kd) で学習<br/>MSE損失 + Adam 最適化"]
        C --> D["<b>3. QUBO変換 (fm_to_qubo.py)</b><br/>FMの重み W, V から<br/>QUBO辞書 {(i, j): weight} を抽出"]
        D --> E["<b>4. 量子最適化 (qaoa_solver.py)</b><br/>Qiskit Aer (C++シミュレータ) + COBYLA<br/>最適解 x_next をサンプリング"]
        E -->|次の一手 x_next| F["<b>5. 実験評価 (bb_function.py)</b><br/>y_new = evaluate_bb(x_next, Q_bb)"]
        F -->|データセットに追加| B
        F -->|最良値の更新判定| H["<b>Best-so-far 履歴の更新</b>"]
    end
    
    H --> I["<b>6. 出力・可視化 (main.py)</b><br/>result/ 配下にログ, JSON, CSV, グラフ(PNG/PDF) を保存"]
```

### クローズドループの処理シーケンス
1. **初期探索**: 未知の関数空間 $\{0, 1\}^d$ からランダムに少数の解候補をサンプリングし、真の関数値を取得。
2. **モデル学習**: 収集された入力 $\bm{x}$ と評価値 $y$ から、特徴量間の2次相互作用を捉える **Factorization Machine (FM)** を学習。
3. **数理最適化問題化**: 学習したFMの重み行列から、QAOAが解ける **QUBO（二次制約なし二値最適化）** 行列を抽出。
4. **量子サンプリング**: QUBOをIsingハミルトニアンに変換し、**QAOA** で基底状態（最小エネルギー状態）をサンプリングして「次に実験すべき有望候補 $\bm{x}_{\mathrm{next}}$」を決定。
5. **評価とフィードバック**: 提案された候補を真の関数で評価し、観測データに追加。このループを繰り返すことで、最小の実験回数で大域的最適解へ到達します。

---

## 2. ファイル一覧と役割対照表

| ファイル名 | 行数 | 主要クラス / 主要関数 | 役割と責務 |
| :--- | :---: | :--- | :--- |
| [`main.py`](./main.py) | 184 | `main()` | BBOループ全体の統括、データ管理、ログ・プロット・CSV/JSON保存 |
| [`config.yaml`](./config.yaml) | 20 | 設定辞書 | 実験条件（次元数、学習率、QAOA設定、反復回数等）の一元管理 |
| [`bb_function.py`](./bb_function.py) | 31 | `create_random_qubo_bb`<br>`evaluate_bb`<br>`generate_dataset`<br>`get_exact_minimum_bb` | 真のブラックボックス関数の定義（ランダムQUBO）、高速評価、正解データの全探索 |
| [`fm.py`](./fm.py) | 34 | `TorchFM(nn.Module)` | PyTorchによる Factorization Machine 実装。2次相互作用の $O(kd)$ 計算 |
| [`fm_to_qubo.py`](./fm_to_qubo.py) | 41 | `fm_to_qubo` | 学習済みFMパラメータ（$W, V$）からQUBO辞書 `{(i, j): weight}` への厳密変換 |
| [`qaoa_solver.py`](./qaoa_solver.py) | 76 | `_safe_eigenvector_to_solutions`<br>`solve_qubo_qaoa` | Qiskit公式QAOAソルバーの実行、$d \ge 20$ における IndexError 回避パッチ |
| [`utils.py`](./utils.py) | 30 | `Logger`<br>`get_fm_minimum` | 標準出力とログファイルの二重書き込み、FM予測上の理論最小値算出 |

---

## 3. 各モジュールのコード詳細解説

---

### 3.1 `main.py`（BBOクローズドループ統括制御）

本スクリプトは、実験の開始から終了までの全工程を制御するエントリポイントです。

#### (1) ディレクトリ作成とロギング設定（18〜34行目）
```python
os.makedirs(os.path.join("result", "txt"), exist_ok=True)
os.makedirs(os.path.join("result", "json"), exist_ok=True)
os.makedirs(os.path.join("result", "png"), exist_ok=True)
os.makedirs(os.path.join("result", "pdf"), exist_ok=True)
os.makedirs(os.path.join("result", "csv"), exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join("result", "txt", f"output_{timestamp}.txt")

original_stdout = sys.stdout
sys.stdout = Logger(log_path)
```
- **詳細**: `result/` 配下に拡張子別の整理用ディレクトリを自動生成します。
- **ポイント**: `sys.stdout` を `utils.py` の `Logger` クラスで置き換えることで、以降のすべての `print()` 出力がターミナル画面に表示されると同時に、タイムスタンプ付きのテキストログファイルへリアルタイムに追記されます。

#### (2) 設定ファイルの絶対パス解決とシード固定（35〜58行目）
```python
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir, "config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

seed_val = config.get("seed", 42)
torch.manual_seed(seed_val)
np.random.seed(seed_val)
```
- **詳細**: `os.path.abspath(__file__)` を起点として `config.yaml` を特定します。これにより、ターミナルがどのカレントディレクトリで開かれていても `FileNotFoundError` を起こさず確実に実行できます。
- **再現性**: PyTorch と NumPy の乱数シードを固定し、実験結果の完全な再現性を保証します。

#### (3) 真の関数の初期化と理論最小値の計算（56〜78行目）
```python
Q_bb = create_random_qubo_bb(d, seed=seed_val)
best_x_bb, min_val_bb = get_exact_minimum_bb(Q_bb, d)
X_train, y_train = generate_dataset(Q_bb, num_initial_samples, d)

X_train_list = [X_train]
y_train_list = [y_train]
current_best_bb = torch.min(y_train).item()
history_bb_min = [current_best_bb]
```
- **詳細**:
  - `create_random_qubo_bb`: $d \times d$ のランダム行列 $Q_{\mathrm{bb}}$ を真の関数として作成。
  - `get_exact_minimum_bb`: 全 $2^d$ 通りの総当たり探索を実行し、ベンチマークの正解値（真の最小値 `min_val_bb`）を事前に算出。
  - `generate_dataset`: 初期観測データ（デフォルト: 10点）を生成。
  - `history_bb_min`: 各サイクル時点での「これまでに見つかった最良の目的関数値（Best-so-far）」を追跡記録するリスト。

#### (4) BBO反復ループ（81〜133行目）
```python
for cycle in range(1, num_bbo_cycles + 1):
    X_current = torch.cat(X_train_list, dim=0)
    y_current = torch.cat(y_train_list, dim=0)
    
    # 1. FMモデルの新規初期化と学習
    model = TorchFM(d=d, k=k)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_current)
        loss = criterion(preds, y_current)
        loss.backward()
        optimizer.step()

    # 2. QUBOへの変換
    qubo_dict, offset = fm_to_qubo(model)
    
    # 3. QAOAによる有望候補の探索
    result = solve_qubo_qaoa(qubo_dict, offset=offset, reps=qaoa_reps, maxiter=qaoa_maxiter)
    best_sample = result.samples[0]
    qaoa_x_tuple = tuple(int(xi) for xi in best_sample.x)
    
    # 4. 真のブラックボックス関数で評価
    y_new_val = evaluate_bb(np.array(qaoa_x_tuple, dtype=np.float32).reshape(1, -1), Q_bb)[0]
    
    # 5. データセットの更新
    X_train_list.append(torch.tensor(qaoa_x_array))
    y_train_list.append(torch.tensor([y_new_val], dtype=torch.float32))
    if y_new_val < current_best_bb:
        current_best_bb = y_new_val
    history_bb_min.append(current_best_bb)
```
- **設計思想**:
  - **モデルの再初期化**: サイクルごとに `TorchFM` を新規インスタンス化しています。これにより、前サイクルの局所解へのパラメータ固着を防ぎ、新たに追加されたデータ点を反映したフラットな学習を実現します。
  - **最良解のサンプリング**: Qiskit の `result.samples` はエネルギー（目的関数値）が低い順にソートされているため、`result.samples[0]` が QAOA の最適予測解となります。

#### (5) 多面的な結果保存とプロット（134〜180行目）
- **JSON保存**: 実験条件、真の最小値、各サイクルの提案ビット列、FM予測値、BB評価値をすべて構造化データとして保存。
- **CSV保存**: サイクル番号と Best-so-far 値の時系列テーブル。論文や外部描画ツールで利用可能。
- **グラフ描画 (PNG / PDF)**: 最適化履歴の折れ線グラフに加え、真の最小値を表す赤い水平点線（`Exact Minimum`）を同時にプロット。

---

### 3.2 `config.yaml`（実験パラメータ一元管理）

```yaml
seed: 42                  # 乱数シード（実験の再現性確保）
d: 5                      # 最適化問題のバイナリ変数数（量子ビット数 N = d）
k: 2                      # FMモデルの潜在因子次元数（特徴量の埋め込みサイズ）
num_initial_samples: 10   # ループ開始前に取得する初期観測データ数
num_bbo_cycles: 5         # BBOを回す反復サイクル数
epochs: 150               # 毎サイクルのFM学習エポック数
lr: 0.1                   # FM学習の学習率 (Adam オプティマイザ)
qaoa_reps: 1              # QAOAのレイヤー数 (p: コスト演算子とミキサーの適用段数)
qaoa_maxiter: 50          # QAOA内部の古典オプティマイザ (COBYLA) の最大反復回数
```

---

### 3.3 `bb_function.py`（ブラックボックス真の目的関数）

本モジュールは、実験室での合成実験や大規模シミュレーションを模した「真の関数」を担当します。

#### `create_random_qubo_bb(d: int, seed: int = 42) -> np.ndarray`
```python
np.random.seed(seed)
Q = np.random.uniform(-1, 1, size=(d, d))
Q = np.triu(Q)
return Q
```
- **数理仕様**: 一様分布 $\mathcal{U}(-1, 1)$ に従う $d \times d$ 行列を生成し、`np.triu()` で上三角化します。
- **理由**: QUBO において $x_i x_j$ ($i \neq j$) の項は $Q_{ij} + Q_{ji}$ としてまとめることができるため、上三角行列 ($i \le j$) に統一しておくことで二重カウントを防止し、パラメータ表現を一意にします。

#### `evaluate_bb(x: np.ndarray, Q: np.ndarray) -> np.ndarray`
```python
return np.einsum('ni,ij,nj->n', x, Q, x)
```
- **数理仕様**:
  $$y_n = \sum_{i=1}^d \sum_{j=1}^d x_{ni} Q_{ij} x_{nj}$$
- **実装の工夫（`np.einsum`）**:
  - 通常 Python の 2 重 for ループで計算すると $O(N d^2)$ の実行時間がかかり極めて低速です。
  - アインシュタインの縮約記法 `np.einsum` を用いることで、C言語の低レベル最適化ループによりバッチサイズ $N$ の入力をミリ秒未満で並列評価します。

#### `generate_dataset(Q: np.ndarray, num_samples: int, d: int, seed: int = 42)`
```python
X = np.random.randint(0, 2, size=(num_samples, d)).astype(np.float32)
y = evaluate_bb(X, Q).astype(np.float32)
return torch.tensor(X), torch.tensor(y)
```
- **出力形状**:
  - `X`: `torch.Tensor` (形状: `(num_samples, d)`)
  - `y`: `torch.Tensor` (形状: `(num_samples,)`)

#### `get_exact_minimum_bb(Q: np.ndarray, d: int)`
```python
all_patterns = np.array(list(itertools.product([0, 1], repeat=d)), dtype=np.float32)
y_all = evaluate_bb(all_patterns, Q)
min_idx = np.argmin(y_all)
return all_patterns[min_idx], y_all[min_idx]
```
- **数理仕様**: 直積 `itertools.product` を用いて、探索空間の全 $2^d$ 通りのバイナリベクトルを一括生成して評価。理論上の最小値 $y^*$ と、それを与える最適解 $\bm{x}^*$ を特定します。

---

### 3.4 `fm.py`（Factorization Machine サロゲートモデル）

Factorization Machine（Rendle, 2010）は、疎なデータや二値特徴量間の**2次相互作用を高精度かつ低計算量でモデル化**する機械学習手法です。

#### 数理モデル
$$\hat{y}(\bm{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle \bm{v}_i, \bm{v}_j \rangle x_i x_j$$
- $w_0 \in \mathbb{R}$: グローバルバイアス（定数項）
- $\bm{w} \in \mathbb{R}^d$: 各特徴量の1次線形重み
- $\bm{v}_i \in \mathbb{R}^k$: 各特徴量に割り当てられた $k$ 次元の潜在因子ベクトル（相互作用の内積モデル）

#### $O(kd)$ 計算トリックの数理導出
通常の多項式回帰では、$\frac{d(d-1)}{2}$ 個のすべての特徴量ペアの重みを個別に学習するため $O(k d^2)$ の計算量が必要となります。FM では以下の等式展開により、**$O(kd)$ の線形時間**で計算します：

$$\begin{aligned}
\sum_{i=1}^d \sum_{j=i+1}^d \langle \bm{v}_i, \bm{v}_j \rangle x_i x_j 
&= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d \left( \sum_{f=1}^k v_{i, f} v_{j, f} \right) x_i x_j - \frac{1}{2} \sum_{i=1}^d \left( \sum_{f=1}^k v_{i, f} v_{i, f} \right) x_i^2 \\
&= \frac{1}{2} \sum_{f=1}^k \left[ \left( \sum_{i=1}^d v_{i, f} x_i \right)^2 - \sum_{i=1}^d v_{i, f}^2 x_i^2 \right]
\end{aligned}$$

#### コード実装（`TorchFM.forward`）
```python
class TorchFM(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(d, k, device=device))
        self.lin = nn.Linear(d, 1).to(device)
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, active_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # term1: (sum_i v_{i,f} * x_i)^2 の f に関する総和
        term1_inner = x @ self.V                                              # (Batch, k)
        term1 = torch.sum(term1_inner * term1_inner, dim=1, keepdim=True)     # (Batch, 1)

        # term2: sum_i (v_{i,f}^2 * x_i^2) の f に関する総和
        term2_inner = x.pow(2) @ self.V.pow(2)                                # (Batch, k)
        term2 = torch.sum(term2_inner, dim=1, keepdim=True)                   # (Batch, 1)

        # 相互作用項の結合
        interaction = 0.5 * (term1 - term2)                                   # (Batch, 1)

        # 線形項とバイアスを加算してフラットな1次元テンソルとして出力
        return (interaction + self.lin(x)).view(-1)
```

---

### 3.5 `fm_to_qubo.py`（FMパラメータからQUBOへの厳密変換）

学習したサロゲートモデル $\hat{y}(\bm{x})$ を量子ソルバー（QAOA）に入力するため、QUBO 行列の各要素へ等価変換します。

#### 数理的対応関係
バイナリ変数 $x_i \in \{0, 1\}$ においては $x_i^2 = x_i$ が常に成立するため、FM の数式はそのまま QUBO 形式 $\bm{x}^T Q \bm{x} + \text{offset}$ と完全に一致します：

$$\hat{y}(\bm{x}) = \underbrace{w_0}_{\text{offset}} + \sum_{i=1}^d \underbrace{w_i}_{Q_{ii}} x_i + \sum_{i < j} \underbrace{\left( \sum_{f=1}^k v_{i, f} v_{j, f} \right)}_{Q_{ij}} x_i x_j$$

#### コード実装
```python
def fm_to_qubo(model: torch.nn.Module) -> Tuple[Dict[Tuple[int, int], float], float]:
    linear_weights = model.lin.weight.detach().cpu().numpy().flatten()
    bias = model.lin.bias.detach().cpu().numpy().item()
    v_matrix = model.V.detach().cpu().numpy()
    d = v_matrix.shape[0]
    
    qubo = {}
    # 1次対角項 (i == j)
    for i in range(d):
        qubo[(i, i)] = float(linear_weights[i])
        
    # 2次非対角項 (i < j): 潜在ベクトルの内積
    for i in range(d):
        for j in range(i + 1, d):
            interaction = np.dot(v_matrix[i], v_matrix[j])
            qubo[(i, j)] = float(interaction)
            
    return qubo, bias
```
- **戻り値**:
  - `qubo`: `{(0, 0): -0.5, (0, 1): 0.35, ...}` のような辞書形式。Qiskit Optimization に直接渡せる構造です。
  - `bias`: オフセット（定数項）。

---

### 3.6 `qaoa_solver.py`（Qiskit公式QAOAソルバー & 大規模バグ回避パッチ）

本モジュールは、QUBO 辞書を受け取り、IBM の量子アルゴリズムライブラリ（Qiskit Algorithms）を用いて基底状態を探索します。

#### (1) $d \ge 20$ における Qiskit 既知バグの回避パッチ（12〜25行目）
```python
import qiskit_optimization.algorithms.optimization_algorithm as oa

_orig_eigenvector_to_solutions = oa.OptimizationAlgorithm._eigenvector_to_solutions

def _safe_eigenvector_to_solutions(eigenvector, qubo, min_probability=1e-06):
    solutions = _orig_eigenvector_to_solutions(eigenvector, qubo, 0.0)
    if not solutions:
        solutions = _orig_eigenvector_to_solutions(eigenvector, qubo, -1.0)
    return solutions

oa.OptimizationAlgorithm._eigenvector_to_solutions = staticmethod(_safe_eigenvector_to_solutions)
```
- **深刻な既知問題の背景**:
  - Qiskit Optimization の `OptimizationAlgorithm._eigenvector_to_solutions` には、`min_probability=1e-06` という測定確率の足切り閾値がハードコードされています。
  - 変数数が $d \ge 20$（$2^{20} = 1,048,576$ 通り）に達すると、状態空間が巨大化するため、どの状態の測定確率も $1/2^{20} \approx 9.5 \times 10^{-7} < 10^{-6}$ となります。
  - この結果、**すべての測定サンプルがノイズと判定されて除外され、解のリストが空（`[]`）になって `IndexError: list index out of range` で異常終了する**という致命的な不具合が存在します。
- **本実装の解決策**:
  - クラスメソッドを動的に差し替え（モンキーパッチ）、閾値を `0.0`（確率が微小でも全て保持）にして実行し、万一空の場合は `-1.0` でフォールバックします。これにより、$d=20 \sim 30$ の規模でもクラッシュすることなく安定して解を出力できます。

#### (2) `solve_qubo_qaoa` 関数の実装（24〜76行目）
```python
def solve_qubo_qaoa(qubo_dict: Dict[Tuple[int, int], float], offset: float = 0.0, reps: int = 1, maxiter: int = 100):
    # 1. 変数数の自動特定
    max_idx = max(max(i, j) for i, j in qubo_dict.keys())
    num_vars = max_idx + 1

    # 2. QuadraticProgram の構築
    qp = QuadraticProgram()
    for i in range(num_vars):
        qp.binary_var(name=f"x_{i}")
    linear = {f"x_{i}": w for (i, j), w in qubo_dict.items() if i == j}
    quadratic = {(f"x_{i}", f"x_{j}"): w for (i, j), w in qubo_dict.items() if i != j}
    qp.minimize(constant=offset, linear=linear, quadratic=quadratic)
    
    # 3. QAOAアルゴリズムの設定
    optimizer = COBYLA(maxiter=maxiter)
    sampler = AerSampler()  # 高速な C++ シミュレータ (qiskit-aer)
    pm = generate_preset_pass_manager(optimization_level=1, target=AerSimulator().target)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, transpiler=pm)
    
    # 4. 最適化の実行
    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)
    return result
```
- **コンポーネント構成**:
  - `QuadraticProgram`: Qiskit Optimization の標準問題形式。
  - `COBYLA`: 勾配を必要としない頑健な古典シンプレックス最適化器。
  - `AerSampler`: `qiskit-aer` による高速な状態ベクトル・サンプリングシミュレータ。
  - `generate_preset_pass_manager`: Qiskit 1.x 推奨のトランスパイルパイプライン（回路のゲート圧縮と最適化を実施）。

---

### 3.7 `utils.py`（ロギング & デバッグ用全探索検証ツール）

#### `Logger` クラス
```python
class Logger:
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
```
- **役割**: `sys.stdout` をフックし、`print()` の出力を画面にそのまま流しながら、裏でログファイルにも即時書き込み（`flush`）します。プログラムが途中で強制終了しても、直前までのログが完全にディスクに残ります。

#### `get_fm_minimum` 関数
```python
def get_fm_minimum(model: torch.nn.Module, d: int):
    all_patterns = list(itertools.product([0, 1], repeat=d))
    all_X = torch.tensor(all_patterns, dtype=torch.float32)
    with torch.no_grad():
        preds = model(all_X)
    min_idx = torch.argmin(preds).item()
    return all_patterns[min_idx], preds[min_idx].item()
```
- **目的（切り分け分析）**:
  - BBO の探索が真の最小値に到達しなかった場合、「サロゲートモデル（FM）の学習が外れているのか？」それとも「サロゲートモデルは合っているが QAOA が最適解を見逃したのか？」という原因特定が極めて重要になります。
  - 本関数で「学習済みFMが予測する理論最小値」を全探索で算出しておくことで、**FMのモデリング誤差**と**QAOAのサンプリング探索誤差**を完全に切り分けて定量分析できます。

---

## 4. データの流れとテンソル・辞書構造の推移

パイプライン内部でデータがどのように変換されるかの詳細な一覧です：

| ステップ | 入力データ構造 | 処理モジュール | 出力データ構造 | 例・形状 |
| :---: | :--- | :---: | :--- | :--- |
| 1 | 次元数 $d$ | `bb_function.create_random_qubo_bb` | 上三角 NumPy 配列 `Q_bb` | `shape: (5, 5)` |
| 2 | 行列 `Q_bb`, サンプル数 $N$ | `bb_function.generate_dataset` | PyTorch テンソル `(X, y)` | `X: (10, 5), y: (10,)` |
| 3 | テンソル `(X, y)` | `fm.TorchFM` | 学習済みモデルパラメータ | `W: (5, 1), V: (5, 2)` |
| 4 | モデル `TorchFM` | `fm_to_qubo.fm_to_qubo` | QUBO辞書 `qubo_dict`, `offset` | `{(0, 1): 0.42, ...}, 1.25` |
| 5 | `qubo_dict`, `offset` | `qaoa_solver.solve_qubo_qaoa` | Qiskit `OptimizationResult` | `result.x: [1, 0, 1, 0, 0]` |
| 6 | 提案ベクトル `x_next` | `bb_function.evaluate_bb` | スカラー値 $y_{\mathrm{new}}$ | `-3.4512` |
| 7 | 全サイクルの記録 | `main.py` | JSON / CSV / PNG / PDF | `result/` 配下のファイル |

---

## 5. 環境構築と実行コマンド

### 1. uv を使用する場合（推奨）
パッケージの解決から実行までが自動化されます：
```bash
uv run python main.py
```

### 2. 通常の pip / venv を使用する場合
```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# パッケージのインストール
pip install -r requirements.txt

# 実行
python main.py
```
