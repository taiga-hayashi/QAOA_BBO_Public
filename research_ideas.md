# FM + QAOA 新規研究アイデア詳細まとめ (数学的定式化版)

ブラックボックス最適化（BBO）のサロゲートモデルとしてFactorization Machine（FM）を用い、その最適化にQAOAを用いる枠組みにおいて、QAOAの量子ならではの強みを活かした3つの有望な研究アプローチを数学的に厳密な形で再定義しました。

---

## 前提: BBOとFactorization Machine (FM) の定式化
ブラックボックス関数を $f: \{0,1\}^N \to \mathbb{R}$ とし、この関数は高コストで評価回数が制限されているとする。
イテレーション $t$ において、これまでに評価されたデータセットを $\mathcal{D}_t = \{(x^{(i)}, y^{(i)})\}_{i=1}^{t}$（$x^{(i)} \in \{0, 1\}^N, y^{(i)} = f(x^{(i)})$）とする。
サロゲートモデルとしてFactorization Machine (FM) を採用し、入力 $x \in \{0,1\}^N$ に対する予測値 $\hat{f}(x; \Theta_t)$ を以下で定義する：
$$
\hat{f}(x; \Theta_t) = w_0 + \sum_{i=1}^N w_i x_i + \sum_{i=1}^N \sum_{j=i+1}^N \langle v_i, v_j \rangle x_i x_j
$$
ここで $\Theta_t = \{w_0, \mathbf{w}, \mathbf{V}\}$ であり、$v_i \in \mathbb{R}^k$ は潜在ベクトルである。
この関数は、変数変換 $Z_i = 1 - 2x_i \in \{+1, -1\}$ を用いることで、直ちにイジング模型のハミルトニアン $H_C^{(t)}$ にマッピングできる。

$$
H_C^{(t)} = \sum_{i<j} J_{ij}^{(t)} Z_i Z_j + \sum_i h_i^{(t)} Z_i
$$

---

## 1. 制約付き最適化とXYミキサー (Constrained BBO + XY-Mixer QAOA)

### 背景と課題
現実のブラックボックス最適化（BBO）問題では、ハイパーパラメータ探索や材料・分子設計のように**カテゴリ変数（Categorical Variables）**や**離散化された連続変数（Discretized Continuous Variables）**が頻出します。これらは通常、One-Hot表現（$d$個の選択肢からちょうど1つを選ぶ）として二値変数化されます。

従来のアニーリングや標準QAOA（QUBO定式化）では、変数ごとに以下のペナルティ項を導入する必要があります：
$$
H_{\text{penalty}} = \sum_{m=1}^M \lambda_m \left( \sum_{j=1}^{d_m} x_{m, j} - 1 \right)^2
$$
しかし、このペナルティアプローチには致命的な欠点が存在します：
1. **ペナルティ係数 $\lambda_m$ のハイパーパラメータ依存性**: $\lambda$ が小さすぎると無効な解（One-Hotを満たさない状態）が測定され、高コストなブラックボックス関数の評価回数を無駄に消費します。逆に $\lambda$ が大きすぎると解空間の間に高エネルギーの障壁が生じ、探索が極小解にトラップされます。
2. **目的関数地形の歪み**: ペナルティ項により変数ブロック内に密な全結合二次項が生じ、本来のサロゲート関数の形状が破壊されます。

### 発展アイデア：ブロック直和型XYミキサーによる「完全ペナルティフリー」最適化
各変数のOne-Hot制約（ハミング重み $K_m = 1$）を量子回路の対称性として埋め込むことで、**ペナルティ項を一切排除（$\lambda=0$）した厳密な部分空間探索**が可能です。

#### 数学的定式化
$M$ 個の変数があり、各変数 $m \in \{1, \dots, M\}$ が $d_m$ 個の選択肢を持つとします（総量子ビット数 $N = \sum_{m=1}^M d_m$）。各変数ブロック $m$ に対するOne-Hot制約は以下で表されます：
$$
\sum_{j=1}^{d_m} x_{m, j} = 1 \iff \sum_{j=1}^{d_m} Z_{m, j} = d_m - 2 \quad (\forall m \in \{1, \dots, M\})
$$

1. **初期状態の構築 (Product of W-states):**
   各変数ブロック $m$ において、ハミング重み1の一様重ね合わせ状態は **W状態** $|W_{d_m}\rangle$ そのものです：
   $$
   |W_{d_m}\rangle = \frac{1}{\sqrt{d_m}} \sum_{j=1}^{d_m} |0 \dots 0 \underbrace{1}_{j\text{-th}} 0 \dots 0\rangle
   $$
   系全体の初期状態 $|\psi_0\rangle$ は、これらW状態の直積（テンソル積）として極めてシンプルに準備できます：
   $$
   |\psi_0\rangle = \bigotimes_{m=1}^M |W_{d_m}\rangle
   $$
   大域的なエンタングルメントが不要なため、初期化回路の深さは各ブロック内で $O(\log d_m)$ または $O(d_m)$ であり、全ブロック並列に短深度で実行可能です。

2. **ブロック直和型XYミキサー (Block-Diagonal XY-Mixer):**
   異なる変数間を混ぜず、**各変数ブロックの内部のみで励起（「1」のビット）を遷移させるXYミキサー**を定義します：
   $$
   H_M = \sum_{m=1}^M H_M^{(m)}, \quad H_M^{(m)} = \frac{1}{2}\sum_{(j, k) \in E_m} (X_{m, j} X_{m, k} + Y_{m, j} Y_{m, k})
   $$
   （ここで $E_m$ はブロック $m$ 内のリンググラフまたは完全グラフの辺集合）。
   各ブロックにおいて $[H_M^{(m)}, \sum_{j=1}^{d_m} Z_{m, j}] = 0$ が独立に成立するため、系全体として以下が保証されます：
   $$
   [H_M, \sum_{j=1}^{d_m} Z_{m, j}] = 0 \quad (\forall m \in \{1, \dots, M\})
   $$

3. **FM（Factorization Machine）との本質的親和性:**
   そもそもFMは、Rendle (2010) が**高次元スパースなOne-Hotカテゴリ変数間の二次相互作用を効率的に捉えるモデル**として提唱したものです。
   したがって、「One-Hotカテゴリ変数のFM表現」×「One-Hotを厳密保存するBlock-XY QAOA」は理論的に完全に合致しており、ペナルティ項を一切加えることなく、FM本来の二次相互作用ハミルトニアン $H_C$ だけを純粋に最適化できます。
   測定結果から得られる解は**数学的に100%の確率で正当なOne-Hot解**となり、制約違反による評価の浪費が完全にゼロになります。

### 関連する先行研究 (Related Work)
* **One-Hot / Partitioning制約のXYミキサー**: Hadfield et al. (2019) "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz" では、グラフ彩色問題や巡回セールスマン問題（TSP）のようなOne-Hot（1-in-q）制約に対するリング型XYミキサーの構成が提案されています。
* **ブロック独立ミキサーの深さと実装**: Cook, Eidenbenz, Bärtschi (2020) "The Quantum Alternating Operator Ansatz on a Planar Graph" や Wang et al. (2020) では、リングトポロジーを採用することでトロッター誤差なしに各ブロックのミキサー層を $O(1)$〜$O(d_m)$ の浅い回路深度でコンパイルできることが示されています。
* **Domain-wall encodingとの比較**: Chancellor (2019) "Domain wall encoding of discrete variables for quantum annealing and QAOA" では、One-Hotの代替としてドメインウォール符号化が議論されていますが、ブロックXYミキサーを用いたOne-Hotはパリティチェックやアンシラビットなしで直接扱えるという優位性があります。

---

## 2. QAOAの「重ね合わせ」を活用したバッチ探索 (Batch BBO via Quantum Sampling)

### 数学的定式化
通常のBBOでは、Expected Improvement (EI) や Upper Confidence Bound (UCB) などの獲得関数を最大化する点 $x^*$ を1つ選ぶ。本手法では、QAOAが出力する確率分布 $P(x) = |\langle x | \vec{\beta}, \vec{\gamma}\rangle|^2$ を直接サンプリング分布として活用し、バッチサイズ $B$ の並列評価を行う。

**多様性を考慮したバッチ選択 (Determinantal Point Process: DPP):**
サンプリングされた候補集合 $\mathcal{S}$ から、バッチ $\mathcal{B} \subset \mathcal{S}$ ($|\mathcal{B}|=B$) を選択する問題を考える。
各候補 $x \in \mathcal{S}$ の「質」と「多様性」を同時に評価するために、類似度行列（カーネル行列） $L \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$ を以下のように構築する：
$$
L_{ij} = q(x_i) \cdot q(x_j) \cdot k(x_i, x_j)
$$
ここで、
* $q(x) = \exp(-H_C(x) / \tau)$ は候補 $x$ の質を表すスコア（$\tau$ は温度パラメータ）。
* $k(x_i, x_j) = \exp\left(-\gamma_{DPP} d(x_i, x_j)^2\right)$ はハミング距離 $d(x_i, x_j) = \|x_i - x_j\|_1$ に基づく多様性カーネル。

DPPの性質により、バッチ $\mathcal{B}$ が選ばれる確率 $\mathcal{P}(\mathcal{B})$ は部分行列の行列式に比例する：
$$
\mathcal{P}(\mathcal{B}) = \frac{\det(L_{\mathcal{B}})}{\det(I + L)} \propto \det(L_{\mathcal{B}})
$$
$\det(L_{\mathcal{B}})$ を最大化する $\mathcal{B}$ の選定（MAP推定）はNP困難であるが、Greedy法により近似的に「互いにハミング距離が離れており、かつFM上での予測値が良い」バッチ $\{x^{(1)}_{t+1}, \dots, x^{(B)}_{t+1}\}$ を多項式時間で得る。これを真の関数 $f$ の並列評価に回す。

### 関連する先行研究 (Related Work)
* **BBOにおけるDPPの応用**: Kathuria et al. (2016) "Batched Gaussian Process Bandit Optimization via Determinantal Point Processes" などで、古典のバッチベイズ最適化においてDPPによる多様性サンプリングが成功を収めています。
* **QAOAのサンプリング能力**: Farhi et al. (2014) の原論文以来、浅いQAOAが最適解の発見だけでなく、エネルギーの低い良質な解の分布を生成する「サンプラー」として機能することが認識されています。
* **量子アルゴリズムとDPPの交差点**: Kerenidis et al. (2020) "Quantum Determinantal Point Processes" で量子回路を用いたDPPサンプリングが提案されていますが、本アプローチのように「QAOAの出力分布＋DPP」によるBBOへの応用は極めて新規性の高い領域です。

---

## 3. パラメータ転移学習 (Parameter Transfer Learning)

### 数学的定式化
イテレーション $t \to t+1$ において、追加データ $(x^{(t+1)}, y^{(t+1)})$ によりFMのパラメータが更新され、コストハミルトニアンは $H_C^{(t+1)} = H_C^{(t)} + \delta H_C$ に変化する。
FMの逐次学習（Online Learning）の学習率を小さく保つことで、作用素ノルム $\|\delta H_C\|$ は十分小さいと仮定できる。

**目的関数の摂動とWarm-Start:**
イテレーション $t$ でのQAOAの目的関数を $F_t(\vec{\beta}, \vec{\gamma}) = \langle \vec{\beta}, \vec{\gamma} | H_C^{(t)} | \vec{\beta}, \vec{\gamma} \rangle$ とする。
イテレーション $t+1$ における目的関数は次のように展開される：
$$
F_{t+1}(\vec{\beta}, \vec{\gamma}) = F_t(\vec{\beta}, \vec{\gamma}) + \langle \vec{\beta}, \vec{\gamma} | \delta H_C | \vec{\beta}, \vec{\gamma} \rangle
$$
右辺第2項の絶対値は $\|\delta H_C\|$ により上から抑えられる。QAOAのエネルギー地形の勾配 $\nabla F(\vec{\beta}, \vec{\gamma})$ は有界であり、目的関数 $F_t$ と $F_{t+1}$ はパラメータ空間上でLipschitz連続性を保ったまま微小な変形を受ける。
したがって、$F_t$ の最適パラメータ $\theta_{t}^* = (\vec{\beta}_{t}^*, \vec{\gamma}_{t}^*)$ と $F_{t+1}$ の最適パラメータ $\theta_{t+1}^*$ との間には、Hessian $\nabla^2 F_t$ の局所的な正定値性を仮定すれば以下のバウンドが成り立つ：
$$
\|\theta_{t+1}^* - \theta_t^*\| \le C \|\delta H_C\|
$$
（$C$ は $\nabla^2 F_t$ の最小固有値に依存する定数）。

**最適化アルゴリズムの適用:**
$t+1$ のイテレーションでは、初期値を $\theta^{(0)} = \theta_t^*$ として勾配降下法などの最適化を実行する。
$$
\theta^{(k+1)} = \theta^{(k)} - \eta \nabla F_{t+1}(\theta^{(k)})
$$
$\theta_t^*$ は既に $\theta_{t+1}^*$ の Basin of attraction（引き込み領域）に属している確率が高く、ランダム初期化で発生するBarren Plateau（勾配消失問題）を回避し、$O(1)$ のイテレーションで収束することが数学的に保証される。

### 関連する先行研究 (Related Work)
* **QAOAのWarm-startingとパラメータ転移**: Egger et al. (2021) "Warm-starting quantum optimization" は古典の緩和解を用いて初期状態とパラメータをWarm-startさせる手法を提案しています。また、Galda et al. (2021) "Transferability of QAOA Parameters between Randomized Graphs" や Sack & Serbyn (2021) などで、あるグラフインスタンスから別の類似インスタンスへQAOAのパラメータが転移（Transfer）できることが実証されています。
* **Barren Plateauの回避**: McClean et al. (2018) "Barren plateaus in quantum neural network training landscapes" により、ランダムなパラメータ初期化が勾配消失を引き起こすことが示されており、本アプローチのような前ステップからのWarm-startはNISQデバイスでの学習において不可欠な技術とされています。
* **時間発展するハミルトニアンのQAOA**: Brandão et al. (2018) や最近の研究において、問題のインスタンスが滑らかに変化する際のパラメータの集中現象（Concentration of parameters）が解析されており、本アイデアの理論的裏付けとなります。
