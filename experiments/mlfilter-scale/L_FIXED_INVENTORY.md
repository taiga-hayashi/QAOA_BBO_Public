# 既存コードで L(層数)が固定されている箇所

- 作成日: 2026-10-20(Phase 0、計画の着手順 1)。コードの読み取りのみで、既存 study のコードは変更していない
- 対象: `Intern_Fujitsu/intern1016/multilayer_filter/` の py(Fujitsu_OpenQarp の `experiments/<study>/py/` と同じもの)と、それが読み込む Intern_Fujitsu 側の共通コード
- 6 層・24 ビット・4,096 候補は、変数・定数ではなくリテラルとして多数の箇所に書かれている。L を変えるには、各 study のコードを直すより L を引数に取る共通モジュールを新しく作る方が安全(Phase 1 の判断事項)

## 1. 評価器

| ファイル | 固定箇所 | Phase 0 での扱い |
|---|---|---|
| `py/evaluator.py` | `FilterEvaluator(layers=6)`、`1 <= layers <= 6` の範囲チェック | `experiments/mlfilter-scale/py/evaluator.py` に複製し、上限だけ 10 に変更。光学計算・FOM・float32 読み込みは同一。L=6 で既存の全件表と照合する(最大差 1e-11 以下) |
| `py/evaluator.py` | 入力を `OUT/json/source_snapshot.json` 経由で読む(Intern_Fujitsu の配置が前提) | `data/inputs/mlfilter/v1-public-code/manifest.json` の SHA-256 照合に置き換え |

## 2. 列挙・監査(`py/workflow.py`、`py/analyze.py`)

- `FilterEvaluator(6)`、`itertools.product(range(4), repeat=6)`、`range(0,4096,16)`、`/4096` の進捗表示
- 不正入力テスト `np.zeros(24)` など、代表構造 `[0,0,0,0,0,0]` などの 6 要素リテラル
- スカラー TMM の厚さ `[np.inf,*([50e-9]*6),np.inf]`
- 回路の事前検証ケース `[4]*6`
- `analyze.py`: `fom.shape==(4096,)`、`'candidates':4096`、`'n_onehot_variables':24`、`np.arange(1,4097)/4096`
- Phase 0 では `phase0.py enumerate` が L を引数に取り、同じ手順(全件、別バッチでの独立再計算、最良構造の全波長スカラー TMM)を行う

## 3. 固定 FM・サンプラー比較(`fixed_fm/py/experiment.py`、`analyze.py`)

- `SIZES=[4]*6`、`candidates()` の `repeat=6` と `FilterEvaluator(6)`
- `TorchFM(24, …)`、`helpers.matrix(qd,24)`、`np.arange(24)` のビット→状態番号の変換、`range(24)` による SA 出力の並べ替え、`reshape(-1,6,4)`
- カテゴリ SA `categorical()`: `range(4096)`、`strides=4**np.arange(5,-1,-1)`、`for g in range(6)`
- 記録値 `'original_onehot_bits':24`、`'simulation_register_bits':12` / `24`
- 通常 QAOA(`standard_search`): 2^24 の全状態ベクトルとエネルギー表を前提にしている。L=8 では 2^32(状態 64 GiB + エネルギー 32 GiB)
- `analyze.py`: `range(4096)`、`np.arange(24)`、`raw.shape==(100,24)`、`range(24)`

## 4. ペナルティ・再学習 FMQA(`fmqa_penalty/`、`fmqa_penalty_validation30/`)

- `fit()`・`sample()`: 学習とサンプルの次元 24、`raw.shape==(100,24)`
- `parent.FilterEvaluator(6)`、`range(4096)`、`random.Random(seed).sample(range(4096),40)`
- 検査用 `testbits.reshape(-1,6,4)`
- `analyze.py`: `data['predictions'].shape==(60,4096)`、`raw_bits.shape==(60,100,24)`、`TorchFM(24, …)`
- 提案サイクル 60・評価予算 100 は L ではなく予算の固定値(計画の「評価予算の扱い」で別に決める)

## 5. XY 診断・深さ(`sampler_diagnostic/`、`xy_depth/`)

- 事前検証ケース `[4]*6`、`np.arange(24)`、`raw.shape==(100,24)`
- `xy_depth/analyze.py` の件数チェック(`angles==61440` など)は Seed 数・予算に依存する定数

## 6. 配置の前提(L 以外)

- 各 study の py は Intern_Fujitsu リポジトリ内の相対パス(`REPO/src/...`、`intern_0924/src/...`、`intern_1009/perovskites/...`)を読み込む。Fujitsu_OpenQarp の `experiments/<study>/py/` は複製で、そのままでは実行できない(各 README にも記載済み)
- Phase 0 では、Intern_Fujitsu のうち必要なコードだけをサーバーの `/home/jovyan/FMA/Intern_Fujitsu_ref` に読み取り専用でコピーし、`config/paths.local.json` の `intern_fujitsu_root` で参照する。使ったファイルの SHA-256 は各 run の `metadata.json` に記録する
- 再利用した共通コード: `src/fm.py`、`src/fm_to_qubo.py`、`src/fmqa_solver.py`(+`bb_function.py`)、`src/qarp_backend.py`(QAOA_BBO_Public の版と SHA-256 が一致)、`intern_0924/src/solvers.py`。`run_pilot.py` の `matrix()` は 3 行で、perovskite 評価器の読み込みを避けるため `phase0.py` に同じ内容で置いた

## 7. Phase 1 に向けた提案(未決定)

- L を引数に取る共通モジュール(候補生成、エンコード、SIZES、カテゴリ SA の strides、ビット変換)を `src/` に作り、各比較の実行コードはそれを使う
- 既存 study のコードは変更しない(結果の意味を変えないため)
- 新しいコードでは L=6 の既存結果(固定 FM の予測値、SA の再現)を再現できることを先に確認する
