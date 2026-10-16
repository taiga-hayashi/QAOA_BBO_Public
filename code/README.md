# Perovskites FMQA / FMQAOA モジュール構成 (`code/`)

## uvでの環境構築・実行

`code/`内で実行します。`pyproject.toml`に依存関係、`uv.lock`に解決済みバージョン、
`.python-version`に既定のPython 3.12を指定しています。`uv sync`で`code/.venv`を作成します。

```bash
cd code
uv sync
uv run python main.py --help
```

光学多層フィルターで、FMQAの固定α=3・100とXY-QAOAのp=1・2・3を各10 Seed、
初期40件＋20回提案の閉ループBBOで比較する場合：

```bash
uv run python main.py --config config_multilayer_filter.json --set \
  mode=bbo n_seeds=10 \
  problem.initial_samples=40 problem.bbo_cycles=20 \
  'methods=["FMQA(alpha=3)","FMQA(alpha=100)","XY-FMQAOA(p=1,mean,powell49)","XY-FMQAOA(p=2,mean,powell49)","XY-FMQAOA(p=3,mean,powell49)"]' \
  output_dir=results/multilayer_fmqa_xy_10runs
```

同じ条件・同じ出力先で再実行すると保存済みチェックポイントから再開します。
既に依存パッケージを導入したPython環境を使う場合は、次のコマンドでも実行できます。
`uv sync`で作成した環境なら、先に`source .venv/bin/activate`を実行してください。

```bash
python3 main.py --config config_multilayer_filter.json --set \
  mode=bbo \
  n_seeds=10 \
  problem.initial_samples=40 \
  problem.bbo_cycles=20 \
  'methods=["FMQA(alpha=3)","FMQA(alpha=100)","XY-FMQAOA(p=1,mean,powell49)","XY-FMQAOA(p=2,mean,powell49)","XY-FMQAOA(p=3,mean,powell49)"]' \
  output_dir=results/multilayer_fmqa_xy_10runs
```

固定FMの1ステップ比較は`mode=fixed_fm`に変更し、別の`output_dir`を指定してください。
サーバーへコピーする際は`pyproject.toml`、`uv.lock`、`.python-version`も含め、
環境自体の`.venv`はコピーせず、コピー先で`uv sync --locked`を実行します。


SSH接続先の計算サーバーへ `code/` ディレクトリごと配置して計算を回し、出力された結果フォルダ（`results/<run_name>/`）をローカルに持ち帰って [`plot_results.ipynb`](plot_results.ipynb) で可視化・統計検定を行うための完全自己完結型（Self-Contained）コードベースです。

---

## ディレクトリ構成

`code/` の外側のファイルには一切依存しないため、`code/` フォルダ単体をそのままサーバーへ `scp` / `rsync` できます。

| ファイル / フォルダ | 役割 |
|---|---|
| [`data/data.csv`](data/data.csv), [`data/config.json`](data/config.json) | ペロブスカイト問題データ（23 One-Hot変数: `[16, 3, 4]`、全192候補） |
| [`config.json`](config.json) | すべての実験条件（Seed数・FM・ペナルティスイープ・XYアブレーション・SA・QAOA・BBO反復数）の設定 |
| [`config.py`](config.py) | JSON設定の読み込み・バリデーション（`XY-FMQAOA` の $\lambda=0$ 強制）・CLI上書き (`--set`) |
| [`evaluator.py`](evaluator.py) | Black-Box評価器、One-Hotエンコード、未評価候補の選択、Normalized Regret、**FMサロゲート精度診断（Spearman順位相関 $\rho_t$・FM最小解の真値ランク）** |
| [`fm_model.py`](fm_model.py) | 2次 Factorization Machine (`TorchFM`) の定義と決定的学習 (`train_fm`) |
| [`qubo.py`](qubo.py) | FMからのQUBO抽出 (`fm_to_qubo`)、内部Isingスケール $S_t$ 算出 (`qubo_to_ising_scale`)、One-Hotペナルティ付きBQM構築 |
| [`sa_sampler.py`](sa_sampler.py) | D-Wave Neal 古典SAサンプラー（任意の正規化ペナルティ係数 $\alpha$ に対応） |
| [`qarp_backend.py`](qarp_backend.py) | OpenQARP 量子回路シミュレータバックエンド（`StandardQAOA`, `XYQAOA`, `CompactXYQAOA`） |
| [`qaoa_sampler.py`](qaoa_sampler.py) | QAOAサンプラー（深さ $p$、目的関数 `mean`/`cvar`、角度探索 `legacy9`/`wide49`/`powell49`/`powell1024`、`shots` 対応） |
| [`main.py`](main.py) | サーバー実行用エントリポイント（Seed完走ごとに `csv/` と `json/` を逐次保存・自動レジューム対応） |
| [`plot_results.ipynb`](plot_results.ipynb) | **ローカル可視化用 Jupyter Notebook**（`results/<run_name>/` を読み込み、統計表とPDF/SVG/PNGプロットを生成） |
| [`requirements.txt`](requirements.txt) | 依存Pythonパッケージ一覧 |

---

## サーバーでの主要3実験コマンド例

### 1. 主実験：5手法（`Exact-FM` を含む）の閉ループBBO比較（10 Seed または 20 Seed）
`Exact-FM`（FMサロゲートの未評価実行可能最小解を厳密列挙で選択する理論上限ベースライン）がデフォルトで含まれます。`n_seeds=10`（デフォルト）や `n_seeds=20` を指定すると、Wilcoxon符号付順位検定で多重比較補正後も統計的有意差（$p < 0.05$）を判定可能な検出力が得られます。
```bash
# 全5手法（Exact-FM, TunedPenalty-FMQA, LargePenalty-FMQA, Penalty-FMQAOA, XY-FMQAOA: 10 Seed × 20サイクル）
python main.py --set n_seeds=10 output_dir=results/bbo_main_10seeds

# 高速な4手法（Penalty-FMQAOA を除く 20 Seed × 20サイクル）
python main.py --set n_seeds=20 'methods=["Exact-FM","TunedPenalty-FMQA","LargePenalty-FMQA","XY-FMQAOA"]' output_dir=results/bbo_fast_20seeds
```

### 2. `XY-FMQAOA` アブレーション実験（深さ $p=1,3 \times \text{Mean}/\text{CVaR}_{0.1} \times \text{角度探索点数}$）
`xy_ablation.enabled=true` を指定すると、`Exact-FM`, `TunedPenalty-FMQA`, `LargePenalty-FMQA` に加えて以下の `XY-FMQAOA` 6条件（すべて $\lambda=0$）を一括で閉ループBBO（または `mode=fixed_fm`）上で比較します：
- `XY-FMQAOA(p=1,mean,legacy9)`
- `XY-FMQAOA(p=1,mean,wide49)`
- `XY-FMQAOA(p=1,cvar,wide49)`
- `XY-FMQAOA(p=3,mean,powell49)`
- `XY-FMQAOA(p=3,cvar,powell49)`
- `XY-FMQAOA(p=3,cvar,powell1024)`

```bash
python main.py --set xy_ablation.enabled=true n_seeds=10 output_dir=results/xy_ablation_bbo
```

### 3. FMQAのペナルティ係数 $\alpha$ 依存性実験 (`penalty_sweep.enabled=true`)
`penalty_sweep.alphas`（`[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]`）とペナルティ不要の `XY-FMQAOA` ($\lambda=0$) を一括比較します。
```bash
# 閉ループBBOでのペナルティ係数依存性
python main.py --set penalty_sweep.enabled=true n_seeds=10 output_dir=results/penalty_sweep_bbo

# 固定FM（1ステップ診断）でのペナルティ係数依存性
python main.py --set mode=fixed_fm problem.initial_samples=20 penalty_sweep.enabled=true n_seeds=10 output_dir=results/penalty_sweep_fixed_fm
```

---

## ローカル [`plot_results.ipynb`](plot_results.ipynb) で出力される表・プロット一覧

サーバーから `results/<run_name>/` をローカルの `code/results/` に取得し、`plot_results.ipynb` の `RUN_DIR` を指定して全セルを実行すると、`RUN_DIR/{pdf,svg,png}/` に以下が自動保存されます：

1. **統計サマリ表 & Wilcoxon符号付順位検定表**（`Final Best`, `Normalized Regret`, `FM-Min Hit Rate`, `Spearman Rho`, `FM-Min True Rank`, `Raw Feasible Rate` の中央値 [IQR] と Bonferroni 補正済み $p$ 値）
2. **`bbo_convergence.{pdf,svg,png}`**: 評価予算に対する Best Objective と Normalized Regret の中央値＋IQR収束曲線
3. **`surrogate_quality_progression.{pdf,svg,png}`**: 各BBOステップにおける未評価候補への **FMサロゲート Spearman 順位相関 $\rho_t$** と **FM最小解の真値ランク** の推移
4. **`sampler_diagnostics_bar.{pdf,svg,png}`**: FM最小解捕捉率・実行可能解率・最終Regretの比較棒グラフ（IQRエラーバー＋Seed個別点）
5. **`per_seed_paired_regret.{pdf,svg,png}`**: Seed別の最終 Normalized Regret ペア比較
6. **`fmqa_penalty_alpha_dependence.{pdf,svg,png}`**: 横軸 $\alpha$ 対数スケールに対する実行可能解率・FM最小解捕捉率・Regret/FM Gapのトレードオフ曲線 ＋ `XY-FMQAOA` ($\lambda=0$) 基準線

## 光学多層フィルター問題の選択

`config_multilayer_filter.json`で、6層×4材料（24 One-Hotビット、実行可能4,096候補）の問題を選択できます。
各層50 nm、全厚300 nm、材料順はSiO2 / Si3N4 / Al2O3 / TiO2、層順は入射側から基板側です。
400–750 nmを透過する目標に対し、公開コードの太陽光スペクトルで重み付けした二乗残差FOMを最小化します。
論文の全厚1,200 nmの設定とは異なります。

```bash
cd code
python main.py --config config_multilayer_filter.json
```

これは実験を開始するコマンドです。問題登録時には実行していません。
設定は初期40点＋最大60提案、既存と共通のFM学習・候補選択を使います。
XYはp=1 / Mean / wide49、λ=0。既存の5手法構成はTunedPenalty-FMQAを含む構成で、
Adaptive-FMQAを含む必須4手法の主比較とは区別します。本実験の実行前に回路の事前検証が必要です。

`data/multilayer_filter/data.csv`は、既存の検証済み光学シミュレーション値の全件を候補順を保って収録した参照表です。
実行時は参照表から値を取得し、新しい光学シミュレーションは行いません。評価回数は参照した候補の回数です。
全件の真値は評価器・事後診断で使用し、FM学習には評価済み候補の値だけを渡します。
参照表方式の所要時間を、光学シミュレーションを含む実時間として解釈しないでください。
出典・ハッシュは`provenance.json`、詳細条件は同じフォルダの`protocol.json`に保存しています。
元コード・光学データの利用条件は未解決のため、ローカル研究用とし、外部公開は行いません。

既定の`config.json`は引き続きペロブスカイト問題です。出力先も問題別です。
Seed記録にデータセットSHA-256を保存し、異なる問題の結果を再利用しないようにしています。
旧形式のSHA-256なしの記録は再利用対象外です。

## 閉ループBBO予備比較（5 Seed、初期40＋20提案、QAOA p=1/2/3）

合意条件は`results/multilayer_filter_bbo_pilot5/json/run_config.json`に固定します。
Seedは42・101・2024・7・19。古典3条件（Exact-FM、Adaptive-FMQA、LargePenalty-FMQA）と、
標準X-Mixer / XY-Mixerそれぞれp=1・2・3の6条件、合計9条件×5 Seed＝45 Runです。
p=1の規定4手法比較を含み、p=2・3は明示的に追加した深さ比較です。
Adaptiveは`intern_0924`の既存規則を`shared_numerics.py`に同一ソースで収録し、初期α=1から実行可能率で更新します。
QAOAは全条件Mean・powell49・100 shots、標準方式α=5、XYはλ=0。SAは100 reads・1,000 sweepsです。
角度探索は9組のp1埋め込み＋16組の全層大域探索＋24回のPowell探索です。
同じ49回でも深い回路は探索する角度が多く、最適化不足の可能性があるため、回路表現力だけの比較とは解釈しません。
`config_multilayer_filter.json`は問題登録時の元設定で、今回の実行は専用の凍結設定を使います。

標準QAOAのエネルギー構築には既存の再帰処理を使い、2^24×24のビット列行列を生成しません。
`tests/test_prevalidation.py`ではN=6/9の各深さの独立回路・符号・ペナルティ・制約保存、FM-QUBO一致、中断再開、および小規模な9条件BBOから監査・3形式の作図までを検証します。
`prepare_multilayer_bbo.py`はその検証と24ビットの実FMによる各深さの動作確認を行い、初期データ・モデルハッシュ・設定・ソース・環境を保存します。

実行は`bbo_job.py calculate --seed 42`のようにSeed別です。各Seedの設定は`json/run_config_seed42.json`などに保存します。
各Seedのデータは`seeds/seed42/`などへ保存します。全Seedの成功後、`bbo_job.py analyze`が監査・集計・作図を行います。
既存停止キューと分けたpueueグループで、1 Seedずつ逐次実行します。既存のキューや並列数は変更しません。
1 Seedにつき48時間で打ち切り、途中データと失敗ログを残します。全体には複数日かかる可能性があります。
提案ごとにJSONを原子的に保存し、各手法の最新モデルを各Seedの`checkpoints/`へ保存します。
再開時は設定とデータのハッシュを確認し、保存済み提案の次から決定的に再構築します。

`analyze_bbo.py`が全Seed・全条件・全提案の評価数・真値・ペナルティ・初期FMの共通性を監査し、
中央値/IQR、全36ペアの対応差とBonferroni補正Wilcoxon、PDF/SVG/PNG、RESULTS/DISCUSSIONを作ります。
参照表による評価回数と提案回数を区別します。候補なしの回に未測定の光学値を補完しません。
五つのSeedは探索的比較であり、有意差・収束・量子優位性を保証しません。
この問題の図には光学FOMを用い、ペロブスカイト用のeV表記を使いません。図生成後の目視確認は別途必要です。

## FMの学習設定比較：学習データ40件を固定

`config_fm_training_fixed40.json`と`evaluate_fm_settings.py`を使用します。
各データSeedの学習40件は、既存の光学フィルター固定FM実験と同じIDを
`data/multilayer_filter/ml_initial_sets.json`に固定しています。
一般BBOランナーのNumPyによる初期点抽出とは別の、明示的に固定したデータです。
全設定で同じ40件を使い、残り4,056件を学習器に渡しません。

| 項目 | 比較値 | 基準 |
|---|---|---|
| rank | 1, 2, 4, 8 | 2 |
| 学習率 | 0.001, 0.01, 0.03, 0.1 | 0.1 |
| weight decay | 0, 0.001, 0.01, 0.1 | 0.01 |
| epochs | 30, 120, 300, 1000 | 120 |

基準＋各項目だけ変更する12設定の合計13設定です。交互作用や全組合せの最適設定は検証しません。
AdamW・全バッチMSE・目的値の標準化なし・固定epoch・scheduler/early stoppingなしです。
weight decayは既存実装と同じくバイアスを含む全パラメータに適用します。
データSeed42/101/2024/7/19×重み初期化3反復＝195回の学習を計画しています。
初期化Seedはdata_seed + rep×1,000,000で、各設定の共通反復番号に同じSeedを使います。

```bash
cd code
# 計画とデータ・ソースハッシュだけ保存。学習しない（既定動作）。
python evaluate_fm_settings.py

# 以下は明示的に学習実行を決めたときのコマンド。今回まだ実行していない。
python evaluate_fm_settings.py --action run
python evaluate_fm_settings.py --action analyze
```

主指標は未学習点のRMSE・Spearman順位相関・FMが選ぶ未学習候補の真のNormalized Regretです。
学習/未学習MSE・RMSE・MAE、真の上位1%/5%との予測上位集合の重なり、選択候補の真の順位、初期最良を含む1提案後の最良値も保存します。
選択候補はFM予測値だけを使った全実行可能点の厳密選択で、SA/QAOAの探索誤差を混ぜません。
このRegretの分母は全4,096候補の最大値−最小値です。予測値同点は候補IDで決めます。
上位集合も候補IDで同点を処理し、k=ceil(割合×未学習件数)とします。

設定別・データSeed別に初期化3反復の中央値を取り、その5データSeedの中央値/IQRを報告します。
基準からの同一データSeedの差を使い、12設定×3主指標＝36比較のWilcoxon検定にBonferroni補正を適用します。
Spearmanが定義できない場合は欠損として残し、その比較は未定義とします。失敗した学習も除外せず、成功時だけ全件監査・集計します。

残り4,056件は設定比較のための探索的評価データです。この結果で設定を選んだ後の独立した最終テストとは呼びません。
設定確定後の確認には、未使用のデータSeedによる新しい学習40件を別途用います。
今回の目的はサロゲート学習設定の評価で、BBO性能・QAOA性能の比較結果ではありません。
出力は`results/fm_training_settings_fixed40/{json,csv,checkpoints,data,pdf,svg,png}/`です。
実際の学習開始時はpueueに学習と成功後の監査・作図を登録します。実行状態とタスクIDは`results/fm_training_settings_fixed40/json/queue.json`と`*_execution.json`を確認してください。

## FMの組合せ比較＋目的値標準化（学習40件を固定）

`config_fm_training_grid_fixed40.json`と`evaluate_fm_grid.py`を使います。
rank=[1,2,4,8]、learning_rate=[0.001,0.01,0.03,0.1]、weight_decay=[0,0.001,0.01,0.1]、
目的値の標準化なし/ありを全組合せで比較します。epochs=120、AdamW、全バッチMSE、初期化方法は固定です。
128設定×同じ5データSeed×3初期化＝1,920学習です。既存195学習の結果と保存先を分けます。

`fm_target_scaling.py`が既存の`fm_model.train_fm`を使います。
標準化は学習40点の目的値だけからmean/population std(ddof=0)を求め、z=(y-mean)/stdで学習します。
std<=1e-12の場合はscale=1とし、その事実を記録します。推論後に必ず元のFOMへ逆変換し、全指標を元の尺度で評価します。
チェックポイントにはFMの重みと変換係数を保存します。将来QUBOを作る場合もQとoffsetを元の尺度に戻す共通関数を用意しています。
今回の比較ではSA/QAOA/BBOは実行しません。

学習40件のID、5データSeed、モデル初期化Seedは前回と同一です。未学習4,056件は学習や標準化に渡しません。
設定ごとに初期化3反復の中央値をデータSeedごとに取り、その5データSeedの中央値/IQRを表示します。
元のraw基準との127対比と同じハイパーパラメータの標準化あり/なし64対比を、3主指標で検定します。
合計573個の対応差Wilcoxon検定を一つのBonferroni補正対象とし、すべてのSeed・失敗・未定義相関を保持します。
この範囲での探索的設定比較であり、設定選択後の独立確認には未使用のデータSeedが必要です。

```bash
# 計画保存だけ（既定）。
python evaluate_fm_grid.py
# 実行を決めた際はpueueから学習と成功後の後処理を登録。
python evaluate_fm_grid.py --action run
python evaluate_fm_grid.py --action analyze
```

出力先は`results/fm_training_grid_standardization_fixed40/`。
全モデル・逆変換係数・予測値・ハッシュ・指標を再読込して監査し、CSV・RESULTS/DISCUSSION・PDF/SVG/PNGを生成します。
図はRMSE/順位相関/選択候補Regretの全組合せヒートマップと、対応する標準化の有無によるRMSE差です。
既存のQAOAジョブや並列数は変更しません。pueueのtask IDと正確なコマンドは`json/queue.json`、実行状態は`*_execution.json`に保存します。

## 光学フィルター用の既定FM設定

`config_multilayer_filter.json`の既定値は、rank=4、learning_rate=0.1、weight_decay=0.1、epochs=120、AdamW、`fm.target_standardization=true`です。
固定40件の組合せ比較で、選択候補の真のRegret中央値が最小だった設定を採用しています。
この推奨は光学フィルターの探索的評価に基づきます。ペロブスカイト用`config.json`や過去の凍結実験設定には適用しません。

通常の`main.py`も、各再学習時点の評価済みデータだけから目的値の平均・標準偏差を計算して学習します。
候補比較・診断・SA/QAOAに渡すQUBOは元の目的値尺度に戻します。
標準化FMの係数をQ_z、切片をb_z、変換係数をμ・sとすると、Q=s Q_z、b=s b_z+μです。
ペナルティ基準Sは、この元尺度のペナルティなしFM-QUBOから求め、λ=Sαとします。XYのλ=0は維持します。
各提案のJSONと最新モデルチェックポイントに`target_transform`を保存します。
旧設定で標準化フラグがない場合は、従来どおり標準化しません。設定ハッシュが異なる結果を再開時に混在させません。

```bash
python main.py --config config_multilayer_filter.json
# 標準化を無効にする場合
python main.py --config config_multilayer_filter.json --set fm.target_standardization=false
```

`prepare_multilayer_bbo.py`もこの学習・逆変換に対応しています。保存済みの旧凍結プロトコルは旧条件の記録です。
新しい条件で本比較を行う際は、新しい保存先で事前検証・プロトコル凍結を行ってください。
実行中の`intern1016/`の固定FM・QAOAジョブには変更を加えていません。

## 新しい既定FMを固定したペナルティ感度

`config_fixed_fm_penalty.json`と`evaluate_fixed_fm_penalty.py`を使います。
学習設定はrank4・lr0.1・weight_decay0.1・120epochs・AdamW・目的値標準化ありです。
完了済みの組合せ比較から、該当する5データSeed×3初期化＝15モデルをそのまま固定して再利用します。再学習しません。
初期40件も学習設定比較と同一です。モデルと変換係数をコピーし、学習済み重み・元尺度予測の一致を確認して凍結します。

α=[0.01,0.03,0.1,0.3,1,3,10,30,100,1000]、100 reads・1000 sweeps・10反復で、計1500 SAバッチ・150000出力です。
Sは元のFOM尺度へ戻した、ペナルティなしFM-QUBOのIsing h/J最大絶対値とし、λ=Sαです。
同じモデル・反復番号では全αに同じSA Seedを与えます。Nealのgeometric auto betaを使い、各バッチのbeta_rangeを保存します。
温度範囲も係数に応じて変わるため、ペナルティだけの因果効果とは解釈しません。

主な指標は、未評価実行可能FM最小解の100出力中の捕捉率、FM gap、One-Hot実行可能率、未評価候補獲得率です。
候補なしのバッチは捕捉率0・条件付きgap=Noneとして残し、失敗スコアを未評価FM値幅+max(1,値幅)と定義します。
図は失敗を含むgap scoreを表示し、条件付きgapは別途CSV/JSONで保持します。選択には元のFM値だけを使います。
モデルごとに10バッチを集計し、同一データSeedの3初期化の中央値、その5データSeedの中央値/IQRを報告します。
α100との9対比×3主指標＝27個の対応差Wilcoxon検定をBonferroni補正します。
この固定FM診断は、予測誤差・再学習BBO・QAOAの性能評価とは別です。光学真値の追加計算も行いません。

```bash
python evaluate_fixed_fm_penalty.py                    # 凍結モデル確認と計画保存
python evaluate_fixed_fm_penalty.py --action run       # 明示的な実行時のみ
python evaluate_fixed_fm_penalty.py --action analyze   # 全計算成功後
```

保存先は`results/fixed_fm_penalty_default_standardized/`です。
`data/models/`に凍結チェックポイント、`data/raw/`に全SA出力、`json/runs/`に全バッチ指標・失敗を保存します。
後処理ではモデル/QUBO/係数スケール・元尺度予測・全rawビット列から再計算した指標を監査し、統計・CSV・PDF/SVG/PNG・結果レポートを作ります。
既存QAOAジョブや並列数は変更せず、pueueの成功依存で後処理を登録します。
