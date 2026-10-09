# 監査と再現性

src/qarp_backend.pyの共有p1標準X回路、固定FMのqaoa_generate（同一9点探索）、shot研究のdraw/scoreを再利用。新規コードはαの走査とエネルギー分散の診断のみ。FMモデルは再学習しない。α5は元の角度探索に対するsourceハッシュとモデル同一性を保存し、選択回路・全rawショット・Cost誤差が前のショット研究と一致することを確認する。

既存N6/N9/N23回路とFM事前検証を参照。新規のN6全状態ペナルティエネルギー検証（α0,0.01,5,100,1000）と、α0/1000の標準QAOA確率を独立Cost-phase/RX計算と照合するテスト2件合格。runnerはprevalidation.jsonの合格・コードハッシュを確認する。

Cost=(baseFM+λP)/S、λ=Sα。大ペナルティを含む最大係数で回路Costを再正規化しない。記録するscaled RMSEは事後の表示診断で、回路の変更ではない。各選択状態のraw norm、実行可能192候補確率、選択期待値を確認。全ショットのCostは独立baseFM+λP式と照合。full2^23×23ビット行列は作らない。エネルギー二次モーメントは65536単位で処理。peak RSSは未計測。

shot RNGはSeedSequence([model_seed,rep,1009,73])、α間は共通uniforms、shots間はnested prefix。無効解を含む全基底indexを保存し、エネルギー精度には全ショットを含める。候補選択は無効・初期・重複を除き最小FM予測1件。初期20＋最大1＝21件の論理評価上限。

```sh
OPENBLAS_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/standard_qaoa_penalty_sweep/py/test_penalties.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/standard_qaoa_penalty_sweep/py/run_penalties.py --workers 3
OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/standard_qaoa_penalty_sweep/py/analyze_penalties.py
```

repository rootで実行。workerはモデルSeed単位の独立プロセスで、各αを逐次保存。既存seedファイルは上書きせず停止する。再実行は原成果物を保全した別フォルダを使う。失敗試行を除外しない。

5モデル内100反復平均（エネルギーはRMSE）と5モデル中央値/IQR。条件付き品質の定義件数も保存。α5基準・10/1000shotsの3指標×12係数×2shots＝72ペア検定をBonferroni補正。5モデルを統計単位にし、32500測定を独立学習モデルとみなさない。9点の最小値は大域的最適角度を保証せず、特に大αでは位相が急変する可能性があり、探索解像度の不足を別に検証する。
