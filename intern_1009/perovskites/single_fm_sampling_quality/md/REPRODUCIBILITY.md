# 再現手順

新規SA・FM学習はしない。元測定はlarge_penalty_diagnosticのfixed_seed42、固定モデルをdata/source_fixed_seed42.jsonとdata/fixed_model.ptに複製し、protocolでSHAを凍結。元rawのうちauto日程だけを使い、fixedbetaの反復は混ぜない。

```sh
MPLCONFIGDIR=/private/tmp/intern1009-mpl XDG_CACHE_HOME=/private/tmp/intern1009-fontcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/single_fm_sampling_quality/py/analyze_single.py > intern_1009/perovskites/single_fm_sampling_quality/data/analysis.log 2>&1
```

共有チェックポイント/FMQUBO/選択ヘルパーは前診断からimport。全70batchesについてsource/model hash、全192FM予測の同一性、FM/QUBO/Torch一致、rawbits・フィルタ・選択・gap・lambda=Salpha、10初期値を検証。N6/N9/N23検証は変更のない共有source SHAと前診断合格を確認して継承。新規回路はない。

summary.jsonには1モデルの全予測、各alphaの出現頻度192件、各反復候補IDs、選択gap、日程、分母を保存。raw7000outputsを再計算せず読み取る。snapshotは未使用fixedbeta反復も保存するが主解析には含めない。CPU1thread、Python/library/OS/commit環境は../large_penalty_diagnostic/json/environment.jsonを参照し、解析SHAと図のSHAを本フォルダdelivery_verificationへ保存。

統計単位はこの1モデルの10sampling repetitions。raw数1000を独立試行数として検定しない。18pairedWilcoxonをBonferroni補正、全差0ならp1、条件付き品質の欠損は保持。FMのモデル間一般化やBB最適化を実験したと報告しない。

全体＋6単独のPDF/SVG/PNG計21filesを生成、全PDFをpdftoppmでレンダリングして目視確認する。構文・source/artifact SHA・git diff --checkを確認する。親benchmarkはdraftのまま。
