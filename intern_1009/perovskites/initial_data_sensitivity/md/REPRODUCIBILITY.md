# 再現手順

cwdはIntern_Fujitsu。条件json/protocol.json、共有source SHAと初期モデル検査はjson/prevalidation.json、環境はjson/environment.json。初期5/10/20は元20件の保存ID順の先頭で固定する。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/initial_data_sensitivity/py/initial_experiment.py prevalidate > intern_1009/perovskites/initial_data_sensitivity/data/prevalidation.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/initial_data_sensitivity/py/initial_experiment.py run > intern_1009/perovskites/initial_data_sensitivity/data/run.log 2>&1
XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/initial_data_sensitivity/py/analyze_initial.py > intern_1009/perovskites/initial_data_sensitivity/data/analysis.log 2>&1
```

runは既存Run JSONを上書きしない。再計算は結果を保管した独立コピーで行う。途中失敗はpartial JSONとtracebackを保持、再開未実装。analyzeは保存結果から再実行できる。45Runのcheckpoint（75/70/60モデルをRunごとにまとめた45pt）、全3075サイクル・307500raw出力をJSON保存。XY ideal角度評価は9225、shotsは102500。SA outputは205000。

BB評価は初期n＋採用候補1件ごとに数え、上限80。候補なし・既評価・invalidに新規評価は割り当てず、no refill。最大サイクル数は80−nで揃える。最適値を見つけても予定サイクルを完走する。実評価数・サイクル数・生成時間を別に記録する。

analyzeは3075モデルhash・checkpoint/protocol・QUBO/Torch予測一致、raw候補フィルタと順位、α更新とλ=Sα、訓練ID/目的値と予算を照合。XY全9225角度の期待値は独立Ring辺回転referenceから再計算し、chosen anglesとsampling CDF乱数も検証。過去20件・100出力15Runの全60サイクルを再現することを確認する。

全192真値を読む最適性照合と共通172の初期FM精度評価は事後監査でsolver budget外。trainには初期nと採用候補の値しか使わない。初期FM評価に3手法分の同じモデルがあるため集計では1手法だけを使う。

統計は5Seed中央値/IQR、共通4Seedで初回到達を比較。未到達81は失敗コードで評価回数の実測ではない。24の事前固定WilcoxonをBonferroni補正。失敗・条件付きGapのnullを隠して集計しない。候補なし率は総サイクル数が違うため割合で比較する。図は全体＋単独6のPDF/SVG/PNG計21ファイル。レンダリングQA、hash、構文チェック、git diff --checkを記録する。
