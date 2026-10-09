# 再現方法

作業ディレクトリはIntern_Fujitsu。実行環境はjson/environment.json、凍結条件はjson/protocol.jsonに保存。元の5 Seedのinitial20とモデル初期値を使用し、データ選択と学習乱数のSeedは同じ。学習乱数を独立に反復した実験ではない。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_regularization_diagnostic/py/diagnose_regularization.py prevalidate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_regularization_diagnostic/py/diagnose_regularization.py run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_regularization_diagnostic/py/diagnose_regularization.py analyze
XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_regularization_diagnostic/py/report.py
```

runは既存seedファイルを上書きしない。再計算する場合は結果全体を別途保管し、結果のない独立コピーで実行する。analyze/reportは保存済み結果から再実行できる。初期化・学習は共有src/fm.py、QUBO変換は共有src/fm_to_qubo.py、評価器・データは既存Perovskites固定lookupを再利用した。今回src/fm.pyにAdamW/weight_decayの任意キーワード引数を追加し、intern/src/fm.pyに同じ変更を反映した。既定Adam wd0を保持。

prevalidationは旧rank1/2の10学習を最大予測差0で再現。零特徴列でのdecay適用、無効引数の拒否、共有ソースと配布ミラーの一致も確認。回路変更がないため、既存N6/N9・23変数の事前検証を継承し、今回は回路を再計算していない。

全30 checkpointのSHA、同一rank/Seed内の初期値SHA、初期データSHA、予測のTorch/QUBO一致、全列挙候補選択、Test RMSE再算出を確認。json/artifact_verification.jsonに記録。intern/README.mdのquick source check（experimet/py/*.py・experimet/*/py/*.py）も実行。全BBO再実行は今回の検証範囲外。

図の一覧・SHAはjson/figure_manifest.json、PDFレンダリング目視確認はjson/delivery_verification.json。集計対象30学習をすべて含め、失敗・除外0。CPU時間は各fitに保存したが、他手法の速度比較には使用しない。
