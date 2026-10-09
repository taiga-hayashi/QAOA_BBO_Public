# 再現手順

リポジトリ直下で、以下のPython環境を使う。runは既存seed JSONがある場合に上書きせず停止する。生結果・checkpointを保存した別フォルダで再計算するか、analyzeのみで保存結果を再集計する。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_learning_diagnostic/py/diagnose.py prevalidate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_learning_diagnostic/py/diagnose.py run
OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_learning_diagnostic/py/diagnose.py analyze
OPENBLAS_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_learning_diagnostic/py/audit_unseen.py
OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fm_learning_diagnostic/py/standalone_panels.py
```

json/environment.jsonにPython・OS・依存版・共有コードSHA、各seedに原結果SHAと初期集合SHA、各fitにcheckpointSHAを保存。CPU型番・peakRSSは未計測。data/checkpointsに全90モデルを保存し、targetのmean/stdもcheckpointへ記録した。設定は実行前にjson/protocol.jsonへ凍結。

prevalidationは元の5FMを再学習し全192候補の予測を照合、最大差0で完全再現。各新規fitでTorch予測とQUBO予測をeV尺度で1e−4以内照合。analyzeでは全件・SHA・候補の独立順位付け・RMSE・QUBO再構成を確認した。N6/N9および23ビットXY保存の既存事前検証は合格済みだが、今回新たに回路を計算していない。

未観測列の監査は元モデルの初期状態と学習済み重みを直接比較し、未観測Vと一次係数の変化が0と確認。訓練真値20件だけをfitへ渡し、未学習172件の真値を学習・候補順位づけに使わない。standardized設定のmean/stdも20件だけで計算。
