# 再現手順

Intern_Fujitsuをcwdにする。Pythonとpackage版はjson/environment.json、全設定はjson/protocol.json、source SHAはjson/prevalidation.jsonに保存。元の評価器はpinned CSV SHAを確認する。initial20は元pilotの保存ID・値と完全一致。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmqa_adamw_bbo/py/run_bbo.py prevalidate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmqa_adamw_bbo/py/run_bbo.py run > intern_1009/perovskites/fmqa_adamw_bbo/data/run.log 2>&1
XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmqa_adamw_bbo/py/analyze.py > intern_1009/perovskites/fmqa_adamw_bbo/data/analysis.log 2>&1
```

既存Runファイルは上書きしない。再実行は保存済み結果を退避した独立コピーで行う。analyzeは保存データから再実行可能。checkpointは各Runに60モデルをまとめたpt（20ファイル）、イベントはサイクルごとにJSON保存。再開機能は未実装で、失敗時はpartial JSONとtracebackを保持する。今回失敗・除外0。

初期20＋採用候補の評価を各Runの論理BB評価として数える。既評価/invalid/pool duplicateにはBB評価を割り当てず、候補なしは0追加。全192候補の真値を事後に読む精度・最適性照合はoffline auditとして別記録し、solver評価予算に含めない。初期データを共有してもRunごとの論理予算は初期20を含む。

独立analyzeで1200モデルのhash、checkpoint SHA、QUBO/Torch予測一致、候補選択、評価数、目的値、α更新・λ=Sαを再検証。全15図ファイルを出力し、PDFからレンダリングして目視確認する。ソース構文検証とgit diff --checkを実行。機械メモリ容量・ピークRSSは未測定。
