# 再現手順

Intern_Fujitsuがcwd。条件・Seed・数式はjson/protocol.json、source SHAと27回路比較はjson/prevalidation.json、環境はjson/environment.json。初期20件は既存fixed_fm_pilotを再利用。元データSHAとevaluatorを変更していない。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmxy_adamw_bbo/py/run_xy.py prevalidate > intern_1009/perovskites/fmxy_adamw_bbo/data/prevalidation.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmxy_adamw_bbo/py/run_xy.py run > intern_1009/perovskites/fmxy_adamw_bbo/data/run.log 2>&1
XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmxy_adamw_bbo/py/analyze_xy.py > intern_1009/perovskites/fmxy_adamw_bbo/data/analysis.log 2>&1
```

runは既存Seed JSONを上書きしない。再実行は結果を保管した独立コピーで行う。partial JSONと失敗tracebackを保持し、再開機能は未実装。analyzeは保存結果から再計算できる。checkpointはSeedごとに60モデルをまとめた5ファイル、全300サイクルの9角度期待値・probability・raw shots・候補選択・初期/訓練ID・評価数・時間をJSONへ保存した。

analyzeは独立辺回転referenceで全2700角度の期待値と選択角度を再検証し、全300モデルhash・FM/QUBO予測・CDF乱数・候補規則・budget・真値を照合する。全192真値を事後に読む最適性診断はsolver budget外のoffline audit。訓練と候補選択・角度選択に全192真値は渡さない。

BB評価は初期20＋各サイクル採用最大1件。invalid/重複に新規評価を割り当てず、候補なしは0評価。最適解を得ても60サイクルを完走。SA比較は保存済みproposed設定の10 Runを読み、同じ初期・予算と揃える。出力数10が等しくても理想角度評価とSA sweepsの費用は等しくない。

PDF/SVG/PNGは全体と単独3パネルの12ファイル。PDFをレンダリングして読みやすさと凡例を確認し、図SHAを保存する。shared sourceと配布ミラーの一致、元のRing7 tests、構文チェック、git diff --checkを実施した。CPU容量・ピークRSSは未測定。
