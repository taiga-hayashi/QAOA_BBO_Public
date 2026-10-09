# 再現方法

Intern_Fujitsuをcwdとする。設定はjson/protocol.json、source SHAはjson/prevalidation.json、環境・package版はjson/environment.json。source/データを変更せず、同じ5 Seedを使用。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/bbo_sample_count_comparison/py/experiment.py prevalidate > intern_1009/perovskites/bbo_sample_count_comparison/data/prevalidation.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/bbo_sample_count_comparison/py/experiment.py run > intern_1009/perovskites/bbo_sample_count_comparison/data/run.log 2>&1
XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/bbo_sample_count_comparison/py/analyze.py > intern_1009/perovskites/bbo_sample_count_comparison/data/analysis.log 2>&1
```

runは既存結果を上書きしない。再計算は結果を保管した独立コピーで行う。analyzeは保存結果から再実行可。途中の失敗はpartial JSONとtracebackを保持し、成功のみを選別しない。今回は30 Runを完了、失敗・除外0。

保存物：30 checkpointファイル（各60モデル）、30 Run JSON（全1800サイクル、99000raw出力、FM予測・角度・選択・λ/α/S・訓練ID・評価数・時間）、事前検証/環境/summary/補助固定FM診断、プロットPDF/SVG/PNG。

analyzeは全1800モデルのhashとTorch/QUBO一致、raw候補選択、Gap、評価予算、訓練IDと真値、Adaptive更新を照合。XY全5400角度の期待値を独立Ring辺回転で検算し、sampling乱数と採用結果も確認。15の10出力Runは過去の全60サイクルとモデルhash・raw出力・採用候補・評価数・bestが一致することを確認する。過去結果の単なるコピーではない。

最適解と最悪値の基準、結果の真値照合に全192lookupを使う事後監査はsolver BB予算の外に記録する。訓練には初期20と採用候補の値のみ。初期20を各Runの論理評価数に含める。最適解を見つけても60サイクルを完走。

固定FM XYの300ペアは候補の予測品質だけを調べ、追加のBB評価や閉ループ最適化には数えない。100乱数の先頭10が元の10出力に一致し、同じFMなら100から選ぶ候補の予測gapが10以下になることを検算。出力数の費用と真評価の費用は区別する。

中央値/IQRと12の事前固定ペア検定を使用。初回到達は初期最適解なし4 Seed、他は5 Seed。未到達の81は失敗コードであり実評価数ではない。Gapは各Runの採用サイクル中央値を5Seedで要約し、候補なしのGapを0に置換しない。図は全体と6単独の計21ファイル。PDFレンダリングを目視確認し、SHAとdelivery確認を保存する。
