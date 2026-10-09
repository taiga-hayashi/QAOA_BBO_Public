# 再現手順

cwdはIntern_Fujitsu。実行前protocol.json、監査AUDIT.md、prevalidation.jsonを確認する。共有FM/SAを再利用し、環境・protocol/source SHAをenvironment.jsonに保存。CPU1thread、Pythonと各ライブラリ版、元checkout commitは環境JSON参照。未測定メモリ容量はunmeasured。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/large_penalty_diagnostic/py/run_penalty.py prevalidate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/large_penalty_diagnostic/py/run_penalty.py fixed > intern_1009/perovskites/large_penalty_diagnostic/data/fixed.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/large_penalty_diagnostic/py/run_penalty.py bbo > intern_1009/perovskites/large_penalty_diagnostic/data/bbo.log 2>&1
MPLCONFIGDIR=/private/tmp/intern1009-mpl XDG_CACHE_HOME=/private/tmp/intern1009-fontcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/large_penalty_diagnostic/py/analyze_penalty.py > intern_1009/perovskites/large_penalty_diagnostic/data/analysis.log 2>&1
```

既存raw JSONを上書きしない。再実行は独立コピーで行う。失敗Runはstatusとtracebackを残す。今回の計画はfixed700batches＋BBO35Run/2450cycles、各100outputs。別途fixed基準beta取得5batchesと事前検証5batchesあり、性能集計には混ぜない。各BBO Runの70state_dictと全cycleのrawbits、FM予測192件、alpha/S/lambda、日程、選択候補、真値、評価回数を保存。fixedモデルは5pt。

初期10件は同一Seedの元20件の先頭。予算は10＋採用新規候補数、上限80。最大70cyclesを完走し、候補なしでは評価数を増やさない。全192真値を読む最適性照合と候補の事後品質評価は探索予算外。solverは評価済み点のみ学習し、採用順位に真値を使わない。

analyzeはモデル/checkpoint/protocol/sourceのhash、FM/QUBO全192予測、rawbitsの制約・重複フィルタ、候補順位、alpha/S/lambda、真値と予算を照合。alpha100の5Runは過去initial10 LargePenalty-FMQAの全70cyclesに対してモデルとサンプル・選択・目的値を完全照合。極端alphaでもfeasibleエネルギー差はprevalidationの丸め誤差を記録。

固定FMは10反復をモデル内平均、5モデル間中央値/IQR。BBOは5Seed中央値/IQR。初期10では全5Seedが初期最適解なし。ペア検定は30件Bonferroni、全差0ならp1。候補がない条件付き品質はnull。予算終了時の未到達はscore81、実評価数と混同しない。図は統合1＋独立6＋軌跡1のPDF/SVG/PNG（24files）、すべて実測JSONから生成。レンダリングQAとSHAはdelivery_verification.jsonに記録。
