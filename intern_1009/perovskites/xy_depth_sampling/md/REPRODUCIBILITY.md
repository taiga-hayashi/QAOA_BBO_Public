# 再現手順

cwdはIntern_Fujitsu、protocol.jsonとAUDIT.mdを実行前に確認。元モデルをsingle_fm_sampling_quality/data/fixed_model.ptからdata/fixed_model.ptへコピーしSHAで凍結。新規学習なし。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/xy_depth_sampling/py/run_depth.py prevalidate > intern_1009/perovskites/xy_depth_sampling/data/prevalidation.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/xy_depth_sampling/py/run_depth.py run > intern_1009/perovskites/xy_depth_sampling/data/run.log 2>&1
MPLCONFIGDIR=/private/tmp/intern1009-mpl XDG_CACHE_HOME=/private/tmp/intern1009-fontcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/xy_depth_sampling/py/analyze_depth.py > intern_1009/perovskites/xy_depth_sampling/data/analysis.log 2>&1
```

runは既存探索JSONを上書きしない。16角度ごとに途中保存、例外はfailedとtracebackを保存、resume未実装。再実行は独立コピーで行う。CPU1thread、Python/OpenQARP等の版・環境・commit・source SHAはenvironment.json。未測定メモリはunmeasured。

共有src/qarp_backend.pyのOpenQARPCompactXYQAOAを用い、ordered Ringの各層Cost＋XYを実行。8bitsは厳密符号化simulatorであり元23bit問題と区別。独立reference_layersは検証専用で、探索に使わない。N6/N9 p1/2/3ではfull/compact/reference、N23各pはcompact/referenceをfresh照合。以前のN23 full-parityは共有sourceの変更がないことを確認して継承し、新p3 full23を計算したとは報告しない。旧p1の9点と同じモデルの確率を再現。

全30探索、3840角度state計算、30best再計算。pごとの探索層評価1280/2560/3840。全300測定batches、30000shots。理想期待値で角度を決め、shotは最後の出力だけ。ハードウェアノイズ・有限shot角度最適化なし。gamma/betaは全層0〜0.8、各p128候補。p1旧9点、p2/3前深さbestゼロ層warmを含む。

samplingは物理LSB整数順のCDF、SeedSequence([42,1009,search_seed,rep,93])、同じ一様乱数100個をp間で対応。invalid/既評価/重複除去して最大1件、真値を順位付けに使わない。BBOを実行しないため新規BB評価予算はなし。品質はFM最小値の全列挙を基準にする。

解析で全3840角度を独立referenceから再計算し、全300測定のCDF乱数・候補IDs・選択gap・制約率を再現、Pminと理想捕捉式・warm期待値非悪化を照合。30JSONに全角度・期待値・選択確率・全shot candidate IDsを保持。検定6件Bonferroni、中央値/IQRは10探索Seed。欠損や失敗を除外しない。

統合＋6単独PDF/SVG/PNG計21files。全PDFをpdftoppmで描画し目視確認。source/図/summary SHAと構文・git diff --checkをdelivery_verification.jsonに記録。
