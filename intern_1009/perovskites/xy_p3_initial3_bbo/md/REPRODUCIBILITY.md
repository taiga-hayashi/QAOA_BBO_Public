# 再現手順

cwdはIntern_Fujitsu。ユーザー指定の初期3件・p3・SA alpha1000をprotocol.jsonで凍結。AUDITとprevalidationを確認してから本実験を実行する。p1対照は同じ初期3とFM設定で再学習する。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/xy_p3_initial3_bbo/py/run_initial3.py prevalidate > intern_1009/perovskites/xy_p3_initial3_bbo/data/prevalidation.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/xy_p3_initial3_bbo/py/run_initial3.py run > intern_1009/perovskites/xy_p3_initial3_bbo/data/run.log 2>&1
MPLCONFIGDIR=/private/tmp/intern1009-mpl XDG_CACHE_HOME=/private/tmp/intern1009-fontcache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/xy_p3_initial3_bbo/py/analyze_initial3.py > intern_1009/perovskites/xy_p3_initial3_bbo/data/analysis.log 2>&1
```

既存Runを上書きしない。再計算は独立コピーで行う。cycleごとJSON保存、10cycleごとcheckpoint更新、終了時は全77state_dictを保存。失敗はpartialモデルとtracebackを保持、resume未実装。計算機・CPU1thread・Python/OpenQARP/Neal版、source SHA・commit・protocol SHAはenvironment.json。メモリ容量はunmeasured。

初期集合は元20件の保存順先頭3件。3＋採用新規候補数が実評価回数で上限80。最大77cycleを完走し、最適解発見後も早期終了しない。候補なしに評価を割り当てず、refillなし。全192真値を読む最適性・候補品質の事後監査はsolver予算外、学習には評価済みデータだけ使用する。真値を候補選択に使わない。

rank1 AdamW wd0.01 lr0.1 120epochsは共有fitを利用。モデルseedはseed+100000*(cycle−1)。全手法の初期モデルSHAが同一であることを確認。2cycle以降は採用候補に応じて学習データが分岐する。候補数は各100、最大1件の未評価実行可能候補をbaseFM順に採用。

XYp1/3は共有OpenQARPCompactXYQAOA、元23One-Hot変数、sim8bit。各cycle128候補:旧p1grid9点（p3はゼロ層埋込）＋119ランダム点、各層gamma/beta0〜0.8。追加p1/p2探索なし。独立多層referenceは検証専用。乱数はprotocol参照。毎cycle共有mixer cacheをクリアし、連続角度の無制限cache増加を避ける。cache破棄は回路や確率を変えない。

SAは共有sampleを使いalpha1000、lambda=Salpha、100reads、1000sweeps、geometric、自動beta。元LargePenalty既定alpha100と区別する。Sはbase Ising h/J最大絶対値、固定lambdaではない。

計画は15Run、1155modelcycles、XY98560角度評価/77000shots、SA38500outputs。角度探索は理想期待値で、最終出力のみ100shots。等しいBB上限とXY anglecallsであって、p3 gate/層評価はp1の3倍で、SAとの計算予算は同等ではない。

事前検証は直前N6/N9/N23 p3検証のsource hash継承＋新初期3の5モデル・p1/p3 anchor一致・ペナルティ代数。解析は全1155モデルをcheckpointから復元、FM/QUBO/Torch予測、hash、全raw選択・真値・予算を照合。XY全98560候補角度をseedから再生成し、保存された期待値でのbest選択を確認。独立referenceで再計算するのは選択best770stateで、全98560期待値の独立再計算ではない。backendは全state評価でraw norm/制約質量を1e-10以内に検査する。

8pairedWilcoxonにBonferroni、中心は5Seed中央値/IQR。未到達81は失敗コード。候補なしの品質はnullで、失敗・不利なSeedを除外しない。図は全体＋6単独＋cycle軌跡のPDF/SVG/PNG計24files。全PDFを描画し視覚検証、source/図/summary hashと構文・git diff --checkをdelivery_verification.jsonに保存。
