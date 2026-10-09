# QAOAのショット数と精度／4手法10出力

状態：全5モデル・6000条件の計算、raw照合、統計集計とPNG/PDF/SVG作成が完了。事前検証2件合格。

標準QAOAとXY-QAOAの測定shots=10,30,100,300,1000を比較。5固定FM・各100測定反復、1候補採用。原実験の厳密期待値9点探索で選んだ角度を固定し、共有OpenQARPの23量子ビットp1回路を再計算。新たに全状態確率から測定を模擬する。保存済みの1000サンプルを間引く計算ではない。

[実行前プロトコル](json/protocol.json) / [結果](md/RESULTS.md) / [考察](md/DISCUSSION.md) / [図の説明](md/FIGURES.md) / [監査・再現性](md/REPRODUCIBILITY.md)

[QAOAショット比較PDF](pdf/qaoa_shot_accuracy_overview.pdf) / [PNG](png/qaoa_shot_accuracy_overview.png)

[4手法10サンプル／10ショット比較PDF](pdf/four_methods_10_overview.pdf) / [PNG](png/four_methods_10_overview.png)

4手法比較のSAは10reads、1000sweepsで新規計算。Adaptiveは3,3,2,2readsの4バッチ間更新、Largeはα100。標準QAOAはα5、XYはλ0。全手法で初期20件・同じFM・最大1件採用・評価上限21を共通化する。

角度探索は理想期待値に基づいた元の条件を引き継ぐ。今回の有限ショットで角度を再最適化した実験ではない。出力数の公平性と計算量の公平性を区別する。実機ノイズ、BBO再学習、量子優位性は評価していない。
