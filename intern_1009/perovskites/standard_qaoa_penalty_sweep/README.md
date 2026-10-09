# 標準QAOAのペナルティ係数と精度

状態：65モデル×α設定・32500測定条件の計算・raw検証・集計・PDF/SVG/PNG作成が完了。事前検証2件合格。

同じ5固定FM・初期20件、23量子ビットp1で、α=0,0.01,0.03,0.1,0.3,1,3,5,10,30,100,300,1000を調べる。SAの12係数に従来の標準QAOA基準α5を追加。各αで同じ9点の厳密期待値探索を行い、shots10,30,100,300,1000で各100回の理想測定を実施。採用候補はFM予測最小の未評価実行可能1件。

[実行前プロトコル](json/protocol.json) / [結果](md/RESULTS.md) / [考察](md/DISCUSSION.md) / [再現性・監査](md/REPRODUCIBILITY.md) / [図の条件](md/FIGURES.md)

[比較PDF](pdf/standard_qaoa_penalty_overview.pdf) / [PNG](png/standard_qaoa_penalty_overview.png) / [理想制約充足確率PDF](pdf/ideal_feasibility.pdf)

内部λ=Sα。係数を変えた標準QAOAの診断であり、既定4手法の標準QAOAα5を再定義しない。XYはλ0のまま別の[ショット研究](../qaoa_shot_accuracy/README.md)に保持する。候補不足や条件付き品質の未定義を隠さない。

全13係数×5モデル＝65角度選択設定、各5ショット数×100反復＝32500測定条件。α5は検証済みの同一9点探索メタデータを再利用し、状態を再計算して前の測定と照合。他のαは新規9点探索。有限ショット角度最適化・実機ノイズ・FM再学習・閉ループBBOは未実施。
