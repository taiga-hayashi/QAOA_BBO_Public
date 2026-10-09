# FMQAのペナルティ感度

状態：全600条件（600,000サンプル）の計算・raw照合・集計・プロット作成が完了。事前検証2件合格。PNG・PDF・SVGを保存。

Perovskites・23 One-Hot変数で、古典SAによる固定αの候補生成を調べる。既存の初期20件・5つの固定FMを再利用し、α=0,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000、各10反復×1000サンプル。内部ペナルティλ=Sα。閉ループBBOとは別の診断。

[実行前プロトコル](json/protocol.json) / [監査](md/AUDIT.md) / [結果](md/RESULTS.md) / [考察](md/DISCUSSION.md) / [再現性](md/REPRODUCIBILITY.md) / [図の条件](md/FIGURES.md)

[比較図 PNG](png/penalty_sensitivity_overview.png) / [PDF](pdf/penalty_sensitivity_overview.pdf) / [SVG](svg/penalty_sensitivity_overview.svg)

「制約充足解を得られるか」と「その解が良いか」を分けて調べる。Raw実行可能割合、1000サンプルで未評価5解を得る成功率、重複・多様性、FM最小値との差、真値Top5%獲得頻度、初期＋提案Regretを保存。真値を使った診断値は学習や提案順位に使用しない。条件付き品質が未定義のケースも記録する。

SAの自動温度範囲がペナルティとともに変化するため、固定温度下のペナルティ単独効果ではない。XYへのペナルティは追加しない。
