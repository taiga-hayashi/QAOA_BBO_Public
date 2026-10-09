# FMQAペナルティ感度：1件採用

採用ルール：1回に生成した1000サンプルから、制約違反・初期データとの重複・候補間重複を除き、**FM予測値が最も低い1件**を採用する。候補がなければ0件、補充サンプリングはしない。同点は従来と同じOne-Hot tuple順。

ユーザー指定による元の[5件採用実験](../fmqa_penalty_sweep/README.md)の事後再集計。全5モデル・12係数・10反復、600条件の保存済みサンプルを使い、SA・学習を再実行しない。元の事前プロトコル・結果を変更しない。

[1件採用のプロトコル](json/protocol.json) / [結果](md/RESULTS.md) / [考察](md/DISCUSSION.md) / [再現性](md/REPRODUCIBILITY.md) / [図の説明](md/FIGURES.md)

[比較図PDF](pdf/penalty_sensitivity_overview.pdf) / [PNG](png/penalty_sensitivity_overview.png) / [SVG](svg/penalty_sensitivity_overview.svg)

初期20件＋新規最大1件＝21件の論理評価上限。初期最良値とは別に、選んだ1件そのもののFM gapと真値を表示する。閉ループBBOではない。
