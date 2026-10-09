# FMQA：サンプル数とペナルティの比較（1件採用）

状態：3000条件の二次解析・独立順位照合・集計・PNG/PDF/SVG作成が完了。

サンプル数10・30・100・300・1000を、α=0〜1000の12条件で比較。初期20件、5固定FM、各10反復、採用はFM予測値の最も低い未評価実行可能候補1件。元のSAは1000 sweeps。

保存済みの[SAペナルティ感度](../fmqa_penalty_sweep/README.md)のrawサンプルを使う二次解析。各1000件を共通乱数で無作為に並べ替え、先頭n件を使用する。小さい集合が大きい集合に含まれるnested比較で、α間にも同じ位置順を適用する。元の[1000件・1件採用](../fmqa_penalty_sweep_top1/README.md)と一致を検証する。

[プロトコル](json/protocol.json) / [結果](md/RESULTS.md) / [考察](md/DISCUSSION.md) / [図の説明](md/FIGURES.md) / [監査・再現性](md/REPRODUCIBILITY.md)

[比較図PDF](pdf/sampling_budget_overview.pdf) / [PNG](png/sampling_budget_overview.png) / [SVG](svg/sampling_budget_overview.svg)

追加のFM学習・SA実行・閉ループBBOは行わない。実際にnum_reads=nで新規生成した分布や時間の測定とは区別する。初期20件＋新規最大1件の論理評価予算上限21。失敗・候補不足も保持。
