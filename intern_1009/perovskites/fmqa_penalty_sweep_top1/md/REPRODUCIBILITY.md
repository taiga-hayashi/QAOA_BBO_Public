# 再現性と監査

候補選択はfixed_fm_pilot/py/run_pilot.pyのscore_candidatesをaccepted_candidates=1で再利用する。SA計算・FM学習は再実行しない。fmqa_penalty_sweepの5seed JSONに含まれるrawビット列と、fixed_fm_pilotのQUBO・初期データを使用する。

各条件で直接FM予測値を計算して全生成候補を順位付けする独立チェックを行い、共有selectorの採用IDと一致を確認。さらに元の5件採用の先頭候補IDと照合。真値の表引き、raw制約充足率の不変、最大1件・合計21件の予算、モデル・初期データ・チェックポイント・sourceハッシュも確認する。

既存の小規模N6/N9/N23回路事前検証とSAペナルティ検証は元実験の合格記録を参照する。今回は回路・SAソルバー・ペナルティの変更はない。環境とコードSHA256をenvironment.jsonに保存。元の計測時間は元実験の測定値として引き継ぎ、新規計算時間とは呼ばない。

```sh
XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmqa_penalty_sweep_top1/py/reanalyze.py
```

実行場所はrepository root。反復モデル間の中央値/IQR、定義可能件数、α100に対するペア差分・Wilcoxon・Bonferroni33比較を保存。今回の指標変更はユーザーの指定による二次解析であり、原実験の事前指定と混同しない。
