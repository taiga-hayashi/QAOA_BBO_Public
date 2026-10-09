# 監査・再現性

ユーザー指定のサンプル数比較をプロトコルに固定した二次解析。既存のFM・SA・ペナルティ実装を変えず、score_candidatesをaccepted_candidates=1で再利用。元の全サンプルをそのまま先頭から取らず、raw行順・最初の250件バッチへの依存を避けるため、モデルSeed・反復に応じた共通のランダム順序を使用。

np.random.SeedSequence([model_seed,repetition,1009])で作ったpermutation(1000)を各αで共有。先頭10・30・100・300・1000位置を取る。実際のビット列の重複は保持し、採用段階で無効・初期候補・重複を除く。各seed JSONに使用位置順と部分集合のrawビット列を保存。

既存のN6/N9/N23回路とペナルティ事前検証の合格記録を引き継ぐ。今回、回路もSAも新規実行しない。全条件で独立した直接FM順位付けと共有selectorの候補を照合。入力ハッシュ・モデル/初期/チェックポイント同一性、n1000と既存top1の一致、nested集合、n増加時の選択FM予測値の非増加、raw分割件数、1件採用/上限21を検証。

```sh
XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmqa_sampling_budget_top1/py/compare_sample_counts.py
```

実行場所はrepository root。元のsource SHA256、コード・環境をJSONへ保存。元の計測時間を小さいnum_readsの時間に換算・捏造しない。ペア検定は同じ5モデルSeedの、1000件を基準にした差分。3指標×12α×4サンプル数＝144比較をBonferroni補正。5モデルの小標本で、3000条件を3000独立学習モデルとして扱わない。
