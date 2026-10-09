# AdamW設定による再学習FMQA最適化

初期20件、60サイクル、10 SA samples/cycle、最大1件採用、実評価上限80。新設定rank1・AdamW・wd0.01と、従来rank2・Adamを、Adaptive-FMQA/LargePenalty-FMQAの各5 Seedで計算した。全20 Run完了。

新設定は両SA手法で5/5 Runが最適値1.5249 eVを保持した。そのうち1 Runは初期データに最適解があり、探索による新規発見は各4/4 Run。これは192候補の固定lookupに対する再学習BBO pilotで、QAOA2手法は今回含めていない。

- [事前条件](json/protocol.json)／[実行前監査](md/AUDIT.md)
- [結果・Seed別数値・補正検定](md/RESULTS.md)／[最適値への初回到達](md/FIRST_OPTIMUM.md)
- [全体PDF](pdf/fmqa_bbo_overview.pdf)／[図の見方](md/FIGURES.md)
- [解釈と限界](md/DISCUSSION.md)／[再現手順](md/REPRODUCIBILITY.md)

FM設定のrankとdecayは同時変更で、個別のBBO寄与は切り分けていない。親benchmarkは既存の未確認事項を保持したdraftのまま。
