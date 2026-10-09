# 同じ初期20件でのFM正則化診断

rank1/2 × Adam wd0 / AdamW wd0 / AdamW wd0.01 ×5 Seedの30学習を完了。lr0.1・120 epochsを固定した。AdamW wd0はAdamと一致し、wd0.01で予測誤差中央値は両ランクとも改善。選択材料の真値はrank1で3/5、rank2で1/5 Seed改善し、残りは同値だった。補正検定の有意差は確認されていない。

- [凍結プロトコル](json/protocol.json)
- [数値・IQR・ペア差分・補正検定](md/RESULTS.md)
- [全体プロットPDF](pdf/fm_regularization_overview.pdf)／[図の見方](md/FIGURES.md)
- [解釈と次の案](md/DISCUSSION.md)
- [再現方法](md/REPRODUCIBILITY.md)

30 checkpoint、生のSeed別予測・選択候補・係数・指標を保存した。これは既に参照したtestを用いる探索的な固定データ診断。SA/QAOAサンプル数、再学習BBO、初期集合の網羅設計は今回の計算に含めない。
