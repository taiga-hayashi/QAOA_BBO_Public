# 初期データを5／10件へ減らす比較

初期5/10/20件、各100出力、同じrank1 AdamW wd0.01・lr0.1・120 epochs。総BB上限80を揃え、最大サイクルを75/70/60にした。3手法×3初期件数×5Seedの45Run完了、20件の既存15Runは全サイクルを完全再現。

初期最適解なし共通4Seedの到達評価数中央値は、Adaptive25/35/30、LargePenalty14.5/38/32、XY27.5/25.5/39（5/10/20の順）。全45Runが最適値1.5249 eVを保持した。XYでは初期10が有力候補だが、初期FM精度は20件の方が良く、補正検定の有意差は未確認。

- [凍結プロトコル](json/protocol.json)／[監査](md/AUDIT.md)
- [結果・Seed別・初期FM・検定](md/RESULTS.md)
- [全体PDF](pdf/initial_count_overview.pdf)／[図の見方](md/FIGURES.md)
- [解釈と次の案](md/DISCUSSION.md)／[再現手順](md/REPRODUCIBILITY.md)

初期集合は元20件の先頭を使うnestedな比較。初期件数の指定がなかったため5/10/20を採用した。初期を減らした分を追加探索に配分する実験であり、60サイクル固定ではない。元問題23 One-Hot変数、XYはλ0・p1・W/Ring・9理想角度点を保持。
