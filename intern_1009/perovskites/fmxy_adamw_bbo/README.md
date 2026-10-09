# 同じAdamW設定でのFM-XYQAOA再学習最適化

rank1・AdamW・weight decay0.01・lr0.1・120 epochs。元の初期20件、60サイクル、10shots/cycle、最大1件採用、評価上限80で5 Seedを実行。XYはp1、W-state初期化、Ring順序付き辺ゲート積、λ0。

全5 Runが192候補表の最適値1.5249 eVに到達。初期最適解ありの1 Runを除き新規4/4。新規発見に40/44/66/80評価を使い、今回のSAより少ない評価数で到達する優位性は示していない。

- [凍結条件](json/protocol.json)／[回路監査](md/AUDIT.md)
- [結果・SA比較・全Seed・補正検定](md/RESULTS.md)
- [全体PDF](pdf/fmxy_bbo_overview.pdf)／[図の見方](md/FIGURES.md)
- [解釈](md/DISCUSSION.md)／[再現手順](md/REPRODUCIBILITY.md)

OpenQARPの厳密な符号化シミュレーションを共有backendに追加して実行した。元23ビット回路との確率をN6/N9/N23で照合し、全300モデル・2700角度・3000shotsを事後独立検証した。シミュレータは8ビットだが、元問題と元回路は23 One-Hotビットである。
