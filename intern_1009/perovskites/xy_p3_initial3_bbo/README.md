# 初期3件でXYp3最適化を再実行

ユーザー指定の初期3、XYp3、LargePenaltyは候補品質が悪化したalpha1000。比較用XYp1も同じ初期集合/FМ条件で再計算。3＋77cycles、BB上限80、100outputs、rank1 AdamW wd0.01 lr0.1 120epochs。15Run・1155cycles完了。

全3条件で5/5Runが最適値1.5249 eVに到達。到達評価数中央値はXYp1:33、p3:29、SAalpha1000:27。p3のFM gapはp1より小さいが、今回のSAも全試行で最適解を得た。補正有意差は未確認。

- [結果・全Seed・検定](md/RESULTS.md)
- [比較プロットPDF](pdf/initial3_p3_BBO_overview.pdf)／[cycle軌跡](pdf/best_cycles.pdf)
- [解釈](md/DISCUSSION.md)／[図の見方](md/FIGURES.md)／[再現手順](md/REPRODUCIBILITY.md)
- [凍結プロトコル](json/protocol.json)／[事前検証](json/prevalidation.json)

XYの128候補は旧p1grid9点（p3はゼロ層埋込み）＋119ランダム点、補助p2探索なし。lambda0、W/Ring、原23bits/sim8。固定FM深さ実験の段階warm探索と区別。SAはalpha1000固定、lambda=Salpha、自動beta。初期3件は変数数3を意味せず、23One-Hot変数の問題を維持する。
