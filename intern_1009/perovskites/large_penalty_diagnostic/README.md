# 大きすぎるペナルティによる品質悪化の検証

初期10件・rank1 AdamW・100readsでalpha1/3/10/100/1000/10000/1000000を比較。固定FM700batchesと、初期10＋70cycles・BB上限80の35Runを完了。

固定FMでは制約充足を維持していても、選択FM gap中央値がalpha1の0から100の0.18548、1000の0.37983 eVへ悪化。再学習BBOは全35Runで最適値1.5249 eVに到達し、最終目的値の悪化は未確認。有限予算の途中bestが不利になる例はあるが、逆の例も保存。補正検定の有意差は未確認。

- [結果・全Seed・検定](md/RESULTS.md)
- [全体PDF](pdf/large_penalty_overview.pdf)／[最適化軌跡](pdf/BBO_best_by_cycle.pdf)
- [地形・障壁・解釈](md/DISCUSSION.md)／[図の見方](md/FIGURES.md)
- [凍結プロトコル](json/protocol.json)／[監査](md/AUDIT.md)／[再現手順](md/REPRODUCIBILITY.md)

通常の自動温度日程が主比較。alpha1日程固定は固定FMだけの補助実験。alpha100のみ既定LargePenalty-FMQA、それ以外は固定alpha感度variant。QAOA未実行、XY lambda0の条件は保持。
