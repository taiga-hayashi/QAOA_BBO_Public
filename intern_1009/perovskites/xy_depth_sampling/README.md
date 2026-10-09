# 1つの固定FMでXY-QAOAのpを変える

ユーザー指定によりXYのp1/2/3を比較。直前と同じSeed42・初期10・rank1 AdamWモデルを凍結し、lambda0、W/Ringを保持。各p128角度候補×10探索Seed、最終100shots×10測定/探索Seed。30探索・3840角度評価・300測定反復を完了。

FM最小値の理想100shot捕捉率中央値はp1/2:6.44%、p3:29.16%。全測定の観測捕捉はp1/2:9/100、p3:26/100。選択gap中央値は0.39828→0.20582 eV。p2は全Seedでp1から引き継いだ解を選んだため、角度探索不足と表現能力を区別する。

- [結果・全探索Seed・検定](md/RESULTS.md)
- [全体PDF](pdf/XY_depth_overview.pdf)／[図の見方](md/FIGURES.md)
- [解釈・限界](md/DISCUSSION.md)／[再現手順](md/REPRODUCIBILITY.md)
- [凍結プロトコル](json/protocol.json)／[事前検証](json/prevalidation.json)

新規FM学習・BBO・標準X-QAOA/SA比較は含めない。理想期待値によるランダム角度探索で大域的最適角度の証明ではない。同じ角度評価件数でも回路層数はpに比例し異なる。元23bits、厳密符号化simulator8bits。
