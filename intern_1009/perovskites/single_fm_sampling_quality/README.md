# 1つの固定FMのサンプリング品質

Seed42・初期10件のrank1 AdamWモデル1つを固定。各alpha1/3/10/100/1000/10000/1000000の既存実測100reads×10反復を再解析した。新規学習・SA・BBOなし。

厳密な未評価FM最小値の100reads捕捉はalpha1で10/10、100で4/10、1000で2/10。最良候補gap中央値は0／0.26457／0.53198 eV。raw制約充足はほぼ1でも、大ペナルティでは良いFM値への集中が弱くなる。FM最小候補の真値は2.2338 eVで、真のBB最適値とは異なる。

- [全体PDF](pdf/single_FM_sampling_overview.pdf)
- [結果・全alpha・検定](md/RESULTS.md)／[解釈](md/DISCUSSION.md)
- [図の見方](md/FIGURES.md)／[再現手順](md/REPRODUCIBILITY.md)
- [凍結プロトコル](json/protocol.json)／[モデル・全反復の検証](json/prevalidation.json)

図ではFM値の経験CDF、FM順位別出現頻度、100reads最良値、最小値捕捉、重複・多様性を区別。これは1モデルのサンプラー品質であり、複数モデル中央値や最終BBO目的値の比較ではない。
