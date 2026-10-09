# 小規模FMQAの文献設定

確認日2026-10-09。論文本文の値を主に確認。ビット数、元変数数、符号化、初期データ数、epochs、BBOサイクルを区別する。今回文献設定の再現学習は実施していない。

| 出典 | 規模・表現 | 初期データ | ランク | Optimizer | 学習率 | epochs |
|---|---|---:|---:|---|---:|---:|
| 現在のPerovskites pilot | One-Hot23bit、実行可能192材料 | 20 | 2 | Adam | 0.1 | 120 |
| Kitai et al., Designing metamaterials with quantum annealing and factorization machines, PRResearch2020 | 24bitのメタマテリアル例（L6,C3） | 50 | 8 | Adam | 本文確認範囲では未確認 | 本文確認範囲では未確認 |
| Distributed QAOA arXiv2407.20212v3, AL-DQAOA | 30bitの光学多層膜例、1層を2bitで表現 | 25（学習/テスト4:1分割） | 本文確認範囲では未確認 | 同左 | 同左 | 同左 |
| Ogawa et al., Stage-dependent integer-binary encoding… arXiv2606.23188v1 | N5,q61のbinary-only対照は30bit（式12から算術）。同問題のOne-Hotは305bit | 64 | 8 | AdamW | 0.5、200epochsごと×0.9 | 最大2000、trainloss<1e−8早期停止 |
| 参考: Nakano et al., Optimization Performance… arXiv2507.21024v1 | LABS、主実験64bit、感度16/49/64/81/101bit。20〜30bitの直接例ではない | 100 | 8 | AdamW | 0.01 | 1000 |

## 一次資料と確認箇所

- [Kitai et al. published PDF](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.013319): II.CにK8/Adam、III.AにL6,C3の24bit例/初期50。II.Dの50readsから最低エネルギー1件採用。学習率・epochsは本調査で未確認。現行GitHubの値を当時の論文値として補完しない。
- [AL-DQAOA v3](https://arxiv.org/html/2407.20212v3): 4.7に初期25と4:1分割、4.8に1層2bit表現、30bitの最大3000最適化サイクル。25件の全てを学習に使う設定ではない。FM学習率・rank・epochsの公開コード照合は未実施。
- [Stage-dependent encoding v1](https://arxiv.org/html/2606.23188v1): III.1式12からceil(log2q)bit/元変数、V.2 Table3と直後に全設定。weight_decay0.01、fullbatch、前回FMをwarm-start、一次係数と因子を含む初期値uniform[-1,1]。2000学習epochsと2000FMQAサイクルは別の数字。目標値はV.1の非線形スケーリングを使用し、生eV学習ではない。提案OhDwのFM入力が30bitという意味ではなく、30bitはbinary-only対照だけ。設定の移植にはこれらを揃える必要がある。
- [Limited training data v1](https://arxiv.org/html/2507.21024v1): IV.B TableIにK8/AdamW/lr0.01/1000epochs/初期100、1500FMAサイクル、15readsから3件追加。AdamWのdefaultを採用する旨があるがweight_decayの数値を論文明示値として本表に補完しない。64bitが基準なので23bitの推奨値とは扱わない。

## 今回の検討への含意

文献にはランク8・初期50/64/100の例があるが、23ビットなら必ずランク8・50点がよいという一般則ではない。現在の問題はOne-Hot23ビットでも実行可能候補192個で、通常の24/30二進ビット空間と候補数・カテゴリ構造が異なる。初期カテゴリ網羅、相互作用の強さ、初期化、正則化、標的スケーリングを併せて検討する。

次の正則化診断では、現在のAdam/no decayとAdamW/weight_decay0.01を別条件にし、まずランク1/2を固定して差を確認する。文献参照のrank8条件を追加する場合も別対照とし、既存20件で改善済みと仮定しない。lr0.01と1000epochsは無正則化Adamでは既に探索したが、AdamW・異なる初期化・スケーリングを含む論文手法の再現にはなっていない。
