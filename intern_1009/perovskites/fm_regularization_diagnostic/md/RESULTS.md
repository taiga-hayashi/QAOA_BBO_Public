# Adam / AdamW・weight decay比較

初期20件・5 Seedを保持し、rank1/2 × Adam wd0 / AdamW wd0 / AdamW wd0.01を比較した。lr0.1、120 epochs、raw eV、全バッチMSE。30学習を完了し、失敗・除外は0。以下は中央値 [Q1,Q3]。

| Rank | Optimizer | wd | Train RMSE eV | Test RMSE eV | Test Spearman | 予測最小の未評価候補の真値 eV |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Adam | 0.0 | 0.0218 [0.0099, 0.0277] | 0.9032 [0.8869, 1.1973] | 0.6882 [0.5628, 0.7845] | 2.2305 [2.1542, 2.3986] |
| 1 | AdamW | 0.0 | 0.0218 [0.0099, 0.0277] | 0.9032 [0.8869, 1.1973] | 0.6882 [0.5628, 0.7845] | 2.2305 [2.1542, 2.3986] |
| 1 | AdamW | 0.01 | 0.0203 [0.0122, 0.0323] | 0.8757 [0.8717, 1.0956] | 0.6982 [0.6009, 0.7826] | 1.8440 [1.8386, 1.8440] |
| 2 | Adam | 0.0 | 0.0084 [0.0047, 0.0100] | 1.4963 [1.2585, 1.5519] | 0.4155 [0.3346, 0.5973] | 3.3912 [2.8957, 3.4691] |
| 2 | AdamW | 0.0 | 0.0084 [0.0047, 0.0100] | 1.4963 [1.2585, 1.5519] | 0.4155 [0.3346, 0.5973] | 3.3912 [2.8957, 3.4691] |
| 2 | AdamW | 0.01 | 0.0126 [0.0075, 0.0131] | 1.3568 [1.2258, 1.3588] | 0.4601 [0.3714, 0.6029] | 2.8957 [2.7329, 3.4691] |

## 同一Seedのペア差分

差分は前者−後者。RMSEと真値は負、相関は正で改善。Wilcoxon検定は事前固定した18比較をBonferroni補正。

| Rank | 比較 | 指標 | 差分中央値 | raw p | 補正p |
|---:|---|---|---:|---:|---:|
| 1 | AdamW_wd0.0 minus Adam_wd0.0 | test_rmse | 0 | 1 | 1 |
| 1 | AdamW_wd0.0 minus Adam_wd0.0 | test_spearman | 0 | 1 | 1 |
| 1 | AdamW_wd0.0 minus Adam_wd0.0 | selected_true_value | 0 | 1 | 1 |
| 1 | AdamW_wd0.01 minus AdamW_wd0.0 | test_rmse | -0.031567 | 0.1875 | 1 |
| 1 | AdamW_wd0.01 minus AdamW_wd0.0 | test_spearman | 0.010073 | 0.3125 | 1 |
| 1 | AdamW_wd0.01 minus AdamW_wd0.0 | selected_true_value | -0.56 | 0.25 | 1 |
| 1 | AdamW_wd0.01 minus Adam_wd0.0 | test_rmse | -0.031567 | 0.1875 | 1 |
| 1 | AdamW_wd0.01 minus Adam_wd0.0 | test_spearman | 0.010073 | 0.3125 | 1 |
| 1 | AdamW_wd0.01 minus Adam_wd0.0 | selected_true_value | -0.56 | 0.25 | 1 |
| 2 | AdamW_wd0.0 minus Adam_wd0.0 | test_rmse | 0 | 1 | 1 |
| 2 | AdamW_wd0.0 minus Adam_wd0.0 | test_spearman | 0 | 1 | 1 |
| 2 | AdamW_wd0.0 minus Adam_wd0.0 | selected_true_value | 0 | 1 | 1 |
| 2 | AdamW_wd0.01 minus AdamW_wd0.0 | test_rmse | -0.096661 | 0.0625 | 1 |
| 2 | AdamW_wd0.01 minus AdamW_wd0.0 | test_spearman | 0.016755 | 0.0625 | 1 |
| 2 | AdamW_wd0.01 minus AdamW_wd0.0 | selected_true_value | 0 | 1 | 1 |
| 2 | AdamW_wd0.01 minus Adam_wd0.0 | test_rmse | -0.096661 | 0.0625 | 1 |
| 2 | AdamW_wd0.01 minus Adam_wd0.0 | test_spearman | 0.016755 | 0.0625 | 1 |
| 2 | AdamW_wd0.01 minus Adam_wd0.0 | selected_true_value | 0 | 1 | 1 |

Seed別の全値はjson/seed_*.json、集計と5個のペア差分はjson/summary.jsonに保存。初期20件を除く172候補で評価し、予測最小候補は真値を使わず全列挙で選択する。真値は保存済みlookup表を事後参照した値。selected_test_regretの基準は各Seedの172候補のmin/maxであり、全192候補のRegretと異なる。

この比較では既に見たtestを再使用している。探索的診断であり、採用設定の独立確認ではない。BBO・SA・QAOA・新規物理評価は実施していない。
