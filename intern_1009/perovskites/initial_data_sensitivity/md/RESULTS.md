# 初期5／10／20件の比較

同じ100出力・rank1 AdamW wd0.01、lr0.1、120epochs。BB上限80を共通化し、初期5/10/20に対し最大75/70/60サイクル。45Runを新規計算。中央値 [Q1,Q3]、初回到達は共通の初期最適解なし4Seed、他は5Seed。

| 手法 | 初期件数 | 新規最適解発見 | 初回到達評価数／未到達81 | 最終best eV | 実評価数 | 候補なし割合 |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive-FMQA | 5 | 5/5 | 25 [20.25,31] | 1.525 [1.525,1.525] | 65 [65,66] | 0.2 [0.1867,0.2] |
| Adaptive-FMQA | 10 | 5/5 | 35 [33.75,36.5] | 1.525 [1.525,1.525] | 66 [66,66] | 0.2 [0.2,0.2] |
| Adaptive-FMQA | 20 | 4/4 | 30 [26.75,32.5] | 1.525 [1.525,1.525] | 63 [61,64] | 0.2833 [0.2667,0.3167] |
| LargePenalty-FMQA | 5 | 5/5 | 14.5 [7.5,21] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] |
| LargePenalty-FMQA | 10 | 5/5 | 38 [31.75,44.5] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] |
| LargePenalty-FMQA | 20 | 4/4 | 32 [27,36.5] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] |
| XY-FMQAOA | 5 | 5/5 | 27.5 [19.5,33.25] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] |
| XY-FMQAOA | 10 | 5/5 | 25.5 [22.75,31] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] |
| XY-FMQAOA | 20 | 4/4 | 39 [35.75,44] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] |

初回到達は初期件数込み。Seed101は20件では初期最適解あり、5/10件では初期になし。到達回数の主比較は共通4Seed42/2024/7/19で揃え、101は下表に残す。未到達81は失敗コードで実評価回数ではない。成功Runだけの中央値を出さない。

| 手法 | 初期 | Seed | 初期best | 初期最適解 | 初回到達 | 最終best | 実評価数 |
|---|---:|---:|---:|---|---:|---:|---:|
| Adaptive-FMQA | 5 | 42 | 2.1431 | False | 9 | 1.5249 | 66 |
| Adaptive-FMQA | 5 | 101 | 2.3129 | False | 26 | 1.5249 | 65 |
| Adaptive-FMQA | 5 | 2024 | 1.8386 | False | 24 | 1.5249 | 66 |
| Adaptive-FMQA | 5 | 7 | 1.9374 | False | 26 | 1.5249 | 63 |
| Adaptive-FMQA | 5 | 19 | 1.5456 | False | 46 | 1.5249 | 65 |
| Adaptive-FMQA | 10 | 42 | 1.9874 | False | 35 | 1.5249 | 66 |
| Adaptive-FMQA | 10 | 101 | 2.3129 | False | 33 | 1.5249 | 69 |
| Adaptive-FMQA | 10 | 2024 | 1.8386 | False | 30 | 1.5249 | 66 |
| Adaptive-FMQA | 10 | 7 | 1.9374 | False | 41 | 1.5249 | 66 |
| Adaptive-FMQA | 10 | 19 | 1.5456 | False | 35 | 1.5249 | 63 |
| Adaptive-FMQA | 20 | 42 | 1.9874 | False | 32 | 1.5249 | 61 |
| Adaptive-FMQA | 20 | 101 | 1.5249 | True | 20 | 1.5249 | 64 |
| Adaptive-FMQA | 20 | 2024 | 1.8386 | False | 34 | 1.5249 | 59 |
| Adaptive-FMQA | 20 | 7 | 1.9090 | False | 28 | 1.5249 | 64 |
| Adaptive-FMQA | 20 | 19 | 1.5456 | False | 23 | 1.5249 | 63 |
| LargePenalty-FMQA | 5 | 42 | 2.1431 | False | 21 | 1.5249 | 80 |
| LargePenalty-FMQA | 5 | 101 | 2.3129 | False | 26 | 1.5249 | 80 |
| LargePenalty-FMQA | 5 | 2024 | 1.8386 | False | 6 | 1.5249 | 80 |
| LargePenalty-FMQA | 5 | 7 | 1.9374 | False | 8 | 1.5249 | 80 |
| LargePenalty-FMQA | 5 | 19 | 1.5456 | False | 21 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 42 | 1.9874 | False | 28 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 101 | 2.3129 | False | 35 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 2024 | 1.8386 | False | 43 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 7 | 1.9374 | False | 33 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 19 | 1.5456 | False | 49 | 1.5249 | 80 |
| LargePenalty-FMQA | 20 | 42 | 1.9874 | False | 28 | 1.5249 | 80 |
| LargePenalty-FMQA | 20 | 101 | 1.5249 | True | 20 | 1.5249 | 80 |
| LargePenalty-FMQA | 20 | 2024 | 1.8386 | False | 38 | 1.5249 | 80 |
| LargePenalty-FMQA | 20 | 7 | 1.9090 | False | 36 | 1.5249 | 80 |
| LargePenalty-FMQA | 20 | 19 | 1.5456 | False | 24 | 1.5249 | 80 |
| XY-FMQAOA | 5 | 42 | 2.1431 | False | 31 | 1.5249 | 80 |
| XY-FMQAOA | 5 | 101 | 2.3129 | False | 23 | 1.5249 | 80 |
| XY-FMQAOA | 5 | 2024 | 1.8386 | False | 6 | 1.5249 | 80 |
| XY-FMQAOA | 5 | 7 | 1.9374 | False | 24 | 1.5249 | 80 |
| XY-FMQAOA | 5 | 19 | 1.5456 | False | 40 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 42 | 1.9874 | False | 22 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 101 | 2.3129 | False | 46 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 2024 | 1.8386 | False | 28 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 7 | 1.9374 | False | 23 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 19 | 1.5456 | False | 40 | 1.5249 | 80 |
| XY-FMQAOA | 20 | 42 | 1.9874 | False | 42 | 1.5249 | 80 |
| XY-FMQAOA | 20 | 101 | 1.5249 | True | 20 | 1.5249 | 80 |
| XY-FMQAOA | 20 | 2024 | 1.8386 | False | 35 | 1.5249 | 80 |
| XY-FMQAOA | 20 | 7 | 1.9090 | False | 36 | 1.5249 | 80 |
| XY-FMQAOA | 20 | 19 | 1.5456 | False | 50 | 1.5249 | 80 |

## 初期モデルの共通172候補診断

元20件の全てを除いた172候補で評価し、test集合を初期件数間で共通化。真値は事後監査のみ。最初のモデルは3手法で同じなので重複集計しない。

| 初期 | 有機カテゴリ網羅数（16中） | Test RMSE eV |
|---:|---:|---:|
| 5 | 4 [4,5] | 1.853 [1.781,2.029] |
| 10 | 7 [7,8] | 1.442 [1.214,1.737] |
| 20 | 12 [10,12] | 0.8757 [0.8717,1.096] |

## 事前固定ペア比較

| 手法 | 比較 | 指標 | ペア差中央値 | raw p | Bonferroni24 p |
|---|---|---|---:|---:|---:|
| Adaptive-FMQA | 5 minus20 | final_regret | 0 | 1 | 1 |
| Adaptive-FMQA | 5 minus20 | first_hit_score81 | -6 | 0.75 | 1 |
| Adaptive-FMQA | 5 minus20 | candidate_none_fraction | -0.083333 | 0.0625 | 1 |
| Adaptive-FMQA | 5 minus20 | median_unseenFMgap | 0 | 1 | 1 |
| Adaptive-FMQA | 10 minus20 | final_regret | 0 | 1 | 1 |
| Adaptive-FMQA | 10 minus20 | first_hit_score81 | 7.5 | 0.375 | 1 |
| Adaptive-FMQA | 10 minus20 | candidate_none_fraction | -0.10952 | 0.0625 | 1 |
| Adaptive-FMQA | 10 minus20 | median_unseenFMgap | 0 | 1 | 1 |
| LargePenalty-FMQA | 5 minus20 | final_regret | 0 | 1 | 1 |
| LargePenalty-FMQA | 5 minus20 | first_hit_score81 | -17.5 | 0.125 | 1 |
| LargePenalty-FMQA | 5 minus20 | candidate_none_fraction | 0 | 1 | 1 |
| LargePenalty-FMQA | 5 minus20 | median_unseenFMgap | -0.003207 | 0.8125 | 1 |
| LargePenalty-FMQA | 10 minus20 | final_regret | 0 | 1 | 1 |
| LargePenalty-FMQA | 10 minus20 | first_hit_score81 | 2.5 | 0.5 | 1 |
| LargePenalty-FMQA | 10 minus20 | candidate_none_fraction | 0 | 1 | 1 |
| LargePenalty-FMQA | 10 minus20 | median_unseenFMgap | 0.020548 | 0.625 | 1 |
| XY-FMQAOA | 5 minus20 | final_regret | 0 | 1 | 1 |
| XY-FMQAOA | 5 minus20 | first_hit_score81 | -11.5 | 0.125 | 1 |
| XY-FMQAOA | 5 minus20 | candidate_none_fraction | 0 | 1 | 1 |
| XY-FMQAOA | 5 minus20 | median_unseenFMgap | -0.012236 | 0.3125 | 1 |
| XY-FMQAOA | 10 minus20 | final_regret | 0 | 1 | 1 |
| XY-FMQAOA | 10 minus20 | first_hit_score81 | -11.5 | 0.125 | 1 |
| XY-FMQAOA | 10 minus20 | candidate_none_fraction | 0 | 1 | 1 |
| XY-FMQAOA | 10 minus20 | median_unseenFMgap | -0.040963 | 0.4375 | 1 |

総BB予算を揃えた初期配分比較で、同じサイクル数の比較ではない。少ない初期件数には追加探索サイクルを割り当てる。全条件で初期集合のカテゴリ分布も変わり、学習件数だけの純粋な因果効果ではない。既評価/invalidは新規BB評価なし、候補なしno refill。XYは9理想角度点・p1・λ0、100は候補生成shotsで角度期待値推定ではない。
