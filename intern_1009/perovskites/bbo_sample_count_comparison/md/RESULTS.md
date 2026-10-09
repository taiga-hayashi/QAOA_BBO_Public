# 10／100出力の再学習最適化比較

rank1・AdamW wd0.01・lr0.1・120 epochs。初期20件、60サイクル、最大1件採用、BB評価上限80を共通化。3手法×2出力数×5Seedの30Runを新規計算し、過去10出力の15Runを全60サイクル完全再現した。中央値 [Q1,Q3]、初回到達だけ初期最適解なし4Seed、他は5Seed。

| 手法 | 出力数 | 新規最適解発見 | 初回到達評価数／未到達81 | 最終best eV | 実評価数 | 候補なしサイクル | 採用候補のFM gap eV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive-FMQA | 10 | 4/4 | 31 [29,33.75] | 1.525 [1.525,1.525] | 59 [57,61] | 21 [19,23] | 0.01422 [0.002218,0.0624] |
| Adaptive-FMQA | 100 | 4/4 | 30 [26.75,32.5] | 1.525 [1.525,1.525] | 63 [61,64] | 17 [16,19] | 0 [0,0] |
| LargePenalty-FMQA | 10 | 4/4 | 28 [25.5,35.75] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] | 0.681 [0.6739,0.74] |
| LargePenalty-FMQA | 100 | 4/4 | 32 [27,36.5] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] | 0.04248 [0.03499,0.0595] |
| XY-FMQAOA | 10 | 4/4 | 55 [43,69.5] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] | 0.9737 [0.9151,1.11] |
| XY-FMQAOA | 100 | 4/4 | 39 [35.75,44] | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] | 0.2303 [0.1938,0.2937] |

採用FM gap＝選択候補のFM予測−その時点の未評価実行可能192候補内FM最小（既評価を除く）。各Runの採用サイクルにおけるgap中央値を計算し、それを5Seedで要約。候補なしをgap0として扱わない。global feasible FM最小との差も各イベントに保存。FM予測の差で、真の目的値の差ではない。

初回到達は初期20を含む。Seed101は初期から最適解を含み、新規発見と到達スコアの集計から分けた。未到達を81という失敗コードで保持し、未到達Runを除外した中央値を出さない。81は実際に行った評価ではない。最適値1.5249 eV、最悪6.3242 eVは全192候補表の固定値。

## Seed別結果

| 手法 | count | Seed | 初期最適解 | 初回到達（None＝未到達） | 最終best | 実評価数 |
|---|---:|---:|---|---:|---:|---:|
| Adaptive-FMQA | 10 | 42 | False | 42 | 1.5249 | 57 |
| Adaptive-FMQA | 10 | 101 | True | 20 | 1.5249 | 63 |
| Adaptive-FMQA | 10 | 2024 | False | 31 | 1.5249 | 55 |
| Adaptive-FMQA | 10 | 7 | False | 31 | 1.5249 | 61 |
| Adaptive-FMQA | 10 | 19 | False | 23 | 1.5249 | 59 |
| Adaptive-FMQA | 100 | 42 | False | 32 | 1.5249 | 61 |
| Adaptive-FMQA | 100 | 101 | True | 20 | 1.5249 | 64 |
| Adaptive-FMQA | 100 | 2024 | False | 34 | 1.5249 | 59 |
| Adaptive-FMQA | 100 | 7 | False | 28 | 1.5249 | 64 |
| Adaptive-FMQA | 100 | 19 | False | 23 | 1.5249 | 63 |
| LargePenalty-FMQA | 10 | 42 | False | 29 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 101 | True | 20 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 2024 | False | 27 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 7 | False | 21 | 1.5249 | 80 |
| LargePenalty-FMQA | 10 | 19 | False | 56 | 1.5249 | 80 |
| LargePenalty-FMQA | 100 | 42 | False | 28 | 1.5249 | 80 |
| LargePenalty-FMQA | 100 | 101 | True | 20 | 1.5249 | 80 |
| LargePenalty-FMQA | 100 | 2024 | False | 38 | 1.5249 | 80 |
| LargePenalty-FMQA | 100 | 7 | False | 36 | 1.5249 | 80 |
| LargePenalty-FMQA | 100 | 19 | False | 24 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 42 | False | 40 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 101 | True | 20 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 2024 | False | 44 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 7 | False | 66 | 1.5249 | 80 |
| XY-FMQAOA | 10 | 19 | False | 80 | 1.5249 | 80 |
| XY-FMQAOA | 100 | 42 | False | 42 | 1.5249 | 80 |
| XY-FMQAOA | 100 | 101 | True | 20 | 1.5249 | 80 |
| XY-FMQAOA | 100 | 2024 | False | 35 | 1.5249 | 80 |
| XY-FMQAOA | 100 | 7 | False | 36 | 1.5249 | 80 |
| XY-FMQAOA | 100 | 19 | False | 50 | 1.5249 | 80 |

## 同一Seedペア差分（100−10）

| 手法 | 指標 | n | 差分中央値 | raw p | Bonferroni12 p |
|---|---|---:|---:|---:|---:|
| Adaptive-FMQA | final_regret | 5 | 0 | 1 | 1 |
| Adaptive-FMQA | first_hit_score81 | 4 | -1.5 | 0.75 | 1 |
| Adaptive-FMQA | zero_candidate_cycles | 5 | -4 | 0.0625 | 0.75 |
| Adaptive-FMQA | median_unseen_FM_gap | 5 | -0.014224 | 0.125 | 1 |
| LargePenalty-FMQA | final_regret | 5 | 0 | 1 | 1 |
| LargePenalty-FMQA | first_hit_score81 | 4 | 5 | 1 | 1 |
| LargePenalty-FMQA | zero_candidate_cycles | 5 | 0 | 1 | 1 |
| LargePenalty-FMQA | median_unseen_FM_gap | 5 | -0.64147 | 0.0625 | 0.75 |
| XY-FMQAOA | final_regret | 5 | 0 | 1 | 1 |
| XY-FMQAOA | first_hit_score81 | 4 | -19.5 | 0.25 | 1 |
| XY-FMQAOA | zero_candidate_cycles | 5 | 0 | 1 | 1 |
| XY-FMQAOA | median_unseen_FM_gap | 5 | -0.81222 | 0.0625 | 0.75 |

## 同じFM・同じ角度でのXY補助診断

10出力Runの各300固定モデルで、同じ100個の乱数から先頭10と全100を比較。閉ループ100Runとは別。真値の追加評価なし。

| Seed | gap10中央値 eV | gap100中央値 eV | 改善サイクル | 同値サイクル |
|---:|---:|---:|---:|---:|
| 42 | 0.91511 | 0.10676 | 50 | 10 |
| 101 | 0.77757 | 0.15024 | 47 | 13 |
| 2024 | 0.97373 | 0.21579 | 55 | 5 |
| 7 | 1.138 | 0.16325 | 53 | 7 |
| 19 | 1.1096 | 0.17131 | 53 | 7 |

100出力は訓練軌跡とAdaptiveのα更新も変えるため、閉ループ平均gapをサンプラー単独の因果効果と呼ばない。SA100は新しいnum_reads=100で実行し、旧1000poolからの部分抽出ではない。XYの角度探索は9理想期待値のままで、100shotsを角度期待値推定へ使っていない。出力数増加の費用をBB予算一致で打ち消したとは主張しない。
