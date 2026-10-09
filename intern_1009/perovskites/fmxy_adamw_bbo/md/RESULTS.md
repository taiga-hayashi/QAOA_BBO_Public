# FM-XYQAOA再学習最適化

同じrank1・AdamW・wd0.01、lr0.1、120 epochs。初期20件、60サイクル、10出力/cycle、最大1件採用、評価上限80。XYはp1・λ0・product W・Ring順序付き辺ゲート積、9角度点の理想期待値で毎サイクル選択。5 Seed・300サイクル・2700角度評価・3000shots完了。数値は中央値 [Q1,Q3]。

| 手法 | 最終best eV | 実評価数 | 候補なしサイクル | 最適解保持 | 新規発見 |
|---|---:|---:|---:|---:|---:|
| XY-FMQAOA | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] | 5/5 | 4/4 |
| Adaptive-FMQA | 1.525 [1.525,1.525] | 59 [57,61] | 21 [19,23] | 5/5 | 4/4 |
| LargePenalty-FMQA | 1.525 [1.525,1.525] | 80 [80,80] | 0 [0,0] | 5/5 | 4/4 |

Seed101は初期20件に最適解があり、新規発見の分母は4。最適値は全192候補表の1.5249 eV、hydrazinium/Sn/I。物理実験の新規最適性を主張しない。

| 手法 | Seed | 初期best | 最終best | 実評価数 | 最適値初回到達評価数 |
|---|---:|---:|---:|---:|---:|
| XY-FMQAOA | 42 | 1.9874 | 1.5249 | 80 | 40 |
| XY-FMQAOA | 101 | 1.5249 | 1.5249 | 80 | 20 |
| XY-FMQAOA | 2024 | 1.8386 | 1.5249 | 80 | 44 |
| XY-FMQAOA | 7 | 1.9090 | 1.5249 | 80 | 66 |
| XY-FMQAOA | 19 | 1.5456 | 1.5249 | 80 | 80 |
| Adaptive-FMQA | 42 | 1.9874 | 1.5249 | 57 | 42 |
| Adaptive-FMQA | 101 | 1.5249 | 1.5249 | 63 | 20 |
| Adaptive-FMQA | 2024 | 1.8386 | 1.5249 | 55 | 31 |
| Adaptive-FMQA | 7 | 1.9090 | 1.5249 | 61 | 31 |
| Adaptive-FMQA | 19 | 1.5456 | 1.5249 | 59 | 23 |
| LargePenalty-FMQA | 42 | 1.9874 | 1.5249 | 80 | 29 |
| LargePenalty-FMQA | 101 | 1.5249 | 1.5249 | 80 | 20 |
| LargePenalty-FMQA | 2024 | 1.8386 | 1.5249 | 80 | 27 |
| LargePenalty-FMQA | 7 | 1.9090 | 1.5249 | 80 | 21 |
| LargePenalty-FMQA | 19 | 1.5456 | 1.5249 | 80 | 56 |

## 補正検定

| 比較 | 指標 | ペア差分中央値 | raw p | Bonferroni p |
|---|---|---:|---:|---:|
| XY minus Adaptive-FMQA | final_regret | 0 | 1 | 1 |
| XY minus Adaptive-FMQA | actual_evaluations | 21 | 0.0625 | 0.25 |
| XY minus LargePenalty-FMQA | final_regret | 0 | 1 | 1 |
| XY minus LargePenalty-FMQA | actual_evaluations | 0 | 1 | 1 |

BB予算と出力数は揃えたが、SAの1000sweepsとQAOAの9理想期待値評価は同じ計算費用ではない。有限shotsの角度探索、大域的角度最適性、実機ゲート合成、標準X-QAOAは未検証。終了は60サイクルの打ち切りであり、収束判定ではない。
