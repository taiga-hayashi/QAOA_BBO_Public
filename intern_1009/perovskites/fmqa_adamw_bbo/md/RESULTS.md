# 再学習FMQAの結果

23 One-Hot変数のPerovskites固定lookupで、2つのFM設定 ×2つのSA手法 ×5 Seedの20 Run、各60サイクルを完了。初期20件、10reads/cycle、最大1件採用、評価予算上限80。候補なしは0評価・追加サンプルなし。以下は5 Seedの中央値 [Q1,Q3]。

| FM設定 | SA手法 | 最終best eV | Regret | 実評価数 | 候補なしサイクル | 最適解保持Run | 新規発見Run |
|---|---|---:|---:|---:|---:|---:|---:|
| proposed | Adaptive-FMQA | 1.525 [1.525,1.525] | 0 [0,0] | 59 [57,61] | 21 [19,23] | 5/5 | 4/4 |
| proposed | LargePenalty-FMQA | 1.525 [1.525,1.525] | 0 [0,0] | 80 [80,80] | 0 [0,0] | 5/5 | 4/4 |
| original | Adaptive-FMQA | 1.525 [1.525,1.525] | 0 [0,0] | 71 [70,72] | 9 [8,10] | 5/5 | 4/4 |
| original | LargePenalty-FMQA | 1.525 [1.525,1.525] | 0 [0,0] | 80 [80,80] | 0 [0,0] | 4/5 | 3/4 |

proposed＝rank1・AdamW・wd0.01、original＝rank2・Adam・wd0。両方lr0.1・120 epochs、生eV・fullbatch・毎サイクル新規初期化。初期20件は同じ元の集合で網羅設計に変えていない。Seed101は最初から真の最適解1.5249 eVを含むため、新規発見率の分母は残り4 Run。

## Seed別結果

| 設定 | 手法 | Seed | 初期best eV | 最終best eV | 実評価数 | 初期最適解 | 新規最適解 |
|---|---|---:|---:|---:|---:|---|---|
| proposed | Adaptive-FMQA | 42 | 1.9874 | 1.5249 | 57 | False | True |
| proposed | Adaptive-FMQA | 101 | 1.5249 | 1.5249 | 63 | True | False |
| proposed | Adaptive-FMQA | 2024 | 1.8386 | 1.5249 | 55 | False | True |
| proposed | Adaptive-FMQA | 7 | 1.9090 | 1.5249 | 61 | False | True |
| proposed | Adaptive-FMQA | 19 | 1.5456 | 1.5249 | 59 | False | True |
| proposed | LargePenalty-FMQA | 42 | 1.9874 | 1.5249 | 80 | False | True |
| proposed | LargePenalty-FMQA | 101 | 1.5249 | 1.5249 | 80 | True | False |
| proposed | LargePenalty-FMQA | 2024 | 1.8386 | 1.5249 | 80 | False | True |
| proposed | LargePenalty-FMQA | 7 | 1.9090 | 1.5249 | 80 | False | True |
| proposed | LargePenalty-FMQA | 19 | 1.5456 | 1.5249 | 80 | False | True |
| original | Adaptive-FMQA | 42 | 1.9874 | 1.5249 | 75 | False | True |
| original | Adaptive-FMQA | 101 | 1.5249 | 1.5249 | 72 | True | False |
| original | Adaptive-FMQA | 2024 | 1.8386 | 1.5249 | 71 | False | True |
| original | Adaptive-FMQA | 7 | 1.9090 | 1.5249 | 70 | False | True |
| original | Adaptive-FMQA | 19 | 1.5456 | 1.5249 | 68 | False | True |
| original | LargePenalty-FMQA | 42 | 1.9874 | 1.5249 | 80 | False | True |
| original | LargePenalty-FMQA | 101 | 1.5249 | 1.5249 | 80 | True | False |
| original | LargePenalty-FMQA | 2024 | 1.8386 | 1.5456 | 80 | False | False |
| original | LargePenalty-FMQA | 7 | 1.9090 | 1.5249 | 80 | False | True |
| original | LargePenalty-FMQA | 19 | 1.5456 | 1.5249 | 80 | False | True |

## ペア比較（提案−従来）

| SA手法 | 指標 | 差分中央値 | raw p | Bonferroni p |
|---|---|---:|---:|---:|
| Adaptive-FMQA | final_regret | 0 | 1 | 1 |
| Adaptive-FMQA | actual_evaluations | -9 | 0.0625 | 0.25 |
| LargePenalty-FMQA | final_regret | 0 | 1 | 1 |
| LargePenalty-FMQA | actual_evaluations | 0 | 1 | 1 |

正規化Regretは全192候補のmin1.5249/max6.3242で計算。探索中に全真値は参照せず、訓練には初期値と採用候補の値のみ。全192真値を使うFM精度・最適性照合は別の事後診断。5 Seed、失敗・除外0。60サイクル打ち切りを収束と解釈しない。
