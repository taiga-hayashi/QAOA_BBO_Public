# Exact-FM／ランダムBBO対照

新規10Runと保存済みXY-p3／SAalpha1000の10Runを同じ初期3件・5Seed・77サイクル・最大80評価で比較。全列挙は真値を使わず、未評価候補のFM予測値のみで選ぶ。Randomは未評価集合から一様に1件を選ぶ。

| 手法 | 成功 | 最終best eV中央値[IQR] | 到達評価数中央値[IQR]（未到達81） |
|---|---:|---:|---:|
| Exact-FM | 5/5 | 1.5249 [1.5249, 1.5249] | 39 [27, 39] |
| Random-unseen | 1/5 | 1.5456 [1.5456, 1.5456] | 81 [81, 81] |
| XY-p3 | 5/5 | 1.5249 [1.5249, 1.5249] | 29 [21, 30] |
| LargePenalty-alpha1000 | 5/5 | 1.5249 [1.5249, 1.5249] | 27 [22, 35] |

## 全Seed

| 手法 | Seed | 最終best | 到達score |
|---|---:|---:|---:|
| Exact-FM | 42 | 1.5249 | 7 |
| Exact-FM | 101 | 1.5249 | 44 |
| Exact-FM | 2024 | 1.5249 | 39 |
| Exact-FM | 7 | 1.5249 | 27 |
| Exact-FM | 19 | 1.5249 | 39 |
| Random-unseen | 42 | 1.5249 | 50 |
| Random-unseen | 101 | 1.5456 | 81 |
| Random-unseen | 2024 | 1.5456 | 81 |
| Random-unseen | 7 | 1.5849 | 81 |
| Random-unseen | 19 | 1.5456 | 81 |
| XY-p3 | 42 | 1.5249 | 21 |
| XY-p3 | 101 | 1.5249 | 31 |
| XY-p3 | 2024 | 1.5249 | 19 |
| XY-p3 | 7 | 1.5249 | 30 |
| XY-p3 | 19 | 1.5249 | 29 |
| LargePenalty-alpha1000 | 42 | 1.5249 | 27 |
| LargePenalty-alpha1000 | 101 | 1.5249 | 22 |
| LargePenalty-alpha1000 | 2024 | 1.5249 | 10 |
| LargePenalty-alpha1000 | 7 | 1.5249 | 38 |
| LargePenalty-alpha1000 | 19 | 1.5249 | 35 |

## 対応Wilcoxon検定（Bonferroni6）

| Exact-FMの対照 | 指標 | 補正p |
|---|---|---:|
| Random-unseen | final_regret | 0.75 |
| Random-unseen | firsthit_score81 | 0.375 |
| XY-p3 | final_regret | 1 |
| XY-p3 | firsthit_score81 | 1 |
| LargePenalty-alpha1000 | final_regret | 1 |
| LargePenalty-alpha1000 | firsthit_score81 | 1 |

81は未到達コードで、実評価数は全Run80。初期3件込み。最適値取得は全192真値の事後照合で判定。全列挙はサンプリング100出力と計算量を揃えた手法ではなく、候補選択の診断用対照。RandomではFMを学習・保存するが選択には使わない。純ランダム探索に必要な処理時間はその学習を除く。失敗Runは削除せず計算失敗で後処理を停止する。補正非有意は同等性の証明ではない。
