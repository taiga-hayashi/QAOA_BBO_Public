# プロットの見方

single_FM_sampling_overviewは6パネル。すべて同じ1モデル、各alpha100reads×10反復を使う。各alphaの総raw数1000、全体7000。単独PDF/SVG/PNGも保存。

- feasibility: 全raw中の制約充足率と未評価実行可能率。分母1000。既評価10件は未評価率に数えない。
- selected_gap: 橙は100readsの最良未評価候補のgap、10反復中央値/IQR。紫はraw未評価サンプルの平均gap。低いほどよい。gapは同一FMの未評価182候補の厳密最小値からの差。
- minimum_capture: 100readsで厳密FM最小値を1件以上取得した反復割合。分母10。有限反復の経験頻度で、厳密な成功確率ではない。
- gap_CDF: raw未評価実行可能サンプルのgapの経験累積分布。横軸の同じgapで曲線が高いほど、良い値へ集中。違反・既評価は分布から除外し、feasibilityで別報告。
- FM_rank_frequency: 未評価182候補を同じFM予測で並べ、候補ごとの経験出現頻度を描画。順位1が最小。頻度の分母は各alphaのraw未評価実行可能件数。0頻度も含む。
- diversity: 1000rawに含まれた未評価候補のユニーク数と、未評価経験分布のexp(Shannon entropy)。候補種類の多さだけでは目的値品質を評価できない。

alpha横軸は対数、その他は線形。CDF/順位図は各alpha色、感度曲線の橙は通常SA、紫は補助指標。4手法の比較図ではない。タイトル・下部注記を置かず、条件と分母は本書とRESULTSに記載。
