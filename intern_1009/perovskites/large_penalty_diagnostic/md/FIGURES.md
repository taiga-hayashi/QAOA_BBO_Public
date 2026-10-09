# 図の見方

全体図large_penalty_overviewは6パネル。各単独図は同名のPDF/SVG/PNG。横軸alpha=lambda/Sは対数。中央線は5Seed中央値、帯はQ1〜Q3。fixedパネルはモデルごと10反復平均の5モデル集計で、反復を50独立モデルとは数えない。

- fixed_feasibility: サンプル100件中の実行可能率。高くても目的値が良いとは限らない。
- fixed_FM_gap: 選択候補のFM値−未評価実行可能候補の厳密FM最小値（eV）。小さい方が良い。FMを固定しているのでサンプラーの選択品質を比較できる。
- fixed_candidate_rate: 最大1件の未評価実行可能候補を獲得した反復割合。候補なし時のgapは未定義。
- BBO_final_regret: 全192真値の最小/最大で正規化した最終best。0が最適値。
- BBO_first_hit: 初期10を含む最適値初回到達評価数。81は未到達コードで、実評価数ではない。
- BBO_candidate_none: 70cycles中の候補なし率。
- BBO_best_by_cycle: 最良真値のサイクル推移。初期集合が同じ10件なので0cycleのbestは共通。同じサイクルでも候補なしにより実評価数は異なる。

橙は通常の自動beta、紫はalpha1で自動算出したbeta範囲を全alphaで固定した補助実験。BBOは橙の日程のみ。自動日程は初期温度を調整するので純粋なペナルティ項だけの効果ではない。fixedbetaは標準FMQA条件を置き換える推奨ではなく、温度調整の影響を検討する補助条件。

タイトルや下部注記を図内に置かず、この説明とRESULTSで条件を示す。BBO軌跡はalpha別色で、4手法比較の色定義とは区別する。alpha100のみLargePenalty-FMQA。
