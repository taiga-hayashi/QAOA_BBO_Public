# 図の見方

initial3_p3_BBO_overviewは6パネル。5Seed、初期3件、最大77cycle、BB上限80、100outputs。中心は5Seed中央値、帯/誤差棒はQ1〜Q3。

- best_evaluations: 初期3を含む実BB評価回数とbest真値（eV）。手法内で5Runがすべて観測した最大共通回数まで描き、未実行域へ外挿しない。
- firsthit_score81: 最適値1.5249eVに初めて到達した実評価数。81は未到達コードで、実評価数ではない。
- final_regret: 全192真値の最小/最大で正規化した最終best。
- median_FM_gap: 各Runの採用cycleの未評価FM最小値からのgapの中央値、さらに5Run間中央値/IQR。真の目的値誤差ではない。
- raw_feasible_fraction: 各Run内全cycleのraw制約充足率平均、5Run間中央値/IQR。候補選択後の制約率とは別。
- generation_seconds: Run全cycleの候補生成時間合計。XYには128角度探索・最終100shots・選択検査、SAには100readsと選択検査を含む。等しい計算予算ではない。
- best_cycles: 初期cycle0から77までのbest真値。候補なしにより同じcycleでも実評価数が違う場合がある。

XYp1は緑菱形、p3は青緑点線の深さ拡張variant、SAalpha1000は橙四角。主4手法のLargePenalty既定alpha100ではない。図内タイトル・下部注記を置かず、本書とRESULTSで条件を説明する。
