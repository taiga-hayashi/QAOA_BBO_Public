# プロットの見方

[全体PDF](../pdf/sample_count_overview.pdf)と6単独パネルをPDF/SVG/PNGで保存。中央値、帯/エラーバーはIQR。青丸Adaptive、橙四角LargePenalty、緑ひし形XY。10出力は点線、100出力は手法の標準線種。タイトル・下部注記は付けず、軸と凡例で区別した。

- best_evaluations：実BB評価数（初期20込み）とbest-so-far。条件ごとに全5Runが到達した最大評価数まで表示し、未実行領域へ外挿しない。
- best_cycles：候補なしも含む60サイクルでのbest-so-far。最適値1.5249 eVは灰線。
- first_hit：初期に最適解を含まない4 Seedの到達評価数。81は未到達の失敗コード。未到達を除外した中央値ではない。
- candidate_shortfall：各Runの候補なしサイクル数（60中）。低いほど機会を得ているが、真値が良いことを保証しない。
- FM_gap：採用候補予測と現在の未評価実行可能FM最小との差。Run内採用サイクルの中央値を5 Seedで集約。低いほどFM最小をよく取り出しているが、FMの予測自体が正しいことを保証しない。

- fixed_FM_XY：10出力RunでFM・角度・訓練集合を固定した300ペア。100乱数の先頭10と全100を比較し、Seed別gap中央値を5Seedで要約する。再学習100Runの曲線とは別。

初回到達以外は5 Seed。固定FM XYの補助診断はRESULTS.mdとjson/fixed_FM_XY_nested.jsonで別途確認する。再学習する100出力Runと同じ結果として扱わない。
