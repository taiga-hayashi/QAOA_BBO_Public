# 図の見方

[全体PDF](../pdf/initial_count_overview.pdf)は全体と単独6パネルをPDF/SVG/PNGで保存。点/曲線は中央値、帯/エラーバーはIQR。青Adaptive、橙LargePenalty、緑XY。bestの点線/破線/実線は初期5/10/20。

- best_evaluations：初期込みの実BB評価数とbest-so-far。初期nの完了時点から表示し、条件ごとに全5Runが到達した評価数までで止める。初期データ収集中の最適化を仮定しない。
- first_hit：共通4Seedの最適値到達評価数。初期に最適解を含む20件Seed101を比較に混ぜない。81は未到達の失敗コード。
- final_best：全5Seedの最終best。中央値だけで全Seed成功とは解釈せず、RESULTS.mdの成否を確認する。
- candidate_fraction：候補なしサイクルの割合。総サイクルが75/70/60と違うので、単純な回数ではなく割合で比べる。
- FM_gap：採用候補のFM予測−当時の未評価実行可能FM最小。Run内採用サイクル中央値を5Seedで要約する。真値の差ではない。
- initial_test_RMSE：3手法共通の最初のFMを、元20件の全てを除く同じ172候補で事後評価したRMSE。初期のカテゴリ網羅数はRESULTS.mdに併記。

総BB上限80を固定し、初期件数を減らした分の探索サイクルを増やしている。等サイクル・等計算費用の比較ではない。タイトルと下部注記は付けず、軸と凡例・本説明で条件を示す。
