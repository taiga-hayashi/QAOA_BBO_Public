# 図の見方

[全体PDF](../pdf/fmqa_bbo_overview.pdf)には4パネルがある。単独図もPDF/SVG/PNGで保存した。曲線は各条件5 Seedの中央値、帯はIQR。青丸＝Adaptive-FMQA、橙四角＝LargePenalty-FMQA。提案設定は手法の標準線種、従来設定は点線。灰色の水平線は全192候補の最適値1.5249 eV。

- best_evaluations：横軸は実際のBB評価数（初期20込み）。各条件5 Seedの全Runが到達した最大評価数まで描き、未実行評価へ補外しない。Runごとの最終実評価数はRESULTS.mdに記載。
- best_cycles：横軸は60サイクル。候補がないサイクルも含み、各時点のbest-so-farを比較する。
- evaluations_cycles：候補不足によって評価数が増えないことを示す。80は上限で、全Runが80件評価したわけではない。
- feasibility_cycles：各サイクル10出力のOne-Hot実行可能割合。経験的SA充足率であり、QAOAの厳密確率ではない。

中央値が最適値へ到達しても全Seed到達を保証しない。全Runの成否と初期最適解の有無をRESULTS.mdで確認する。統計はjson/summary.json、生の全候補・選択理由・λ/α/S・Seed・予測・時間は各Runのjsonに保存。
