# 図の見方

XY_depth_overviewは6パネル、単独PDF/SVG/PNGも保存。1つの固定FMに対して各p128角度候補×10探索Seed、最終100shots×10測定/探索Seed。全曲線の中央値/IQRは探索Seed間の集計で、FMモデル間ではない。

- expected_gap: 選択角度の理想期待FM値−全実行可能FM最小値（eV）。角度選択に使った指標。低い方が良い。
- ideal_100shot_capture: 未評価FM最小候補の理想確率Pminから計算した1−(1−Pmin)^100。実測測定頻度とは区別する。
- median_selected_gap: 100shotsから選ぶ最良未評価候補のgap。探索Seed内10測定の中央値を取り、さらに10探索Seed間中央値/IQRを示す。
- empirical_capture_fraction: 各探索Seedでの最小値捕捉回数/10測定。その10探索Seed間中央値/IQR。全測定の捕捉回数/100はRESULTSに別記。
- gap_CDF: 既評価10点を除く未評価候補に条件付けた理想確率CDF。10探索SeedのCDFの中央値/IQR。サンプルの経験CDFではなく、選択した回路の理想確率から算出。
- angle_search: 評価済み角度候補の最良期待gapの推移。p2/3の第1候補は前深さbestをゼロ層で埋め込んだwarm start。初期点が異なる理由を明示し、独立した同一初期角度の学習曲線とは呼ばない。

横軸pまたは角度評価件数、CDF横軸はeV。p別線色で区別、深さ感度図はXYの緑菱形。誤差は中央値とQ1〜Q3、タイトル・下部注記なし。全p lambda0、同じW/Ring。角度評価件数は同じだがゲート・層評価数はpに比例し異なる。
