# 図の条件・指標

penalty_sensitivity_overviewは6パネル、各指標は独立した個別図としても保存。加えてeffective_diversity、duplicates、generation_timeを保存。形式はPNG（300dpi）、PDF、SVG。タイトル・下部注釈は付けない。

横軸は正規化係数α。0を含めるためsymlog（線形領域0〜0.01、以降対数）を使用。各モデルで10回のサンプリング反復を平均した値を薄い線で示し、5モデルの中央値を青丸、IQRを帯で示す。Seedは42、101、2024、7、19。FMは初期20件、rank2、120epochs、Adam lr0.1で学習した既存固定モデル。各反復1000 reads、1000 sweeps、4バッチのSA。内部ペナルティλ=Sα、Sはbase FM Ising h,Jの最大絶対値。

- raw_feasibility：未加工1000サンプル中のOne-Hot実行可能割合。
- five_candidate_success：1000サンプルから未評価実行可能解を5種類以上得た反復の割合。10反復による頻度。
- unique_candidates：初期20件を除く実行可能候補の種類数。
- surrogate_quality：新規実行可能サンプルに条件付けた平均FM予測値と、全実行可能192候補のFM最小値との差（eV）。重複サンプルも頻度として重み付け。該当サンプル0件なら未定義。定義件数はRESULTS.mdとsummary.jsonで示す。
- true_top5_success：未評価の真値Top5%（全192候補中10件）を1件以上生成した反復の割合。真値は事後診断にのみ使用。
- true_regret：初期20件とFM順位で採用した最大5件の最良真値を、全表の最小1.5249、最大6.3242で正規化。候補不足を保持。
- effective_diversity：新規実行可能サンプル内のShannon entropyの指数。該当0件なら未定義。
- duplicates：新規実行可能解のうち2回目以降の出現数 / 全1000サンプル。
- generation_time：SA候補生成の壁時計秒、縦軸対数。FM学習・真値診断を除く。

Nealの自動beta rangeはペナルティ付きBQMに応じて変化する。温度を固定したペナルティのみの因果比較ではない。閉ループBBOの収束図ではない。
