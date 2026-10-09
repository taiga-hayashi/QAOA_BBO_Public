# 図の読み方

fixed_fm_overviewは3パネル、raw_feasibility・true_regret・generation_timeは独立に描画した個別図。PDF/SVG/PNGを保存。タイトル・下部注釈は付けず、この文書に条件を記載した。

各点は同じ5 Seedの実測値、黒い線は中央値とIQR（Q1〜Q3）。各手法1,000サンプル、初期20件、未評価候補を最大5件採用。無効候補や不足は除外せず、Regretには初期候補の最良値も含める。

Raw feasibleはサンプル中の実行可能割合（線形0〜1）。Regretは監査済みの192候補表の最小・最大で正規化。線形軸で今回の値を読み取れるよう表示範囲を約0〜0.12にした。時間は候補生成の壁時計秒（対数軸）、FM学習・真値診断の時間を含まない。計算資源を同等にした速度比較ではない。

QAOAはOpenQARPの理想23量子ビット・p=1・9点探索・1,000ショット。SAは1,000 reads×1,000 sweeps。固定FMでの診断であり、BBO軌跡や量子優位性の図ではない。全データ・指標・条件はjson/summary.json、json/seed_*.json、json/protocol.jsonにある。
