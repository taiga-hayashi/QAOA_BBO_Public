# 10／100サンプル・ショットの閉ループ比較

同じrank1 AdamW wd0.01・初期20・60サイクル・最大1件採用・BB上限80。Adaptive SA / LargePenalty SA / XYの各10/100出力、5Seed、30Runを新規計算した。過去10出力の15Runを全60サイクルで完全再現。

初期最適解なし4Seedの到達評価数中央値はAdaptive31→30、LargePenalty28→32、XY55→39。全条件で新規4/4が最適値1.5249 eVに到達した。100出力はFM gapを減らしたが、真の目的の到達が必ず速くなるとは言えず、補正検定の有意差は未確認。

- [事前プロトコル](json/protocol.json)／[監査](md/AUDIT.md)
- [結果・IQR・Seed別・検定・固定FM補助診断](md/RESULTS.md)
- [全体PDF](pdf/sample_count_overview.pdf)／[図の見方](md/FIGURES.md)
- [解釈と次の案](md/DISCUSSION.md)／[再現手順](md/REPRODUCIBILITY.md)

コードはpy/、生Run・集計・検証はjson/、checkpoint/logはdata/、図はpdf/svg/png/。XYはλ0・W初期化・Ring・p1・9理想角度点を保持。元23ビット回路と照合済み共有OpenQARP厳密符号化シミュレーションを使用（元問題23、simulator8）。標準X-QAOAは今回含めない。
