# analysis/mlfilter-scale/paper: 論文用の図

- 入力: 確認実験(c-e0b866ab4433/r-91f13e039c1e)、探索の閉ループ(bbo-xy、L=6・7・8)、追加解析(paper-extra、c-9ddf1efaebc1/r-f2433d20d4d3)。読んだ全ファイルの SHA-256 は `csv/input_manifest.json`
- 入口: `python plot.py`、`python plot.py --replot`、`plot.ipynb`(各図の読み方つき)。図の設定は `src/aps_style.py`

| 図 | 内容 |
|---|---|
| `concept` | 概念図: ペナルティの大小によるエネルギー地形と XY ミキサー |
| `paired_ranks` | 確認実験の Seed ごとの対応(XY vs α=0.3・100・1) |
| `loop_alpha_sweep` | 閉ループの α 掃引(U 字)と XY |
| `retune_robustness` | 温度を調整し直した SA(固定 FM と閉ループ) |
| `energy_hist` | サンプルの FM 分布(L=7) |
| `loop_mechanism` | 閉ループのサイクルごとの選択の質・多様性 |
| `si_closed_loop_L678` | 補足: L=6・7・8 の閉ループ(探索、5 Seed) |
| `csv/compute_cost.csv` | 1 サイクルあたりの計算時間 |

状態: 2026-10-22 実データから作図。Notebook は `.venv` の Python で先頭から実行(エラーなし)。PNG 目視確認済み、PDF 未確認。
