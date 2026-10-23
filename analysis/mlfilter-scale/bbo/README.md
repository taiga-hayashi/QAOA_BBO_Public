# analysis/mlfilter-scale/bbo: FMQA 閉ループの確認実験

L=7・予算 100・新しい 15 Seed(5001〜5015)で、XY-FMQAOA(ペナルティ不要)と、ペナルティ係数の調整に失敗した FMQA(SA α=0.3、α=100)、調整済み(α=1)、ランダムを比べる。

- 入力: `data/runs/mlfilter-scale/c-e0b866ab4433/r-91f13e039c1e`(90 run、SHA-256 は [analysis_config.json](analysis_config.json))
- 条件・仮説: [experiments/mlfilter-scale/bbo_confirm.json](../../../experiments/mlfilter-scale/bbo_confirm.json)(実行前に固定)
- 入口: `python plot.py`、`python plot.py --replot`、`plot.ipynb`(読み方つき)。図の設定は [src/aps_style.py](../../../src/aps_style.py)

| ファイル | 内容 |
|---|---|
| `bbo_l7_confirm` | (a) 最良解の真の順位の推移(中央値・IQR)、(b) 最適解を見つけた Seed の割合 |
| `panels/` | 各パネルの独立図 |
| `csv/best_rank_curves.csv` | Seed・評価回数ごとの最良順位 |
| `csv/best_rank_summary.csv` | 図の元の値(条件・評価回数ごとの中央値・IQR・到達割合) |
| `csv/final_per_seed.csv` | Seed ごとの最終値と実際の評価回数 |

注意: α=0.3 は候補が出ないサイクルが多く 50〜65 評価しか使っていない。その後は最良値を延ばして描いた。

## 状態

- 2026-10-22: 実データから作図。Notebook は `.venv` の Python で先頭から実行(エラーなし)。PNG 目視確認済み、PDF 未確認
