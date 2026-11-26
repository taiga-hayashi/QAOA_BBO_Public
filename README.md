# XY-FMQAOA for Multilayer Optical Filter Design — 再現コード・データ一式

発表スライドに掲載している実験結果およびプロットを再現するための自己完結パッケージです。  
**研究サーバーで計算・CSV 生成を行い、ローカルでは `plot_slides.ipynb` で可視化**します。  
**各プロット用の CSV ごとに `config.json` の設定も実行も個別に独立**しています。

## フォルダ構成

```text
deliverable/
├── run_main.py          # ★メイン: 30ラン最適化 (closedloop_l678_fom_curves.csv)
├── run_ablation.py      # 閉ループ補助 (4本の CSV を個別または一括実行)
├── run_diag.py          # 固定FM診断 (5本の CSV を個別または一括実行)
├── plot_slides.ipynb    # ローカルで開いて図（PNG/PDF）を描く Notebook
├── config.json          # 全10プロットCSVごとの個別実験設定
├── requirements.txt     # 依存パッケージ一覧
├── src/                 # 共通モジュール（XY-QAOA は OpenQARP 必須）
└── results/
    ├── csv/             # サーバーで生成された集計済み CSV（10本）
    └── figures/         # plot_slides.ipynb が出力する PNG / PDF
```

## 使い方

### 1. 環境構築（OpenQARP 前提）

```bash
python3 -m venv ~/.venvs/xy-fmqaoa
source ~/.venvs/xy-fmqaoa/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. サーバーで計算する（引数なしで全体、または `--target` で個別実行）

どのコマンドも **真値テーブル生成・並列実行（最大8並列）・中断再開スキップ・CSV 出力まで自動**で行います。

```bash
# ★メイン: 30ラン最適化 (L=6, 7, 8)
python run_main.py
python run_main.py --layers 6 7          # 一部の層数だけ回す場合

# 閉ループ補助（全部、またはプロットCSVごとに個別実行）
python run_ablation.py
python run_ablation.py --target closedloop_l7_alpha_sweep

# 固定FM診断（全部、またはプロットCSVごとに個別実行）
python run_diag.py
python run_diag.py --target fixedfm_sa_alpha_sweep
```

- 動作確認だけしたいときは `--smoke` を付けてください（`outputs/smoke/` に出力されるため、`results/csv/` は上書きされません）。
- 個別プロットだけ別設定で試したい場合は、[config.json](config.json) 内の `targets.<CSV名>` を編集するか、`--config my_config.json` で別設定ファイルを渡せます。

### 3. ローカルで図を描画する（`plot_slides.ipynb`）

サーバーから `results/csv/` を取得し、ローカルの Jupyter / VS Code で `plot_slides.ipynb` を開いて先頭から全セルを実行すると `results/figures/{png,pdf}/` に全図が出力されます。

### ターゲット名（`--target`）・CSV・出力図の対応表

| 実行スクリプト | `--target` 名（`config.json` キー） | 出力 CSV (`results/csv/`) | 出力図 (`results/figures/`) |
|---|---|---|---|
| `run_main.py` | `closedloop_l678_fom_curves` | `closedloop_l678_fom_curves.csv` | `closedloop_l678_fom_curves` |
| `run_ablation.py` | `closedloop_l7_alpha_sweep` | `closedloop_l7_alpha_sweep.csv` | `closedloop_l7_alpha_sweep`, `*_mean` |
| `run_ablation.py` | `closedloop_l7_paired_ranks` | `closedloop_l7_paired_ranks.csv` | `closedloop_l7_paired_ranks` |
| `run_ablation.py` | `closedloop_l7_schedule_retune` | `closedloop_l7_schedule_retune.csv` | `sa_schedule_retune_robustness` |
| `run_ablation.py` | `closedloop_l678_explore_5seeds` | `closedloop_l678_explore_5seeds.csv` | `closedloop_l678_explore_5seeds` |
| `run_diag.py` | `fixedfm_sa_alpha_sweep` | `fixedfm_sa_alpha_sweep.csv` | `fixedfm_sa_alpha_sweep`, `*_mean` |
| `run_diag.py` | `fixedfm_xy_depth_sweep` | `fixedfm_xy_depth_sweep.csv` | `fixedfm_xy_depth_sweep`, `*_mean` |
| `run_diag.py` | `fixedfm_l7_percentile_hist` | `fixedfm_l7_percentile_hist.csv` | `fixedfm_l7_percentile_hist` |
| `run_diag.py` | `fixedfm_sa_schedule_retune` | `fixedfm_sa_schedule_retune.csv` | `sa_schedule_retune_robustness` |
| `run_diag.py` | `fixedfm_l7_sa_reads_sweep` | `fixedfm_l7_sa_reads_sweep.csv` | `fixedfm_l7_sa_reads_sweep` |
| `plot_slides.ipynb` 末尾 | （OpenQARP 回路構築） | — | `openqarp_xy_qaoa_macro`, `openqarp_xy_mixer_decomp` |
