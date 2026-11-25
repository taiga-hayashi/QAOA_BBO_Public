# XY-FMQAOA for Multilayer Optical Filter Design — 再現コード・データ一式

発表スライドに掲載している実験結果およびプロットを再現するための自己完結パッケージです。

## フォルダ構成

```text
deliverable/
├── README.md              # 本ファイル
├── requirements.txt       # 依存パッケージ一覧
├── config.json            # 全実験の条件・Seed・予算・ハイパーパラメータ
├── main.py                # 実験実行エントリポイント（固定FM・閉ループBBO・reads掃引）
├── fm.py                  # Factorization Machine（学習・QUBO変換・Isingスケール S・候補選択）
├── xy_qaoa.py             # XY-QAOA（OpenQARP / NumPy 両バックエンド・CVaR・INTERP/Powell）
├── sa.py                  # 比較手法（ペナルティ付きSA・Adaptive・温度再調整・ランダム探索）
├── optics.py              # 多層膜フィルタの伝達行列法（TMM）と FOM 計算
├── plot_slides.ipynb      # results/csv/ から全図を描画する Notebook（実行済み出力付き）
├── inputs/                # 光学定数・太陽スペクトル・manifest.json
└── results/
    ├── csv/               # 実験内容ごとの集計済み CSV
    └── figures/           # plot_slides.ipynb が出力する PNG / PDF
```

## 環境構築

標準の `python3 -m venv` で構築できます（`openqarp` 未導入の環境でも、同一ユニタリの純粋 NumPy バックエンドに自動フォールバックします）。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# OpenQARP を利用する場合（任意）:
# pip install openqarp
```

## 1. 図の再現（CSV → 図、数秒）

重い実験を再実行せずに、`results/csv/` から全プロットを再描画できます。

- `plot_slides.ipynb` を Jupyter / VS Code で開き、先頭から全セルを実行してください（実行済み出力も保存済みです）。
- 生成された PNG / PDF は `results/figures/png/` および `results/figures/pdf/` に出力されます。

### 実験内容と CSV・出力図の対応表

| 実験カテゴリ | 内容 | 元 CSV (`results/csv/`) | 出力図 (`results/figures/`) |
|---|---|---|---|
| 固定 FM 評価 | ペナルティ係数 $\alpha$ と SA の出力品質（$L=6\text{--}8$） | `fixedfm_sa_alpha_sweep.csv` | `fixedfm_sa_alpha_sweep` |
| 固定 FM 評価 | XY-QAOA の深さ $p$ と正規化 FM ギャップ（$L=6\text{--}8$） | `fixedfm_xy_depth_sweep.csv` | `fixedfm_xy_depth_sweep` |
| 固定 FM 評価 | 実行可能サンプルの FM 百分位分布（$L=7$） | `fixedfm_l7_percentile_hist.csv` | `fixedfm_l7_percentile_hist` |
| 固定 FM 評価 | SA `reads` 掃引（$L=7$, 100〜10,000 reads） | `fixedfm_l7_sa_reads_sweep.csv` | `fixedfm_l7_sa_reads_sweep` |
| 閉ループ BBO | $\alpha$ 掃引による最良順位と最適解到達数（$L=7$, 15 seeds） | `closedloop_l7_alpha_sweep.csv` | `closedloop_l7_alpha_sweep` |
| 閉ループ BBO | 3層数統一比較（$L=6, 7, 8$, 各 30 seeds） | `closedloop_l678_fom_curves.csv` | `closedloop_l678_fom_curves` |
| 閉ループ BBO | 第1確認実験の Seed ごとの順位対応（$L=7$, 15 seeds） | `closedloop_l7_paired_ranks.csv` | `closedloop_l7_paired_ranks` |
| 閉ループ BBO | 探索段階の推移（$L=6, 7, 8$, 5 seeds） | `closedloop_l678_explore_5seeds.csv` | `closedloop_l678_explore_5seeds` |
| 頑健性検証 | SA 温度スケジュール再設定・10倍 sweeps（固定 FM & 閉ループ） | `fixedfm_sa_schedule_retune.csv`, `closedloop_l7_schedule_retune.csv` | `sa_schedule_retune_robustness` |
| 量子回路可視化 | OpenQARP による XY-QAOA 回路図（`openqarp` 導入環境のみ） | （回路構築） | `openqarp_xy_qaoa_macro`, `openqarp_xy_mixer_decomp` |

## 2. 実験の実行（`main.py`）

### 軽量スモークテスト（約 30〜60 秒）

```bash
python main.py --suite all --smoke --workers 4 --backend auto
```

### 本番スイートごとの実行（中断再開対応）

完了した `(condition, seed)` の JSON は自動スキップされます。`--backend auto` は `openqarp`（`qarpx`）が利用可能なら OpenQARP、無ければ厳密一致する NumPy シミュレータを使用します。

```bash
python main.py --suite truth
python main.py --suite fixed_fm --workers 8 --backend auto
python main.py --suite reads_sweep
python main.py --suite confirm1 --workers 8 --backend auto
python main.py --suite l6_30seeds --workers 8 --backend auto
python main.py --suite l7_30seeds --workers 8 --backend auto
python main.py --suite l8_30seeds --workers 8 --backend auto
python main.py --suite explore_L678 --workers 8 --backend auto
```

## OpenQARP の利用箇所

- `xy_qaoa.py`: 4材料/層を 2量子ビットにコンパクト符号化し、`SimpleBlock.diagonal_unitary`（コスト層）、`SimpleBlock.unitary_synthesis`（XY リングミキサー層）、`QarpSimulator().statevector()` を用いて厳密状態ベクトルを計算しています。
- `plot_slides.ipynb` 末尾セル: `qarp.plotting.plot()` を用いて XY-QAOA 全体回路と XY ミキサーのゲート分解図を再現します。
