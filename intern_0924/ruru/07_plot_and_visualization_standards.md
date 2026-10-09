# 07. 必須プロットと可視化仕様

## 1. 原文指定プロットと論文図面対応

| プロット番号 | 仕様書原文での指定内容 | 論文採択図面番号 / ファイル名 |
| :---: | :--- | :--- |
| **Fig. 1** | $\alpha$ 対 最終 Normalized Regret（充足率・受容率を併記） | `fig09_penalty_sensitivity` |
| **Fig. 2** | $N$ 対 最終 Normalized Regret（目標到達回数を併記） | `fig07_qubit_scaling` |
| **Fig. 3** | $N$ 対 Raw Feasible Rate（未重複実行可能候補率、Top-5% 候補割合と Normalized Regret の品質パレート面） | `fig08_feasibility_and_pareto` |
| **Fig. 4** | FMQA 反復 対 $\alpha_t$（$\alpha_t$ 対 Feasible Rate および Best-so-far） | `fig10_adaptive_dynamics` |
| **Fig. 9** | 主要サイズでの Boxplot とペア差分図 | `fig11_boxplot_paired_differences` |
| **Main 1** | $N=27$ 目的関数値収束軌跡（100サイクル） | `fig01_trajectory_n27_convergence` |
| **Main 2** | 累積制約違反数と経済的実験損失推移 | `fig02_cumulative_wasted_evaluations` |
| **Main 3** | サンプル効率データプロファイル (ECDF) | `fig03_sample_efficiency_ecdf` |
| **Main 4** | 量子測定 Shot 数感度とスケーラビリティ | `fig04_quantum_shot_efficiency` |
| **Main 5** | サロゲートモデル予測精度の進化推移 | `fig05_surrogate_accuracy_progression` |
| **Main 6** | 変分パラメータ空間エネルギーランドスケープ | `fig06_xy_qaoa_energy_landscape` |

---

## 2. 統一スタイル規定

全図面で以下のスタイル（色、線種、マーカー）を完全に統一すること：

| 手法名 | 略称 | カラーHEX | マーカー | 線種 |
| :--- | :---: | :---: | :---: | :---: |
| **`Adaptive-FMQA`** | 手法1 | `#1f77b4` (Blue) | `o` (丸, $\bullet$) | 実線 (`-`) |
| **`LargePenalty-FMQA`** | 手法2 | `#ff7f0e` (Orange) | `s` (四角, $\blacksquare$) | 破線 (`--`) |
| **`Penalty-FMQAOA`** | 手法3 | `#d62728` (Red) | `^` (三角, $\blacktriangle$) | 一点鎖線 (`-.`) |
| **`XY-FMQAOA`** | 手法4 | `#2ca02c` (Green) | `D` (ひし形, $\blacklozenge$) | 実線 (`-`) |

---

## 3. フォーマットとキャプション要件

- **出力フォーマット**:
  - 本文・論文用: **PDF**（ベクター形式）および **SVG**（Web・ベクター形式）
  - 確認・プレビュー用: **PNG**（解像度 300 DPI）
- **キャプション要件**:
  - 試行数（Seed数、評価回数）の明記
  - 誤差棒の定義（中央値と IQR、または 95% Bootstrap CI）の明記
  - 縦軸・横軸の物理量・単位・対数/線形スケールの明記
  - 図単体で理解可能な自己完結した説明文の付記
