# slide_ja.pptx のプロットと元データ

更新: 2026-11-12(2 回目)。`docs/slide/slide_ja.pptx`(24 枚、`docs/slide/build/build4.js` で生成)に貼られたプロットと、その元データ。

## スライド用に描き直した図(本編)

`plot_slide.py` が `csv/` だけを読んで `png/`・`pdf/` の `slide_*` を出力する(`python3 plot_slide.py`)。スライドに貼る実寸で描き、日本語ラベル、文字 13〜14 pt、ヒラギノ角ゴ。手法の色は全図で固定(XY-FMQAOA 青、SA α=1 黒破線、SA α=100 橙、SA α=0.3 緑一点鎖線、カテゴリ型 SA 紫点線、ランダム提案 灰破線)。

| スライド | 図 | 元データ |
|---|---|---|
| 16 | `slide_sweetspot` 罰則係数 α と FM 最良解の発見率・FM 百分位・制約を満たす割合(固定 FM、L=6–8) | `solver_summary.csv` |
| 17 | `slide_depth` XY-QAOA の深さと FM ギャップ(固定 FM、L=6–8、SA の各 α を横線) | `solver_summary.csv` |
| 18 | `slide_loop_alpha` 閉ループの α 掃引と最適解到達数(L=7、15 Seed) | `loop_alpha_sweep.csv` |
| 19 | `slide_confirm2` 第 2 確認実験の推移(L=7、30 Seed) | `confirm2_curves.csv`(到達数の表示は `confirm2_final.csv` と同じ) |

## 論文用の図をそのまま使った補足スライド

| スライド | 図 | 元データ | 元 |
|---|---|---|---|
| 21 | `s18_energy_hist` FM 百分位の分布 | `energy_hist.csv`(表は `solver_summary.csv`) | `paper/` |
| 22 | `s25_retune_robustness` 温度の再設定 | `fixedfm_schedules.csv`、`loop_alpha_sweep.csv` | `paper/` |
| 23 | `s26_paired_ranks` 第 1 確認実験の Seed ごとの対応 | `confirm_final_ranks.csv` | `paper/` |
| 24 | `s27_si_closed_loop_L678` 探索段階の閉ループ | `explore_curves.csv` | `paper/` |

ファイル名の `sNN` は 2026-11-12 の改訂前のスライド番号。旧版で使っていた `s16_penalty_sweetspot`、`s17_xy_depth_L8`、`s19_loop_alpha_sweep`、`s20_confirm2`、`s22_loop_mechanism` は記録として残しているが、今のスライドでは使っていない。

## 作図用 CSV と plot.py / plot.ipynb(2026-11-12 追加)

`plot.py` と `plot.ipynb` は同じコード(Notebook は `plot.py` の `# %%` セルから生成)。8 図すべてを `csv/` から作る。

1. 元データ(下の表の `csv/*.csv`)を集計し、図に描く値そのものを `csv/slideNN_*.csv` に書き出す
2. `csv/slideNN_*.csv` だけを読んで `png/`・`pdf/` に `slide_*` を出力する

| スライド | 作図用 CSV | 図 |
|---|---|---|
| 16 | `slide16_sweetspot.csv`(L・α ごとの発見率・FM 百分位・制約を満たす割合、5 FM 中央値) | `slide_sweetspot` |
| 17 | `slide17_depth.csv`(XY の深さ p ごとの FM ギャップ中央値・四分位、SA・ランダムの参照値) | `slide_depth` |
| 18 | `slide18_loop_alpha.csv`(条件ごとの最良順位の中央値・四分位・最適解到達数) | `slide_loop_alpha` |
| 19 | `slide19_confirm2.csv`(条件・評価回数ごとの最良順位中央値と到達割合) | `slide_confirm2` |
| 21 | `slide21_energy_hist.csv`(L=7、FM 百分位のビンごとの合算度数と密度) | `slide_energy_hist` |
| 22 | `slide22_retune_fixedfm.csv`(a–c)、`slide22_retune_loop.csv`(d、Seed ごとの順位) | `slide_retune_robustness` |
| 23 | `slide23_paired_ranks.csv`(Seed ごとの比較手法と XY の順位、勝敗) | `slide_paired_ranks` |
| 24 | `slide24_explore.csv`(L・条件・評価回数ごとの最良順位中央値) | `slide_explore_L678` |

- 実行:`/opt/homebrew/bin/python3 plot.py`、または `plot.ipynb` を先頭から実行(2026-11-12 に Homebrew Python で全セル実行、エラーなし、図 8 枚を出力)
- 検証:スライド 16〜19 の図は、デッキ内の画像とピクセル単位で一致。21〜24 は描画内容は同じで、外周の余白が数ピクセル違う(デッキの画像は `paper/plot.py` を `fas` 仮想環境(matplotlib 3.11)で描いたもの、16〜19 は Homebrew Python(matplotlib 3.10)で描いたもの。同じ環境では両方を同時に再現できない)。`fas` で実行すると 21〜24 が一致し、16〜19 の文字配置が少しずれる
- 到達数(18:α=1 で 9/15、XY 13/15、自動調整 8/15。19:23・18・16・4・2/30)は Notebook 末尾のセルで確認し、`confirm2_final.csv` とも一致
- `plot_slide.py` は 16〜19 だけを描く旧版(出力は同じ)。`docs/slide/build/build4.js` は 21〜24 を今も `paper/png/` から読む
- スライド 21 の表(正規化 FM ギャップ)は図ではないため対象外

## 2026-11-12 デザイン統一

- 8 図とも同じ枠(幅 7.4 × 高さ 5.0 インチ = スライド上の実寸)、日本語ラベル、文字 11〜14 pt、ヒラギノ角ゴで描くよう `plot.py` を変更した。スライド 21〜24 も論文用の英語の図ではなくなった。
- 手法の色・線種は全図で共通(XY-FMQAOA 青、SA α=1 黒破線、α=100 橙、α=0.3 緑一点鎖線、カテゴリ型 SA 紫点線、ランダム提案 灰破線)。
- スライド 19(第 2 確認実験)は横 2 枚から上下 2 段に、22 は横 4 枚から 2×2 に変えた。
- スライドでは 16〜19・21〜24 を同じ配置(左に図、右に結果と ⇒ 結論)で置く(`docs/slide/build/build4.js` の `resultSlide`)。21〜24 も `paper/png/` ではなく、このフォルダの `png/slide_*` を読む。
- 上の「検証」の行(デッキの画像とピクセル一致)は統一前の版についての記録。統一後はデッキの画像 = このフォルダの `png/slide_*`。

## 注意

- 元の解析(`solver/`、`paper/`)を描き直しても、`csv/` とこのフォルダの図には反映されない
- `confirm2_final.csv` には図に出していない温度付き分布の行も含まれる
- `csv/` は元の解析からのコピー。SHA-256 は `SHA256SUMS.txt`(`slide_*` の図は含まない)
- Drive への保存検証は未実施
