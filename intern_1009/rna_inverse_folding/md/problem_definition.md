# RNA逆折りたたみ：問題定義の確認

確認日: 2026-10-09。公開コードを静的確認。ViennaRNAの評価は未実行。

## 入出力と目的

入力は長さnのRNA配列、各位置4択。目標二次構造sを固定し、ViennaRNAのnormalized ensemble defect（NED）を最小化する。単一の最小自由エネルギー構造との不一致率を目的に置き換えない。

[公開コード保存版](../data/reference/fmqa_rna.py.txt) 40–43行の既定目標は `..((((((((.....)).))))))..`（26塩基）。入力文字の対応は必須引数base_allocationで指定され、例AUGCを暗黙の固定順として扱わない。将来の4手法比較では選択した順を同じに固定する。

## 評価手順と設定

65–79行では温度37 °C、dangles=2でfold_compoundを構成し、mfe → exp_params_rescale → pf → ensemble_defect(s)の順に呼び出す。取得値を[0,1]へクリップする。コード全体にはRNA.cvar.pf_scale=0.0の設定と結果キャッシュもある。

原コードは不正な整数デコードやtry内の例外を1.0へ置換する。この処理は評価失敗と正常な目的値1.0を区別できないため、移植時には失敗statusと理由を保存し、4手法共通の予算規則を定義する。RNA.fold_compound等のtry外まで全例外を捕捉しているわけではない。

## 符号化と規模

関連研究・コードはone-hot、domain-wall、binary、unaryを扱う。One-HotではN=4n。論文の12塩基例は48、コード既定26塩基は104論理量子ビットで、いずれも既定27上限外。12塩基例の正確な目標構造は未転記であり、実行設定は未選択。短い人工目標を導入する場合は原問題と別に記録する。

One-Hotにより塩基4択は保存できるが、目標構造の実現を保証するものではない。これは評価すべき目的である。

## 依存と次の作業

README記載はPython3.12、ViennaRNA2.7.2、amplify>=1.4.1、torch>=2。[README保存版](../data/reference/README.md.txt)。目的評価部分を独立させる場合と、Amplify tokenを要する元の全FMQA実行を区別する。依存互換性・再現値は未検証。

[MITライセンス](../data/reference/LICENSE.txt)、[commit・SHA256](../json/source_snapshot.json)。次に目標構造・塩基順・ViennaRNAパラメータ版を固定し、評価器だけを照合する。

## 変数数による候補判定（ユーザー指定）

One-Hot変数数が100を超える設定は不採用。100以下は候補に残すが、採用・実行可能性は未確定。既定の27量子ビット上限は別途確認する。

| 設定 | One-Hot変数数 | 判定 |
|---|---:|---|
| `gc_placement_12nt` | 48 | 候補継続 |
| `stickshift_26nt` | 104 | 不採用（100超） |
