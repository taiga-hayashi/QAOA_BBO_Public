# COMBO Centroid：問題定義の確認

確認日: 2026-10-09。原実装の静的照合。コードは実行していない。

## 原問題

格子上の3つのIsing分布P_kを用意し、各辺eの結合をいずれか1つの元モデルから選ぶ。入力x_e∈{0,1,2}、混合結合J_e(x)=J_{x_e,e}。分布は場なしで

$$P_J(s)=Z_J^{-1}\exp(2\sum_eJ_es_us_v),\quad s_u\in\{-1,1\}.$$

目的は

$$f(x)=\frac13\sum_{k=0}^2D_{KL}(P_k\Vert P_{J(x)}).$$

KLの向きは元モデル→混合モデル。[binary_categorical.py](../data/reference/binary_categorical.py.txt) の式は2Σ(J_original−J_mixed)Cov_original + logZ_mixed−logZ_original。相関と分配関数は全スピン列の列挙で計算する。

元結合は符号±と大きさuniform[0.05,5)を乱数生成する。モデル生成Seedと結合値を保存しなければProblem IDが定まらない。解析できる分布関数をBBとして扱う比較であり、材料シミュレータ問題とは区別する。

## 符号化と縮小案

h×w格子の辺数E=h(w−1)+w(h−1)、N=3E。原4×4はE=24、72論理量子ビット。2×2はE=4、12ビット、2×3はE=7、21ビットで、ともに新しい縮小設定案。辺の並べ方・水平垂直結合の配置を固定する。各辺の3択以外に追加制約はない。

## 公開コードの不整合

[multiple_categorical.py](../data/reference/multiple_categorical.py.txt) 62–64行のising_dense呼出しはpartition_sparsified、partition_original、grid_hをキーワード引数にする。一方[import先の保存版](../data/reference/binary_categorical.py.txt) 63行の定義はlog_partition_original、log_partition_sparsifiedを受け取り、grid_hはない。取得commitでインターフェース不一致がある。これは静的指摘で、実行時例外を測定した報告ではない。

補助関数はグローバルISING_GRID_Hも参照する。縮小・長方形格子の移植では行列幅と頂点番号対応を監査する。KLと因子2を保った独立評価器へ直し、元4×4の意味と小格子での全列挙を照合する。

## 根拠と未確定事項

[設定ファイル](../data/reference/experiment_configuration.py.txt)、[commit・SHA256](../json/source_snapshot.json)、[BSD系ライセンス保存版](../data/reference/LICENSE.txt)。格子・辺順・モデルSeed・結合・移植の検証は未確定。最適解・時間・評価値は未測定。

## 変数数による候補判定（ユーザー指定）

One-Hot変数数が100を超える設定は不採用。100以下は候補に残すが、採用・実行可能性は未確定。既定の27量子ビット上限は別途確認する。

| 設定 | One-Hot変数数 | 判定 |
|---|---:|---|
| `grid_4x4` | 72 | 候補継続 |
| `grid_2x2` | 12 | 候補継続 |
| `grid_2x3` | 21 | 候補継続 |
