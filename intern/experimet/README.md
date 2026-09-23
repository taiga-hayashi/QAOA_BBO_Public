# Experiment validation inventory

各実験フォルダは `py/`, `json/`, `png/`, `pdf/`, `md/` に分類されている．

QAOA 回路は `py/openqarp_qaoa.py` に集約し、OpenQARP 0.1.0 の
`qarpx` ネイティブ回路・状態ベクトル実行器を用いる。Standard QAOA は
X-mixer、FM-XY-QAOA は One-Hot 初期状態と完全グラフ `RXX/RYY` mixer を
同じ角度規約で実装している。既存の JSON/PDF は移行前の出力なので、
OpenQARP 実行済みを示す `qaoa_backend` メタデータが付くまでは新実装の
結果として扱わない。

## 既存検証

| Folder | Validation |
|---|---|
| `optimization/` | 2つのBB関数に対する直接最適化と30-cycle BBO |
| `penalty_sensitivity/` | FMQAのペナルティ係数感度 |
| `std_qaoa_sensitivity/` | Standard QAOAのペナルティ係数感度 |
| `qubit_scaling/` | 量子ビット数に対する制約充足率・目的関数値・計算量 |
| `xy_qaoa_accuracy_scaling/` | 現行の完全グラフXY mixerで、BB-1/BB-2・N=8,12,16,20におけるp=1探索密度、現在のp=2候補探索、実行時間、状態ベクトルメモリを厳密計算で比較 |
| `xy_qaoa_normalization/` | BB-1/BB-2・N=8,12,16,20で、無正規化・max-abs・RMS正規化が現行XY-QAOA分布へ与える影響を、元の目的関数で比較 |
| `xy_qaoa_bbo_scaling/` | 最高性能設定（p=2, 49 candidates）での100-cycle BBOにおけるN=8,12,16,20スケーリング特性と各ビット数trajectory個別比較 |

## 今回追加した検証

| Folder | Validation | Main finding |
|---|---|---|
| `bbo_seed_robustness/` | 3 seedでのBBO平均・標準偏差 | 単一seedの順位は安定しない．BB-1の最適到達はFMQA 2/3，FM-XY-QAOA 2/3，Standard QAOA 0/3 |
| `shot_budget_sensitivity/` | Standard QAOAが未評価の実行可能候補を得る確率とshotsの関係 | BB-1では500 shots時の成功確率が約0.60% |
| `adaptive_lambda_validation/` | 50ランダムインスタンスで経験的適応λと厳密必要下限を比較 | 48/50では十分だが2例で不足し，普遍的な十分条件ではない |

## 主要な残課題

- BBOのseed数を3から10以上へ増やし，信頼区間と統計検定を追加する．
- 複数のBBインスタンスにまたがるBBO性能を評価する．
- QAOA深さ `p`，回路ノイズ，読み出し誤差に対する頑健性を検証する．
- 実材料または公開材料データに対する外的妥当性を確認する．
