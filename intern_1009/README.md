# 2026-10-09 XY-FMQAOA対象問題の検討・実験

公開版の収録範囲・依存関係・実行手順は[PUBLICATION.md](PUBLICATION.md)を参照。

候補7問題の検討場所。**One-Hot変数数は100以下に限定し、100超の設定は不採用。**ユーザー指定によりPestControlは対象外。

| 問題 | One-Hot変数数＝論理量子ビット数 | 難易度見積もり | 状態 |
|---|---|---|---|
| [多層光学フィルター](multilayer_filter/README.md) | 4L。3〜6層で12〜24 | 中 | draft / 未実施 |
| [Olympus Perovskites](perovskites/README.md) | 16+3+4=23 | 低〜中 | local pilot完了 / 正式benchmarkはdraft |
| [水素分子の基底状態エネルギー](hydrogen_molecule/README.md) | 小規模Q=8で16。128・384変数版は不採用 | 低〜中 | draft / 未実施 |
| [3材料の平面多層放射冷却構造](radiative_cooling/README.md) | 3L。5〜9層で15〜27 | 中 | draft / 未実施 |
| [Olympus Redoxmers](redoxmers/README.md) | 2+8+8+11=29。2択の位置固定で27 | 低〜中 | draft / 未実施 |
| [RNA逆折りたたみ](rna_inverse_folding/README.md) | 12塩基で48は候補。26塩基104は不採用 | 中 | draft / 未実施 |
| [COMBO Centroid](centroid/README.md) | 3E。4×4格子72。縮小2×2は12、2×3は21 | 中 | draft / 未実施 |

各フォルダの`README.md`に問題定義、参照元、最初の確認事項を記載し、`json/problem_manifest.json`に設定案と未確定値を保存する。正式benchmarkの採用manifestは`draft`を保持する。Perovskitesの評価器・各ローカル実験は実施済みで、問題別READMEに記録する。他の6候補は初期検討段階。

[共通検討手順](INITIAL_REVIEW.md) / [作業ルール](AGENTS.md) / [先行調査メモ](../report/xy_fmqaoa_candidate_problems.md)

## ステップ1の調査記録

[7候補の問題定義確認表](md/problem_definition_review.md)を作成した。各問題の詳細定義・公開コードとの差・未確定項目・取得資料の版を保存済み。評価器と最適化は未実施で、全設定はdraftのまま。

## 着手の選択肢

- 3択仕様を保つ基礎検証: Centroid縮小版、3材料平面多層案。
- 公開データを使う材料問題: Perovskites、Redoxmers固定版。
- 既存FMQAの符号化研究との接続: 水素分子、RNA。
- 光学構造のサイズ検証: 多層光学フィルター。

採用順は未決定。固定版・縮小版・9層平面拡張は新しい問題設定として記録する。放射冷却5層の平面比較は原資料に存在する。29以上の回路は既定の27上限外であり、検討候補の記載を実行設定と解釈しない。

## 完了条件

最初の到達点は、各問題で「評価器を再現できるか」「XYが保存する制約で十分か」「必要な回路幅で実行可能か」を根拠付きで判断すること。その後に採用設定・共通比較条件を凍結する。

## 優先する問題規模

ユーザー指定によりOne-Hot変数数20〜30を優先する。[候補設定表](md/target_size_candidates.md)を参照。既定回路上限27のため、まず20〜27の設定から着手する。28〜30は拡張扱い。範囲外の参照設定は保存するが優先しない。
