# Perovskites評価器・符号化の確認結果

確認日: 2026-10-09。[事前プロトコル](../json/evaluator_validation_protocol.json)に従い、評価器と古典的符号化のみを検証した。FM学習・BBO・QAOAは実行していない。

## 結果

| 確認項目 | 結果 |
|---|---|
| CSV形式 | ヘッダーなし、organic/cation/anion/hse_gapの4列 |
| 行数・一意な組成数 | 192件・192組成 |
| 選択肢の直積との一致 | 16×3×4の全192組成を網羅 |
| 欠損セル・重複キー・未知カテゴリ | いずれも0 |
| 非有限値・負のバンドギャップ | いずれも0 |
| カテゴリ→One-Hot→カテゴリ | 全192候補で一致 |
| カテゴリ評価／One-Hot評価／CSV | 全192候補でPython float値が厳密一致 |
| 共有one_hot_patternsとの候補順 | 全192候補で一致 |
| 回帰テスト | 6テスト合格。破損データ7種類や不正入力を含む |

根拠：[データ監査JSON](../json/data_audit.json)、[検証JSON](../json/evaluator_validation.json)、[全192候補の照合記録](../json/validated_candidates.json)、[テストログ](evaluator_validation.log)。ソース版・SHA256は[source_snapshot](../json/source_snapshot.json)、実行環境とコードSHA256は[validation_environment](../json/validation_environment.json)。

## 保存表の参照最適値

全192候補を網羅しているため、この固定表については全列挙による最小値を確定できる。

- 最小値: 1.5249、hydrazinium / Sn / I。
- 最大値: 6.3242、tetramethylammonium / Pb / F。
- 採用単位: eV。元データ論文のHSE06バンドギャップの単位を根拠とする。Olympus config/CSV自身には単位が明示されていない。

これは新たなDFT計算やBBOで発見した値ではない。元論文の1346構造全体の最適性を主張するものでもない。Olympusが複数構造から1組成1値へ縮約した規則は独立再構成していない。[単位・利用条件の確認状況](../json/unit_and_license_review.json)に記録した。

## 評価器とビット順

[PerovskitesEvaluator](../py/perovskites_evaluator.py)はPython標準ライブラリのみで動く。入力はconfig順の3カテゴリ、出力は目的値のfloat。encode/decode/evaluate_onehotを用意した。不正な長さ、非二値、各グループの0個／複数個選択、未知カテゴリはValueErrorとし、補修や目的値への置換をしない。

23要素のベクトルでは、organicがq0〜q15、cationがq16〜q18、anionがq19〜q22。配列先頭がq0で、状態番号はΣ bits[q]2^q。この表記を画面上のMSB先頭文字列と混同しない。

[共有src/qarp_backend.py](../../../src/qarp_backend.py)のone_hot_patternsを検証に再利用した。評価器用にFM・QUBO・回路実装は複製していない。既存src/bb_function.pyの実行可能判定は長さ・二値の厳格な検査をしないため、問題のデコーダで入力検査を行った。

## 未実施と次の作業

- 共有OpenQARPXYQAOAは現時点で全対ペアのXYを使う実装。ユーザー指定のRingと一致するとは扱わない。次の回路監査で共有実装を確認する。
- N=6,9の事前検証と[16,3,4]のW-state・Ring制約保存検証は未実施。
- 実行環境にはnumpyがあるがtorch・qarpxは未導入。今回の表引き・符号化検証では不要。
- 元の外部データ配布元の独立した利用条件、Olympusの構造選択規則は未確認。取得リポジトリのMITライセンスと元論文のCC BY 4.0は確認した。
- BB予算、初期データ、Seed、FM条件、重複規則は未凍結。全候補と最適値は監査専用とし、探索器の学習へ事前投入しない。

manifestはdraft、benchmark_ready=falseを維持した。評価器validated=trueはQAOAやBBO実験の検証完了を意味しない。

## 回路検証の更新

上記は評価器検証時点の未実施項目。その後、共有回路のRing対応とN=6/9/23の理想回路事前検証を実施した。現状は[CIRCUIT_REVIEW](CIRCUIT_REVIEW.md)を参照。利用環境は既存のfas環境（OpenQARP 0.1.1）を確認して使い、新規インストールは行っていない。
