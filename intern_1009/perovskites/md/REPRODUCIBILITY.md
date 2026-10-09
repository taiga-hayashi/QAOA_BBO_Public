# 評価器検証の再実行

対象は固定Olympus表の監査、表引き評価器、23変数One-Hot変換。BBOや回路の実験手順ではない。

リポジトリのルートから：

```sh
python3 intern_1009/perovskites/py/validate_evaluator.py
```

該当フォルダからなら：

```sh
python3 py/validate_evaluator.py
```

評価器自体はPython標準ライブラリのみ。検証スクリプトは共有src/qarp_backend.pyのone_hot_patternsを使うためnumpyが必要。今回の環境はPython 3.14.4 / numpy 2.4.6 / macOS arm64。torch・qarpxは不要。

事前に[プロトコル](../json/evaluator_validation_protocol.json)を読み、今回の範囲を確認する。スクリプトはprotocolの状態を検査してから実行し、同名の監査JSON・検証JSON・全候補記録・環境JSON・テストログを更新する。データ・config・コードのハッシュは記録されるので比較できる。参照ファイル自体は書き換えない。

回帰テストのみ：

```sh
python3 -m unittest discover -s intern_1009/perovskites/py -p test_perovskites_evaluator.py -v
```

評価器の利用例（`py/`をPythonのimport pathへ追加した環境）：

```python
from perovskites_evaluator import PerovskitesEvaluator
bb = PerovskitesEvaluator()
labels = ('ethylammonium', 'Ge', 'F')
value = bb.evaluate(labels)  # 5.3704
bits = bb.encode(labels)    # 23要素、q0先頭
assert bb.decode(bits) == labels
assert bb.evaluate_onehot(bits) == value
```

データ取得元は[保存したURL・commit・SHA256](../json/source_snapshot.json)で固定している。値の独立したDFT再計算や元1346構造との照合は含まない。全候補記録は監査用であり、将来のBBOの初期学習データとして全件を渡さない。

## Ring XY回路の事前検証

[回路プロトコル](../json/circuit_validation_protocol.json)を読み、理想回路だけの検証であることを確認する。既存環境にOpenQARP 0.1.1 / numpy 1.26.4 / scipyがあることを確認して使った。

リポジトリのルートから：

```sh
/Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/py/validate_xy_ring.py
```

該当フォルダから：

```sh
/Users/hayashitaiga/.venvs/fas/bin/python py/validate_xy_ring.py
```

別のPCではopenqarp・numpy・scipyを利用できるPythonに置き換える。元のパスが必須ではない。検証はN=6/9/23の全状態ベクトルを使い、結果JSON・環境JSON・テストログ・進捗JSONを更新する。過去結果を残す場合は出力を退避してから実行する。

今回の実行ログは`md/circuit_validation_run.log`。状態の測定記録は`json/circuit_validation.json`、環境・コード版は`json/circuit_validation_environment.json`。`circuit_validation_progress.json`は最後の条件の中間記録であり、全条件の集計には使わない。

回路・評価器の回帰テストをまとめて実行する場合：

```sh
/Users/hayashitaiga/.venvs/fas/bin/python -m unittest discover -s intern_1009/perovskites/py -p 'test_*.py' -v
```

初期W状態は理想ベクトルとして注入する。実機用の準備ゲート、ノイズ、有限ショット、FM学習・BBOはこの手順には含まない。
