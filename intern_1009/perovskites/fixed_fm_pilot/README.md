# Perovskites：固定FMの候補生成診断

初期20件で学習したFMを固定し、4手法の候補生成を5 Seedで比較した。全20条件完了。チェックポイント・真値・予算記録の独立照合も合格。閉ループBBOの実験ではない。元問題は23変数、16/3/4択。条件は実行前に[protocol.json](json/protocol.json)で固定した。

[結果](md/RESULTS.md) / [解釈と次の課題](md/DISCUSSION.md) / [実装監査](md/AUDIT.md) / [再現手順](md/REPRODUCIBILITY.md) / [集計JSON](json/summary.json) / [コード](py/run_pilot.py)

リポジトリルートから：

```sh
/Users/hayashitaiga/.venvs/fas/bin/python -m unittest discover -s intern_1009/perovskites/fixed_fm_pilot/py -p test_pilot.py -v
/Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fixed_fm_pilot/py/run_pilot.py
/Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fixed_fm_pilot/py/summarize_pilot.py
```

該当フォルダからは各コマンドのパスを`py/test_pilot.py`等に変更する。Python環境にはtorch、numpy、scipy、openqarp、dwave-neal、dimodが必要。共有FM・QUBO・SA・OpenQARP回路、intern_0924の適応alphaとメモリを抑える汎用処理を再利用する。サロゲート・回路を問題ごとに複製していない。

run_pilotは完成したSeedだけを、プロトコルとコードハッシュが一致する場合に再利用する。失敗・未完成・ハッシュ不一致の記録がある場合は自動上書きしない。条件変更時は元の結果を残し、別バージョンの出力を作る。失敗Seedを黙って除外しない。

学習済みFMはdata/checkpoints/、生結果はjson/、ログと報告はmd/。監査用の全表参照を、ソルバーのBB予算や学習データに混ぜない。各方式の候補確認予算は初期20＋最大5、実際の採用数は別途記録する。
