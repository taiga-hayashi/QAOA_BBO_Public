# 再現性

実験条件は../json/protocol.jsonに実行前に固定。既存fixed_fm_pilotの5モデル・初期20件・QUBOを使用し、チェックポイントと各入力のSHA256を保存。既存モデルを再学習しない。src/fmqa_solver.pyのペナルティBQMとfixed_fm_pilotの候補採用規則を再利用する。

リポジトリrootで、環境Python `/Users/hayashitaiga/.venvs/fas/bin/python` を使用。

```sh
/Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmqa_penalty_sweep/py/test_sweep.py
/Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmqa_penalty_sweep/py/run_sweep.py
XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/fmqa_penalty_sweep/py/analyze_sweep.py
```

runnerはprevalidation.jsonの合格状態とrunnerハッシュを確認して実行する。変更時はテストを実行し、その結果・コードハッシュを新たに記録する。既存seedファイルがあれば上書きせず停止する。再計算は原結果を残した新規フォルダで行う。失敗run・起動失敗ログも残す。

1反復のsampler_seed=model_seed×100+repetition×100000+batch。α間で共通。4×250 reads、1000 sweeps、geometric。自動beta rangeは各バッチのinfoへ保存。反復ごと最大5件の候補確認であり、600反復を一本の評価軌跡として連結しない。全192件のオフライン真値は診断用でBB予算に混入させない。

environment.jsonにPython・パッケージ・コードハッシュ・OSを保存。summary.jsonに全5モデルの反復平均、中央値/IQR、定義可能件数、ペア差分・補正検定を保存。artifact_verification.jsonにraw再計算と入力照合の結果を保存。
