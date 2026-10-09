# 再現手順

リポジトリ直下で実行する。Python・プラットフォームは[environment.json](../json/environment.json)、依存版・解析コードSHAは[delivery_verification.json](../json/delivery_verification.json)、原FM・データ・コードSHAは環境記録および各seed JSONに保存。プロトコルは実行前にreadyとして固定した。raw実行、解析、事前検証ログを本mdに保存。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/angle_search_validation/py/test_angles.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/angle_search_validation/py/run_angles.py
OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/angle_search_validation/py/analyze_angles.py
```

raw実行は既存seed JSONがあれば停止し、上書きしない。再計算は既存成果物を保存した別フォルダで実施する。解析だけなら保存済みraw JSONから実行できる。全440設定と47,080回路評価の完全性、探索点の最小Cost選択、入れ子探索の最小期待値非増加、選択状態の規格化を確認する。標準X確率は独立phase/RX計算で照合した。XY全探索点の実行可能率はrunnerで検査。共有backendに変更は加えていない。

3つの事前テストはFM制限の全状態エネルギー照合、N6/N9 XY保存、厳密best-of-shots式の小さな独立列挙を確認。検証結果は[prevalidation.json](../json/prevalidation.json)と[artifact_verification.json](../json/artifact_verification.json)。元の5モデルに欠測・失敗runはない。計算は理想状態ベクトル、CPU上。実機ノイズ、全23変数の角度探索、BB追加評価と再学習を実施していない。
