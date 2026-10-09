# 監査と再現性

共有src/qarp_backend.pyの標準X/p1およびRing XY/productW/p1を使用。XY λ0厳守。既存N6/N9/N23の回路事前検証と固定FM監査合格を参照。新規テストでCDF境界・nested prefix・ペナルティ込みエネルギーの独立数式照合を確認。runnerはprevalidation.jsonの合格とコードSHA256を実行前に検証。

各FMのQUBO・初期候補・チェックポイントは元の固定FM実験と同じ。元の9点探索で決定済みの角度を用いて全23量子ビット状態を再計算し、raw norm、192実行可能候補の確率、正規化Cost期待値を元結果と照合。full2^23×23ビット行列は作らない。complex128 state・float64 probabilities/CDF/energiesを用い、二次モーメントは65536単位で計算する。配列サイズは実装情報であり、peak RSSは未測定。

新規測定uniformsはSeedSequence([model_seed,rep,1009,73])による1000個。method間は同じuniforms、shots間はnested prefix。CDF逆変換で全基底状態を直接サンプリング。無効解もfull基底indexとして保存し、全ショットのCostに含める。各ショットのCostは独立のFM+λP数式と比較。平均Cost誤差と制約充足率誤差を保存。候補は共有score_candidatesでFM最小の未評価実行可能1件を選ぶ。

SA10比較は4バッチ3,3,2,2、1000sweeps、geometric default。seed=model_seed×100+rep×100000+batch。Largeはα100、Adaptiveは各反復にα1へ戻し、共有AdaptivePenaltyTrackerでバッチ間更新。原1000reads実験の結果を再ラベルしない。

```sh
/Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/qaoa_shot_accuracy/py/test_shots.py
OPENBLAS_NUM_THREADS=1 /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/qaoa_shot_accuracy/py/run_shots.py
OPENBLAS_NUM_THREADS=1 XDG_CACHE_HOME=/private/tmp/intern1009-fontcache MPLCONFIGDIR=/private/tmp/intern1009-mpl /Users/hayashitaiga/.venvs/fas/bin/python intern_1009/perovskites/qaoa_shot_accuracy/py/analyze_shots.py
```

repository rootで実行。再実行は既存rawを保全した別フォルダで行う。設定・コード変更時は事前検証とSHA256の新規記録が必要。元のseed JSONが存在すればrunnerは上書きせず停止。失敗記録は保持する。

5モデルで統計集計し、6000反復を6000独立学習モデルとは扱わない。補正Wilcoxonとペア差を保存。QAOAの9点角度探索コストは元実験の条件、今回の10shots出力や測定精度とは別。標準とXYのCostを同一とみなさない。有限ショット角度探索・ハードウェアノイズ・再学習BBOは未実施。
