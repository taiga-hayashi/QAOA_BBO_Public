# 固定FMパイロットの実装監査

## 共有実装の再利用

- FMと学習: src/fm.pyのTorchFM / train_factorization_machine。
- FM→QUBO: src/fm_to_qubo.py。全192実行可能候補で直接FM出力との誤差を検査。
- SAのOne-Hot BQM: src/fmqa_solver.pyのbuild_penalty_bqm、nealの明示1,000 sweeps。
- Adaptive alpha: intern_0924/src/penalty.pyのAdaptivePenaltyTracker。固定FMの250-readバッチごとに更新する診断であり、再学習を伴うBBO反復の実行ではない。
- QAOA: src/qarp_backend.pyのOpenQARPStandardQAOA / OpenQARPXYQAOA。標準方式にも生statevectorを取得するAPIを加え、回路やゲート角は維持した。intern/srcの同ファイルへ同期。
- メモリ処理: intern_0924/src/solvers.pyの汎用_build_qubo_energies、_state_energy_expectation、_sample_statevector_streamingを再利用。3択専用の問題・Mixer関数は呼ばない。

23変数の2²³×23の全ビット行列を作らない。エネルギー配列はfloat64の2²³要素（64 MiB）、状態はcomplex128の2²³要素（1本128 MiB）。これは配列サイズの算術でありピークRSSの実測ではない。角度探索は共有OpenQARPの全状態シミュレーションで、候補192状態へ投影したシミュレーションではない。

## 事前確認

test_pilot.pyで4テストを実行。標準X回路と独立のCost/RX計算、分割QUBOエネルギーと全ビット参照、FM→QUBOの全二値入力照合、ストリーミングサンプリング、無効・重複拒否と予算不足を検証した。既存評価器・Ring回路の13テストも再実行して合格。ログはprevalidation.log / shared_regression.log。

## フェアネスと解釈

各Seedに対して初期データとFMのハッシュを全4手法で一致させる。train20件以外の真値はFMや角度選択には渡さず、診断指標と選択済み候補の評価にのみ使う。初期候補とプール内重複を共通に除外し、FM予測順で最大5件を採用する。不足は再サンプルで埋めず、実使用BB予算を記録する。

QAOA方式間は同じ深さ・角度候補数・角度範囲・サンプル数。SAもサンプル数は同じだが、CPU時間・計算操作数を一致させた比較ではない。固定9点での標準QAOAの不調を、角度探索や深さ一般の不可能性と解釈しない。

同じ固定FMは4手法で同じ真値順位誤差を持つ。FMの最小予測値を探す能力と実際によい材料を提案する能力を分ける。Uniformの平均真値などは全表からの診断参照であり、追加の最適化手法のBBO結果とは呼ばない。
