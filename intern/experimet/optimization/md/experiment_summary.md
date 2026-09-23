# 2つのBB関数に対する最適化比較（100サイクルBBO）

N=16・4ブロックOne-Hot QUBOの2問題について，直接サンプリング比較と，seed=42の閉ループBBOを100サイクル実行した．Part 1の直接サンプリング値は既存の検証済み結果を保持し，Part 2のみを100サイクルで再計算している．

ペナルティ係数 $\lambda_{\mathrm{adaptive}}$ は負の相互作用に基づく利得ベースの目安であり，制約充足や最適性を保証する理論下限値ではない．

## Part 2: BBOの結果

| 問題 | 手法 | 最終最良値 | Gap | 最適値への初到達cycle | 新規実行可能候補なし | 実行時間 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BB-1 | FMQA | -15.7736 | 0.0000 | 12 | 0 / 100 | 99.21 s |
| BB-1 | Standard QAOA | -8.4284 | 7.3451 | — | 100 / 100 | 42.90 s |
| BB-1 | FM-XY-QAOA | -15.7736 | 0.0000 | 12 | 0 / 100 | 0.72 s |
| BB-2 | FMQA | -5.8090 | 0.0000 | 5 | 0 / 100 | 84.39 s |
| BB-2 | Standard QAOA | -5.8090 | 0.0000 | 21 | 27 / 100 | 46.66 s |
| BB-2 | FM-XY-QAOA | -5.8090 | 0.0000 | 5 | 0 / 100 | 1.86 s |

この表はseed=42の単一試行である。手法の一般的な優劣は，[seed頑健性実験](../../bbo_seed_robustness/)を含む複数seedの結果で評価する。

## 成果物

- [結果JSON](../json/results_two_bb_optimization.json)
- [100サイクル比較PDF](../pdf/two_bb_optimization_comparison.pdf)
- [実行スクリプト](../py/run_optimization_experiment.py)
- [プロットスクリプト](../py/plot_optimization_results.py)
