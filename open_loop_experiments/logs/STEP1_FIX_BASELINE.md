# 1. 比較基準の固定

- **p=1 と p=3 の差異の記録**:
  以前の `XY-FMQAOA` の保存済み結果は `p=1` (`best_params` の長さが 2) であったのに対し、現在の `src/runner.py` は `solve_xy_fmqa_p_layers` に `p=3` をハードコードして呼び出していました。この差異による影響と互換性を保つため、`runner.py` の `run_bbo_experiment` 関数に新たに `xy_p` パラメータを追加し、デフォルト値を `1` とすることで、既存の p=1 ベースラインを維持しつつ将来の p の変更にも対応できるように改修しました。

- **N=6/9 での検証結果**:
  `verify_p1_baseline.py` による短縮サイクル検証で、`N=6` および `N=9` について、`raw feasible rate = 1.0000` (100% 実行可能) であり、one-hot 制約が適切に保存され、候補の提案元 (solver_best, solver_alternative, uniform_feasible_regeneration) も正しく記録されていることを確認しました。エネルギーや保存済み軌跡との整合性も期待通りに機能しています。

- **Penalty-FMQAOA N=24/27 の扱い**:
  計画書に定められた通り、Penalty-FMQAOA の N=24 および N=27 の結果は、修正済みの計算リソース（run）が完全に揃うまで、未測定として扱います。既存結果を上書きしません。
