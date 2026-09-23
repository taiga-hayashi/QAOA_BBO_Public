# Python ファイル一覧と役割

`intern/` の Python ファイルを、実行目的ごとに整理した一覧です。通常の実行入口は `main.py`、検証結果の完全再現は `main.py --experiments` です。QAOA の実行バックエンドは OpenQARP/`qarpx` に統一されています。

## 最初に使う入口

| ファイル | 役割 | 実行 |
| --- | --- | --- |
| `main.py` | 通常 BBO デモ、または実験再現へ振り分けるラッパー | `python main.py` / `python main.py --experiments` |
| `run_all_experiments.py` | 固定シードの全実験、100-cycle BBO、図・独立パネルを再生成 | `python main.py --experiments [--fresh]` |

## `src/`: 実装・比較・解析コード

| ファイル | 役割 |
| --- | --- |
| `src/__init__.py` | 既存の直接実行互換のためのパッケージ初期化。 |
| `src/main.py` | FM 学習、OpenQARP QAOA 提案、BB評価、結果保存を行う小規模BBOデモ。 |
| `src/runtime.py` | YAML設定検証、乱数固定、結果ディレクトリ作成の共通処理。 |
| `src/fm.py` | PyTorch Factorization Machine と共通FM学習関数。 |
| `src/fm_to_qubo.py` | 学習済みFMを QUBO 辞書と定数項へ変換。 |
| `src/qubo.py` | 上三角QUBO行列を共通QUBO辞書へ変換し、max-abs/RMS係数正規化と正規化係数の記録を提供する。 |
| `src/bb_function.py` | ランダムQUBO BB関数、One-Hot実行可能解、厳密解、データセット生成。 |
| `src/fmqa_solver.py` | Simulated Annealing とペナルティによる古典FMQAベースライン。 |
| `src/qarp_backend.py` | Standard/XY QAOA の OpenQARP 回路、One-Hot初期状態、Ising係数変換。 |
| `src/qaoa_solver.py` | OpenQARP-only の Standard QAOA、FM-XY-QAOA、通常QUBOの統一API。 |
| `src/subspace_xy_qaoa_solver.py` | 従来のリングXY部分空間ベースライン。基底・ハミルトニアンは拡張版と共有。 |
| `src/enhanced_subspace_xy_qaoa_solver.py` | p層、リング/完全グラフXY、CVaR、局所探索を含む部分空間拡張ソルバー。 |
| `src/compare_solvers.py` | FMQA・Standard QAOA・FM-XY-QAOAの分布とペナルティ掃引比較。 |
| `src/compare_bbo.py` | 3手法のBBO軌跡と制約違反提案を比較。 |
| `src/scaling_experiment.py` | Nごとの制約充足率・最適解確率・計算量スケーリングを評価。 |
| `src/boundary_analysis.py` | ペナルティ、問題スケール、QAOA層数の境界条件を調べる。 |
| `src/benchmark_multi_instance.py` | 複数ランダムインスタンスでの標準・部分空間手法の集計。 |
| `src/benchmark_large_scale_comparison.py` | N=20/24の大規模One-Hot比較。 |
| `src/benchmark_accuracy_enhancement.py` | 拡張部分空間QAOAの精度改善要素を比較。 |
| `src/materials_scaling_benchmark.py` | 材料設計型QUBOのスケール比較と図作成。 |
| `src/practical_materials_bbo.py` | 多元触媒・材料設計を模したBBO閉ループ。 |
| `src/fmqa_penalty_sweep_experiment.py` | FMQAのペナルティ係数掃引。 |
| `src/fmqa_penalty_sweet_spot_analysis.py` | FMQAペナルティのスイートスポット解析。 |
| `src/fmqa_lambda_with_objective_scale.py` | 目的関数スケールに対するFMQAペナルティ感度。 |
| `src/scale_up_kitai_benchmark.py` | Kitaiスケールを意識したN=4〜60の比較用計算。 |
| `src/generate_materials_scaling_plots.py` | 保存済み材料スケールJSONから図を作成。 |
| `src/plot_extreme_scales.py` | 極端なNに対する概念的スケール図を作成。 |
| `src/utils.py` | stdoutログ複製とFM全探索補助。 |

`src/*_benchmark.py`、`src/*_experiment.py`、`src/compare_*.py` は個別解析用です。現在の検証済み成果物の再現には、下記の `experimet/` 側を使用してください。

## `experimet/`: 検証済みOpenQARP実験

| ファイル | 役割 |
| --- | --- |
| `experimet/py/openqarp_qaoa.py` | `src/qarp_backend.py` を実験スクリプトから使うためのp=1互換ラッパー。 |
| `experimet/py/run_openqarp_bbo_chunk.py` | 100-cycle BBOを途中チェックポイント付きで分割実行。 |
| `experimet/py/finalize_openqarp_bbo_results.py` | 完走済みチェックポイントだけを結果JSONへ昇格。 |
| `experimet/py/rerun_openqarp_direct_comparison.py` | N=16直接比較（1,000 shots）の再計算。 |
| `experimet/py/export_individual_panels.py` | 統合図から切り抜かず、独立した各パネル図を新規描画。 |
| `experimet/optimization/py/run_optimization_experiment.py` | BB-1/BB-2の直接比較とBBO閉ループ本体。 |
| `experimet/optimization/py/plot_optimization_results.py` | 最適化比較JSONからPDF/PNGを描画。 |
| `experimet/bbo_seed_robustness/py/run_bbo_seed_robustness.py` | 複数seedのBBO頑健性データを構成。 |
| `experimet/bbo_seed_robustness/py/plot_bbo_seed_robustness.py` | seed頑健性JSONを図へ変換。 |
| `experimet/qubit_scaling/py/run_qubit_scaling.py` | N別のOpenQARP QAOA、FMQA、状態空間・目的値比較を実行。 |
| `experimet/qubit_scaling/py/plot_qubit_scaling.py` | 量子ビットスケーリングJSONを統合図と目的値図へ変換。 |
| `experimet/std_qaoa_sensitivity/py/run_std_qaoa_sensitivity.py` | Standard QAOAのOne-Hotペナルティ λ 感度を測定。 |
| `experimet/std_qaoa_sensitivity/py/plot_std_qaoa_sensitivity.py` | Standard QAOA感度の図を描画。 |
| `experimet/penalty_sensitivity/py/run_penalty_sensitivity.py` | FMQAペナルティ感度を測定。 |
| `experimet/penalty_sensitivity/py/plot_penalty_sensitivity.py` | FMQAペナルティ感度の図を描画。 |
| `experimet/shot_budget_sensitivity/py/run_shot_budget_sensitivity.py` | OpenQARP直接比較を基にショット数依存性を集計。 |
| `experimet/shot_budget_sensitivity/py/plot_shot_budget_sensitivity.py` | ショット数感度JSONを描画。 |
| `experimet/adaptive_lambda_validation/py/run_adaptive_lambda_validation.py` | 適応ペナルティλの制約充足検証を実行。 |
| `experimet/xy_qaoa_accuracy_scaling/py/run_xy_qaoa_accuracy_scaling.py` | 現行の完全グラフXY mixerを保ち、BB-1/BB-2のN=8,12,16,20でp=1探索密度と現在のp=2候補探索を厳密状態ベクトルで比較する。 |
| `experimet/xy_qaoa_accuracy_scaling/py/plot_xy_qaoa_accuracy_scaling.py` | 上記JSONから問題別の最適解確率・期待gap、実行時間、状態ベクトルメモリの独立PDF/PNGと集計MDを作成する。 |
| `experimet/xy_qaoa_normalization/py/run_xy_qaoa_normalization.py` | BB-1/BB-2・N=8,12,16,20で、QUBOの無正規化・最大係数正規化・RMS正規化を、p=1/p=2の現行XY-QAOAへ適用して比較する。評価は常に元の目的関数で行う。 |
| `experimet/xy_qaoa_normalization/py/plot_xy_qaoa_normalization.py` | 正規化感度JSONから、問題・深さごとの最適解確率と元の目的関数gapの独立PDF/PNG、および集計MDを作成する。 |
| `experimet/xy_qaoa_bbo_scaling/py/run_xy_qaoa_bbo_scaling.py` | BB-1/BB-2・N=8,12,16,20で、最高性能XY-QAOA（p=2, 49 candidates）の100サイクルBBOを実行し、Standard QAOAおよびFMQAと性能・軌跡を比較する。 |
| `experimet/xy_qaoa_bbo_scaling/py/plot_xy_qaoa_bbo_scaling.py` | 上記BBO結果から、各ビット数の100サイクルtrajectory個別図（8パネル＋統合図）およびスケーリング比較図（4パネル＋統合図）をタイトルなし・独立パネル形式で作成する。 |

## 教材・報告書・参考資料

| ファイル | 役割 |
| --- | --- |
| `qaoa_tutorial/06_hands_on_qiskit_implementation.py` | Qiskitを使う過去の教育用ハンズオン。現在の実験バックエンドではない。 |
| `qaoa_tutorial/build_textbook.py` | QAOA教材LaTex全体をビルド。 |
| `qaoa_tutorial/generate_textbook.py` | 教材TeXの体裁を生成・更新。 |
| `qaoa_tutorial/patch_tikz.py` | 教材TikZ図の重なり・はみ出し修正。 |
| `qaoa_tutorial/perfect_patch_tikz.py` | 教材TikZ図の仕上げ修正。 |
| `quarp/scripts/generate_qaoa_oss_comparison_figures.py` | OpenQARPと量子OSSの比較説明図を作成。 |
| `report/generate_accuracy_enhancement_figures.py` | 精度向上ベンチマークの報告図を作成。 |
| `report/generate_fmqa_lambda_sweep_figures.py` | FMQA λ掃引の報告図を作成。 |
| `report/generate_fmqa_objective_scale_figure.py` | 目的関数スケール対λの報告図を作成。 |
| `report/generate_high_visibility_figures.py` | 視認性重視の総合報告図を作成。 |
| `report/generate_large_scale_benchmark_figures.py` | 大規模比較の報告図を作成。 |
| `report/generate_multi_instance_benchmark_figures.py` | 複数インスタンス比較の報告図を作成。 |
| `report/generate_qiskit_oss_scaling_figures.py` | 過去のQiskit/OSSスケーリング説明図を作成。実験再計算には使わない。 |
| `report/generate_symmetry_scaling_explanation_figure.py` | 対称性・部分空間スケーリングの説明図を作成。 |

## 実行時の注意

- `experimet/` のJSON・PDF・PNGは現在の検証済みスナップショットです。完全再計算は長時間かかるため、通常は閲覧のみで十分です。
- 現在のQAOA実行コードは OpenQARP/`qarpx` を使います。Qiskit名を含む教材・過去の説明図は参照資料であり、通常実行や実験再現の依存には含まれません。
- 個別スクリプトは `intern/` をカレントディレクトリとして実行する前提のものがあります。推奨入口は `python main.py` と `python main.py --experiments` です。
