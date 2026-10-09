# 実験仕様・ルール集 (ruru/)

本ディレクトリ `ruru/` は、長大な実験仕様書（`calc.md`）をAIおよび研究者が目的・内容ごとに素早く参照・読解できるように分割・構造化したルール集です。

---

## 📑 分割ファイル一覧（クイックインデックス）

| 番号 | ファイル名 | 内容概要 |
| :---: | :--- | :--- |
| **01** | [`01_original_prompt.md`](./01_original_prompt.md) | 原文プロンプトの完全書き起こし（忠実テキスト） |
| **02** | [`02_objectives_and_research_questions.md`](./02_objectives_and_research_questions.md) | 実験目的および4つの主要研究課題 |
| **03** | [`03_methods_and_problem_setup.md`](./03_methods_and_problem_setup.md) | 比較4手法の定義（Adaptive, Large, Penalty-FMQAOA, XY-FMQAOA）と公平比較条件 |
| **04** | [`04_prevalidation_and_code_audit.md`](./04_prevalidation_and_code_audit.md) | コード監査項目および事前検証（$N=6, 9$ での $P_{\text{feasible}}=1$ 厳密検証） |
| **05** | [`05_math_definitions_and_metrics.md`](./05_math_definitions_and_metrics.md) | 数理定義（係数スケール $S_t$、正規化ペナルティ $\alpha_t$）および主要評価指標 |
| **06** | [`06_experiments_specification.md`](./06_experiments_specification.md) | 実験1〜実験6の目的・パラメータ・対象規模・詳細仕様 |
| **07** | [`07_plot_and_visualization_standards.md`](./07_plot_and_visualization_standards.md) | 必須プロット定義（Fig 1〜4, 9, Main 1〜6）および色・マーカー・線種の統一書式 |
| **08** | [`08_statistical_analysis_guidelines.md`](./08_statistical_analysis_guidelines.md) | 統計処理基準（中央値、IQR、ペア差分、Wilcoxon検定、95% Bootstrap CI） |
| **09** | [`09_artifacts_and_metadata.md`](./09_artifacts_and_metadata.md) | 保存成果物一覧（JSON、ログ、環境情報、再現性メタデータ） |
| **10** | [`10_strict_prohibitions.md`](./10_strict_prohibitions.md) | 厳格な禁止事項（捏造禁止、チェリーピック禁止、XYへのペナルティ混入禁止等） |
| **11** | [`11_workflow_steps.md`](./11_workflow_steps.md) | 推奨ワークフロー（監査 → 事前検証 → 本実験 → プロット → レポート） |
