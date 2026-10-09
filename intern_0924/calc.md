# Multiple One-Hot 制約付き FMQA への XY-Mixer 導入効果検証実験 仕様書 (calc.md)

> [!NOTE]
> 本仕様書は、AIや研究者が各タスクに応じたルールをピンポイントかつ効率的に参照できるよう、**内容ごとに [`ruru/`](./ruru/) フォルダ配下に分割・整理** されています。各項目をクリックして詳細を参照してください。

---

## 📑 仕様・ルール分割インデックス (`ruru/`)

| 番号 | ルール・仕様項目 | 主な内容 | 参照ファイル |
| :---: | :--- | :--- | :--- |
| **01** | **原文プロンプト** | 指示書画像の完全な原文書き起こしテキスト | [`ruru/01_original_prompt.md`](./ruru/01_original_prompt.md) |
| **02** | **目的と研究課題** | 実験目的および4つの主要リサーチクエスチョン | [`ruru/02_objectives_and_research_questions.md`](./ruru/02_objectives_and_research_questions.md) |
| **03** | **比較手法と問題設定** | 4手法の定義（Adaptive, Large, Penalty-FMQAOA, XY-FMQAOA）および公平比較条件 | [`ruru/03_methods_and_problem_setup.md`](./ruru/03_methods_and_problem_setup.md) |
| **04** | **コード監査と事前検証** | 監査チェックリストおよび $N=6, 9$ での $P_{\text{feasible}}=1$ 厳密検証基準 | [`ruru/04_prevalidation_and_code_audit.md`](./ruru/04_prevalidation_and_code_audit.md) |
| **05** | **数理定義と評価指標** | 係数スケール $S_t$、正規化ペナルティ $\alpha_t$、Regret $R$、制約充足率の数式定義 | [`ruru/05_math_definitions_and_metrics.md`](./ruru/05_math_definitions_and_metrics.md) |
| **06** | **実験詳細仕様** | 実験1（感度）〜実験6（計算資源）の各実験パラメータ・プロトコル | [`ruru/06_experiments_specification.md`](./ruru/06_experiments_specification.md) |
| **07** | **プロット可視化仕様** | 必須プロット定義（Fig 1〜4, 9）および色（青/橙/赤/緑）・マーカー・線種の統一規程 | [`ruru/07_plot_and_visualization_standards.md`](./ruru/07_plot_and_visualization_standards.md) |
| **08** | **統計処理基準** | 中央値・IQR、同一Seedペア差分解析、Wilcoxon検定、Bootstrap CI 基準 | [`ruru/08_statistical_analysis_guidelines.md`](./ruru/08_statistical_analysis_guidelines.md) |
| **09** | **保存成果物とメタデータ** | 保存すべき JSON、ログ、環境情報、再現性手順書一覧 | [`ruru/09_artifacts_and_metadata.md`](./ruru/09_artifacts_and_metadata.md) |
| **10** | **厳格な禁止事項** | 架空データ禁止、チェリーピック禁止、XYへのペナルティ混入禁止など 7大原則 | [`ruru/10_strict_prohibitions.md`](./ruru/10_strict_prohibitions.md) |
| **11** | **実行ステップ** | 監査 → 事前検証 → 本実験 → 可視化 → 報告の5段階ワークフロー | [`ruru/11_workflow_steps.md`](./ruru/11_workflow_steps.md) |

---

## 🚀 クイックスタート・再実行コマンド
完全な実験再実行・再現手順は [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) を参照してください。

```bash
# 事前検証（N=6, 9 厳密数学検証）
"intern/.venv/bin/python" intern_0924/tests/test_prevalidation.py

# 全実験および全図面の一括生成
"intern/.venv/bin/python" intern_0924/experiments/run_all.py
```
