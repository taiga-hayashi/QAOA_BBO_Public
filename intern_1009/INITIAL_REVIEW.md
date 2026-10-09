# 共通の初期検討手順

この文書は検討計画。実験結果は含まない。

ユーザー指定の候補条件: One-Hot変数数が100を超える設定は不採用。100以下でも実行可能性は別途検証し、既定27量子ビット上限を自動的に変更しない。

## 1. 原資料と問題定義を確認

原論文・公式データ・公開コードを読み、設計変数、選択肢、目的、追加制約、元の符号化を記録する。コードのcommit、データのSHA256、利用条件、依存環境を保存する。設定変更は原問題と区別する。

## 2. 評価器を再現

カテゴリ値を受け取ってBB目的値を返す評価器を作る。表引きデータは欠損・重複・未収録候補を監査する。シミュレータは代表例を原資料と照合し、決定性または共通乱数を確認する。解析できる関数をBBとして扱う場合はその限界を明記する。

小規模の評価器確認と本BBO実験を区別する。時間計測も測定条件と実行ログを残す。

## 3. 符号化と回路の監査

各グループK_g択でN=sum(K_g)。選択肢順序、MSB/LSB、One-Hotのデコードを固定する。既存3択固定コードを不均一サイズへそのまま流用しない。

XY-FMQAOAはp=1、各グループW-state、グループ内Ring、One-Hotペナルティλ=0。既定のN=6,9検証に加え、対象グループサイズで|1−P_feasible|<1e−10を検証する。追加の実行可能条件は別に扱う。部分空間シミュレーションと全状態シミュレーションを混同しない。

共有FM、QUBO、回路実装を監査して再利用する。問題別に共通処理を複製しない。新規問題のデコーダと評価器を分離する。

## 4. 比較プロトコルを凍結

比較はAdaptive-FMQA（古典SA、alpha動的）、LargePenalty-FMQA（古典SA、alpha=100）、Penalty-FMQAOA（X-Mixer、p=1、alpha=5）、XY-FMQAOA（λ=0）。

同じProblem IDとSeedで初期候補・学習データ・FM構造と学習条件・BB予算・重複破棄ルールを完全に統一する。初期解・候補一覧・評価値とハッシュを保存する。無効候補と重複の再サンプル／停止／予算カウントも統一する。

S_t=max(max_i|h_i|,max_{j<i}|J_ij|)、λ_t=S_t alpha_tを記録し、固定alphaと固定λを区別する。S_t=0の場合の扱いは実装前に定義する。

BB予算、Seed、FMハイパーパラメータ、評価関数版、採用設定は現在未確定。`problem_manifest.json`のnullを仮の既定値で埋めて実行しない。採用したvariant、ライセンス／版、評価器検証、比較条件、事前検証を記録して初めて`status: ready`、`benchmark_ready: true`とする。readyは結果の成功を意味しない。

3択以外は既定仕様からの拡張を明示する。27量子ビットを超える設定は本実験の既定上限外。元の変数数と部分回路幅を両方記載する。

## 5. 性能評価・報告の計画

Best-so-far、raw/accepted feasible rate、未評価候補率、重複率、FM誤差・順位相関、評価器／学習／候補生成の時間、成功率を記録する。

真のf_optが確認できる場合のみR=(f_best−f_opt)/(f_worst−f_opt)を用い、f_worstの定義を事前に固定する。連続最適値と離散候補の最適値を混同せず、未知の最適値をbest-foundで代用しない。分母ゼロの場合も別途定義する。

中央値とIQRまたは95%Bootstrap CI、同一Seedのペア差、Wilcoxonと多重比較補正を用いる。失敗runを除外せず理由・打切りを報告し、予算終了を収束と呼ばない。

既定の監査→事前検証→実験1〜6→可視化→報告の順を守る。PDF/SVG/PNGは実測JSONから生成する。RESULTS.md、DISCUSSION.md、REPRODUCIBILITY.mdは実測・検証後に作り、架空の結果で埋めない。

## 根拠となるルール

- [ruru全体](../intern_0924/ruru/README.md)
- [手法・フェアネス](../intern_0924/ruru/03_methods_and_problem_setup.md)
- [監査・事前検証](../intern_0924/ruru/04_prevalidation_and_code_audit.md)
- [数理定義と指標](../intern_0924/ruru/05_math_definitions_and_metrics.md)
- [統計](../intern_0924/ruru/08_statistical_analysis_guidelines.md)
- [保存物](../intern_0924/ruru/09_artifacts_and_metadata.md)
- [禁止事項](../intern_0924/ruru/10_strict_prohibitions.md)
- [実行順序](../intern_0924/ruru/11_workflow_steps.md)

## 優先する問題規模

ユーザー指定によりOne-Hot変数数20〜30を優先する。[候補設定表](md/target_size_candidates.md)を参照。既定回路上限27のため、まず20〜27の設定から着手する。28〜30は拡張扱い。範囲外の参照設定は保存するが優先しない。
