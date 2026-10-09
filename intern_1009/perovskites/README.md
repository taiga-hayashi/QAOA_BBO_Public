# Olympus Perovskites

状態: **固定FM診断と古典SAとXYの再学習pilotが完了 / benchmarkはdraft**。標準X-QAOAの閉ループBBOは未実施。

[回路の確認結果](md/CIRCUIT_REVIEW.md) / [コード監査](md/audit_report.md) / [評価器の確認結果](md/EVALUATOR_REVIEW.md) / [再実行手順](md/REPRODUCIBILITY.md) / [評価器コード](py/perovskites_evaluator.py)

[問題定義の確認記録](md/problem_definition.md) / [原資料の取得版・SHA256](json/source_snapshot.json)

[全体一覧](../README.md) / [共通検討手順](../INITIAL_REVIEW.md) / [設定・未確認事項](json/problem_manifest.json)

## 問題定義

有機成分16択、金属Ge/Sn/Pbの3択、ハロゲンF/Cl/Br/Iの4択から材料を選ぶ。

目的: HSE06バンドギャップ hse_gap の最小化。

評価方法: `lookup`。実装難易度の見積もり: **低〜中**。

## 符号化と規模

One-Hot変数数: **16+3+4=23**。論理量子ビット数であり、物理量子ビット数ではない。

各グループ内で1つの選択肢を選ぶ。W-stateとグループ内Ring XY-Mixerを用い、One-Hotペナルティはλ=0。候補がOne-Hotでも評価器固有の無効条件を満たすことがあるため、別途検証する。

| 設定案 | 元の設計変数数 | One-Hot変数数 | 由来 | 27上限 |
|---|---:|---:|---|---|
| `full` | 3 | 23 | original_dataset | 範囲内 |

公式descriptionは192サンプルを記載。設計変数は3個で、記述子の数を量子ビット数に加えない。

## 最初に確認すること

- [x] data.csvの192組合せ、欠損・重複を確認する。単位eVは元論文に基づき、Olympus CSVへの明示はない
- [x] データのcommit・SHA256と利用条件の確認状況を保存する（外部元データの条件は未確認）
- [x] 全192候補でカテゴリ順序、表引き、One-Hot往復と共有候補順を確認する
- [x] 16・3・4択のW-state／Ringを理想全状態ベクトルで検証する（実機用準備ゲートは未合成）
- [ ] 全データから最適値を確認し、未評価候補数以内の予算を決める
- [ ] 実装コード、データ、ライセンス、評価関数版を記録する。
- [x] デコーダ、全候補の評価、無効入力と破損データの拒否を検証する。
- [ ] 原設定と変更した設定を別Problem IDで記録する。
- [ ] 共通手順の評価予算・初期データ・Seed・FM条件を確定する。
- [x] N=6・9および23変数の回路事前検証を通す。BB予算カウントはBBO実装時に確認する。

## 限界・未確認

全192候補の小規模問題。EvoDuet側の課題条件は未確認。全候補表を探索器・FMの学習に事前公開せず、BB呼出し経由で取得する。

論文の結果は本フォルダでの実測結果ではない。公開資料の調査内容を引き継いでおり、参照資料はdata/referenceへ保存済み。数値データ監査・評価器・One-Hot変換・回路検証と固定FMでの候補生成比較は完了。閉ループBBOは未実施。

## 参照元

- [公式config](https://raw.githubusercontent.com/the-matter-lab/olympus/main/src/olympus/datasets/dataset_perovskites/config.json)
- [公式description](https://raw.githubusercontent.com/the-matter-lab/olympus/main/src/olympus/datasets/dataset_perovskites/description.txt)
- [データ](https://github.com/the-matter-lab/olympus/tree/main/src/olympus/datasets/dataset_perovskites)

参照確認日: 2026-10-09。実装時には資料のcommit／版と利用条件を再確認する。

## 初期検討の記録

- 採用する設定と理由: 未決定
- 評価値の照合: 全192件でCSV値と一致。保存表の最小値1.5249、最大値6.3242（eV）
- 評価時間・メモリ: 性能計測は未実施。テストログの所要時間は評価器ベンチマークではない
- FM表現力・候補選択品質: 未測定
- 次の作業: 共通BBOプロトコルの凍結と23変数の候補集計・メモリ監査

実装時に必要な `py/`、`data/`、`md/`、`png/`、`pdf/`、`svg/` を追加する。生データと集計データは`json/`で区別し、ログ・チェックポイントも保存する。

## 変数数による候補判定（ユーザー指定）

One-Hot変数数が100を超える設定は不採用。100以下は候補に残すが、採用・実行可能性は未確定。既定の27量子ビット上限は別途確認する。

| 設定 | One-Hot変数数 | 判定 |
|---|---:|---|
| `full` | 23 | 候補継続 |

## 固定FMの候補生成診断

[fixed_fm_pilot](fixed_fm_pilot/README.md)で、同じ初期20件・同じ学習済みFMに対して4手法の比較を完了した。5 Seed、各1,000サンプル、未評価候補を最大5件採用。閉ループBBOとは区別する。条件は[事前プロトコル](fixed_fm_pilot/json/protocol.json)で固定した。結果は[RESULTS.md](fixed_fm_pilot/md/RESULTS.md)、図は[FIGURES.md](fixed_fm_pilot/md/FIGURES.md)を参照。

## FMQAペナルティ係数の感度

[fmqa_penalty_sweep](fmqa_penalty_sweep/README.md)で、同じ5固定FMの古典SAについてα=0〜1000の12条件・各10反復を計算した。全600条件の制約充足率、新規実行可能解獲得頻度、候補品質と多様性を保存。固定α診断であり、動的更新や閉ループBBOではない。

ユーザー指定により、今後の候補採用は**未評価実行可能候補のFM予測値が最も低い1件**とする。保存済みSAサンプルの[1件採用再集計](fmqa_penalty_sweep_top1/README.md)を別フォルダに保存。従来の5件採用結果は原実験として保持する。今後の4手法比較にも同じ1件採用規則を適用し、比較プロトコルは別途凍結する。

[サンプル数10・30・100・300・1000の比較](fmqa_sampling_budget_top1/README.md)も保存済みSAプールから二次解析した。全3000条件。新規num_reads設定でのSA実行時間や閉ループBBOは未測定。

[QAOAショット数・精度と4手法10出力比較](qaoa_shot_accuracy/README.md)では、既存角度でOpenQARP全状態を再計算し、10〜1000shotsの新規測定を各100回模擬。SA10readsも新規計算。全6000条件の結果・図を保存。有限ショットによる角度再最適化と閉ループBBOは未実施。

[標準QAOAのペナルティ感度](standard_qaoa_penalty_sweep/README.md)も全5モデル・13α・32500測定条件の検証を完了。各αで同じ9点の角度選択を実施し、理想充足確率と有限ショット精度を保存。角度探索の大域最適性は未証明。[次の検討方向（提案）](md/NEXT_STEPS.md)は角度探索の検証→FM初期データ感度→再学習BBOとし、実施済み結果とは区別する。

[N6/N9の角度探索検証](angle_search_validation/README.md)は5モデル・440設定・47,080回路評価で完了。9/9/49/361点の共通探索予算、標準α1〜1000とCost再正規化対照、XY λ0を比較した。元の9点の探索依存性と、制約充足率・FM最小解獲得の違いを確認。全23変数の角度検証・有限ショット角度探索・閉ループBBOは未実施。

[FM学習改善の初期診断](fm_learning_diagnostic/README.md)は同じ初期20件の18条件・90学習で完了。ランク・学習率・epochs・train-only標準化を比較し、未観測有機カテゴリの重みが初期値のまま残ることも照合した。探索的な固定データ診断であり、この段階では正則化・初期集合変更・再学習BBOを含めていない。

[Adam/AdamW・weight decay診断](fm_regularization_diagnostic/README.md)は同じ初期20件、rank1/2、lr0.1・120 epochsの30学習で完了。予測誤差と選択材料の真値を分けて報告し、初期データ網羅性の変更と再学習BBOは未実施。

[AdamW設定の再学習FMQA](fmqa_adamw_bbo/README.md)は初期20＋60サイクル・10reads・最大1件採用、2FM設定×2SA手法×5Seedの20Runで完了。新設定の両SA手法で5/5が最適値1.5249eV、初期最適解ありを除く新規発見は4/4。QAOA BBO・網羅設計は未実施。

[FM-XYQAOAの再学習最適化](fmxy_adamw_bbo/README.md)は同じrank1 AdamW、初期20＋60サイクル・10shots・最大1件採用で5Run完了。新規最適値発見4/4、初期最適解ありを含む最終5/5。元23ビット回路と確率を照合した共有OpenQARP厳密符号化シミュレーションを使用。標準X-QAOA BBOは未実施。

[10／100出力の再学習比較](bbo_sample_count_comparison/README.md)は同じrank1 AdamWで3手法×2counts×5Seedの30Run、1800サイクルを完了。初期最適解なし4Seedの到達回数中央値はAdaptive31→30、LargePenalty28→32、XY55→39。全条件新規4/4。予測FM gapは改善したが、補正有意差は未確認。固定FM・角度のXY300ペアも別診断として保存。

[初期5／10／20件の比較](initial_data_sensitivity/README.md)を45Runで完了。100出力・総BB上限80を固定し、75／70／60サイクルに配分。共通4SeedのXY最適解到達評価数中央値は27.5／25.5／39。全45Runで最適値1.5249 eVに到達。初期FM精度は20件が良く、補正有意差は未確認。次の共通設定候補は初期10件＋70サイクル。[全体PDF](initial_data_sensitivity/pdf/initial_count_overview.pdf)。

[大ペナルティ診断](large_penalty_diagnostic/README.md)は初期10・100reads、alpha1〜1000000、固定FM700batches＋再学習35Runで完了。制約充足を維持しても固定FM gapが悪化（alpha100:0.18548、1000:0.37983 eV）。BBO最終は全35Run最適値、途中予算では悪化例と逆の例がある。補正有意差未確認。[全体PDF](large_penalty_diagnostic/pdf/large_penalty_overview.pdf)。

[1つの固定FMの分布解析](single_fm_sampling_quality/README.md)はSeed42・初期10の1モデルに限定し、各alpha100reads×10反復の既存生データを再解析。FM最小捕捉はalpha1で10/10、100で4/10、1000で2/10。CDF・順位別頻度・最良候補gap・多様性を保存。新規学習/SA/BBOなし。[全体PDF](single_fm_sampling_quality/pdf/single_FM_sampling_overview.pdf)。

[固定FMのXY深さ比較](xy_depth_sampling/README.md)は同じSeed42・初期10モデルでp1/2/3、128角度候補×10探索Seed、100shots×10測定/探索Seedを完了。観測FM最小捕捉はp1/2:9/100、p3:26/100。p2は全探索でp1埋込解を選択、角度探索不足と能力を区別。BBOなし、サンプル品質の補正有意差は未確認。[PDF](xy_depth_sampling/pdf/XY_depth_overview.pdf)。

[初期3件・XY p＝3の再学習最適化](xy_p3_initial3_bbo/README.md)を、共通XY-p1・SA alpha1000と合わせて15Runで完了。各77サイクル・100出力・総BB上限80。最適解到達評価数の中央値はp1:33、p3:29、SA:27、全条件5/5で最適値1.5249 eV。補正有意差は未確認。XYはlambda0、SAはlambda＝1000S。128角度候補は各サイクルの総予算。[全体PDF](xy_p3_initial3_bbo/pdf/initial3_p3_BBO_overview.pdf)。

[FM最小解と実最適解の一致診断](fm_optimum_alignment/README.md)を保存済み1155モデルで完了。初期3件の5FMは全て不一致。全77モデルの一致率の5Seed中央値はXYp1/p3:16.9%、SAalpha1000:22.1%。p3の実最適解発見時は4/5Runで未評価FM最小解ではなかった。事後診断であり新規BBOなし。

[全列挙FM最小／ランダムBBO対照](exact_random_bbo/README.md)をpueueに登録。初期3件・77サイクル・最大80評価・5Seed、新規10Run。保存済みXYp3／SAalpha1000との比較は正常終了後に監査・プロットする。結果は未確認。
