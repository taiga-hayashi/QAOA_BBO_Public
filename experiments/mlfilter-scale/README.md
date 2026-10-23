# experiments/mlfilter-scale: 光学フィルターの層数拡大(Phase 0: 実現性の確認)

- 計画: [docs/mlfilter-scale-plan.md](../../docs/mlfilter-scale-plan.md)
- 区分: Phase 0 は diagnostic。手法の比較・優劣の判定はしない
- 条件: [config.json](config.json)。config_hash は `{config: config.json, layers: L}` の正規化 JSON の SHA-256 で、L ごとに config_id が分かれる
- 既存 study(mlfilter など)のコードと結果は変更しない

## コード

| ファイル | 役割 |
|---|---|
| `py/evaluator.py` | 既存評価器の層数上限を 10 に広げた版。光学計算・FOM は同一 |
| `py/phase0.py` | `init` / `enumerate` / `timing` / `xstate`。各 run の `metadata.json` に使用コードの SHA-256 を記録し、変更後の再実行を拒否する |
| `py/run_pinned.sh` | 空いているコアに固定して 1 ジョブを実行(flock)。`NCORES=n` で n コア・n スレッド |
| `../../src/qarp_pair_xy.py` | `OpenQARPPairXYQAOA`: コスト層をグループ・グループ対の対角に分けた compact XY(結果は旧版と同一、L=8 で約 300 倍速い) |
| `L_FIXED_INVENTORY.md` | 既存コードの L 固定箇所の一覧(着手順 1) |
| `PHASE0_RESULTS.md` | Phase 0 の実測結果と計画への影響 |

## run(Phase 0、サーバー)

| L | 候補数 | 内容 | config_id | run_id |
|---:|---:|---|---|---|
| 6(基準の再確認) | 4,096 | 全列挙・旧 compact XY・通常 QAOA | c-266f9e02f5c8 | r-2295f36082ca |
| 7 | 16,384 | 同上 | c-5e9cbb5483f7 | r-f492fe73e4f9 |
| 8 | 65,536 | 同上(旧 XY の 1024 回探索は停止、通常 QAOA は 16 スレッド) | c-94cc85879e73 | r-00239d4f5cc3 |
| 6 | 4,096 | 新 pair 版 XY の時間(全列挙表は上の run を再利用) | c-f6c6ad42128d | r-ab0dcf5c7ca0 |
| 7 | 16,384 | 同上 | c-64d0d19a8681 | r-7922ac09200a |
| 8 | 65,536 | 同上 | c-3a1fbd63bd25 | r-9270f95cb06e |

### Phase 1(探索): ペナルティ係数の感度と層数

条件: [phase1_penalty.json](phase1_penalty.json)。コード: `py/phase1_penalty.py`(init / run / analyze)、共通ループ `../../src/mlfilter_fmqa.py`、`py/run_pinned_py.sh`。pueue グループ `mlfilter-p1`(20 並列、2026-10-20 投入)。

| L | 内容 | config_id | run_id |
|---:|---|---|---|
| 6 | 探索 5 Seed × 7 条件(予算 100) | c-002e69f22ae5 | r-340fa2f9b2b9 |
| 6 | mlfilter-pen30 のサーバー再現(3001〜3030 × α1/α3/α100/Random) | c-002e69f22ae5 | r-3c8320e5abcb |
| 7 | 探索 5 Seed × 7 条件(予算 400、先頭 100 評価も集計) | c-8665d43bc952 | r-3fa991b2856f |
| 8 | 探索 5 Seed × 7 条件(予算 1600、先頭 100 評価も集計) | c-b36e0d935f9d | r-c6204a8b9cb7 |

コードの等価性: L=6 で保存済み mlfilter-pen30 の FM 状態から SA・選択を再実行し、900 サイクル全一致。FM 学習は Mac と x86 で float32 丸めが異なり、軌跡そのものはビット一致しない(そのため pen30 の結論をサーバーで再現する run を置いた)。

出力: `data/runs/mlfilter-scale/<config_id>/<run_id>/`(`raw/enumeration.json`、`raw/truth_table.npz`、`raw/timing.json`、`raw/xstate.json`、`logs/`、`status.json`)。ジョブは [job_record.json](job_record.json)。

## 実行環境

- サーバー `jupyter-taiga0806--taiga-instance1`(cgroup のメモリ上限 256 GiB)。各ジョブ 1 スレッド、1 コアに固定
- Python: `/home/jovyan/FMA/.venv/bin/python`(3.12.3、numpy 1.26.4、torch 2.13.0)。openqarp 0.1.1 は `PYTHONPATH=/home/jovyan/FMA/QAOA_BBO_Public/code/.deps`
- 共通コード: `/home/jovyan/FMA/Intern_Fujitsu_ref`(Drive の Intern_Fujitsu から必要分だけコピー。`config/paths.local.json`)
- pueue: 専用デーモン `/home/jovyan/FMA/.pueue-openqarp/pueue.yml`、グループ `mlfilter-scale`

## 状態

- 2026-10-20: Phase 0 の計測は完了(L=8 の旧 XY 探索のみ停止)。結果は [PHASE0_RESULTS.md](PHASE0_RESULTS.md)
- XY の実装を pair 版に変更(利用者の判断)。config.json の `timing.xy.backend_history` に記録
- 未決定事項(最大の L、評価予算、比較手法、問題構造への対処)は利用者の判断待ち
- パッケージ化・Drive 保存は未実施
