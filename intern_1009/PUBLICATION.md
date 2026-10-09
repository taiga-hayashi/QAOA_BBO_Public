# 公開スナップショット（2026-10-09）

`taiga-hayashi/QAOA_BBO_Public`用。ローカル実験コード、測定済みJSON、モデルcheckpoint、PDF/SVG/PNGを収録する。結果の再計算やSeedの選別はしていない。履歴・設定・ハッシュを保つため、保存済み実験ファイルは原本のままコピーしている。

## 見る場所

- [問題候補の一覧](README.md)
- [Perovskitesの全実験](perovskites/README.md)
- [初期3件・XY p3比較](perovskites/xy_p3_initial3_bbo/md/RESULTS.md)
- [FM最小解と実最適解の一致](perovskites/fm_optimum_alignment/md/RESULTS.md)
- [全列挙FM最小／ランダムBBO対照](perovskites/exact_random_bbo/md/RESULTS.md)

Perovskitesはローカル表引きpilotの計算が完了している。元のベンチマークmanifestのdraftは、正式なベンチマークの採用・来歴照合等が未完了という意味で、計算未実施を意味しない。他の6候補は初期検討段階。

## 実行環境

Python 3.11以上。リポジトリのルートで専用環境を用意する。

```sh
python3 -m venv .venv-intern1009
source .venv-intern1009/bin/activate
python -m pip install -r intern_1009/requirements.txt
python intern_1009/py/run_public.py --check-imports intern_1009/perovskites/exact_random_bbo/py/run.py
```

公開リポジトリのルートには旧版のflat modulesも存在する。`run_public.py`を使うと収録したcanonical `src/`を優先し、実験当時の共有FM/QUBO/OpenQARPモジュールを利用できる。元の実験コードは変更していない。`src/`、`intern_0924/src/`と実験ルールを同時に収録し、`intern/src/fm.py`・`qarp_backend.py`・`qaoa_solver.py`の既存mirrorも同期した。既存の他実験結果を再計算したものではない。

記録済みの絶対パス、Python環境パス、pueue task IDは実行時の来歴であり、clone先でそのまま使う設定ではない。完了済みRunは上書きを拒否する。再計算には別の作業コピーを使い、各実験のREPRODUCIBILITY.mdとreadyプロトコルを確認する。実験実行にはpueue、正常終了後の監査・プロットには依存taskを使う。公開時の確認は構文・import・ハッシュの照合で、新規BBOは実行していない。

## 再配布する第三者資料

- Olympusのconfig、description、Perovskites CSV: 保存した版と出典は各`source_snapshot.json`。OlympusのMITライセンス・著作権表示を保持している。
- 元の材料データ: Kim, Huan, Krishnan, Ramprasad, *Data from: A hybrid organic-inorganic perovskite dataset*, Dryad, [doi:10.5061/dryad.gq3rg](https://datadryad.org/dataset/doi:10.5061/dryad.gq3rg)。[Dryad利用条件](https://datadryad.org/terms)で公開DatasetのCC0適用と再利用を確認した。ただしOlympusの192候補への縮約と1346構造の対応を検証したわけではない。既存metadataの未確認フラグは過去の記録として保持する。
- 元論文: Kim et al., *A hybrid organic-inorganic perovskite dataset*, Scientific Data 4, 170057 (2017), [doi:10.1038/sdata.2017.57](https://doi.org/10.1038/sdata.2017.57)。論文全文は収録しない。
- RNA参照コード: 同梱MITライセンスを保持。
- COMBO参照コード: 同梱BSDライセンス・著作権表示を保持。

## 公開版で省いたファイル

`__pycache__`、`.pyc`、`.DS_Store`、実行ログと、再配布条件未確認の第三者文献PDF2件・DQAOA参照コード／README3件は省いた。取得先・版・SHA256は残している。省略一覧は[publication_manifest.json](json/publication_manifest.json)。ローカル原本を削除したものではない。

フォルダ全体へ新しい包括的なライセンスを付けたものではない。第三者資料は各ライセンス・出典に従う。秘密キー・token等のパターン照合では該当を検出していない。
