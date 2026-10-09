# 再現手順

`/Users/hayashitaiga/.venvs/fas/bin/python py/analyze.py compute` が正常終了してから同じ環境で `py/analyze.py plot` を実行する。作業ディレクトリはこの診断フォルダ。OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。元データ・checkpointは変更しない。計算は共有checked関数で全予測と選択を照合。正常終了後のみpueue依存タスクで集計結果の図を生成。
