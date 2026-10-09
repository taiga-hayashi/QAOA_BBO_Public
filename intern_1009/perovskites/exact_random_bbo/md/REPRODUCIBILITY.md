# 再現手順

Python: /Users/hayashitaiga/.venvs/fas/bin/python。OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1、MPLCONFIGDIR=/private/tmp/intern1009-mpl、XDG_CACHE_HOME=/private/tmp/intern1009-fontcache。

順序: `py/run.py prevalidate` → pueueで `py/run.py run` → 正常終了依存タスク `py/analyze.py`。未完了/既存Runの上書きは拒否。10サイクルごとと例外時にcheckpoint保存。baseline JSONとcheckpointは読み取りのみ。図は中央値/IQR、タイトル・下部注記なし。目視確認は次の対話で実施する。
