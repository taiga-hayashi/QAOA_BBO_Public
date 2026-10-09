# 再現手順と版管理

実行条件はjson/protocol.json。実行前に毎回このプロトコルを読む。固定FMの診断としてstatus=readyを付けたもので、親の閉ループBBO manifestはdraftのまま。

READMEの順にtest_pilot.py、run_pilot.py、summarize_pilot.pyを実行する。全Seed完了後にplot_pilot.pyを実行すればPDF/SVG/PNGと独立パネルを再生成できる。PythonはOpenQARP等を備えた既存fas環境を使用した。

環境とソースハッシュはjson/environment.json。各Seedの初期候補・値・ハッシュ、モデルハッシュ、checkpoint hashはjson/seed_*.json。モデルはdata/checkpoints/seed_<seed>.pt。モデル再学習や全表によるハイパーパラメータ選択を行わない。

完全に同じプロトコルとソースの場合だけ、完成Seedは再利用される。不完全・失敗Seedは記録を保存したまま自動上書きを止める。再計算や条件変更はこのフォルダを別版として複製し、結果とプロトコルを区別する。

Sourceデータは../data/reference/data.csvと同じ親Perovskites表。SHA256はprotocolと照合される。192候補の全表評価はホールドアウト診断であり、各手法の20+最大5件のBB予算に含まれる探索評価と区別する。表引き値の独立DFT再計算・元構造の選択規則再構築は含まない。
