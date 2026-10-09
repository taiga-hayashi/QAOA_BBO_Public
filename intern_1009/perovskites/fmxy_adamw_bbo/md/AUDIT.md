# 回路監査・事前検証

共有src/qarp_backend.pyにOpenQARPCompactXYQAOAを追加し、intern/src/qarp_backend.pyへ同じ変更を反映した。既存OpenQARPXYQAOA/Standardの挙動は変更しない。元の23 One-Hot回路のbasis・Ring辺順序を共通定義から取得する。新経路はOne-Hot192状態を各群の符号化レジスタへ写し、OpenQARPのdiagonal_unitaryで同じFM cost、unitary_synthesisで同じ順序付き辺回転の積を計算する。射影の近似・Hamiltonianの近似・ペナルティ追加はない。

16/3/4の群はシミュレータ上4/2/2ビットの8ビットレジスタへ写る。未使用符号は初期振幅0、群mixerは未使用符号へ移らない。元問題の変数数と元回路の幅は23のまま。この計算時間短縮を実機8量子ビットでの同等資源・量子優位の根拠として扱わない。

独立reference_evolutionは各Ring辺の2状態回転を直接適用する。N6/N9/N23で全9角度点について、(1)元の全状態OpenQARPの確率、(2)新しい符号化OpenQARPの確率、(3)独立辺回転の振幅（全体位相補正のみ）を照合。確率差・振幅差・norm/feasible mass誤差を1e-10以内で確認。N23は今回のSeed42・初期20から学習したrank1 AdamWモデルを使用。N9ではp2の追加一致試験も行った。全27条件はjson/prevalidation.jsonとdata/prevalidation.logに保存。

元のRing実装の7 regression tests、intern/README.mdのquick source checkも合格した。現在の角度gridは従来のgamma0.05/0.4/0.8、beta0.1/0.4/0.8の9点を保持し、FM係数SでCostを正規化する。角度目的は理想FM期待値、真値を使わない。9点以外の大域最適性は未証明。

共有学習とSA pilotの選択関数を再利用し、初期20・学習条件・modelseed・評価上限・候補規則をSA proposed設定と一致させる。全5 Seedで最初のFMモデルhashがSAと同じことを事後監査。以後の訓練集合は各手法が選ぶ候補によって分岐する。sampling CDFは元23ビットLSBの基底順へ並べて10乱数を適用する。
