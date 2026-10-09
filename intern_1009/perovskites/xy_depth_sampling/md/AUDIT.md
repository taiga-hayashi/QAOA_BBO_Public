# 実行前監査

ユーザーがXY-QAOAのp1/2/3を選択したため、既定p1の主比較とは別の深さvariantを作る。直前のSeed42・初期10固定FMをそのまま使い、lambda0、W、Ringのordered RXX/RYY productを維持。共有src/qarp_backend.pyは多層角度列を受け取り、変更不要。角度探索・保存だけ新規実装し、FM、QUBO、選択、回路は再利用。

元23bit回路を8bitで厳密に符号化したシミュレーション。物理量子ビット8に圧縮できたという主張ではない。N6/N9多層ではfull OpenQARPとcompactをfresh照合、対象23bitでは独立部分空間referenceに照合する。モデルSHA、全192予測、p1旧9点も検証してから実行。

角度範囲は各層0〜0.8、各深さ128候補×10search seeds。p1旧9点を含む。前深さbestをゼロ層で埋め込むwarm startはcandidate1件として数える。理想期待FM値で角度を選び、100shots×10反復で出力品質を評価する。深さごとの回路評価回数は同じだが総層数は違う。
