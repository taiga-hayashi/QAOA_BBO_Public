# 図の読み方

比較図fm_learning_overviewの横軸はランク。0は一次項と切片だけの加法モデル、1/2/4はFM因子数。左Test RMSE、中Test Spearman、右予測最小未評価材料の真の値。RMSE・右の真値は低いほど良く、順位相関は高いほど良い。青は元のlr0.1/120epochs、橙はlr0.01/1000epochsで、後者は学習率と回数を両方変更した対照。両者の差を片方だけの効果として読まない。

各線は5Seed中央値、帯はIQR。独立図test_rmse/test_spearman/selected_true_valueもPDF/SVG/PNGで保存。図の2系列以外を含む全18条件はRESULTS.mdとjson/summary.jsonに保存した。

真値は全未評価172候補のFM予測を列挙して最小候補を選んだ後に参照する。QAOA/SAの有限サンプルによる選択結果や、再学習BBOの最終値ではない。図中にタイトル・下部注釈を入れず、本書で条件を明示する。
