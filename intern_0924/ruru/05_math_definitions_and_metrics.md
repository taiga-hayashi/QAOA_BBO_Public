# 05. 数理定義と主要評価指標

## 1. ペナルティ係数の正規化定義

FM-QUBO の内部係数スケール $S_t$：
$$S_t = \max\left( \max_i |h_i|, \max_{j < i} |J_{ij}| \right)$$

正規化ペナルティ係数 $\alpha_t$：
$$\alpha_t = \frac{\lambda_t}{S_t} \iff \lambda_t = S_t \cdot \alpha_t$$

- **固定ペナルティ感度評価で使用する $\alpha$ グリッド**:
  $$\alpha \in \{0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100\}$$
- 各反復で $\lambda_t = S_t \cdot \alpha_t$ を保存し、$\lambda$ 固定と $\alpha$ 固定を混同しないこと。
- LargePenalty-FMQA の推定根拠を明記すること。

---

## 2. 主要評価指標

### 最適化性能
- **最終 Normalized Regret ($R$)**:
  $$R = \frac{f_{\text{best}} - f_{\text{opt}}}{f_{\text{worst}} - f_{\text{opt}}} \quad (R = 0 \text{ が最良})$$
  ※最小化問題において $f_{\text{opt}}$ は大域的最小値、$f_{\text{worst}}$ は初期ランダムサンプルの最悪値または全空間最大値。
- **探索推移**: BB 評価回数に対する Best-so-far 値推移
- **目標到達性能**: 目標精度（例: $R \le 0.05$ や $R \le 0.02$）到達までの BB 評価回数、目標到達成功率

### 制約充足性
- **Raw One-Hot Feasible Rate ($P_{\text{feasible}}$)**: ソルバーがサンプリングした候補群における One-Hot 実行可能解の割合。
- **Accepted Feasible Rate**: 最終的に BBO ループで受容されブラックボックス評価に送られた候補の実行可能率。
- **未評価実行可能候補率**: サンプリングされた実行可能候補のうち、過去に評価されていない新規解の割合。

### 候補品質と多様性
- **未重複 Top-5% 候補生成率**: 全実行可能解の中で上位 5% に入る良質解が、未重複で生成された確率。
- **重複候補率、候補再生成回数**: 探索が局所解に停滞して同一解を連続提案した頻度。

### FM サロゲートモデル精度
- **予測誤差**: RMSE、MAE
- **順位相関**: Spearman 順位相関係数 $\rho$、Kendall 順位相関係数 $\tau$
- **上位捕捉率 (Recall)**: Top-1%, Top-5%, Top-10% Recall

### 計算コスト
- **時間**: 候補1件当たりの生成時間、FMQA 全体処理時間、FM 学習時間、QAOA 実行時間
- **安定性**: 実行成功率、失敗理由ログ
