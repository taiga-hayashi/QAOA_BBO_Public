# 量子近似最適化アルゴリズム（QAOA）体系的学習教材
## 〜 Farhi 2014 原論文から Hadfield 2019 XYミキサー論文、実問題応用まで 〜

本教材は、量子アルゴリズムの最重要分野の一つである**量子近似最適化アルゴリズム（QAOA: Quantum Approximate Optimization Algorithm）**について、基礎物理・線形代数から、原著論文の完全数式読解、XYミキサーによる制約付き最適化への拡張、そしてPython/Qiskitによる実践コードまでを体系的かつ網羅的に学ぶための自習・研究用テキストです。

---

## 読者対象と前提知識
- **対象**: 量子情報・量子コンピューティング、最適化アルゴリズム、または機械学習・物性計算に関心のある学生・研究者・エンジニア
- **前提知識**: 
  - 基本的な線形代数（固有値・固有ベクトル、エルミート行列、ユニタリ行列、テンソル積）
  - 量子力学のブラケット表記の基礎（$|0\rangle, |1\rangle, \langle \psi |$）
  - Pythonプログラミングの基礎

---

## 教材の構成とカリキュラム（目次）

本教材は以下の7つのモジュール（章）から構成されています。順番に読み進めることで、未経験から最先端のXYミキサー研究レベルまで到達できるように設計されています。

| 章番号 | ファイル名 | 主要テーマ・内容 | 学習目標 |
| :--- | :--- | :--- | :--- |
| **第1章** | [`01_foundations_qubo_and_ising.md`](./01_foundations_qubo_and_ising.md) | **組合せ最適化とイジング模型・断熱量子計算 (AQC)** | 古典問題を物理のスピン模型に変換する手法と、QAOAの原型である断熱時間発展を理解する。 |
| **第2章** | [`02_farhi_qaoa_original_paper.md`](./02_farhi_qaoa_original_paper.md) | **QAOA原論文（Farhi et al., 2014）完全徹底解読** | 原論文（arXiv:1411.4028）の数式展開、MaxCut近似比0.6924の導出、変分ハイブリッドループを習得する。 |
| **第3章** | [`03_penalty_limits_and_constraint_break.md`](./03_penalty_limits_and_constraint_break.md) | **ペナルティ法の限界と「制約の壁」** | 実問題で必須となるOne-Hot制約等に対し、なぜ従来のペナルティ法（Standard QAOA）が破綻するのかを数理的に解明する。 |
| **第4章** | [`04_hadfield_xy_mixer_paper.md`](./04_hadfield_xy_mixer_paper.md) | **XYミキサー原著（Hadfield et al., 2017/2019）完全解読** | QAO Ansatzの原著論文（Algorithms 2019）を読解し、部分空間探索、対称性保存、W状態初期化を完全マスターする。 |
| **第5章** | [`05_wang_analytical_results_xy_mixer.md`](./05_wang_analytical_results_xy_mixer.md) | **XYミキサーの数理解析とNISQ実装（Wang et al., 2020）** | 論文（Phys. Rev. A, 2020）に基づき、XYミキサーのゲート分解、リング型トポロジー、パラメータ対称性を理解する。 |
| **第6章** | [`06_hands_on_qiskit_implementation.py`](./06_hands_on_qiskit_implementation.py) | **実践: Qiskitで組むStandard QAOA vs XY-QAOA** | 実際に動くPythonスクリプト。W状態回路、XYミキサーゲートの実装、サンプリング結果の可視化を体験する。 |
| **第7章** | [`07_exercises_and_solutions.md`](./07_exercises_and_solutions.md) | **理解度チェック演習問題（解答・解説付き）** | 交換関係の証明、行列指数関数の計算、近似比計算などの腕試し問題集。 |

---

## 参照する基幹論文（Academic References）

本教材は、以下の査読付き学術論文・原著論文の厳密な数式と理論構成を直接下敷きにして執筆されています：

1. **[Farhi 2014 原論文]**
   Edward Farhi, Jeffrey Goldstone, Sam Gutmann, 
   *"A Quantum Approximate Optimization Algorithm"*, 
   arXiv:1411.4028 (2014).
2. **[Hadfield 2019 XYミキサー基礎論文]**
   Stuart Hadfield, Zhihui Wang, Bryan O'Gorman, Eleanor G. Rieffel, Davide Venturelli, Rupak Biswas, 
   *"From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz"*, 
   *Algorithms*, 12(2), 34 (2019) / arXiv:1709.03489.
3. **[Wang 2020 XYミキサー解析論文]**
   Zhihui Wang, Nicholas C. Rubin, Jason M. Dominy, Eleanor G. Rieffel, 
   *"XY-mixers: Analytical and numerical results for QAOA"*, 
   *Physical Review A*, 101, 012320 (2020) / arXiv:1904.09314.
4. **[Cook 2020 平面グラフ実装論文]**
   Jeremy Cook, Stephan Eidenbenz, Andreas Bärtschi, 
   *"The Quantum Alternating Operator Ansatz on a Planar Graph"*, 
   *IEEE High Performance Extreme Computing Conference (HPEC)* (2020).
5. **[Kitai 2020 FMQA先行研究]**
   Koki Kitai, Jiang Guo, Shenghao Ju, Shu Tanaka, Koji Tsuda, 
   *"Designing Metamaterials with Quantum Annealing and Factorization Machines"*, 
   *Physical Review Research*, 2(1), 013319 (2020).

---

## 推奨される学習の進め方
1. **ステップ 1**: まず `01_foundations_qubo_and_ising.md` と `02_farhi_qaoa_original_paper.md` を読み、無制約QAOAの基本原理と原論文の思想を把握します。
2. **ステップ 2**: `03_penalty_limits_and_constraint_break.md` で、実問題に直面した際の「ペナルティ法の壁」を実感します。
3. **ステップ 3**: `04_hadfield_xy_mixer_paper.md` と `05_wang_analytical_results_xy_mixer.md` を精読し、XYミキサーがどのように制約を克服するのかを数理的に解き明かします。
4. **ステップ 4**: `06_hands_on_qiskit_implementation.py` を実行して、シミュレータ上で両者の挙動の違いを目撃します。
5. **ステップ 5**: `07_exercises_and_solutions.md` の演習問題を解き、数式導出力を定着させます。
