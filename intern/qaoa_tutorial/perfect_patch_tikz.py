#!/usr/bin/env python3
"""
Final perfection patch for all TikZ figures in qaoa_tutorial.tex.
"""

from pathlib import Path

tex_path = Path("/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/qaoa_tutorial/qaoa_tutorial.tex")
content = tex_path.read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# 1. Perfect Cover Page (Titlepage)
# -----------------------------------------------------------------------------
# Ensure the scope shift is exactly centered relative to current page.north
new_cover_scope = r"""    % TikZ による量子回路・幾何学アートワーク（中央白地エリアに正確に配置）
    \begin{scope}[shift={([yshift=-175mm]current page.north)}]
        % 背景装飾サークル
        \draw[dashed, booksecondary!25, line width=1.2pt] (0, 0) circle (3.6cm);
        \draw[dotted, bookgold!35, line width=1.5pt] (0, 0) circle (4.2cm);

        % 量子回路ワイヤー
        \draw[very thick, booksecondary!50] (-4.8, 1.1) -- (4.8, 1.1);
        \draw[very thick, booksecondary!50] (-4.8, 0.0) -- (4.8, 0.0);
        \draw[very thick, booksecondary!50] (-4.8, -1.1) -- (4.8, -1.1);

        % ゲート装飾 H
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.0, 0.75) rectangle (-3.1, 1.45) node[midway, font=\bfseries\color{booksecondary}] {$H$};
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.0, -0.35) rectangle (-3.1, 0.35) node[midway, font=\bfseries\color{booksecondary}] {$H$};
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.0, -1.45) rectangle (-3.1, -0.75) node[midway, font=\bfseries\color{booksecondary}] {$H$};

        % 相互作用 コスト層
        \draw[fill=bookaccent!18, draw=bookaccent!80, thick, rounded corners=2mm] (-2.0, -1.4) rectangle (0.2, 1.4) node[midway, font=\bfseries\color{bookaccent}, align=center] {コスト層\\$e^{-i\gamma C}$};

        % ミキサー層
        \draw[fill=bookprimary!18, draw=bookprimary!80, thick, rounded corners=2mm] (1.2, -1.4) rectangle (3.4, 1.4) node[midway, font=\bfseries\color{bookprimary}, align=center] {ミキサー層\\$e^{-i\beta B}$};

        % CNOT
        \filldraw[bookprimary] (4.2, 1.1) circle (3pt);
        \draw[thick, bookprimary] (4.2, 1.1) -- (4.2, 0.0);
        \draw[thick, fill=white, draw=bookprimary] (4.2, 0.0) circle (5pt);
        \draw[thick, bookprimary] (4.2, -0.18) -- (4.2, 0.18);
        \draw[thick, bookprimary] (4.02, 0.0) -- (4.38, 0.0);
    \end{scope}"""

import re
content = re.sub(
    r"    % TikZ による量子回路・幾何学アートワーク.*?\\end\{scope\}",
    lambda m: new_cover_scope,
    content,
    flags=re.DOTALL
)

# -----------------------------------------------------------------------------
# 2. Perfect Fig 2.1 (Single Qubit Gates)
# -----------------------------------------------------------------------------
new_fig21 = r"""\begin{tikzpicture}[scale=0.95, >=latex]
    % X Gate
    \node[left] at (0.2, 2.5) {$\ket{\psi}$};
    \draw[thick] (0.2, 2.5) -- (2.8, 2.5) node[right] {$X\ket{\psi}$};
    \draw[fill=blue!12, draw=blue!75!black, thick] (1.0, 2.05) rectangle (2.0, 2.95) node[midway, font=\bfseries\large] {$X$};
    \node[below, font=\footnotesize, text width=3.2cm, align=center] at (1.5, 1.8) {NOTゲート\\($\ket{0} \leftrightarrow \ket{1}$ 反転)};

    % H Gate
    \node[left] at (5.4, 2.5) {$\ket{0}$};
    \draw[thick] (5.4, 2.5) -- (7.8, 2.5) node[right] {$\ket{+}$};
    \draw[fill=green!12, draw=green!65!black, thick] (6.1, 2.05) rectangle (7.1, 2.95) node[midway, font=\bfseries\large] {$H$};
    \node[below, font=\footnotesize, text width=3.4cm, align=center] at (6.6, 1.8) {アダマールゲート\\($\ket{+} = \frac{\ket{0}+\ket{1}}{\sqrt{2}}$ 生成)};

    % Rz Gate
    \node[left] at (10.6, 2.5) {$\ket{\psi}$};
    \draw[thick] (10.6, 2.5) -- (13.0, 2.5) node[right] {$R_Z(\theta)\ket{\psi}$};
    \draw[fill=orange!12, draw=orange!75!black, thick] (11.3, 2.05) rectangle (12.3, 2.95) node[midway, font=\bfseries\small] {$R_Z(\theta)$};
    \node[below, font=\footnotesize, text width=3.4cm, align=center] at (11.8, 1.8) {Z回転ゲート\\(位相差 $e^{-i\theta/2}$ 付加)};
\end{tikzpicture}"""

content = re.sub(
    r"\\begin\{figure\}\[htbp\]\s*\\centering\s*\\begin\{tikzpicture\}\[scale=0\.95, >=latex\].*?\\caption\{代表的な1量子ビットゲート",
    lambda m: "\\begin{figure}[htbp]\n\\centering\n" + new_fig21 + "\n\\caption{代表的な1量子ビットゲート",
    content,
    flags=re.DOTALL
)

# -----------------------------------------------------------------------------
# 3. Perfect Fig 3.1 (Avoided Level Crossing)
# -----------------------------------------------------------------------------
new_fig31 = r"""\begin{tikzpicture}[scale=1.05, >=latex]
    % Axes
    \draw[->, thick] (0, 0) -- (9.0, 0) node[right, font=\small] {時間進行 $t / T$ ($0 \to 1$)};
    \draw[->, thick] (0, 0) -- (0, 5.2) node[above, font=\small] {エネルギー $E(t)$};

    % Energy levels with avoided crossing
    % Excited state curve
    \draw[very thick, red!80!black] (0.8, 4.4) .. controls (3.0, 3.8) and (4.0, 2.9) .. (4.5, 2.8) .. controls (5.0, 2.9) and (6.0, 3.8) .. (8.2, 4.5);
    \node[above right, font=\small\bfseries, text=red!80!black] at (8.2, 4.4) {第1励起状態 $E_1(t)$};

    % Ground state curve
    \draw[very thick, blue!80!black] (0.8, 0.9) .. controls (3.0, 1.2) and (4.0, 1.9) .. (4.5, 2.0) .. controls (5.0, 1.9) and (6.0, 1.1) .. (8.2, 0.8);
    \node[above right, font=\small\bfseries, text=blue!80!black] at (8.2, 0.8) {基底状態 $E_0(t)$ (最適解 $\ket{x^*}$)};

    % Initial state label with helper line
    \node[above, font=\footnotesize\bfseries, text=blue!75!black] at (1.2, 2.2) {初期基底状態 $\ket{s} = \ket{+}^{\otimes n}$};
    \draw[->, thin, blue!70!black] (1.2, 2.1) -- (1.2, 1.1);

    % Phase transition point dashed line
    \draw[dashed, gray!80] (4.5, 0) -- (4.5, 2.0);
    \node[below, font=\footnotesize] at (4.5, 0) {$t^*$ (相転移点)};

    % Minimum gap arrow and label
    \draw[<->, very thick, purple] (4.5, 2.05) -- (4.5, 2.75);
    \node[right, font=\footnotesize\bfseries, text=purple] at (4.65, 2.4) {最小ギャップ $\Delta_{\min}$};

    % Landau-Zener non-adiabatic transition arrow
    \draw[->, very thick, dashed, red!70!black] (3.7, 1.7) -- (5.3, 3.1)
        node[pos=0.75, above left, font=\scriptsize\bfseries, text=red!75!black, fill=white, inner sep=1pt] {非断熱遷移（局所解へ脱落）};

    % Bottom warning note
    \node[below, font=\footnotesize, text width=8cm, align=center, text=booktext] at (4.5, -0.6) {
        $T$ が不足すると狭いボトルネック $\Delta_{\min}$ で励起状態へ跳び移り探索が失敗する
    };
\end{tikzpicture}"""

content = re.sub(
    r"\\begin\{figure\}\[htbp\]\s*\\centering\s*\\begin\{tikzpicture\}\[scale=1\.05, >=latex\].*?\\caption\{断熱量子計算における準位の反交差",
    lambda m: "\\begin{figure}[htbp]\n\\centering\n" + new_fig31 + "\n\\caption{断熱量子計算における準位の反交差",
    content,
    flags=re.DOTALL
)

# -----------------------------------------------------------------------------
# 4. Perfect Fig 4.2 (VQA Loop)
# -----------------------------------------------------------------------------
new_fig42 = r"""\begin{tikzpicture}[scale=0.92, >=latex]
    % QPU Box
    \draw[fill=cyan!8, draw=cyan!80!black, thick, rounded corners=3mm] (0, 0) rectangle (5.8, 4.3);
    \node[font=\bfseries\large, text=cyan!80!black] at (2.9, 3.85) {量子プロセッサ (QPU)};
    \node[font=\footnotesize, align=center] at (2.9, 2.7) {
        \textbf{量子回路の実行}\\
        $\ket{\vec{\gamma}, \vec{\beta}} = \prod_{k=1}^p U_B(\beta_k) U_C(\gamma_k) \ket{s}$
    };
    \node[font=\footnotesize, align=center] at (2.9, 1.1) {
        \textbf{計算基底で高速サンプリング}\\
        ($1,000 \sim 10,000$ shots)\\
        測定サンプル列 $\{z^{(1)}, z^{(2)}, \dots\}$ を出力
    };

    % CPU Box
    \draw[fill=orange!8, draw=orange!85!black, thick, rounded corners=3mm] (8.2, 0) rectangle (14.0, 4.3);
    \node[font=\bfseries\large, text=orange!85!black] at (11.1, 3.85) {古典コンピュータ (CPU)};
    \node[font=\footnotesize, align=center] at (11.1, 2.7) {
        \textbf{期待値（エネルギー）の推定}\\
        $F_p(\vec{\gamma}, \vec{\beta}) \approx \frac{1}{K}\sum_{k=1}^K C(z^{(k)})$
    };
    \node[font=\footnotesize, align=center] at (11.1, 1.1) {
        \textbf{古典オプティマイザ (COBYLA/SPSA)}\\
        変分パラメータ更新:\\
        $(\vec{\gamma}, \vec{\beta}) \leftarrow (\vec{\gamma}', \vec{\beta}')$
    };

    % Clean, well-spaced communication arrows
    \draw[->, very thick, purple!80!black] (5.8, 2.9) -- (8.2, 2.9)
        node[midway, above, font=\footnotesize\bfseries, text=purple!85!black] {測定サンプル列};
    \draw[<-, very thick, blue!80!black] (5.8, 1.2) -- (8.2, 1.2)
        node[midway, above, font=\footnotesize\bfseries, text=blue!85!black] {新パラメータ $(\vec{\gamma}, \vec{\beta})$};
\end{tikzpicture}"""

content = re.sub(
    r"\\begin\{figure\}\[htbp\]\s*\\centering\s*\\begin\{tikzpicture\}\[scale=0\.92, >=latex\].*?\\caption\{QAOAにおける量子・古典ハイブリッド最適化ループ",
    lambda m: "\\begin{figure}[htbp]\n\\centering\n" + new_fig42 + "\n\\caption{QAOAにおける量子・古典ハイブリッド最適化ループ",
    content,
    flags=re.DOTALL
)

# -----------------------------------------------------------------------------
# 5. Perfect Fig 6.2 (Subspace Comparison)
# -----------------------------------------------------------------------------
new_fig62 = r"""\begin{tikzpicture}[scale=0.88, >=latex]
    % Left side: Standard QAOA
    \draw[fill=red!3, draw=red!70!black, thick, rounded corners=2mm] (0, 0) rectangle (6.2, 5.2);
    % Title bar
    \draw[fill=red!15, draw=red!70!black, thick, rounded corners=2mm] (0, 4.4) rectangle (6.2, 5.2)
        node[midway, font=\bfseries\small, text=red!85!black] {Standard QAOA (全空間探索)};
    
    % Gray dashed invalid space
    \draw[fill=gray!12, draw=gray!60, dashed] (0.5, 0.4) rectangle (5.7, 4.1);
    \node[font=\scriptsize\bfseries, text=gray!80!black] at (3.1, 3.8) {全状態空間 $2^N$ (指数爆発)};
    
    % Feasible solution dots
    \filldraw[blue!75!black] (1.6, 1.2) circle (3pt) node[below, font=\scriptsize] {$\ket{1001}$};
    \filldraw[blue!75!black] (4.6, 1.2) circle (3pt) node[below, font=\scriptsize] {$\ket{0110}$};
    \filldraw[blue!75!black] (1.6, 2.7) circle (3pt) node[above, font=\scriptsize] {$\ket{1010}$};
    \filldraw[red!85!black] (4.6, 2.7) circle (4pt) node[above, font=\scriptsize\bfseries] {$\ket{0101}^*$};

    % Wasteful dispersal arrows
    \draw[->, thick, gray!70, dashed] (3.1, 2.0) -- (1.0, 1.9) node[pos=1.0, left, font=\tiny, text=red!70!black] {違反 $\ket{0000}$};
    \draw[->, thick, gray!70, dashed] (3.1, 2.0) -- (5.2, 1.9) node[pos=1.0, right, font=\tiny, text=red!70!black] {違反 $\ket{1111}$};

    \node[below, font=\footnotesize, text width=6.0cm, align=center] at (3.1, -0.2) {
        ペナルティ障壁により波動関数が無効空間（$99\%$以上）に散逸
    };

    % Right side: FM-XY-QAOA
    \draw[fill=green!3, draw=green!65!black, thick, rounded corners=2mm] (7.2, 0) rectangle (13.4, 5.2);
    % Title bar
    \draw[fill=green!15, draw=green!65!black, thick, rounded corners=2mm] (7.2, 4.4) rectangle (13.4, 5.2)
        node[midway, font=\bfseries\small, text=green!75!black] {FM-XY-QAOA (部分空間探索)};

    % Subspace Ring
    \draw[thick, blue!65!black, dashed] (10.3, 2.1) circle (1.4cm);
    \node[font=\scriptsize\bfseries, text=blue!75!black] at (10.3, 2.1) {許容部分空間 $\mathcal{H}_{\text{feas}}$};

    % Feasible dots along ring
    \filldraw[blue!75!black] (8.9, 2.1) circle (3pt) node[left, font=\scriptsize] {$\ket{1001}$};
    \filldraw[blue!75!black] (11.7, 2.1) circle (3pt) node[right, font=\scriptsize] {$\ket{0110}$};
    \filldraw[blue!75!black] (10.3, 3.5) circle (3pt) node[above, font=\scriptsize] {$\ket{1010}$};
    \filldraw[red!85!black] (10.3, 0.7) circle (4pt) node[below, font=\scriptsize\bfseries] {$\ket{0101}^*$};

    % XY mixer rotation arrows
    \draw[->, very thick, green!60!black] (9.1, 2.4) arc (155:115:1.4cm) node[midway, above left, font=\tiny\bfseries] {XY};
    \draw[->, very thick, green!60!black] (10.6, 3.5) arc (65:25:1.4cm);
    \draw[->, very thick, green!60!black] (11.5, 1.8) arc (-25:-65:1.4cm);
    \draw[->, very thick, green!60!black] (10.0, 0.7) arc (-115:-155:1.4cm);

    \node[below, font=\footnotesize, text width=6.0cm, align=center] at (10.3, -0.2) {
        無効空間を物理排除し、有効解の間だけをトンネリング遷移
    };
\end{tikzpicture}"""

content = re.sub(
    r"\\begin\{figure\}\[htbp\]\s*\\centering\s*\\begin\{tikzpicture\}\[scale=0\.88, >=latex\].*?\\caption\{探索空間の幾何学的対比",
    lambda m: "\\begin{figure}[htbp]\n\\centering\n" + new_fig62 + "\n\\caption{探索空間の幾何学的対比",
    content,
    flags=re.DOTALL
)

tex_path.write_text(content, encoding="utf-8")
print("All TikZ figures perfectly updated in", tex_path)
