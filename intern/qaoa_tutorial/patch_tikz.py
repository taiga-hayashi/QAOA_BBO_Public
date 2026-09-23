#!/usr/bin/env python3
"""
Fix all TikZ overlap and clipping issues in qaoa_tutorial.tex.
"""

import re
from pathlib import Path

tex_path = Path("/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/qaoa_tutorial/qaoa_tutorial.tex")
content = tex_path.read_text(encoding="utf-8")

# -----------------------------------------------------------------------------
# 1. Fix Titlepage (Cover Page)
# -----------------------------------------------------------------------------
old_cover = r"""\begin{titlepage}
\begin{tikzpicture}[remember picture, overlay]
    % 背景グラデーション風の帯
    \fill[bookprimary] (current page.north west) rectangle ([yshift=-85mm]current page.north east);
    \fill[booksecondary] ([yshift=-85mm]current page.north west) rectangle ([yshift=-90mm]current page.north east);
    \fill[bookgold] ([yshift=-90mm]current page.north west) rectangle ([yshift=-92mm]current page.north east);

    % シリーズ名
    \node[anchor=north west, font=\gtfamily\sffamily\bfseries\color{white!85}, yshift=-25mm, xshift=25mm] at (current page.north west) {
        \large アドバンスト量子コンピューティング教科書シリーズ 第1巻
    };

    % メインタイトル
    \node[anchor=north west, font=\gtfamily\bfseries\fontsize{28}{36}\selectfont\color{white}, yshift=-37mm, xshift=25mm] at (current page.north west) {
        量子近似最適化アルゴリズム\\（QAOA）体系的教科書
    };

    % サブタイトル
    \node[anchor=north west, font=\gtfamily\bfseries\large\color{white!90}, yshift=-64mm, xshift=25mm] at (current page.north west) {
        ― 基礎数理・回路図解から原論文完全解読、XYミキサー、古典角度最適化まで ―
    };

    % TikZ による量子回路・ブロッホ球の幾何学アートワーク
    \begin{scope}[xshift=105mm, yshift=65mm]
        % 量子回路ワイヤー
        \draw[very thick, booksecondary!40] (-6, 0) -- (6, 0);
        \draw[very thick, booksecondary!40] (-6, -1.2) -- (6, -1.2);
        \draw[very thick, booksecondary!40] (-6, -2.4) -- (6, -2.4);

        % ゲート装飾
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.5, -0.4) rectangle (-3.5, 0.4) node[midway, font=\bfseries\color{booksecondary}] {$H$};
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.5, -1.6) rectangle (-3.5, -0.8) node[midway, font=\bfseries\color{booksecondary}] {$H$};
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.5, -2.8) rectangle (-3.5, -2.0) node[midway, font=\bfseries\color{booksecondary}] {$H$};

        % 相互作用
        \draw[fill=bookaccent!20, draw=bookaccent!80, thick, rounded corners=2mm] (-2.2, -2.7) rectangle (0.2, 0.3) node[midway, font=\bfseries\color{bookaccent}, align=center] {コスト層\\$e^{-i\gamma C}$};

        \draw[fill=bookprimary!20, draw=bookprimary!80, thick, rounded corners=2mm] (1.5, -2.7) rectangle (3.9, 0.3) node[midway, font=\bfseries\color{bookprimary}, align=center] {ミキサー層\\$e^{-i\beta B}$};

        % CNOT
        \filldraw[bookprimary] (4.8, 0) circle (3pt);
        \draw[thick, bookprimary] (4.8, 0) -- (4.8, -1.2);
        \draw[thick, fill=white, draw=bookprimary] (4.8, -1.2) circle (5pt);

        % 装飾サークル
        \draw[dashed, booksecondary!30, line width=1pt] (0, -1.2) circle (4.2cm);
        \draw[dotted, bookgold!40, line width=1.2pt] (0, -1.2) circle (4.8cm);
    \end{scope}

    % 下部：著者・出版情報
    \node[anchor=south west, font=\gtfamily\large\color{bookprimary}, yshift=35mm, xshift=25mm] at (current page.south west) {
        \textbf{著者・編者}: 量子コンピューティング・最適化アルゴリズム研究グループ\\
        \textbf{対象読者}: 大学部・大学院生、量子情報研究者、材料探索・最適化エンジニア
    };

    \node[anchor=south west, font=\small\color{booktext!70}, yshift=20mm, xshift=25mm] at (current page.south west) {
        発行: 2026年最新版（第1版） $\mid$ 完全収録版
    };

    \draw[line width=1.5pt, bookprimary!30] ([yshift=30mm, xshift=25mm]current page.south west) -- ([yshift=30mm, xshift=-25mm]current page.south east);
\end{tikzpicture}
\end{titlepage}"""

new_cover = r"""\begin{titlepage}
\begin{tikzpicture}[remember picture, overlay]
    % 背景グラデーション風の帯
    \fill[bookprimary] (current page.north west) rectangle ([yshift=-92mm]current page.north east);
    \fill[booksecondary] ([yshift=-92mm]current page.north west) rectangle ([yshift=-97mm]current page.north east);
    \fill[bookgold] ([yshift=-97mm]current page.north west) rectangle ([yshift=-99mm]current page.north east);

    % シリーズ名
    \node[anchor=north west, font=\gtfamily\sffamily\bfseries\color{white!85}, yshift=-22mm, xshift=22mm] at (current page.north west) {
        \large アドバンスト量子コンピューティング教科書シリーズ 第1巻
    };

    % メインタイトル
    \node[anchor=north west, font=\gtfamily\bfseries\fontsize{23}{31}\selectfont\color{white}, yshift=-34mm, xshift=22mm, text width=166mm, align=left] at (current page.north west) {
        量子近似最適化アルゴリズム（QAOA）体系的教科書
    };

    % サブタイトル
    \node[anchor=north west, font=\gtfamily\bfseries\large\color{white!90}, yshift=-58mm, xshift=22mm, text width=166mm, align=left] at (current page.north west) {
        ― 基礎数理・回路図解から原論文完全解読、XYミキサー、古典角度最適化まで ―
    };

    % TikZ による量子回路・幾何学アートワーク（中央白地エリアに正確に配置）
    \begin{scope}[xshift=105mm, yshift=100mm]
        % 背景装飾サークル
        \draw[dashed, booksecondary!25, line width=1.2pt] (0, 0) circle (3.8cm);
        \draw[dotted, bookgold!35, line width=1.5pt] (0, 0) circle (4.4cm);

        % 量子回路ワイヤー
        \draw[very thick, booksecondary!50] (-5.2, 1.2) -- (5.2, 1.2);
        \draw[very thick, booksecondary!50] (-5.2, 0.0) -- (5.2, 0.0);
        \draw[very thick, booksecondary!50] (-5.2, -1.2) -- (5.2, -1.2);

        % ゲート装飾 H
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.3, 0.8) rectangle (-3.3, 1.6) node[midway, font=\bfseries\color{booksecondary}] {$H$};
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.3, -0.4) rectangle (-3.3, 0.4) node[midway, font=\bfseries\color{booksecondary}] {$H$};
        \draw[fill=booksecondary!15, draw=booksecondary!80, thick, rounded corners=1mm] (-4.3, -1.6) rectangle (-3.3, -0.8) node[midway, font=\bfseries\color{booksecondary}] {$H$};

        % 相互作用 コスト層
        \draw[fill=bookaccent!18, draw=bookaccent!80, thick, rounded corners=2mm] (-2.2, -1.5) rectangle (0.2, 1.5) node[midway, font=\bfseries\color{bookaccent}, align=center] {コスト層\\$e^{-i\gamma C}$};

        % ミキサー層
        \draw[fill=bookprimary!18, draw=bookprimary!80, thick, rounded corners=2mm] (1.3, -1.5) rectangle (3.7, 1.5) node[midway, font=\bfseries\color{bookprimary}, align=center] {ミキサー層\\$e^{-i\beta B}$};

        % CNOT
        \filldraw[bookprimary] (4.5, 1.2) circle (3pt);
        \draw[thick, bookprimary] (4.5, 1.2) -- (4.5, 0.0);
        \draw[thick, fill=white, draw=bookprimary] (4.5, 0.0) circle (5pt);
        \draw[thick, bookprimary] (4.5, -0.18) -- (4.5, 0.18);
        \draw[thick, bookprimary] (4.32, 0.0) -- (4.68, 0.0);
    \end{scope}

    % 下部：著者・出版情報
    \node[anchor=south west, font=\gtfamily\normalsize\color{bookprimary}, yshift=35mm, xshift=22mm, text width=166mm, align=left] at (current page.south west) {
        \textbf{著者・編者}: 量子コンピューティング・最適化アルゴリズム研究グループ\\[2mm]
        \textbf{対象読者}: 大学部・大学院生、量子情報研究者、材料探索・最適化エンジニア
    };

    \node[anchor=south west, font=\small\color{booktext!70}, yshift=20mm, xshift=22mm, text width=166mm, align=left] at (current page.south west) {
        発行: 2026年最新版（第1版） $\mid$ 完全収録版
    };

    \draw[line width=1.2pt, bookprimary!30] ([yshift=30mm, xshift=22mm]current page.south west) -- ([yshift=30mm, xshift=-22mm]current page.south east);
\end{tikzpicture}
\end{titlepage}"""

content = content.replace(old_cover, new_cover)

# -----------------------------------------------------------------------------
# 2. Fix Fig 2.1 (Single Qubit Gates)
# -----------------------------------------------------------------------------
old_fig21 = r"""\begin{tikzpicture}[scale=1.0, >=latex]
    % X Gate
    \node at (-0.5, 2.5) {$\ket{\psi}$};
    \draw[thick] (0, 2.5) -- (2.5, 2.5) node[right] {$X\ket{\psi}$};
    \draw[fill=blue!10, draw=blue!70!black, thick] (0.8, 2.1) rectangle (1.7, 2.9) node[midway, font=\bfseries] {$X$};
    \node[below, font=\small, text width=3.5cm, align=center] at (1.25, 1.9) {NOTゲート\\($\ket{0} \leftrightarrow \ket{1}$ の反転)};

    % H Gate
    \node at (4.0, 2.5) {$\ket{0}$};
    \draw[thick] (4.5, 2.5) -- (7.0, 2.5) node[right] {$\ket{+} = \frac{\ket{0}+\ket{1}}{\sqrt{2}}$};
    \draw[fill=green!10, draw=green!60!black, thick] (5.2, 2.1) rectangle (6.3, 2.9) node[midway, font=\bfseries] {$H$};
    \node[below, font=\small, text width=3.5cm, align=center] at (5.75, 1.9) {アダマールゲート\\(均等な重ね合わせ作成)};

    % Rz Gate
    \node at (8.5, 2.5) {$\ket{\psi}$};
    \draw[thick] (9.0, 2.5) -- (11.5, 2.5) node[right] {$R_Z(\theta)\ket{\psi}$};
    \draw[fill=orange!10, draw=orange!70!black, thick] (9.6, 2.1) rectangle (10.9, 2.9) node[midway, font=\bfseries\small] {$R_Z(\theta)$};
    \node[below, font=\small, text width=3.5cm, align=center] at (10.25, 1.9) {Z回転ゲート\\(位相差 $e^{-i\theta/2}$ を付加)};
\end{tikzpicture}"""

new_fig21 = r"""\begin{tikzpicture}[scale=0.95, >=latex]
    % X Gate (x: 0 to 4)
    \node[left] at (0.2, 2.5) {$\ket{\psi}$};
    \draw[thick] (0.2, 2.5) -- (3.2, 2.5) node[right] {$X\ket{\psi}$};
    \draw[fill=blue!12, draw=blue!75!black, thick] (1.1, 2.05) rectangle (2.1, 2.95) node[midway, font=\bfseries\large] {$X$};
    \node[below, font=\footnotesize, text width=3.2cm, align=center] at (1.6, 1.85) {NOTゲート\\($\ket{0} \leftrightarrow \ket{1}$ 反転)};

    % H Gate (x: 5.2 to 9.2)
    \node[left] at (5.2, 2.5) {$\ket{0}$};
    \draw[thick] (5.2, 2.5) -- (8.0, 2.5) node[right] {$\ket{+} = \frac{\ket{0}+\ket{1}}{\sqrt{2}}$};
    \draw[fill=green!12, draw=green!65!black, thick] (6.0, 2.05) rectangle (7.0, 2.95) node[midway, font=\bfseries\large] {$H$};
    \node[below, font=\footnotesize, text width=3.4cm, align=center] at (6.5, 1.85) {アダマールゲート\\(均等重ね合わせ生成)};

    % Rz Gate (x: 11.2 to 14.8)
    \node[left] at (11.0, 2.5) {$\ket{\psi}$};
    \draw[thick] (11.0, 2.5) -- (13.6, 2.5) node[right] {$R_Z(\theta)\ket{\psi}$};
    \draw[fill=orange!12, draw=orange!75!black, thick] (11.8, 2.05) rectangle (12.9, 2.95) node[midway, font=\bfseries\small] {$R_Z(\theta)$};
    \node[below, font=\footnotesize, text width=3.4cm, align=center] at (12.35, 1.85) {Z回転ゲート\\(位相 $e^{-i\theta/2}$ 付加)};
\end{tikzpicture}"""

content = content.replace(old_fig21, new_fig21)

# -----------------------------------------------------------------------------
# 3. Fix Fig 2.2 (CNOT Gate)
# -----------------------------------------------------------------------------
old_fig22 = r"""\begin{tikzpicture}[scale=1.0, >=latex]
    % Control wire
    \node at (-0.8, 1) {$\ket{c}$};
    \draw[thick] (-0.3, 1) -- (3.5, 1) node[right] {$\ket{c}$ (不変)};
    % Target wire
    \node at (-0.8, 0) {$\ket{t}$};
    \draw[thick] (-0.3, 0) -- (3.5, 0) node[right] {$\ket{t \oplus c}$ (反転)};
    
    % Control dot
    \filldraw[black] (1.6, 1) circle (3pt);
    % Vertical line
    \draw[thick] (1.6, 1) -- (1.6, 0);
    % Target circle with cross
    \draw[thick, fill=white] (1.6, 0) circle (6pt);
    \draw[thick] (1.6, -0.21) -- (1.6, 0.21);
    \draw[thick] (1.39, 0) -- (1.81, 0);
    
    \node[right, font=\small, text width=7cm] at (4.5, 0.5) {
        $\ket{00} \to \ket{00}$ \\
        $\ket{01} \to \ket{01}$ \\
        $\ket{10} \to \ket{11}$ \quad ($\ket{c}=1$ なので標的が反転) \\
        $\ket{11} \to \ket{10}$ \quad ($\ket{c}=1$ なので標的が反転)
    };
\end{tikzpicture}"""

new_fig22 = r"""\begin{tikzpicture}[scale=1.0, >=latex]
    % Control wire
    \node[left] at (-0.2, 1) {$\ket{c}$};
    \draw[thick] (-0.2, 1) -- (2.6, 1) node[right] {$\ket{c}$};
    % Target wire
    \node[left] at (-0.2, 0) {$\ket{t}$};
    \draw[thick] (-0.2, 0) -- (2.6, 0) node[right] {$\ket{t \oplus c}$};
    
    % Control dot
    \filldraw[black] (1.3, 1) circle (3pt);
    % Vertical line
    \draw[thick] (1.3, 1) -- (1.3, 0);
    % Target circle with cross
    \draw[thick, fill=white] (1.3, 0) circle (6pt);
    \draw[thick] (1.3, -0.21) -- (1.3, 0.21);
    \draw[thick] (1.09, 0) -- (1.51, 0);
    
    % Explanatory Truth Table
    \node[right, font=\small, text width=8cm] at (4.6, 0.5) {
        $\ket{00} \to \ket{00}$ \quad (制御=0: 標的不変) \\
        $\ket{01} \to \ket{01}$ \quad (制御=0: 標的不変) \\
        $\ket{10} \to \ket{11}$ \quad (\textbf{制御=1: 標的反転}) \\
        $\ket{11} \to \ket{10}$ \quad (\textbf{制御=1: 標的反転})
    };
\end{tikzpicture}"""

content = content.replace(old_fig22, new_fig22)

# -----------------------------------------------------------------------------
# 4. Fix Fig 3.1 (Avoided Level Crossing)
# -----------------------------------------------------------------------------
old_fig31 = r"""\begin{tikzpicture}[scale=1.1, >=latex]
    % Axes
    \draw[->, thick] (0, 0) -- (8.5, 0) node[right, font=\small] {時間進行 $t / T$ ($0 \to 1$)};
    \draw[->, thick] (0, 0) -- (0, 5.0) node[above, font=\small] {エネルギー $E(t)$};

    % Energy levels with avoided crossing
    % Excited state
    \draw[very thick, red!80!black] (0.5, 4.2) .. controls (3.0, 3.5) and (3.8, 2.5) .. (4.25, 2.45) .. controls (4.7, 2.5) and (5.5, 3.8) .. (8.0, 4.4);
    \node[above, font=\small\bfseries, text=red!80!black] at (8.0, 4.4) {第1励起状態 $E_1(t)$};

    % Ground state
    \draw[very thick, blue!80!black] (0.5, 0.8) .. controls (3.0, 1.2) and (3.8, 2.1) .. (4.25, 2.15) .. controls (4.7, 2.1) and (5.5, 1.0) .. (8.0, 0.6);
    \node[below, font=\small\bfseries, text=blue!80!black] at (8.0, 0.6) {基底状態 $E_0(t)$ (大域的最適解 $\ket{x^*}$)};

    % Minimum gap arrow
    \draw[<->, very thick, purple] (4.25, 2.17) -- (4.25, 2.43);
    \node[right, font=\footnotesize\bfseries, text=purple] at (4.3, 2.3) {最小ギャップ $\Delta_{\min}$};
    \draw[dashed, gray] (4.25, 0) -- (4.25, 2.17) node[below, font=\footnotesize, text=black] {$t^*$ (相転移点)};

    % Landau-Zener non-adiabatic transition arrow
    \draw[->, very thick, dashed, red!70!black] (4.0, 1.95) -- (4.5, 2.65) node[midway, above left, font=\scriptsize\bfseries, text=red!70!black] {非断熱遷移（誤った局所解へ脱落）};

    % Labels
    \node[above, font=\small] at (1.0, 0.8) {初期基底状態 $\ket{s} = \ket{+}^{\otimes n}$};
    \node[below, font=\footnotesize, text width=7cm, align=center] at (4.25, -0.6) {
        変化が速すぎる（$T$ が不足）と、狭いボトルネック $\Delta_{\min}$ で\\
        励起状態へと飛び移り、探索が破綻する
    };
\end{tikzpicture}"""

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
    \node[above, font=\footnotesize\bfseries, text=blue!75!black] at (1.8, 1.8) {初期基底状態 $\ket{s} = \ket{+}^{\otimes n}$};
    \draw[->, thin, blue!70!black] (1.8, 1.7) -- (1.5, 1.05);

    % Phase transition point dashed line
    \draw[dashed, gray!80] (4.5, 0) -- (4.5, 2.0);
    \node[below, font=\footnotesize] at (4.5, 0) {$t^*$ (相転移点)};

    % Minimum gap arrow and label (distinctly separated)
    \draw[<->, very thick, purple] (4.5, 2.05) -- (4.5, 2.75);
    \node[right, font=\footnotesize\bfseries, text=purple] at (4.65, 2.4) {最小ギャップ $\Delta_{\min}$};

    % Landau-Zener non-adiabatic transition arrow
    \draw[->, very thick, dashed, red!70!black] (3.8, 1.7) -- (5.2, 3.1)
        node[pos=0.8, above left, font=\scriptsize\bfseries, text=red!75!black, fill=white, inner sep=1pt] {非断熱遷移（局所解へ脱落）};

    % Bottom warning note
    \node[below, font=\footnotesize, text width=8cm, align=center, text=booktext] at (4.5, -0.6) {
        $T$ が不足すると狭いボトルネック $\Delta_{\min}$ で励起状態へ跳び移り探索が失敗する
    };
\end{tikzpicture}"""

content = content.replace(old_fig31, new_fig31)

# -----------------------------------------------------------------------------
# 5. Fix Fig 4.1 (QAOA Circuit)
# -----------------------------------------------------------------------------
old_fig41 = r"""\begin{tikzpicture}[scale=0.9, >=latex]
    % Wires
    \foreach \y/\lab in {3/{q_0}, 2/{q_1}, 0.5/{q_{n-1}}} {
        \node at (-0.8, \y) {$\ket{0}$};
        \draw[thick] (-0.3, \y) -- (13.5, \y);
    }
    \node at (-0.8, 1.3) {$\vdots$};
    \node at (6.8, 1.3) {$\vdots$};
    \node at (13.8, 1.3) {$\vdots$};

    % Initial Hadamard layer
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=green!15, draw=green!60!black, thick] (0.2, \y-0.35) rectangle (0.9, \y+0.35) node[midway, font=\bfseries] {$H$};
    }
    \draw[dashed, red!60!black, thick] (0.0, -0.1) rectangle (1.1, 3.6);
    \node[above, font=\small\bfseries, text=red!60!black] at (0.55, 3.6) {初期化 $\ket{s}$};

    % Layer 1: Cost unitary
    \draw[fill=purple!15, draw=purple!70!black, thick] (1.6, 0.1) rectangle (3.2, 3.4) node[midway, font=\bfseries, align=center] {コスト層\\$e^{-i\gamma_1 C}$};
    
    % Layer 1: Mixer unitary
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=blue!15, draw=blue!70!black, thick] (3.8, \y-0.35) rectangle (5.3, \y+0.35) node[midway, font=\footnotesize\bfseries] {$R_X(2\beta_1)$};
    }
    \draw[dashed, blue!60!black, thick] (1.4, -0.1) rectangle (5.5, 3.6);
    \node[above, font=\small\bfseries, text=blue!60!black] at (3.45, 3.6) {第 1 レイヤー ($p=1$)};

    % Dots
    \node[font=\Large\bfseries] at (6.2, 2) {$\cdots$};
    \node[font=\Large\bfseries] at (6.2, 0.5) {$\cdots$};

    % Layer p: Cost unitary
    \draw[fill=purple!15, draw=purple!70!black, thick] (7.2, 0.1) rectangle (8.8, 3.4) node[midway, font=\bfseries, align=center] {コスト層\\$e^{-i\gamma_p C}$};

    % Layer p: Mixer unitary
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=blue!15, draw=blue!70!black, thick] (9.4, \y-0.35) rectangle (10.9, \y+0.35) node[midway, font=\footnotesize\bfseries] {$R_X(2\beta_p)$};
    }
    \draw[dashed, blue!60!black, thick] (7.0, -0.1) rectangle (11.1, 3.6);
    \node[above, font=\small\bfseries, text=blue!60!black] at (9.05, 3.6) {第 $p$ レイヤー};

    % Measurement
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=gray!20, draw=black, thick] (11.7, \y-0.35) rectangle (12.7, \y+0.35);
        \draw[thick] (11.9, \y-0.15) arc (180:0:3mm);
        \draw[thick, ->] (12.2, \y-0.15) -- (12.5, \y+0.2);
    }
    \node[above, font=\small\bfseries] at (12.2, 3.6) {サンプリング測定};
\end{tikzpicture}"""

new_fig41 = r"""\begin{tikzpicture}[scale=0.88, >=latex]
    % Wires
    \foreach \y/\lab in {3/{q_0}, 2/{q_1}, 0.5/{q_{n-1}}} {
        \node at (-0.6, \y) {$\ket{0}$};
        \draw[thick] (-0.2, \y) -- (13.2, \y);
    }
    \node at (-0.6, 1.25) {$\vdots$};
    \node at (6.2, 1.25) {$\vdots$};
    \node at (11.2, 1.25) {$\vdots$};

    % Initial Hadamard layer
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=green!15, draw=green!60!black, thick] (0.2, \y-0.35) rectangle (0.9, \y+0.35) node[midway, font=\bfseries] {$H$};
    }
    \draw[dashed, red!60!black, thick] (0.0, -0.1) rectangle (1.1, 3.6);
    \node[above, font=\small\bfseries, text=red!60!black] at (0.55, 3.6) {初期化 $\ket{s}$};

    % Layer 1: Cost unitary
    \draw[fill=purple!15, draw=purple!70!black, thick] (1.6, 0.1) rectangle (3.1, 3.4) node[midway, font=\bfseries, align=center] {コスト層\\$e^{-i\gamma_1 C}$};
    
    % Layer 1: Mixer unitary
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=blue!15, draw=blue!70!black, thick] (3.6, \y-0.35) rectangle (4.9, \y+0.35) node[midway, font=\scriptsize\bfseries] {$R_X(2\beta_1)$};
    }
    \draw[dashed, blue!60!black, thick] (1.4, -0.1) rectangle (5.1, 3.6);
    \node[above, font=\small\bfseries, text=blue!60!black] at (3.25, 3.6) {第 1 レイヤー ($p=1$)};

    % Dots
    \node[font=\Large\bfseries] at (5.7, 2) {$\cdots$};
    \node[font=\Large\bfseries] at (5.7, 0.5) {$\cdots$};

    % Layer p: Cost unitary
    \draw[fill=purple!15, draw=purple!70!black, thick] (6.7, 0.1) rectangle (8.2, 3.4) node[midway, font=\bfseries, align=center] {コスト層\\$e^{-i\gamma_p C}$};

    % Layer p: Mixer unitary
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=blue!15, draw=blue!70!black, thick] (8.7, \y-0.35) rectangle (10.0, \y+0.35) node[midway, font=\scriptsize\bfseries] {$R_X(2\beta_p)$};
    }
    \draw[dashed, blue!60!black, thick] (6.5, -0.1) rectangle (10.2, 3.6);
    \node[above, font=\small\bfseries, text=blue!60!black] at (8.35, 3.6) {第 $p$ レイヤー};

    % Measurement
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=gray!20, draw=black, thick] (11.7, \y-0.35) rectangle (12.7, \y+0.35);
        \draw[thick] (11.9, \y-0.15) arc (180:0:3mm);
        \draw[thick, ->] (12.2, \y-0.15) -- (12.5, \y+0.2);
    }
    \node[above, font=\small\bfseries] at (12.2, 3.6) {測定};
\end{tikzpicture}"""

content = content.replace(old_fig41, new_fig41)

# -----------------------------------------------------------------------------
# 6. Fix Fig 4.2 (VQA Hybrid Loop)
# -----------------------------------------------------------------------------
old_fig42 = r"""\begin{tikzpicture}[scale=0.95, >=latex]
    % QPU Box
    \draw[fill=cyan!10, draw=cyan!80!black, thick, rounded corners=3mm] (0, 0) rectangle (5.5, 3.5);
    \node[font=\bfseries\large, text=cyan!70!black] at (2.75, 3.0) {量子プロセッサ (QPU)};
    \node[font=\small, align=center] at (2.75, 1.8) {
        量子回路の実行\\
        $\ket{\vec{\gamma}, \vec{\beta}} = \prod U_B(\beta_k) U_C(\gamma_k) \ket{s}$
    };
    \node[font=\small, align=center] at (2.75, 0.7) {
        計算基底で測定 ($1,000 \sim 10,000$ shots)\\
        サンプル列 $\{z^{(1)}, z^{(2)}, \dots\}$ を取得
    };

    % CPU Box
    \draw[fill=orange!10, draw=orange!80!black, thick, rounded corners=3mm] (7.5, 0) rectangle (13.0, 3.5);
    \node[font=\bfseries\large, text=orange!80!black] at (10.25, 3.0) {古典コンピュータ (CPU)};
    \node[font=\small, align=center] at (10.25, 2.0) {
        期待値（エネルギー）の推定\\
        $F_p(\vec{\gamma}, \vec{\beta}) \approx \frac{1}{K}\sum_{k=1}^K C(z^{(k)})$
    };
    \node[font=\small, align=center] at (10.25, 0.8) {
        古典オプティマイザ (COBYLA / SPSA)\\
        パラメータ更新: $(\vec{\gamma}, \vec{\beta}) \leftarrow (\vec{\gamma}', \vec{\beta}')$
    };

    % Arrows
    \draw[->, very thick, purple!70!black] (5.5, 2.3) -- (7.5, 2.3) node[midway, above, font=\footnotesize\bfseries] {測定サンプル列};
    \draw[<-, very thick, blue!70!black] (5.5, 1.0) -- (7.5, 1.0) node[midway, below, font=\footnotesize\bfseries] {新パラメータ $(\vec{\gamma}, \vec{\beta})$};
\end{tikzpicture}"""

new_fig42 = r"""\begin{tikzpicture}[scale=0.92, >=latex]
    % QPU Box (Expanded height: y=0 to 4.2)
    \draw[fill=cyan!8, draw=cyan!80!black, thick, rounded corners=3mm] (0, 0) rectangle (5.6, 4.2);
    \node[font=\bfseries\large, text=cyan!80!black] at (2.8, 3.75) {量子プロセッサ (QPU)};
    \node[font=\footnotesize, align=center] at (2.8, 2.6) {
        \textbf{量子回路の実行}\\
        $\ket{\vec{\gamma}, \vec{\beta}} = \prod_{k=1}^p U_B(\beta_k) U_C(\gamma_k) \ket{s}$
    };
    \node[font=\footnotesize, align=center] at (2.8, 1.0) {
        \textbf{計算基底で高速サンプリング}\\
        ($1,000 \sim 10,000$ shots)\\
        サンプル列 $\{z^{(1)}, z^{(2)}, \dots\}$ を出力
    };

    % CPU Box (Expanded height: y=0 to 4.2)
    \draw[fill=orange!8, draw=orange!85!black, thick, rounded corners=3mm] (8.2, 0) rectangle (13.8, 4.2);
    \node[font=\bfseries\large, text=orange!85!black] at (11.0, 3.75) {古典コンピュータ (CPU)};
    \node[font=\footnotesize, align=center] at (11.0, 2.6) {
        \textbf{期待値（エネルギー）の推定}\\
        $F_p(\vec{\gamma}, \vec{\beta}) \approx \frac{1}{K}\sum_{k=1}^K C(z^{(k)})$
    };
    \node[font=\footnotesize, align=center] at (11.0, 1.0) {
        \textbf{古典オプティマイザ (COBYLA/SPSA)}\\
        パラメータ更新:\\
        $(\vec{\gamma}, \vec{\beta}) \leftarrow (\vec{\gamma}', \vec{\beta}')$
    };

    % Clean, non-overlapping communication arrows
    \draw[->, very thick, purple!80!black] (5.6, 2.8) -- (8.2, 2.8)
        node[midway, above, font=\footnotesize\bfseries, text=purple!85!black] {測定サンプル列};
    \draw[<-, very thick, blue!80!black] (5.6, 1.2) -- (8.2, 1.2)
        node[midway, below, font=\footnotesize\bfseries, text=blue!85!black] {新パラメータ $(\vec{\gamma}, \vec{\beta})$};
\end{tikzpicture}"""

content = content.replace(old_fig42, new_fig42)

# -----------------------------------------------------------------------------
# 7. Fix Fig 6.2 (Subspace Comparison)
# -----------------------------------------------------------------------------
old_fig62 = r"""\begin{tikzpicture}[scale=0.9, >=latex]
    % Left side: Standard QAOA
    \draw[fill=red!5, draw=red!60!black, thick, rounded corners=2mm] (0, 0) rectangle (6, 5);
    \node[font=\bfseries, text=red!70!black] at (3, 4.6) {Standard QAOA (全空間探索)};
    \draw[fill=gray!20, draw=gray, dashed] (0.5, 0.5) rectangle (5.5, 4.0);
    \node[font=\footnotesize, text=gray!80!black] at (3, 2.3) {全状態空間 $2^N$ (指数的爆発)};
    
    % Feasible dots
    \filldraw[blue!70!black] (1.5, 1.5) circle (3pt) node[below, font=\scriptsize] {$\ket{1001}$};
    \filldraw[blue!70!black] (4.5, 1.5) circle (3pt) node[below, font=\scriptsize] {$\ket{0110}$};
    \filldraw[blue!70!black] (1.5, 3.2) circle (3pt) node[above, font=\scriptsize] {$\ket{1010}$};
    \filldraw[red!80!black] (4.5, 3.2) circle (4pt) node[above, font=\scriptsize\bfseries] {$\ket{0101}^*$};

    % Wasteful arrows
    \draw[->, thick, gray, dashed] (3, 2.3) -- (0.8, 3.5) node[left, font=\tiny, text=red!70!black] {制約違反 $\ket{0000}$};
    \draw[->, thick, gray, dashed] (3, 2.3) -- (5.2, 0.8) node[right, font=\tiny, text=red!70!black] {制約違反 $\ket{1111}$};
    \node[below, font=\small, text width=5.5cm, align=center] at (3, -0.2) {
        ペナルティ障壁により波動関数が無効空間（$99\%$以上）に散逸
    };

    % Right side: FM-XY-QAOA
    \draw[fill=green!5, draw=green!60!black, thick, rounded corners=2mm] (7.5, 0) rectangle (13.5, 5);
    \node[font=\bfseries, text=green!60!black] at (10.5, 4.6) {FM-XY-QAOA (部分空間探索)};

    % Ring of feasible states
    \draw[thick, blue!60!black, dashed] (10.5, 2.3) circle (1.5cm);
    \node[font=\footnotesize, text=blue!70!black] at (10.5, 2.3) {許容部分空間 $\mathcal{H}_{\text{feas}}$};

    \filldraw[blue!70!black] (9.0, 2.3) circle (3pt) node[left, font=\scriptsize] {$\ket{1001}$};
    \filldraw[blue!70!black] (12.0, 2.3) circle (3pt) node[right, font=\scriptsize] {$\ket{0110}$};
    \filldraw[blue!70!black] (10.5, 3.8) circle (3pt) node[above, font=\scriptsize] {$\ket{1010}$};
    \filldraw[red!80!black] (10.5, 0.8) circle (4pt) node[below, font=\scriptsize\bfseries] {$\ket{0101}^*$};

    % XY mixer rotation arrows
    \draw[->, very thick, green!50!black] (9.2, 2.6) arc (160:110:1.5cm) node[midway, above left, font=\tiny\bfseries] {XYミキサー};
    \draw[->, very thick, green!50!black] (10.8, 3.8) arc (70:20:1.5cm);
    \draw[->, very thick, green!50!black] (11.8, 2.0) arc (-20:-70:1.5cm);
    \draw[->, very thick, green!50!black] (10.2, 0.8) arc (-110:-160:1.5cm);

    \node[below, font=\small, text width=5.5cm, align=center] at (10.5, -0.2) {
        無効空間を物理的に排除し、有効解の間だけをトンネリング遷移
    };
\end{tikzpicture}"""

new_fig62 = r"""\begin{tikzpicture}[scale=0.88, >=latex]
    % Left side: Standard QAOA
    \draw[fill=red!4, draw=red!65!black, thick, rounded corners=2mm] (0, 0) rectangle (6.2, 5.4);
    \node[font=\bfseries, text=red!75!black] at (3.1, 5.0) {Standard QAOA (全空間探索)};
    
    % Gray dashed invalid space
    \draw[fill=gray!15, draw=gray!70, dashed] (0.5, 0.6) rectangle (5.7, 4.4);
    \node[font=\scriptsize\bfseries, text=gray!80!black] at (3.1, 2.5) {全状態空間 $2^N$ (指数爆発)};
    
    % Feasible solution dots
    \filldraw[blue!75!black] (1.6, 1.4) circle (3pt) node[below, font=\scriptsize] {$\ket{1001}$};
    \filldraw[blue!75!black] (4.6, 1.4) circle (3pt) node[below, font=\scriptsize] {$\ket{0110}$};
    \filldraw[blue!75!black] (1.6, 3.6) circle (3pt) node[above, font=\scriptsize] {$\ket{1010}$};
    \filldraw[red!85!black] (4.6, 3.6) circle (4pt) node[above, font=\scriptsize\bfseries] {$\ket{0101}^*$};

    % Wasteful dispersal arrows
    \draw[->, thick, gray!80, dashed] (3.1, 2.5) -- (1.2, 2.7) node[pos=1.0, left, font=\tiny, text=red!70!black] {違反 $\ket{0000}$};
    \draw[->, thick, gray!80, dashed] (3.1, 2.5) -- (5.0, 2.3) node[pos=1.0, right, font=\tiny, text=red!70!black] {違反 $\ket{1111}$};

    \node[below, font=\footnotesize, text width=6.0cm, align=center] at (3.1, -0.2) {
        ペナルティ障壁により波動関数が無効空間（$99\%$以上）に散逸
    };

    % Right side: FM-XY-QAOA
    \draw[fill=green!4, draw=green!65!black, thick, rounded corners=2mm] (7.2, 0) rectangle (13.4, 5.4);
    \node[font=\bfseries, text=green!65!black] at (10.3, 5.0) {FM-XY-QAOA (部分空間探索)};

    % Subspace Ring
    \draw[thick, blue!65!black, dashed] (10.3, 2.4) circle (1.4cm);
    \node[font=\scriptsize\bfseries, text=blue!75!black] at (10.3, 2.4) {許容部分空間 $\mathcal{H}_{\text{feas}}$};

    % Feasible dots along ring
    \filldraw[blue!75!black] (8.9, 2.4) circle (3pt) node[left, font=\scriptsize] {$\ket{1001}$};
    \filldraw[blue!75!black] (11.7, 2.4) circle (3pt) node[right, font=\scriptsize] {$\ket{0110}$};
    \filldraw[blue!75!black] (10.3, 3.8) circle (3pt) node[above, font=\scriptsize] {$\ket{1010}$};
    \filldraw[red!85!black] (10.3, 1.0) circle (4pt) node[below, font=\scriptsize\bfseries] {$\ket{0101}^*$};

    % XY mixer rotation arrows
    \draw[->, very thick, green!60!black] (9.1, 2.7) arc (155:115:1.4cm) node[midway, above left, font=\tiny\bfseries] {XY};
    \draw[->, very thick, green!60!black] (10.6, 3.8) arc (65:25:1.4cm);
    \draw[->, very thick, green!60!black] (11.5, 2.1) arc (-25:-65:1.4cm);
    \draw[->, very thick, green!60!black] (10.0, 1.0) arc (-115:-155:1.4cm);

    \node[below, font=\footnotesize, text width=6.0cm, align=center] at (10.3, -0.2) {
        無効空間を物理排除し、有効解の間だけをトンネリング遷移
    };
\end{tikzpicture}"""

content = content.replace(old_fig62, new_fig62)

tex_path.write_text(content, encoding="utf-8")
print("Successfully patched TikZ diagrams in", tex_path)
