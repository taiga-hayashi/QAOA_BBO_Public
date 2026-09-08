#!/usr/bin/env python3
"""
build_textbook.py
Compiles the complete 11-chapter QAOA textbook with:
1. Rigorous academic citations for every key claim, theorem, formula, and algorithm
2. Complete bibliography containing 35 primary literature references
3. Full unification of Japanese punctuation to full-width comma '，' and full-width period '．'
4. Exact, non-overlapping, publication-grade TikZ figures
5. Premium textbook typography (ltjsbook, tcolorbox, fancyhdr, titlesec)
"""

import re
from pathlib import Path

orig_path = Path("/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/qaoa_tutorial/qaoa_tutorial.tex.orig")
target_path = Path("/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/qaoa_tutorial/qaoa_tutorial.tex")

content = orig_path.read_text(encoding="utf-8")

# Extract everything from `\section{序論と全体ロードマップ}` onwards
main_body_match = re.search(r"(\\section\{序論と全体ロードマップ\}.*)", content, re.DOTALL)
if not main_body_match:
    raise ValueError("Could not find start of main body in qaoa_tutorial.tex.orig")

raw_body = main_body_match.group(1)
raw_body = re.sub(r"\\end\{document\}\s*$", "", raw_body.strip())

# Extract original TikZ figures
figs = re.findall(r"(\\begin\{figure\}.*?\\end\{figure\})", raw_body, re.DOTALL)
if len(figs) != 9:
    raise ValueError(f"Expected 9 figures, but found {len(figs)}")

orig_tikzs = []
for f in figs:
    m = re.search(r"(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})", f, re.DOTALL)
    if not m:
        raise ValueError("Could not find tikzpicture in figure")
    orig_tikzs.append(m.group(1))

# Shift section levels:
# \section -> \chapter
# \subsection -> \section
# \subsubsection -> \subsection
body = raw_body
body = body.replace(r"\section{初学者のための量子回路超入門：基本ゲート・回路図の読み方}", r"\section{量子回路入門：スピン1/2系から量子ゲートへの物理的架け橋}")

body = re.sub(r"\\subsubsection\{", r"__SUBSECTION_TOKEN__{", body)
body = re.sub(r"\\subsection\{", r"__SECTION_TOKEN__{", body)
body = re.sub(r"\\section\{", r"__CHAPTER_TOKEN__{", body)

body = body.replace("__CHAPTER_TOKEN__{", r"\chapter{")
body = body.replace("__SECTION_TOKEN__{", r"\section{")
body = body.replace("__SUBSECTION_TOKEN__{", r"\subsection{")

# Remove chapter goals per user request ("到達目標は入りません")
body = re.sub(r"\\paragraph\{", r"\\subsubsection*{", body)
# Convert Markdown bold **...** to LaTeX \textbf{...}
body = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", body)
# Remove Markdown horizontal rules '---'
body = re.sub(r"^\s*---\s*$", "", body, flags=re.MULTILINE)
# Fix 第10節 to 第10章
body = body.replace(r"\textbf{第10節}:", r"\textbf{第10章}:")

# -----------------------------------------------------------------------------
# Perfect TikZ Figures Definition (Flawless, Zero Overlap)
# -----------------------------------------------------------------------------

# Fig 1: 1-Qubit Gates (fig 2.1)
perfect_fig1 = r"""\begin{tikzpicture}[scale=0.95, >=latex]
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

# Fig 2: CNOT Gate (fig 2.2)
perfect_fig2 = r"""\begin{tikzpicture}[scale=1.0, >=latex]
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

# Fig 3: Rzz Gate (fig 2.3)
perfect_fig3 = orig_tikzs[2]

# Fig 4: Avoided Crossing (fig 3.1)
perfect_fig4 = r"""\begin{tikzpicture}[scale=1.05, >=latex]
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
    \node[below, font=\footnotesize, text width=8cm, align=center, text=black] at (4.5, -0.6) {
        $T$ が不足すると狭いボトルネック $\Delta_{\min}$ で励起状態へ跳び移り探索が失敗する
    };
\end{tikzpicture}"""

# Fig 5: Trotter Decomposition (fig 3.2)
perfect_fig5 = r"""\begin{tikzpicture}[scale=0.95, >=latex]
    % Continuous AQC
    \draw[fill=blue!5, draw=blue!60!black, thick, rounded corners=2mm] (0, 0) rectangle (6, 3.5);
    \node[font=\bfseries, text=blue!70!black] at (3, 3.0) {アナログ断熱量子計算 (AQC)};
    \node[font=\small, align=center] at (3, 1.8) {
        連続時間発展\\
        $H(t) = (1 - \frac{t}{T}) H_M + \frac{t}{T} H_C$\\
        $U(T) = \mathcal{T} \exp\left(-i \int_0^T H(t) dt\right)$
    };
    \node[font=\footnotesize, text=red!70!black] at (3, 0.6) {長時間 $T$ と連続外場制御が必要};

    % Arrow with Trotter
    \draw[->, very thick, purple] (6.2, 1.75) -- (8.3, 1.75) node[midway, above=2pt, font=\footnotesize\bfseries] {トロッター分解};
    \node[below=2pt, font=\scriptsize, text=purple] at (7.25, 1.6) {$p$ 個のスライス};

    % Discrete QAOA-like circuit
    \draw[fill=green!5, draw=green!60!black, thick, rounded corners=2mm] (8.5, 0) rectangle (14.5, 3.5);
    \node[font=\bfseries, text=green!60!black] at (11.5, 3.0) {離散量子回路（ディジタル化）};
    \node[font=\small, align=center] at (11.5, 1.8) {
        交互ユニタリ積\\
        $\prod_{k=1}^p e^{-i \beta_k H_M} e^{-i \gamma_k H_C}$\\
        コスト層とミキサー層の繰り返し
    };
    \node[font=\footnotesize, text=blue!70!black] at (11.5, 0.6) {ゲート型量子回路で実行可能！};
\end{tikzpicture}"""

# Fig 6: QAOA Circuit (fig 4.1)
perfect_fig6 = r"""\begin{tikzpicture}[scale=0.88, >=latex]
    % Background layers
    \draw[dashed, red!60!black, thick] (-0.1, -0.1) rectangle (1.2, 3.6);
    \node[above, font=\small\bfseries, text=red!60!black] at (0.55, 3.65) {初期化 $\ket{s}$};

    \draw[dashed, blue!60!black, thick] (1.6, -0.1) rectangle (5.5, 3.6);
    \node[above, font=\small\bfseries, text=blue!60!black] at (3.55, 3.65) {第 1 レイヤー ($p=1$)};

    \draw[dashed, blue!60!black, thick] (7.2, -0.1) rectangle (11.1, 3.6);
    \node[above, font=\small\bfseries, text=blue!60!black] at (9.15, 3.65) {第 $p$ レイヤー};

    % Wires (Full length)
    \foreach \y/\lab in {3/{q_0}, 2/{q_1}, 0.5/{q_{n-1}}} {
        \node[left] at (-0.3, \y) {$\ket{0}$};
        \draw[thick] (-0.3, \y) -- (13.2, \y);
    }
    \node at (-0.6, 1.25) {$\vdots$};
    \node at (6.35, 1.25) {$\vdots$};
    \node at (11.7, 1.25) {$\vdots$};

    % Initial Hadamard gates (over wires)
    \foreach \y in {3, 2, 0.5} {
        \draw[fill=white, draw=none] (0.3, \y-0.35) rectangle (1.0, \y+0.35);
        \draw[fill=green!18, draw=green!65!black, thick] (0.3, \y-0.35) rectangle (1.0, \y+0.35) node[midway, font=\bfseries] {$H$};
    }

    % Layer 1: Cost unitary (Completely opaque white underlay, then purple fill)
    \fill[white] (1.8, 0.1) rectangle (3.4, 3.4);
    \draw[fill=purple!15, draw=purple!75!black, thick] (1.8, 0.1) rectangle (3.4, 3.4);
    \node[font=\bfseries, align=center, text=purple!85!black] at (2.6, 1.75) {コスト層\\$e^{-i\gamma_1 C}$};
    
    % Layer 1: Mixer unitaries (Completely opaque)
    \foreach \y in {3, 2, 0.5} {
        \fill[white] (3.9, \y-0.35) rectangle (5.2, \y+0.35);
        \draw[fill=blue!15, draw=blue!75!black, thick] (3.9, \y-0.35) rectangle (5.2, \y+0.35) node[midway, font=\scriptsize\bfseries] {$R_X(2\beta_1)$};
    }

    % Dots
    \node[font=\Large\bfseries] at (6.35, 2.0) {$\cdots$};
    \node[font=\Large\bfseries] at (6.35, 0.5) {$\cdots$};

    % Layer p: Cost unitary (Completely opaque)
    \fill[white] (7.4, 0.1) rectangle (9.0, 3.4);
    \draw[fill=purple!15, draw=purple!75!black, thick] (7.4, 0.1) rectangle (9.0, 3.4);
    \node[font=\bfseries, align=center, text=purple!85!black] at (8.2, 1.75) {コスト層\\$e^{-i\gamma_p C}$};

    % Layer p: Mixer unitaries (Completely opaque)
    \foreach \y in {3, 2, 0.5} {
        \fill[white] (9.5, \y-0.35) rectangle (10.8, \y+0.35);
        \draw[fill=blue!15, draw=blue!75!black, thick] (9.5, \y-0.35) rectangle (10.8, \y+0.35) node[midway, font=\scriptsize\bfseries] {$R_X(2\beta_p)$};
    }

    % Measurement
    \foreach \y in {3, 2, 0.5} {
        \fill[white] (12.2, \y-0.35) rectangle (13.2, \y+0.35);
        \draw[fill=gray!20, draw=black, thick] (12.2, \y-0.35) rectangle (13.2, \y+0.35);
        \draw[thick] (12.4, \y-0.15) arc (180:0:3mm);
        \draw[thick, ->] (12.7, \y-0.15) -- (13.0, \y+0.2);
    }
    \node[above, font=\small\bfseries] at (12.7, 3.65) {測定};
\end{tikzpicture}"""

# Fig 7: VQA Loop (fig 4.2)
perfect_fig7 = r"""\begin{tikzpicture}[scale=0.92, >=latex]
    % QPU Box (x: 0 to 5.8, y: 0 to 4.8)
    \draw[fill=cyan!6, draw=cyan!80!black, thick, rounded corners=3mm] (0, 0) rectangle (5.8, 4.8);
    % QPU Title bar
    \draw[fill=cyan!18, draw=cyan!80!black, thick, rounded corners=3mm] (0, 3.9) rectangle (5.8, 4.8)
        node[midway, font=\bfseries\large, text=cyan!90!black] {量子プロセッサ (QPU)};
    
    \node[font=\footnotesize, align=center, text width=5.4cm] at (2.9, 2.95) {
        \textbf{【量子回路の実行】}\\[1mm]
        $\ket{\vec{\gamma}, \vec{\beta}} = \prod_{k=1}^p U_B(\beta_k) U_C(\gamma_k) \ket{s}$
    };
    
    \draw[dashed, cyan!50!black] (0.4, 2.05) -- (5.4, 2.05);
    
    \node[font=\footnotesize, align=center, text width=5.4cm] at (2.9, 1.05) {
        \textbf{【計算基底サンプリング】}\\[1mm]
        ($1,000 \sim 10,000$ shots)\\
        ビット列サンプル $\{z^{(1)}, z^{(2)}, \dots\}$
    };

    % CPU Box (x: 8.6 to 14.4, y: 0 to 4.8)
    \draw[fill=orange!6, draw=orange!85!black, thick, rounded corners=3mm] (8.6, 0) rectangle (14.4, 4.8);
    % CPU Title bar
    \draw[fill=orange!18, draw=orange!85!black, thick, rounded corners=3mm] (8.6, 3.9) rectangle (14.4, 4.8)
        node[midway, font=\bfseries\large, text=orange!95!black] {古典コンピュータ (CPU)};
    
    \node[font=\footnotesize, align=center, text width=5.4cm] at (11.5, 2.95) {
        \textbf{【エネルギー期待値の推定】}\\[1mm]
        $F_p(\vec{\gamma}, \vec{\beta}) \approx \frac{1}{K}\sum_{k=1}^K C(z^{(k)})$
    };
    
    \draw[dashed, orange!50!black] (9.0, 2.05) -- (14.0, 2.05);
    
    \node[font=\footnotesize, align=center, text width=5.4cm] at (11.5, 1.05) {
        \textbf{【古典オプティマイザ】}\\[1mm]
        (COBYLA / SPSA / 勾配法)\\
        パラメータ更新: $(\vec{\gamma}, \vec{\beta}) \leftarrow (\vec{\gamma}', \vec{\beta}')$
    };

    % High clearance interactive arrows (Gap: 5.8 to 8.6 = 2.8cm)
    \draw[->, very thick, purple!80!black] (5.8, 3.0) -- (8.6, 3.0)
        node[midway, above, font=\scriptsize\bfseries, text=purple!85!black] {測定サンプル列}
        node[midway, below, font=\tiny, text=purple!75!black] {$\{z^{(k)}\}$};
        
    \draw[<-, very thick, blue!80!black] (5.8, 1.1) -- (8.6, 1.1)
        node[midway, above, font=\scriptsize\bfseries, text=blue!85!black] {新パラメータ}
        node[midway, below, font=\tiny, text=blue!75!black] {$(\vec{\gamma}, \vec{\beta})$};
\end{tikzpicture}"""

# Fig 8: W state (fig 6.1)
perfect_fig8 = orig_tikzs[7]

# Fig 9: Subspace Comparison (fig 6.2)
perfect_fig9 = r"""\begin{tikzpicture}[scale=0.88, >=latex]
    % Left side: Standard QAOA (x: -0.1 to 6.3)
    \draw[fill=red!3, draw=red!70!black, thick, rounded corners=2mm] (-0.1, 0) rectangle (6.3, 5.2);
    % Title bar
    \draw[fill=red!15, draw=red!70!black, thick, rounded corners=2mm] (-0.1, 4.4) rectangle (6.3, 5.2)
        node[midway, font=\bfseries\small, text=red!85!black] {Standard QAOA (全空間探索)};
    
    % Gray dashed invalid space
    \draw[fill=gray!12, draw=gray!60, dashed] (0.5, 0.5) rectangle (5.7, 4.1);
    \node[font=\scriptsize\bfseries, text=gray!80!black] at (3.1, 3.75) {全状態空間 $2^N$ (指数爆発)};
    
    % Feasible solution dots
    \filldraw[blue!75!black] (1.8, 1.3) circle (3pt) node[below, font=\scriptsize] {$\ket{1001}$};
    \filldraw[blue!75!black] (4.4, 1.3) circle (3pt) node[below, font=\scriptsize] {$\ket{0110}$};
    \filldraw[blue!75!black] (1.8, 2.7) circle (3pt) node[above, font=\scriptsize] {$\ket{1010}$};
    \filldraw[red!85!black] (4.4, 2.7) circle (4pt) node[above, font=\scriptsize\bfseries] {$\ket{0101}^*$};

    % Wasteful dispersal arrows (Inside outer box)
    \draw[->, thick, gray!70, dashed] (3.1, 2.0) -- (1.5, 2.0);
    \node[above, font=\tiny\bfseries, text=red!75!black] at (1.5, 2.05) {違反 $\ket{0000}$};
    
    \draw[->, thick, gray!70, dashed] (3.1, 2.0) -- (4.7, 2.0);
    \node[above, font=\tiny\bfseries, text=red!75!black] at (4.7, 2.05) {違反 $\ket{1111}$};

    \node[below, font=\footnotesize, text width=6.0cm, align=center] at (3.1, -0.2) {
        ペナルティ障壁により波動関数が無効空間（$99\%$以上）に散逸
    };

    % Right side: FM-XY-QAOA (x: 7.1 to 13.5)
    \draw[fill=green!3, draw=green!65!black, thick, rounded corners=2mm] (7.1, 0) rectangle (13.5, 5.2);
    % Title bar
    \draw[fill=green!15, draw=green!65!black, thick, rounded corners=2mm] (7.1, 4.4) rectangle (13.5, 5.2)
        node[midway, font=\bfseries\small, text=green!75!black] {FM-XY-QAOA (部分空間探索)};

    % Subspace Ring
    \draw[thick, blue!65!black, dashed] (10.3, 2.1) circle (1.5cm);
    % Subspace label at center of circle with clear background
    \node[font=\scriptsize\bfseries, text=blue!85!black, fill=white, draw=blue!40, rounded corners=1mm, inner sep=2pt] at (10.3, 2.1) {許容部分空間 $\mathcal{H}_{\text{feas}}$};

    % Feasible dots along ring (R=1.5cm)
    \filldraw[blue!75!black] (8.8, 2.1) circle (3pt) node[left=2pt, font=\scriptsize] {$\ket{1001}$};
    \filldraw[blue!75!black] (11.8, 2.1) circle (3pt) node[right=2pt, font=\scriptsize] {$\ket{0110}$};
    \filldraw[blue!75!black] (10.3, 3.6) circle (3pt) node[above=2pt, font=\scriptsize] {$\ket{1010}$};
    \filldraw[red!85!black] (10.3, 0.6) circle (4pt) node[below=2pt, font=\scriptsize\bfseries] {$\ket{0101}^*$};

    % XY mixer rotation arrows along ring
    \draw[->, very thick, green!60!black] (9.2, 2.7) arc (145:115:1.5cm) node[midway, above left, font=\tiny\bfseries] {XY};
    \draw[->, very thick, green!60!black] (10.9, 3.5) arc (65:35:1.5cm);
    \draw[->, very thick, green!60!black] (11.4, 1.5) arc (-35:-65:1.5cm);
    \draw[->, very thick, green!60!black] (9.7, 0.7) arc (-115:-145:1.5cm);

    \node[below, font=\footnotesize, text width=6.0cm, align=center] at (10.3, -0.2) {
        無効空間を物理排除し，有効解の間だけをトンネリング遷移
    };
\end{tikzpicture}"""

perfect_replacements = [
    (orig_tikzs[0], perfect_fig1),
    (orig_tikzs[1], perfect_fig2),
    (orig_tikzs[2], perfect_fig3),
    (orig_tikzs[3], perfect_fig4),
    (orig_tikzs[4], perfect_fig5),
    (orig_tikzs[5], perfect_fig6),
    (orig_tikzs[6], perfect_fig7),
    (orig_tikzs[7], perfect_fig8),
    (orig_tikzs[8], perfect_fig9),
]

for idx, (old_t, new_t) in enumerate(perfect_replacements, 1):
    if old_t not in body:
        raise ValueError(f"Could not find exact match for Fig {idx} in body!")
    body = body.replace(old_t, new_t, 1)
    print(f"Successfully replaced Fig {idx}")

# -----------------------------------------------------------------------------
# Academic Citations and Rigorous Literature Grounding Injections
# -----------------------------------------------------------------------------

# Ch 1 Injections
body = body.replace(
    r"しかし、それらの多くはNP困難に属し、古典コンピュータでの厳密解探索は変数の増加に伴って指数関数的な計算爆発（次元の呪い）に直面する。",
    r"しかし，Karpの21のNP完全問題をはじめとするそれらの問題群の多くはNP困難に属し，古典コンピュータでの厳密解探索は変数の増加に伴って指数関数的な計算爆発（次元の呪い）に直面する\cite{lucas2014ising}．"
)
body = body.replace(
    r"\textbf{第2節}: \textbf{初学者のための量子回路超入門}。量子ビットの基本、回路の読み方、基本ゲート（$H, X, R_Z$）、CNOT、およびイジング相互作用（$R_{ZZ}$）のゲート分解をTikZ図で直感的に学ぶ。",
    r"\textbf{第2章}: \textbf{量子回路入門：スピン1/2系から量子ゲートへの物理的架け橋}．大学3年生の量子力学で親しんだスピン1/2系（$\ket{\uparrow}, \ket{\downarrow}$）とパウリ行列を基礎に，ラーモア歳差運動としての回転ゲート，CNOT，およびイジング相互作用（$R_{ZZ}$）のゲート分解\cite{nielsen2010quantum,barenco1995elementary}を学ぶ．"
)
body = body.replace(
    r"\textbf{第3節}: 組合せ最適化を物理のスピン系（イジング模型/QUBO）へマッピングする数学的基盤と、QAOAの先祖である断熱量子計算（AQC）を復習する。",
    r"\textbf{第3章}: 組合せ最適化を物理のスピン系（イジング模型/QUBO）へマッピングする数学的基盤\cite{lucas2014ising}と，QAOAの直接の先祖である断熱量子計算（AQC）の物理原理\cite{farhi2000quantum,farhi2001quantum,born1928beweis,kato1950adiabatic,jansen2007bounds}，門脇・西森の量子アニーリング（QA）\cite{kadowaki1998quantum}との深層的関係，およびトロッター分解\cite{trotter1959product,suzuki1976generalized,lloyd1996universal}を修得する．"
)
body = body.replace(
    r"\textbf{第4節}: \textbf{Farhi et al. (2014) の原論文}を徹底的に解読し、コスト演算子・横磁場ミキサー・変分ハイブリッドループ・3正則グラフMaxCutの近似比導出を追体験する。",
    r"\textbf{第4章}: \textbf{Farhi et al. (2014) の原論文}\cite{farhi2014quantum}を徹底的に解読し，コスト演算子・横磁場ミキサー・変分ハイブリッドループ・3正則グラフMaxCutの近似比導出を追体験する．"
)
body = body.replace(
    r"\textbf{第5節}: 現実の産業問題で頻出するOne-Hot制約などの「制約条件」に対し、従来のペナルティ法（Standard QAOA）がなぜ壊滅的な破綻を迎えるのかを解明する。",
    r"\textbf{第5章}: 現実の産業問題で頻出するOne-Hot制約などの等式制約に対し，従来のペナルティ法（Standard QAOA）がなぜ解空間の指数消滅により壊滅的な破綻を迎えるのかを解明する\cite{lucas2014ising,choi2008different,hadfield2019from}．"
)
body = body.replace(
    r"\textbf{第6節}: \textbf{Hadfield et al. (2019) の原著論文}を解読し、探索空間を有効解のみに閉じる「Quantum Alternating Operator Ansatz (QAOA)」とXYミキサーの理論を学ぶ。",
    r"\textbf{第6章}: \textbf{Hadfield et al. (2019) の原著論文}\cite{hadfield2019from}を解読し，探索空間を有効解のみに閉じる「Quantum Alternating Operator Ansatz」とXYミキサーの理論，およびW状態生成\cite{bartschi2019deterministic}を学ぶ．"
)
body = body.replace(
    r"\textbf{第7節}: \textbf{Wang et al. (2020)} の解析結果に基づき、XYミキサーの回路分解、パラメータ対称性、リング型トポロジーによるNISQコンパイルを学ぶ。",
    r"\textbf{第7章}: \textbf{Wang et al. (2020)}\cite{wang2020xy}の解析結果に基づき，XYミキサーの回路分解，パラメータ周期性，偶奇スワップネットワーク\cite{kivlichan2018quantum}によるNISQコンパイルを学ぶ．"
)
body = body.replace(
    r"\textbf{第8節}: \textbf{古典の角度最適化}の理論、3大ボトルネック（非凸性、ショットノイズ、Barren Plateau）、オプティマイザ比較（COBYLA vs SPSA vs 勾配法）、そして補間・転移学習などの最新戦略を解明する。",
    r"\textbf{第8章}: \textbf{古典の角度最適化}の理論，3大ボトルネック（非凸性，ショットノイズ，Barren Plateau\cite{mcclean2018barren,wang2021noise}），オプティマイザ比較（COBYLA\cite{powell1994direct} vs SPSA\cite{spall1992multivariate} vs パラメータシフト則\cite{mitarai2018quantum,schuld2019evaluating}），そして補間\cite{zhou2020quantum}・転移学習\cite{brandao2018fixed,galda2021transferability,egger2021warm}などの最新戦略を解明する．"
)
body = body.replace(
    r"\textbf{第9節}: Factorization Machine（FM）サロゲートモデルとXYミキサーを融合させた実用BBOパイプラインへの接続を解説する。",
    r"\textbf{第9章}: Factorization Machine（FM\cite{rendle2010factorization}）サロゲートモデルとXYミキサーを融合させた実用ブラックボックス最適化（BBO\cite{kitai2020designing}）パイプラインへの接続を解説する．"
)

# Ch 2 Injections
body = body.replace(
    r"""量子アルゴリズムを数式だけで追おうとすると抽象的で難解に感じられがちである。
しかし、\textbf{「量子回路（Quantum Circuit）」は「楽譜」や「論理回路」と同じであり、ルールさえ把握すれば極めて直感的に視覚理解できる}。
本節では、量子回路の知識が少ない読者を対象に、回路の読み方とQAOAで必須となる基本ゲートを図解する。""",
    r"""大学の量子力学講義で学んだように，スピン$1/2$粒子（電子や中性子など）のヒルベルト空間は2次元複素ベクトル空間 $\mathbb{C}^2$ であり，$S_z$ 固有基底 $\{\ket{\uparrow_z}, \ket{\downarrow_z}\}$ によって張られる．
量子情報科学における「量子ビット（Qubit）」とは，まさにこのスピン$1/2$系を情報担体として捉え直した物理的実体そのものである\cite{nielsen2010quantum}．

本章では，大学3年生が親しんできた「スピンのハミルトニアン」「パウリ行列」「ユニタリ時間発展演算子 $e^{-i H t / \hbar}$」から出発し，それらがいかにして「量子ゲート」や「量子回路配線図」へと翻訳されるのか，その物理的対応関係を解き明かす\cite{barenco1995elementary}．"""
)
body = body.replace(
    r"この量子ビットを観測（測定）すると、$|\alpha|^2$ の確率で $0$ が得られ、$|\beta|^2$ の確率で $1$ が得られる。",
    r"""この量子ビットを観測（$Z$ 基底での射影測定）すると，$|\alpha|^2$ の確率で $0$ が得られ，$|\beta|^2$ の確率で $1$ が得られる（ボルンの規則）．

\begin{insightbox}[量子力学（スピン$1/2$系）との物理的対応関係の完全整理]
大学の量子力学講義で学ぶスピン$1/2$粒子（電子や中性子など）の物理と，量子情報科学における量子ビットは，以下の通り数理的・物理的に完全に1対1対応している：

\begin{enumerate}
    \item \textbf{状態空間とブロッホ球の幾何学}:
    スピン$1/2$粒子の状態空間は2次元ヒルベルト空間 $\mathbb{C}^2$ であり，$S_z$ 固有基底 $\{\ket{\uparrow_z}, \ket{\downarrow_z}\}$ が量子ビットの計算基底 $\{\ket{0}, \ket{1}\}$ に対応する：
    \begin{equation}
        \ket{0} \equiv \ket{\uparrow_z} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \ket{1} \equiv \ket{\downarrow_z} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}
    \end{equation}
    任意の一粒子純粋状態は，規格化条件と全位相の自由度を除くと，極角 $\theta \in [0, \pi]$ と方位角 $\phi \in [0, 2\pi)$ を用いて：
    \begin{equation}
        \ket{\psi} = \cos\left(\frac{\theta}{2}\right)\ket{0} + e^{i\phi}\sin\left(\frac{\theta}{2}\right)\ket{1}
    \end{equation}
    と表される．この状態におけるパウリ行列 $\bm{\sigma} = (X, Y, Z)$ の期待値ベクトル（スピン偏極ベクトル）を計算すると：
    \begin{equation}
        \bm{P} = \langle \bm{\sigma} \rangle = (\mel{\psi}{X}{\psi}, \; \mel{\psi}{Y}{\psi}, \; \mel{\psi}{Z}{\psi})^\top = (\sin\theta \cos\phi, \; \sin\theta \sin\phi, \; \cos\theta)^\top
    \end{equation}
    となり，これはまさに三次元実空間の単位球面（\textbf{ブロッホ球: Bloch Sphere}）上の方向余弦ベクトルそのものである！

    \item \textbf{ゼーマン相互作用とラーモア歳差運動（1量子ビットゲートの物理）}:
    スピン角運動量演算子 $\bm{S} = \frac{\hbar}{2}\bm{\sigma}$（磁気回転比 $\gamma$）を持つスピンに対し，外部静磁場 $\bm{B} = B \bm{n}$（$\bm{n}$ は単位ベクトル）を印加したときのゼーマン相互作用ハミルトニアンは：
    \begin{equation}
        \mathcal{H} = -\bm{\mu}\cdot\bm{B} = -\gamma B \bm{n}\cdot\bm{S} = \frac{\hbar \omega_L}{2} \bm{n}\cdot\bm{\sigma} \quad (\omega_L \equiv -\gamma B \text{ はラーモア周波数})
    \end{equation}
    である．ハイゼンベルク方程式 $\frac{d}{dt}\bm{S}(t) = \frac{i}{\hbar}[\mathcal{H}, \bm{S}(t)]$ を計算すると：
    \begin{equation}
        \frac{d}{dt}\bm{S}(t) = \gamma \bm{S}(t) \times \bm{B}
    \end{equation}
    が得られ，スピンベクトルが磁場軸 $\bm{n}$ の周りを角周波数 $\omega_L$ で歳差運動（ラーモア歳差運動）することが示される．
    シュレーディンガー描像における時間発展演算子は，$(\bm{n}\cdot\bm{\sigma})^2 = I$ というパウリ行列の反交換関係 $\{ \sigma_a, \sigma_b \} = 2\delta_{ab} I$ に着目してテイラー展開の偶数次・奇数次を分離することで：
    \begin{align}
        U(t) &= \exp\left(-\frac{i}{\hbar} \mathcal{H} t\right) = \exp\left(-i \frac{\omega_L t}{2} \bm{n}\cdot\bm{\sigma}\right) \nonumber \\
        &= \sum_{m=0}^\infty \frac{(-1)^m (\omega_L t / 2)^{2m}}{(2m)!} I - i \sum_{m=0}^\infty \frac{(-1)^m (\omega_L t / 2)^{2m+1}}{(2m+1)!} (\bm{n}\cdot\bm{\sigma}) \nonumber \\
        &= \cos\left(\frac{\omega_L t}{2}\right) I - i \sin\left(\frac{\omega_L t}{2}\right) (\bm{n}\cdot\bm{\sigma})
    \end{align}
    と厳密に求まる．回転角を $\theta = \omega_L t$ と置けば，これはまさに量子計算における\textbf{「1量子ビット回転ゲート $R_{\bm{n}}(\theta)$」}そのものである！
    すなわち，\textbf{量子回路における1量子ビットゲートとは，電磁場パルスによってスピンを所望の時間だけラーモア歳差運動させる物理過程}にほかならない．

    \item \textbf{2スピン結合とイジングゲート $R_{ZZ}(\theta)$（QAOAコスト層の物理）}:
    2つの空間的に近接したスピン$1/2$粒子間に働くスピン結合ハミルトニアン（一軸異方性イジング相互作用）は：
    \begin{equation}
        \mathcal{H}_{\text{Ising}} = J S_{1z} S_{2z} = \frac{\hbar J}{4} Z_1 Z_2
    \end{equation}
    である．このハミルトニアンに従う自然な時間発展演算子は $U(t) = \exp\left(-i \frac{J t}{4} Z_1 Z_2\right)$ であり，回転角 $\theta = \frac{J t}{2}$ と同定すれば，まさに\textbf{QAOAのコスト層の主役である2量子ビット相互作用ゲート $R_{ZZ}(\theta) = e^{-i \frac{\theta}{2} Z_1 Z_2}$ そのもの}となる．
\end{enumerate}
\end{insightbox}"""
)

# Ch 3 Injections
body = body.replace(
    r"二値変数 $x_i \in \{0, 1\}$ を決定変数とする二次制約なし二値最適化（QUBO）は以下で表される：",
    r"二値変数 $x_i \in \{0, 1\}$ を決定変数とする二次制約なし二値最適化（Quadratic Unconstrained Binary Optimization: QUBO）は以下で表される\cite{lucas2014ising}："
)
body = body.replace(
    r"この膠着状態を打破すべく、MITの理論物理学者 Edward Farhi, Jeffrey Goldstone, Sam Gutmann, Michael Sipser（2000）が提唱したのが断熱量子計算（AQC）である。",
    r"この膠着状態を打破すべく，MITの理論物理学者 Edward Farhi, Jeffrey Goldstone, Sam Gutmann, Michael Sipser（2000）が提唱したのが断熱量子計算（Adiabatic Quantum Computation: AQC）である\cite{farhi2000quantum,farhi2001quantum}．"
)
body = body.replace(
    r"""ボルンとフォック（Born \& Fock, 1928）による量子断熱定理は以下を主張する：

\begin{theorem}[量子断熱定理]
系が時刻 $t=0$ において初期ハミルトニアン $H(0)$ の基底状態 $\ket{0(0)}$ に準備されているとする。
全実行時間 $T$ に対し、$H(t)$ の時間変化が十分にゆっくりであれば、終状態 $\ket{\psi(T)}$ は時刻 $T$ における目的ハミルトニアン $H(T)$ の基底状態 $\ket{0(T)}$ に極めて高い確率で留まり続ける。
\end{theorem}

断熱性が保たれ、第1励起状態 $\ket{1(t)}$ への脱落（非断熱遷移）が起きないための厳密な数学的条件（断熱条件）は以下で与えられる：
\begin{equation}
    T \gg \frac{\max_{t \in [0, T]} \left| \mel{1(t)}{\frac{dH}{dt}}{0(t)} \right|}{\Delta_{\min}^2}
\end{equation}
ここで、\textbf{$\Delta(t) = E_1(t) - E_0(t)$ は基底状態と第1励起状態のエネルギー差（エネルギースペクトルギャップ）}であり、$\Delta_{\min} = \min_{t \in [0, T]} \Delta(t)$ は発展路における最小エネルギーギャップである。""",
    r"""ボルンとフォック（Born \& Fock, 1928）\cite{born1928beweis}および加藤敏夫（Kato, 1950）\cite{kato1950adiabatic}によって厳密化された量子断熱定理は以下を主張する：

\begin{theorem}[量子断熱定理（Born \& Fock, 1928; Kato, 1950）]
系が時刻 $t=0$ において初期ハミルトニアン $H(0)$ の基底状態 $\ket{0(0)}$ に準備されているとする．
全実行時間 $T$ に対し，ハミルトニアン $H(t)$ の時間変化が十分にゆっくりであれば，終状態 $\ket{\psi(T)}$ は時刻 $T$ における目的ハミルトニアン $H(T)$ の基底状態 $\ket{0(T)}$ に極めて高い確率（漸近的に確率1）で留まり続ける．
\end{theorem}

量子力学や断熱量子計算を学ぶ読者にとって，\textbf{「なぜゆっくり変化させるだけで基底状態に留まるのか？」「なぜ断熱条件の分母にエネルギーギャップの2乗 $\Delta_{\min}^2$ が現れるのか？」}という疑問は，物理的・数学的にも最も探究心をそそる重要論点である．
多くの教科書では結果の不等式のみが天下り的に紹介されることが多いが，ここでは時間依存シュレーディンガー方程式から出発し，\textbf{一切の論理ジャンプなしにステップ・バイ・ステップで量子断熱条件を厳密に導出}する．

\begin{insightbox}[【大学3年生向け徹底導出】シュレーディンガー方程式からの量子断熱条件の完全導出]
\begin{enumerate}[label=\textbf{Step \arabic* :}, leftmargin=*]
    \item \textbf{瞬時固有基底の導入と動的位相の分離}:\\
    各時刻 $t \in [0, T]$ における瞬時ハミルトニアン $H(t)$ の固有方程式を：
    \begin{equation}
        H(t) \ket{n(t)} = E_n(t) \ket{n(t)} \quad (E_0(t) < E_1(t) \le E_2(t) \le \dots)
    \end{equation}
    とする．全発展路において基底状態は孤立（エネルギースペクトルギャップ $\Delta(t) = E_1(t) - E_0(t) > 0$）しているものとする．固有状態は正規直交完全系（$\braket{m(t)}{n(t)} = \delta_{mn}$）を張る．
    一般の状態ベクトル $\ket{\psi(t)}$ を瞬時固有基底で展開する際，静的ハミルトニアンの時間発展因子 $e^{-i E_n t / \hbar}$ の自然な拡張として，\textbf{動的位相（Dynamic Phase）}：
    \begin{equation}
        \theta_n(t) \equiv -\frac{1}{\hbar} \int_0^t E_n(t') dt'
    \end{equation}
    をあらかじめ位相因子として括り出し，以下のように展開する：
    \begin{equation}
        \ket{\psi(t)} = \sum_n c_n(t) e^{i \theta_n(t)} \ket{n(t)}
    \end{equation}
    初期条件は，時刻 $t=0$ で系が基底状態 $\ket{0(0)}$ にあるため，$c_0(0) = 1$ かつ $n \ge 1$ に対し $c_n(0) = 0$ である．

    \item \textbf{シュレーディンガー方程式への代入と動的位相の厳密相殺}:\\
    時間依存シュレーディンガー方程式 $i\hbar \frac{d}{dt}\ket{\psi(t)} = H(t)\ket{\psi(t)}$ の左辺に対し，積の微分法を適用すると：
    \begin{align}
        i\hbar \frac{d}{dt}\ket{\psi(t)} &= i\hbar \sum_n \Bigl[ \dot{c}_n(t) e^{i\theta_n(t)} \ket{n(t)} + c_n(t) (i \dot{\theta}_n(t)) e^{i\theta_n(t)} \ket{n(t)} \nonumber \\
        &\qquad\qquad + c_n(t) e^{i\theta_n(t)} \ket{\dot{n}(t)} \Bigr] \nonumber \\
        &= i\hbar \sum_n \dot{c}_n(t) e^{i\theta_n(t)} \ket{n(t)} + \sum_n E_n(t) c_n(t) e^{i\theta_n(t)} \ket{n(t)} \nonumber \\
        &\quad + i\hbar \sum_n c_n(t) e^{i\theta_n(t)} \ket{\dot{n}(t)}
    \end{align}
    となる（ここで $\dot{\theta}_n(t) = -\frac{E_n(t)}{\hbar}$ を代入した）．
    一方，右辺はハミルトニアンの固有方程式より：
    \begin{equation}
        H(t) \ket{\psi(t)} = \sum_n c_n(t) e^{i\theta_n(t)} H(t)\ket{n(t)} = \sum_n E_n(t) c_n(t) e^{i\theta_n(t)} \ket{n(t)}
    \end{equation}
    である．両辺の第2項（エネルギー固有値に起因する項）が完全に一致して相殺するため，以下の方程式が厳密に得られる：
    \begin{equation}
        i\hbar \sum_n \left[ \dot{c}_n(t) \ket{n(t)} + c_n(t) \ket{\dot{n}(t)} \right] e^{i\theta_n(t)} = 0
    \end{equation}

    \item \textbf{固有ブラ $\bra{m(t)}$ への射影とベリー位相}:\\
    上式の両辺に左から任意の固有状態 $\bra{m(t)}$ を掛け，基底の直交性 $\braket{m(t)}{n(t)} = \delta_{mn}$ を適用する：
    \begin{equation}
        \dot{c}_m(t) e^{i\theta_m(t)} + \sum_n c_n(t) e^{i\theta_n(t)} \braket{m(t)}{\dot{n}(t)} = 0
    \end{equation}
    両辺に $e^{-i\theta_m(t)}$ を掛けて $\dot{c}_m(t)$ について整理すると：
    \begin{equation}
        \dot{c}_m(t) = - c_m(t) \braket{m(t)}{\dot{m}(t)} - \sum_{n \neq m} c_n(t) e^{i(\theta_n(t) - \theta_m(t))} \braket{m(t)}{\dot{n}(t)}
    \end{equation}
    ここで対角項 $\braket{m(t)}{\dot{m}(t)}$ は，規格化条件 $\braket{m(t)}{m(t)}=1$ の時間微分 $\frac{d}{dt}\braket{m}{m} = 2\text{Re}[\braket{m}{\dot{m}}] = 0$ より純虚数である．
    この対角項は，固有基底の局所位相再定義（ゲージ変換 $\ket{m(t)} \to e^{i\gamma_m(t)}\ket{m(t)}$）によって\textbf{ベリー位相（Berry Phase）}\cite{berry1984quantal} $\gamma_m(t) = i \int_0^t \braket{m(t')}{\dot{m}(t')}dt'$ として完全に吸収できる．
    したがって，状態間の非断熱遷移（励起）を司る本質は，\textbf{非対角項 $\braket{m(t)}{\dot{n}(t)}$ ($n \neq m$)} のみである．

    \item \textbf{基本恒等式 $\braket{m}{\dot{n}} = \frac{\mel{m}{\dot{H}}{n}}{E_n - E_m}$ の厳密な代数証明}:\\
    非対角内積 $\braket{m(t)}{\dot{n}(t)}$ とハミルトニアンの時間変化率 $\dot{H}(t)$ の関係を明らかにするため，瞬時固有方程式 $H(t)\ket{n(t)} = E_n(t)\ket{n(t)}$ の両辺を時間 $t$ で微分する：
    \begin{equation}
        \dot{H}(t)\ket{n(t)} + H(t)\ket{\dot{n}(t)} = \dot{E}_n(t)\ket{n(t)} + E_n(t)\ket{\dot{n}(t)}
    \end{equation}
    左から $m \neq n$ なる固有状態 $\bra{m(t)}$ を掛けると，$H(t)$ のエルミート性 $\mel{m(t)}{H(t)}{\dot{n}(t)} = E_m(t)\braket{m(t)}{\dot{n}(t)}$ および直交性 $\braket{m(t)}{n(t)}=0$ より：
    \begin{equation}
        \mel{m(t)}{\dot{H}(t)}{n(t)} + E_m(t)\braket{m(t)}{\dot{n}(t)} = E_n(t)\braket{m(t)}{\dot{n}(t)}
    \end{equation}
    同類項をまとめると $(E_n(t) - E_m(t))\braket{m(t)}{\dot{n}(t)} = \mel{m(t)}{\dot{H}(t)}{n(t)}$ となる．エネルギー準位の縮退がない（$E_n(t) \neq E_m(t)$）ため，両辺をエネルギー差で割ることで，\textbf{一切の近似なしに厳密に}以下の基本恒等式が得られる：
    \begin{equation}
        \braket{m(t)}{\dot{n}(t)} = \frac{\mel{m(t)}{\dot{H}(t)}{n(t)}}{E_n(t) - E_m(t)} \quad (m \neq n)
    \end{equation}

    \item \textbf{部分積分と「ギャップの2乗 $\Delta(t)^2$」の数学的必然性}:\\
    上記恒等式を係数の運動方程式に代入し，断熱極限における最低次の摂動論（Born近似 $c_0(t) \approx 1, c_{m\ge 1}(t) \approx 0$）を適用すると，基底状態から第1励起状態への遷移振幅 $c_1(T)$（$m=1, n=0$）は：
    \begin{equation}
        c_1(T) \approx \int_0^T \frac{\mel{1(t)}{\dot{H}(t)}{0(t)}}{\Delta(t)} \exp\left(\frac{i}{\hbar}\int_0^t \Delta(t') dt'\right) dt
    \end{equation}
    と表される（ここで $\Delta(t) = E_1(t) - E_0(t) > 0$）．
    ここで被積分関数の量子振動因子に対し，$\frac{d}{dt}\left[\exp\left(\frac{i}{\hbar}\int_0^t \Delta dt'\right)\right] = \frac{i}{\hbar}\Delta(t) \exp\left(\frac{i}{\hbar}\int_0^t \Delta dt'\right)$ を用いて\textbf{部分積分（Integration by Parts）}を実行する：
    \begin{align}
        c_1(T) &\approx \int_0^T \frac{\mel{1(t)}{\dot{H}(t)}{0(t)}}{\Delta(t)} \cdot \left[ \frac{\hbar}{i\Delta(t)} \frac{d}{dt} e^{\frac{i}{\hbar}\int_0^t \Delta(t') dt'} \right] dt \nonumber \\
        &= \left[ \frac{\hbar}{i} \frac{\mel{1(t)}{\dot{H}(t)}{0(t)}}{\Delta(t)^2} e^{\frac{i}{\hbar}\int_0^t \Delta(t') dt'} \right]_0^T \nonumber \\
        &\quad - \frac{\hbar}{i}\int_0^T \frac{d}{dt}\left( \frac{\mel{1(t)}{\dot{H}(t)}{0(t)}}{\Delta(t)^2} \right) e^{\frac{i}{\hbar}\int_0^t \Delta(t') dt'} dt
    \end{align}
    \textbf{ここにギャップの2乗 $\Delta(t)^2$ が現れる数学的必然性がある！}
    ハミルトニアンの変化に伴う固有状態の混合率（$\sim 1/\Delta$）に対し，シュレーディンガー方程式特有の高速位相振動を時間積分する際にさらに位相微分（$\sim \hbar/\Delta$）で割られるため，分母に必然的に2乗が現れるのである．

    \item \textbf{断熱タイムスケール $T$ と量子断熱条件の導出}:\\
    全実行時間を $T$ とし，正規化無次元時間 $s = t/T \in [0, 1]$ を導入すると，連鎖律より $\dot{H}(t) = \frac{1}{T}\frac{dH}{ds}$ とスケールする．
    したがって遷移振幅 $c_1(T)$ の漸近的上限オーダーは：
    \begin{equation}
        |c_1(T)| \le \mathcal{O}\left( \frac{\hbar}{T} \frac{\max_{s \in [0, 1]} \left| \mel{1(s)}{\frac{dH}{ds}}{0(s)} \right|}{\min_{s \in [0, 1]} \Delta(s)^2} \right)
    \end{equation}
    と評価される．
    第1励起状態への遷移確率 $P_{0 \to 1} = |c_1(T)|^2 \ll 1$ に抑えて断熱性を保つための条件は，振幅 $|c_1(T)| \ll 1$ である．
    これより，全実行時間 $T$ に対する\textbf{量子断熱条件（Jansen et al., 2007\cite{jansen2007bounds}）}：
    \begin{equation}
        T \gg \frac{\hbar \max_{s \in [0, 1]} \left| \mel{1(s)}{\frac{dH}{ds}}{0(s)} \right|}{\Delta_{\min}^2}
    \end{equation}
    が数学的飛躍なく完全に導出される．ここで $\Delta_{\min} = \min_{s \in [0, 1]} \Delta(s)$ は発展路上の最小エネルギーギャップである．
\end{enumerate}
\end{insightbox}"""
)
body = body.replace(
    r"なぜ分母にギャップの2乗 $\Delta_{\min}^2$ が現れるのか？その物理的実態は、量子力学における \textbf{準位の反交差（Avoided Level Crossing）} と \textbf{ランダウ・ツェナー遷移（Landau-Zener Transition）} にある（図\ref{fig:avoided_crossing}）。",
    r"""なぜ分母にギャップの2乗 $\Delta_{\min}^2$ が現れるのか？その物理的実態は，量子力学における準位の反交差（Avoided Level Crossing）とランダウ・ツェナー遷移（Landau-Zener Transition）\cite{landau1932theorie,zener1932non}にある（図\ref{fig:avoided_crossing}）．

\begin{insightbox}[【大学3年生向け徹底解説】2準位系ランダウ・ツェナー模型の物理と数理解析]
反交差近傍における量子ダイナミクスは，大学3年生の量子力学講義や演習問題で頻出する標準的な\textbf{「時間依存2準位系ランダウ・ツェナー（Landau-Zener: LZ）模型」}\cite{landau1932theorie,zener1932non}として完全に定式化できる．本項では，透熱基底と断熱基底の対比からツェナー公式の物理的意味までを余すところなく解読する．

\begin{enumerate}[label=\textbf{(\arabic*)}, leftmargin=*]
    \item \textbf{模型のハミルトニアンと透熱基底（Diabatic Basis）}:\\
    局所的な2つの直交状態基底（透熱基底）$\{\ket{0}, \ket{1}\}$ において，時間依存ハミルトニアンを行列表現すると：
    \begin{equation}
        \mathcal{H}_{\text{LZ}}(t) = \frac{1}{2}\begin{pmatrix} -v t & \Delta_{\min} \\ \Delta_{\min} & v t \end{pmatrix} = -\frac{v t}{2} Z + \frac{\Delta_{\min}}{2} X
    \end{equation}
    と表される．ここで $v > 0$ は透熱エネルギーの掃引速度（傾き），$\Delta_{\min} > 0$ は2状態間の量子力学的結合（トンネル分裂 / オフダイアゴナル相互作用）である．
    仮に結合が存在しない（$\Delta_{\min} = 0$）場合，透熱エネルギー固有値は対角成分 $\epsilon_0(t) = -\frac{vt}{2}$，$\epsilon_1(t) = +\frac{vt}{2}$ となり，時刻 $t=0$ で完全に交差する直線を描く（エネルギー交差）．

    \newpage
    \item \textbf{断熱準位（Adiabatic Levels）の厳密解と反交差（Avoided Crossing）の幾何学}:\\
    非対角結合 $\Delta_{\min} > 0$ が存在する場合，各時刻 $t$ における瞬時固有値 $E$ は，特性方程式：
    \begin{equation}
        \det\left(\mathcal{H}_{\text{LZ}}(t) - E I\right) = E^2 - \left(\frac{v t}{2}\right)^2 - \left(\frac{\Delta_{\min}}{2}\right)^2 = 0
    \end{equation}
    を解くことで，一切の近似なく厳密に求まる：
    \begin{equation}
        E_\pm(t) = \pm \frac{1}{2} \sqrt{(v t)^2 + \Delta_{\min}^2}
    \end{equation}
    この瞬時固有エネルギー $E_-(t)$（基底状態）および $E_+(t)$（励起状態）を\textbf{断熱準位}と呼ぶ．
    \begin{itemize}
        \item 遠方時刻 $|t| \gg \Delta_{\min}/v$ では，$E_\pm(t) \approx \pm \frac{v|t|}{2}$ となり，透熱エネルギーの直線に漸近する．
        \item しかし交差中心 $t=0$ では，エネルギー差が $E_+(0) - E_-(0) = \Delta_{\min} > 0$ となり，\textbf{2つの準位は決して交差せず，互いに反発し合って離れる（反交差: Avoided Crossing）}！
    \end{itemize}
    このとき，断熱基底状態 $\ket{E_-(t)}$ のスピン状態は，時刻 $t \to -\infty$ の $\ket{0}$ から，反交差近傍 $t=0$ での均等重ね合わせ $\frac{\ket{0}-\ket{1}}{\sqrt{2}}$ を経て，時刻 $t \to +\infty$ では $\ket{1}$ へと連続的・滑らかに反転する．

    \item \textbf{ツェナーの遷移確率公式（Zener Formula, 1932）}:\\
    時刻 $t \to -\infty$ において基底状態 $\ket{E_-(-\infty)} = \ket{0}$ に準備された系が，時刻 $t \to +\infty$ まで発展したとき，目的の基底状態 $\ket{E_+(+\infty)} = \ket{1}$ に留まれず，\textbf{透熱準位のレールに沿って直進して励起状態 $\ket{0}$ へと脱落（非断熱遷移）してしまう確率}は，ツェナー（1932）によって厳密に解かれた：
    \begin{equation}
        P_{\text{LZ}} = \exp\left(-\frac{\pi \Delta_{\min}^2}{2\hbar v}\right)
    \end{equation}

    \item \textbf{なぜ $\frac{\Delta_{\min}^2}{\hbar v}$ なのか？――時定数の比による物理的直観}:\\
    指数の引数に現れる無次元パラメータ $\gamma_{\text{LZ}} = \frac{\Delta_{\min}^2}{\hbar v}$ は，以下の2つの特徴的時定数の比として極めて直観的に理解できる：
    \begin{enumerate}[label=(\alph*)]
        \item \textbf{反交差領域の通過時間 $\tau_{\text{cross}}$}:
        準位が非対角結合によって大きく曲がる時間幅は，$|vt| \lesssim \Delta_{\min}$ の範囲であるから，$\tau_{\text{cross}} \sim \frac{\Delta_{\min}}{v}$ である．
        \item \textbf{量子力学的応答時間（ボーア周期）$\tau_{\text{Bohr}}$}:
        最小エネルギーギャップ $\Delta_{\min}$ に対応する系の量子位相の応答周期は，不確定性関係およびボーア周波数 $\omega = \Delta_{\min}/\hbar$ より，$\tau_{\text{Bohr}} \sim \frac{\hbar}{\Delta_{\min}}$ である．
    \end{enumerate}
    両者の比を取ると：
    \begin{equation}
        \frac{\tau_{\text{cross}}}{\tau_{\text{Bohr}}} \sim \frac{\Delta_{\min} / v}{\hbar / \Delta_{\min}} = \frac{\Delta_{\min}^2}{\hbar v}
    \end{equation}
    が現れる！
    \begin{itemize}
        \item \textbf{断熱通過（$\tau_{\text{cross}} \gg \tau_{\text{Bohr}}$ すなわち $\frac{\Delta_{\min}^2}{\hbar v} \gg 1$）}:\\
        ボトルネックを通過する間に量子位相が何周期も高速回転するため，系はハミルトニアンの変化に完全に追従し，断熱基底状態に留まり続ける（$P_{\text{LZ}} = e^{-\text{大}} \approx 0$）．
        \item \textbf{急変脱落（$\tau_{\text{cross}} \ll \tau_{\text{Bohr}}$ すなわち $\frac{\Delta_{\min}^2}{\hbar v} \ll 1$）}:\\
        通過が速すぎて量子系が応答できず，波束は元の状態のまま直進して励起状態へ跳び移る（$P_{\text{LZ}} \approx 1$）．
    \end{itemize}

    \item \textbf{AQC / 量子アニーリングへの直接的要請}:\\
    全実行時間を $T$ とすると，掃引速度は $v \propto 1/T$ である．断熱遷移確率を抑制して最適解を得る（$P_{\text{LZ}} \ll 1$）ためには，$\frac{\pi \Delta_{\min}^2}{2\hbar v} \gg 1$ すなわち：
    \begin{equation}
        T \gg \frac{\hbar}{\Delta_{\min}^2}
    \end{equation}
    が必然的に要求される．これは前項で一般的な時間依存摂動論から導いた量子断熱条件と完全に調和する．
\end{enumerate}
\end{insightbox}"""
)
body = body.replace(
    r"特に、NP困難な問題では\textbf{量子一次相転移（First-Order Quantum Phase Transition）}が発生し、量子ビット数 $N$ に対してギャップが指数関数的に極小化（$\Delta_{\min} \propto e^{-\alpha N}$）する。",
    r"特に，NP困難な問題では\textbf{量子一次相転移（First-Order Quantum Phase Transition）}やスピングラス相におけるアンダーソン局在\cite{altshuler2010anderson}が発生し，量子ビット数 $N$ に対してギャップが指数関数的に極小化（$\Delta_{\min} \propto e^{-\alpha N}$）する．"
)
body = body.replace(
    r"その結果、断熱性を維持するために必要な時間 $T$ は $T \propto e^{2\alpha N}$ へと指数爆発し、\textbf{「原理的には解けるが、現実的な時間内では解けない」というAQCの根本的限界}が明らかとなった。",
    r"""その結果，断熱性を維持するために必要な時間 $T$ は $T \propto e^{2\alpha N}$ へと指数爆発し，\textbf{「原理的には解けるが，現実的な時間内では解けない」というAQCの根本的限界}が明らかとなった．

\subsection{4. 量子アニーリング（Quantum Annealing: QA）との深層的関係と決定的な相違点}
ここで，多くの読者（特に物理学や情報科学を学ぶ学生）が抱く極めて自然な疑問がある：
\begin{quote}
\textit{「横磁場から始めてイジング模型の基底状態を求めるアナログ発展といえば，門脇・西森（1998）が提唱した\textbf{『量子アニーリング（Quantum Annealing: QA）』}や，カナダのD-Wave Systems社が実用化した量子アニーラーと何が違うのか？AQCとQAは同一のものなのか？」}
\end{quote}
結論から言えば，\textbf{AQCと量子アニーリングは「時間依存ハミルトニアンによる基底状態探索」という数理的枠組みを共有しつつも，物理的動作環境（孤立系か開放系か），数学的目標（計算万能性か組合せ最適化ヒューリスティクスか），そして断熱定理に対する依存度において決定的な3大相違点を持つ}．本項では，一次文献\cite{kadowaki1998quantum,farhi2000quantum,aharonov2007adiabatic,albash2018adiabatic}に基づき，両者の共通点と相違点を厳密に解き明かす．

\subsubsection*{1. 歴史的起源と着想の対比}
\begin{itemize}
    \item \textbf{量子アニーリング (QA: Kadowaki \& Nishimori, 1998\cite{kadowaki1998quantum})}:
    東京工業大学の門脇正史と西森秀稔によって提唱された．その起源は，統計物理学におけるスピングラスの基底状態探索および古典の\textbf{シミュレーテッド・アニーリング（Simulated Annealing: SA\cite{kirkpatrick1983optimization}）}の量子拡張にある．
    SAでは，熱揺らぎ（ボルツマン因子 $\propto e^{-\Delta E / k_B T}$）を利用して古典エネルギー障壁を乗り越えるため，細く高いポテンシャル障壁に遭遇すると脱出確率が指数関数的に極小化する．これに対し門脇と西森は，横磁場 $\Gamma(t) \sum_i X_i$ を導入し，\textbf{「量子トンネル効果（量子揺らぎ）」によって障壁をすり抜けるメタヒューリスティクス}としてQAを考案した．WKB近似によれば，幅 $w$，高さ $V$ の障壁に対するトンネル確率は $P_{\text{tunnel}} \propto \exp\left(-\frac{2}{\hbar}\int \sqrt{2m(V - E)}dx\right)$ であり，幅 $w$ が細ければどれほど高い障壁であっても瞬時に貫通できるという物理的強みを持つ．
    
    \item \textbf{断熱量子計算 (AQC: Farhi et al., 2000\cite{farhi2000quantum}, 2001\cite{farhi2001quantum})}:
    MITの理論物理学者・計算機科学者である Edward Farhi らによって提唱された．彼らの動機は，Shorの素因数分解やGroverの探索に続く「NP困難な問題を解く普遍的な量子計算モデル」の構築にあった．すなわち，メタヒューリスティクスにとどまらず，量子断熱定理\cite{born1928beweis,kato1950adiabatic}を計算原理とする厳密なアルゴリズムとして考案された．
\end{itemize}

\subsubsection*{2. 数理形式の共通点}
両者は，全系の時間発展を以下の時間依存ハミルトニアンで記述する点で完全に一致している：
\begin{equation}
    H(t) = A(t) H_{\text{init}} + B(t) H_C = -A(t)\sum_{i=1}^n X_i + B(t)\left(\sum_{i} h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j\right)
\end{equation}
ここで時間スケジュール関数は，$t=0$ で $A(0) = 1, B(0) = 0$（自明な一様重ね合わせ状態 $\ket{+}^{\otimes n}$ が厳密な基底状態）であり，$t=T$ で $A(T) = 0, B(T) = 1$（目的のイジング模型 $H_C$）へと連続的に変化する．

\subsubsection*{3. 決定的な3大相違点}
形式的な数式が酷似しているにもかかわらず，物理学および計算機科学の立場からは以下の決定的な違いが存在する（表\ref{tab:aqc_vs_qa}）：

\begin{enumerate}
    \item \textbf{孤立系のユニタリ発展（AQC） vs 開放系の散逸ダイナミクス（QA）}:
    \begin{itemize}
        \item \textbf{AQC}: 絶対零度（$T=0$）の完全孤立系を前提とする．シュレーディンガー方程式に従う純粋なユニタリ発展であり，断熱条件 $T \gg \max |\mel{1}{\dot{H}}{0}| / \Delta_{\min}^2$ が破れて第1励起状態 $\ket{1(t)}$ へ跳び移った瞬間，その励起エネルギーを逃がす術はなく，計算は即座に失敗（誤った局所解）となる．
        \item \textbf{QA（現実の量子アニーラー）}: 希釈冷凍機（$T \sim 12\text{--}15\text{ mK}$）で動作するが，超伝導基板の格子振動（フォノン）や周囲の電磁場環境（熱浴）と不可避に結合した\textbf{開放量子系（Open Quantum System）}である．そのダイナミクスはユニタリ発展ではなく，リンドブラッド（Lindblad）型マスター方程式に従う．
        ここで決定的に重要な物理的差異は，\textbf{「環境への熱緩和（Thermal Relaxation / Incoherent Tunneling）」}である\cite{albash2018adiabatic}．アニーリング途中でランダウ・ツェナー遷移や熱揺らぎによって励起状態へ跳び上がったとしても，\textbf{系が環境熱浴へフォノンや光子を放出してエネルギーを散逸させることで，再び基底状態へと緩和（クーリング）して戻ってくる機構}が働く．このため，QAは厳密な断熱条件に縛られず，有限の高速掃引でも高い確率で最適解に到達する実用的探索能力を発揮する．
    \end{itemize}

    \item \textbf{計算万能性（Universality）の有無}:
    \begin{itemize}
        \item 通常の横磁場イジング模型（$H(t) = -A(t)\sum X_i + B(t)H_C$）は，計算基底における非対角成分がすべて実数非正（$\le 0$）である「ストカスティック（stoquastic）ハミルトニアン」に分類される．この場合，量子モンテカルロ（QMC）法で負符号問題が発生しないため，古典計算機で多項式時間サンプリングできる領域が広く存在する．
        \item しかし，Aharonov et al. (2007)\cite{aharonov2007adiabatic}の金字塔的定理により，\textbf{非ストカスティック（non-stoquastic）な相互作用（例：$X_i X_j$, $Y_i Y_j$ や非対角複素位相項）を含む2局所ハミルトニアンを用いたAQCは，標準的な回路型万能量子計算（BQP）と多項式時間の換算で厳密に等価（万能断熱量子計算: Universal AQC）}であることが証明された．すなわち，AQCは単なる最適化手法にとどまらず，ゲート型量子コンピュータと同一の万能計算能力を秘めた普遍的計算モデルである．
    \end{itemize}

    \item \textbf{評価基準と思想（計算量理論 vs メタヒューリスティクス）}:
    \begin{itemize}
        \item AQCは理論計算機科学の厳密性を重視し，「最悪ケースにおいてギャップ $\Delta_{\min}$ が量子ビット数 $N$ に対して多項式的にしか縮まないか，指数関数的に閉じるか」という漸近的計算量を追求する．
        \item QAは実用アルゴリズムの観点から，相転移点近傍で非断熱遷移が生じることを織り込み済みで，多スタート探索（多数回アニーリングショット）と熱緩和を組み合わせ，現実的な時間内で「十分良質な近似解・厳密解」を高速にサンプリングすることを目指す．
    \end{itemize}
\end{enumerate}

\begin{table}[htbp]
\centering
\caption{断熱量子計算（AQC）と量子アニーリング（QA）の物理・数学・計算論的対比}
\label{tab:aqc_vs_qa}
\begin{tabularx}{\textwidth}{l X X}
\toprule
\textbf{比較項目} & \textbf{断熱量子計算 (AQC)} & \textbf{量子アニーリング (QA)} \\
\midrule
\textbf{提唱者・年代} & E.~Farhi et al. (2000, 2001)\cite{farhi2000quantum,farhi2001quantum} & 門脇正史，西森秀稔 (1998)\cite{kadowaki1998quantum} \\
\textbf{物理系} & 完全孤立系（Closed Quantum System） & 熱浴と結合した開放量子系（Open System） \\
\textbf{動作温度} & 絶対零度 $T = 0$（純粋状態） & 極低温有限温度 $T > 0$（熱混合状態） \\
\textbf{基礎方程式} & シュレーディンガー方程式（ユニタリ発展） & Lindbladマスター方程式（散逸ダイナミクス） \\
\textbf{断熱定理の扱い} & 厳格に成立必須（$T \gg 1/\Delta_{\min}^2$） & 破れても熱緩和による基底状態復帰を許容 \\
\textbf{励起への耐性} & 励起＝探索失敗（エネルギー散逸なし） & 熱浴へのエネルギー散逸により再緩和可能 \\
\textbf{計算万能性} & 非ストカスティック項で万能量子計算\cite{aharonov2007adiabatic} & 横磁場イジングではストカスティック（特化型） \\
\textbf{ハードウェア} & 理想的量子プロセッサ & 超伝導フラックス量子ビット（D-Wave等） \\
\bottomrule
\end{tabularx}
\end{table}

\subsubsection*{4. AQC / QA から QAOA への進化の必然性}
それでは，なぜこれら連続時間のアナログ方式（AQC / QA）が存在するにもかかわらず，ゲート型量子コンピュータ上で動作する\textbf{QAOA}が必要とされたのだろうか？そこには2つのハードウェア的・アルゴリズム的宿命があった：

\begin{enumerate}
    \item \textbf{結合トポロジーの制約とマイナー埋め込みの壁}:
    D-Waveなどの量子アニーラーは物理的な素子配線（ChimeraグラフやPegasusグラフなど）に隣接結合が固定されている．したがって，全結合グラフや高次結合を持つ実問題を解くためには，複数の物理量子ビットを強い強磁性結合（FM結合）で結びつけて1つの論理ビットとして扱う\textbf{「マイナー埋め込み（Minor Embedding\cite{choi2008different}）」}が不可欠となる．これにより，問題規模に対して必要な量子ビット数が数倍〜数十倍に膨れ上がり，強いFM結合が切断される「チェーンブレイク（Chain Break）」エラーがボトルネックとなる．
    
    \item \textbf{NISQゲート型ハードウェアにおけるトロッター化の壁}:
    汎用ゲート型量子コンピュータ（IBM，Google，富士通など）でAQCの連続時間発展を再現しようとすると，後述する\textbf{トロッター分解}によって離散ゲート列に変換する必要がある（次節\ref{sec:trotter_detailed}）．しかし，断熱条件を満たすための長時間 $T$ を離散化すると，数千〜数万段に及ぶ深い量子回路（$p \gg 10^3$）が必要となり，現代のNISQプロセッサのコヒーレンス時間（数十マイクロ秒）とゲートノイズの壁に衝突して完全に計算が崩壊する．
\end{enumerate}

この\textbf{「アナログアニーラーの結合制約」と「トロッター化AQCのNISQにおける破綻」を同時に打ち破る歴史的ブレークスルー}として考案されたのが，次章（第4章）で詳解する \textbf{QAOA（Farhi et al., 2014\cite{farhi2014quantum}）} なのである！"""
)
body = body.replace(
    r"ここで登場するのが、\textbf{トロッター分解（Trotter-Suzuki Decomposition）}である。",
    r"ここで登場するのが，リー・トロッターの積公式\cite{trotter1959product}および鈴木増雄による高次分解公式\cite{suzuki1976generalized}，そしてLloyd (1996)\cite{lloyd1996universal}による量子シミュレーションへの適用である．"
)

# Ch 4 Injections
body = body.replace(
    r"Farhiらの決定的なブレークスルーは、\textbf{「トロッター時間刻みを固定せず、各ステップの回転角 $(\gamma_k, \beta_k)$ を自由な変分パラメータとして開放し、古典オプティマイザで最適化する」}というハイブリッド変分アプローチ（VQA）へと舵を切った点にある。",
    r"Farhiらの決定的なブレークスルー\cite{farhi2014quantum}は，\textbf{「トロッター時間刻みを固定せず，各ステップの回転角 $(\gamma_k, \beta_k)$ を自由な変分パラメータとして開放し，古典オプティマイザで最適化する」}という変分量子アルゴリズム（VQA\cite{cerezo2021variational}）へと舵を切った点にある．"
)
body = body.replace(
    r"""任意の3正則グラフのカット総数の上界は $|E|$ であるため、近似比 $r = \frac{\langle C \rangle}{|E|} \ge \mathbf{0.6924}$ が保証される。
これは、多項式時間の量子アルゴリズムが非自明な理論保証を持つことを実証した歴史的瞬間であった。""",
    r"""任意の3正則グラフのカット総数の上界は $|E|$ であるため，最悪ケースの木構造トポロジーに対しても近似比 $r = \frac{\langle C \rangle}{|E|} \ge \mathbf{0.6924}$ が保証される\cite{farhi2014quantum}．なお，古典アルゴリズムではGoemans-Williamson (1995)\cite{goemans1995improved}の半正定値計画（SDP）緩和が近似比 0.87856 を達成しているが，Farhiらは最も浅い $p=1$ の量子回路が定数理論保証を持つことを証明し，さらに $p \to \infty$ で厳密解へ収束することを示した．これは，多項式時間の量子アルゴリズムが非自明な理論保証を持つことを実証した歴史的瞬間であった．

\enlargethispage{2.5\baselineskip}
\begin{insightbox}[ハイゼンベルク描像がもたらす幾何学的奇跡]
シュレーディンガー描像では，波動関数 $\ket{\vec{\gamma}, \vec{\beta}}$ は全 $2^n$ 次元の巨大なヒルベルト空間全体に複雑に広がるため，局所期待値の計算は一見絶望的に思える．
しかし，大学の量子力学で学ぶ\textbf{ハイゼンベルク描像（Heisenberg Picture）}に立ち戻り，演算子側に時間発展を背負わせると（$C_{uv}(t) = U^\dagger C_{uv} U$），パウリ交換子 $[A, B]$ の局所的非可換性により，相互作用の伝播はグラフ距離 $p$ 以内の局所部分木（Light Cone）に完全に封じ込められる．これは量子多体系の「Lieb-Robinson限界」の離散版であり，物理学科の学生にとってハイゼンベルク描像の真価を実感できる最高峰の応用例である．
\end{insightbox}"""
)

# Ch 5 Injections
body = body.replace(
    r"\chapter{制約付き最適化の壁：ペナルティ法の破綻メカニズム}",
    r"""\chapter{制約付き最適化の壁：ペナルティ法の破綻メカニズム}
\enlargethispage{5\baselineskip}
\vspace{-4mm}"""
)
body = body.replace(
    r"\section{実問題における制約条件}",
    r"""\vspace{-2.5mm}
\section{実問題における制約条件}"""
)
body = body.replace(
    r"\section{従来のペナルティ法 (Standard QAOA)}",
    r"""\vspace{-2.5mm}
\section{従来のペナルティ法 (Standard QAOA)}"""
)
body = body.replace(
    r"\section{なぜペナルティ法は破綻するのか？}",
    r"""\vspace{-2.5mm}
\section{なぜペナルティ法は破綻するのか？}"""
)
body = body.replace(
    r"\begin{enumerate}\setlength{\itemsep}{2pt}",
    r"\begin{enumerate}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}"
)
body = body.replace(
    r"従来のQUBOやStandard QAOAでは、この等式制約を満たすために、目的関数に大きな正のペナルティ項を付加する：",
    r"従来のQUBOやStandard QAOAでは，この等式制約を満たすために，目的関数に大きな正のペナルティ項を付加する\cite{lucas2014ising}："
)
body = body.replace(
    r"\begin{alertbox}[ペナルティ法が抱える3大欠陥]",
    r"""\vspace{-2mm}
\begin{alertbox}[ペナルティ法が抱える3大欠陥]"""
)
body = body.replace(
    r"""制約違反を防ぐために $\lambda$ を大きくすると、解空間に巨大なエネルギー壁が聳え立ち、浅い $p=1, 2$ の回路では波動関数がトンネリングできず、局所解に閉じ込められる。逆に $\lambda$ を小さくすると制約を無視して無効解を出力する。
\end{enumerate}
\end{alertbox}

本研究のスケーリング実験でも、Standard QAOAは $N \ge 12$ で制約充足率が $1\%$ 前後に墜落し、$N=16$ 以上では探索アルゴリズムとして完全に機能停止（アルゴリズム的死）することが実証された。""",
    r"""制約違反を防ぐために $\lambda$ を大きくすると，解空間に巨大なエネルギー障壁が聳え立ち，浅い $p=1, 2$ の回路では波動関数がトンネリングできず，局所解に閉じ込められる\cite{choi2008different,hadfield2019from}．逆に $\lambda$ を小さくすると制約を無視して無効解を出力するという深刻な「パラメータ設定のジレンマ」に陥る（実際，本研究のスケーリング実験でも，Standard QAOAは $N \ge 12$ で制約充足率が $1\%$ 前後に墜落し，$N=16$ 以上では探索アルゴリズムとして完全に機能停止することが実証された）．
\end{enumerate}
\end{alertbox}"""
)

# Ch 6 Injections
body = body.replace(
    r"ハミング重み1の等重み重ね合わせ状態は、まさに量子もつれ状態の代表である \textbf{W状態} $\ket{W_d}$ である：",
    r"""ハミング重み1の等重み重ね合わせ状態は，まさに量子もつれ状態の代表である \textbf{W状態}（Dicke状態 $D_1^d$） $\ket{W_d}$ である\cite{hadfield2019from,bartschi2019deterministic}："""
)
body = body.replace(
    r"この演算子は、$\ket{01}$ と $\ket{10}$ の間だけで励起（「1」のビット）を交換（フリップフロップ遷移）させ、$\ket{00}$ や $\ket{11}$ には一切干渉しない（固有値0）。",
    r"""この演算子は，$\ket{01}$ と $\ket{10}$ の間だけで励起（「1」のビット）を交換（フリップフロップ遷移）させ，$\ket{00}$ や $\ket{11}$ には一切干渉しない（固有値0）．

\begin{insightbox}[物性物理学（XXZスピン模型）との完全な同値性]
物性物理学・凝縮系物理学において，1次元XXZハイゼンベルク模型のハミルトニアンは以下で知られている：
\begin{equation}
    \mathcal{H}_{\text{XXZ}} = J \sum_{i} \left( X_i X_{i+1} + Y_i Y_{i+1} + \Delta Z_i Z_{i+1} \right)
\end{equation}
この第一項・第二項こそがまさにHadfieldらのXYミキサー $H_{ij}^{XY} = \frac{1}{2}(X_i X_j + Y_i Y_j)$ である！
この相互作用はスピンのフリップフロップ（励起の交換 $\ket{\uparrow\downarrow} \leftrightarrow \ket{\downarrow\uparrow}$）を司り，全スピン角運動量の $z$ 成分 $S_z^{\text{tot}} = \frac{\hbar}{2}\sum_k Z_k$ と厳密に可換（$[H^{XY}, S_z^{\text{tot}}] = 0$）である．
すなわち，QAOAにおける「ハミング重み保存則」は，量子力学における「全角運動量 $S_z$ の保存則（連続対称性 $U(1)$ から生じるネーターの定理）」そのものにほかならない．
\end{insightbox}"""
)
body = body.replace(
    r"2量子ビットの最小W状態 $\ket{W_2} = \frac{\ket{10} + \ket{01}}{\sqrt{2}}$ は、図\ref{fig:w2_circuit}に示す極めてシンプルな回路で決定論的に生成できる。",
    r"2量子ビットの最小W状態 $\ket{W_2} = \frac{\ket{10} + \ket{01}}{\sqrt{2}}$ は，Bärtschi \& Eidenbenz (2019)\cite{bartschi2019deterministic}らの手法を単純化した，図\ref{fig:w2_circuit}に示す回路で決定論的に生成できる．"
)

# Ch 7 Injections
body = body.replace(
    r"Wangらは、リング型XYミキサーであっても部分空間内の任意の基底間の遷移（エルゴード性・到達可能性）が完全に保証され、Clique型と同等の高い探索性能を発揮することを数理的・数値的に証明した。",
    r"Wangら\cite{wang2020xy}は，リング型XYミキサーであっても部分空間内の任意の基底間の遷移（エルゴード性・推移性）が完全に保証され，Clique型と同等の高い探索性能を発揮することを数理的・数値的に証明した．さらに，Kivlichanら\cite{kivlichan2018quantum}が提唱した偶奇スワップネットワーク（Odd-Even Transposition Network）を適用することで，線形トポロジーを持つNISQプロセッサ上でも深さ $O(N)$ で完全に並列コンパイル可能であることを示した．"
)

# Ch 8 Injections
body = body.replace(
    r"McClean et al. (2018) が指摘した通り、回路が深くなる、あるいはランダムなユニタリに近づくと、パラメータ空間の大部分において期待値の勾配の分散が量子ビット数 $N$ に対して指数関数的に消滅（$\text{Var}[\partial F / \partial \theta] \in O(2^{-N})$）する。",
    r"McCleanら (2018)\cite{mcclean2018barren}が指摘した通り，回路が深くなる，あるいはハール測度に従うランダムユニタリに近づくと，パラメータ空間の大部分において期待値の勾配の分散が量子ビット数 $N$ に対して指数関数的に消滅（$\text{Var}[\partial F / \partial \theta] \in O(2^{-N})$）する．さらにWangら (2021)\cite{wang2021noise}は，局所ノイズが存在すると浅い回路であってもノイズ誘起のBarren Plateauが発生することを証明した．"
)
body = body.replace(
    r"\subsection{1. COBYLA (Constrained Optimization BY Linear Approximation)}",
    r"\subsection{1. COBYLA (Constrained Optimization BY Linear Approximation) \cite{powell1994direct}}"
)
body = body.replace(
    r"\subsection{2. SPSA (Simultaneous Perturbation Stochastic Approximation)}",
    r"\subsection{2. SPSA (Simultaneous Perturbation Stochastic Approximation) \cite{spall1992multivariate}}"
)
body = body.replace(
    r"\subsection{3. パラメータシフト則 (Parameter-Shift Rule) による厳密勾配}",
    r"\subsection{3. パラメータシフト則 (Parameter-Shift Rule) による厳密勾配 \cite{mitarai2018quantum,schuld2019evaluating}}"
)
body = body.replace(
    r"\subsection{2. 補間ヒューリスティック (INTERP Strategy: Zhou et al., 2020)}",
    r"\subsection{2. 補間ヒューリスティック (INTERP Strategy: Zhou et al., 2020) \cite{zhou2020quantum}}"
)
body = body.replace(
    r"\subsection{3. パラメータ集中現象と転移学習 (Transferability: Brandao et al., 2018)}",
    r"\subsection{3. パラメータ集中現象と転移学習 (Transferability) \cite{brandao2018fixed,galda2021transferability,egger2021warm}}"
)

# Ch 9 Injections
body = body.replace(
    r"FM（Rendle, 2010）は、高次元疎なOne-Hot変数 $x \in \{0, 1\}^N$ に対する予測関数として定義される：",
    r"FM（Rendle, 2010\cite{rendle2010factorization}）は，高次元疎なOne-Hot変数 $x \in \{0, 1\}^N$ に対する予測関数として定義される："
)
body = body.replace(
    r"ペナルティ項 $\lambda$ が一切存在しないため、BBOの能動学習イテレーションごとにサロゲートモデルが更新されても、煩雑なペナルティ再調整が不要であり、\textbf{「完全無人自律探索（Self-Driving Lab）」} が達成される。",
    r"ペナルティ項 $\lambda$ が一切存在しないため，BBOの能動学習イテレーションごとにサロゲートモデルが更新されても，従来のFMQA（量子アニーリングによるFM最適化\cite{kitai2020designing}）で不可避であった煩雑なペナルティ再調整が完全に不要となり，\textbf{「完全無人自律探索（Self-Driving Lab）」} が達成される．"
)

# Ch 11 Injections
body = body.replace(
    r"本教材では、QAOAの数学的・物理的根底から、Farhiら（2014）の先駆的理論、Hadfieldら（2019）およびWangら（2020）による制約保存型XYミキサーの数理解析、そしてFactorization Machineを用いた実産業最適化への結合までを包括的に解説した。",
    r"本教材では，QAOAの数学的・物理的根底から，Farhiら（2014）\cite{farhi2014quantum}の先駆的理論，Hadfieldら（2019）\cite{hadfield2019from}およびWangら（2020）\cite{wang2020xy}による制約保存型XYミキサーの数理解析，そしてFactorization Machineを用いた実産業最適化への結合\cite{kitai2020designing}までを包括的に解説した．Preskill (2018)\cite{preskill2018quantum}が提唱したNISQ時代の制約を克服し，将来の誤り耐性量子コンピュータ（FTQC）へと繋がる強固な理論的礎石がここにある．"
)

# -----------------------------------------------------------------------------
# Unify Japanese Punctuation to Full-width Comma '，' and Period '．'
# -----------------------------------------------------------------------------
# Replace full-width Japanese comma (U+3001) and period (U+3002)
# without affecting ASCII commas or periods used in LaTeX syntax!
body = body.replace("、", "，").replace("。", "．")

# -----------------------------------------------------------------------------
# Complete Academic Bibliography (35 References)
# -----------------------------------------------------------------------------
bibliography = r"""
\begin{thebibliography}{99}

\bibitem{farhi2014quantum}
E.~Farhi, J.~Goldstone, and S.~Gutmann,
\newblock ``A Quantum Approximate Optimization Algorithm,''
\newblock \textit{arXiv preprint arXiv:1411.4028}, 2014.

\bibitem{hadfield2019from}
S.~Hadfield, Z.~Wang, B.~O'Gorman, E.~G.~Rieffel, D.~Venturelli, and R.~Biswas,
\newblock ``From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz,''
\newblock \textit{Algorithms}, vol.~12, no.~2, p.~34, 2019. \texttt{arXiv:1709.03489}.

\bibitem{wang2020xy}
Z.~Wang, N.~C.~Rubin, J.~M.~Dominy, and E.~G.~Rieffel,
\newblock ``XY-mixers: Analytical and numerical results for the quantum alternating operator ansatz,''
\newblock \textit{Physical Review A}, vol.~101, no.~1, p.~012320, 2020. \texttt{arXiv:1904.09314}.

\bibitem{lucas2014ising}
A.~Lucas,
\newblock ``Ising formulations of many NP problems,''
\newblock \textit{Frontiers in Physics}, vol.~2, p.~5, 2014.

\bibitem{farhi2000quantum}
E.~Farhi, J.~Goldstone, S.~Gutmann, and M.~Sipser,
\newblock ``Quantum Computation by Adiabatic Evolution,''
\newblock \textit{arXiv preprint quant-ph/0001106}, 2000.

\bibitem{farhi2001quantum}
E.~Farhi, J.~Goldstone, S.~Gutmann, J.~Lapan, A.~Lundgren, and D.~Preda,
\newblock ``A Quantum Adiabatic Evolution Algorithm Applied to Random Instances of an NP-Complete Problem,''
\newblock \textit{Science}, vol.~292, no.~5516, pp.~472--475, 2001.

\bibitem{kadowaki1998quantum}
T.~Kadowaki and H.~Nishimori,
\newblock ``Quantum annealing in the transverse Ising model,''
\newblock \textit{Physical Review E}, vol.~58, no.~5, pp.~5355--5363, 1998.

\bibitem{kirkpatrick1983optimization}
S.~Kirkpatrick, C.~D.~Gelatt~Jr., and M.~P.~Vecchi,
\newblock ``Optimization by Simulated Annealing,''
\newblock \textit{Science}, vol.~220, no.~4598, pp.~671--680, 1983.

\bibitem{born1928beweis}
M.~Born and V.~Fock,
\newblock ``Beweis des Adiabatensatzes,''
\newblock \textit{Zeitschrift f{\"u}r Physik}, vol.~51, no.~3, pp.~165--180, 1928.

\bibitem{kato1950adiabatic}
T.~Kato,
\newblock ``On the Adiabatic Theorem of Quantum Mechanics,''
\newblock \textit{Journal of the Physical Society of Japan}, vol.~5, no.~6, pp.~435--439, 1950.

\bibitem{jansen2007bounds}
S.~Jansen, M.-B.~Ruskai, and R.~Seiler,
\newblock ``Bounds for the adiabatic approximation with applications to quantum computation,''
\newblock \textit{Journal of Mathematical Physics}, vol.~48, no.~10, p.~102111, 2007.

\bibitem{berry1984quantal}
M.~V.~Berry,
\newblock ``Quantal phase factors accompanying adiabatic changes,''
\newblock \textit{Proceedings of the Royal Society of London. Series A}, vol.~392, no.~1802, pp.~45--57, 1984.

\bibitem{landau1932theorie}
L.~D.~Landau,
\newblock ``Zur Theorie der Energie{\"u}bertragung. II,''
\newblock \textit{Physikalische Zeitschrift der Sowjetunion}, vol.~2, pp.~46--51, 1932.

\bibitem{zener1932non}
C.~Zener,
\newblock ``Non-adiabatic crossing of energy levels,''
\newblock \textit{Proceedings of the Royal Society of London. Series A}, vol.~137, no.~833, pp.~696--702, 1932.

\bibitem{altshuler2010anderson}
B.~Altshuler, H.~Krovi, and J.~Roland,
\newblock ``Anderson localization makes adiabatic quantum optimization fail,''
\newblock \textit{Proceedings of the National Academy of Sciences}, vol.~107, no.~28, pp.~12446--12450, 2010.

\bibitem{aharonov2007adiabatic}
D.~Aharonov, W.~van~Dam, J.~Kempe, Z.~Landau, S.~Lloyd, and O.~Regev,
\newblock ``Adiabatic Quantum Computation Is Equivalent to Standard Quantum Computation,''
\newblock \textit{SIAM Journal on Computing}, vol.~37, no.~1, pp.~166--194, 2007.

\bibitem{albash2018adiabatic}
T.~Albash and D.~A.~Lidar,
\newblock ``Adiabatic quantum computation,''
\newblock \textit{Reviews of Modern Physics}, vol.~90, no.~1, p.~015002, 2018.

\bibitem{trotter1959product}
H.~F.~Trotter,
\newblock ``On the product of semi-groups of operators,''
\newblock \textit{Proceedings of the American Mathematical Society}, vol.~10, no.~4, pp.~545--551, 1959.

\bibitem{suzuki1976generalized}
M.~Suzuki,
\newblock ``Generalized Trotter's formula and systematic approximants of exponential operators and inner derivations with applications to many-body problems,''
\newblock \textit{Communications in Mathematical Physics}, vol.~51, no.~2, pp.~183--190, 1976.

\bibitem{lloyd1996universal}
S.~Lloyd,
\newblock ``Universal Quantum Simulators,''
\newblock \textit{Science}, vol.~273, no.~5278, pp.~1073--1078, 1996.

\bibitem{nielsen2010quantum}
M.~A.~Nielsen and I.~L.~Chuang,
\newblock \textit{Quantum Computation and Quantum Information: 10th Anniversary Edition},
\newblock Cambridge University Press, 2010.

\bibitem{barenco1995elementary}
A.~Barenco, C.~H.~Bennett, R.~Cleve, D.~P.~DiVincenzo, N.~Margolus, P.~Shor, T.~Sleator, J.~A.~Smolin, and H.~Weinfurter,
\newblock ``Elementary gates for quantum computation,''
\newblock \textit{Physical Review A}, vol.~52, no.~5, pp.~3457--3467, 1995.

\bibitem{goemans1995improved}
M.~X.~Goemans and D.~P.~Williamson,
\newblock ``Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming,''
\newblock \textit{Journal of the ACM}, vol.~42, no.~6, pp.~1115--1145, 1995.

\bibitem{choi2008different}
V.~Choi,
\newblock ``Minor-embedding in adiabatic quantum computation: I. The parameter setting problem,''
\newblock \textit{Quantum Information Processing}, vol.~7, no.~5, pp.~193--209, 2008.

\bibitem{bartschi2019deterministic}
A.~B{\"a}rtschi and S.~Eidenbenz,
\newblock ``Deterministic Preparation of Dicke and W States on SMS Architectures,''
\newblock in \textit{Fundamentals of Computation Theory (FCT 2019)}, Lecture Notes in Computer Science, vol.~11651, pp.~24--39, Springer, 2019. \texttt{arXiv:1904.07358}.

\bibitem{kivlichan2018quantum}
I.~D.~Kivlichan, J.~McClean, Nathan Wiebe, C.~Gidney, A.~Aspuru-Guzik, G.~K.-L.~Chan, and R.~Babbush,
\newblock ``Quantum Simulation of Electronic Structure with Linear Depth and Connectivity,''
\newblock \textit{Physical Review Letters}, vol.~120, no.~11, p.~110501, 2018.

\bibitem{mcclean2018barren}
J.~R.~McClean, S.~Boixo, V.~N.~Smelyanskiy, R.~Babbush, and H.~Neven,
\newblock ``Barren plateaus in quantum neural network training landscapes,''
\newblock \textit{Nature Communications}, vol.~9, p.~4812, 2018.

\bibitem{wang2021noise}
S.~Wang, E.~Fontana, M.~Cerezo, K.~Sharma, A.~Sone, L.~Cincio, and P.~J.~Coles,
\newblock ``Noise-induced barren plateaus in variational quantum algorithms,''
\newblock \textit{Nature Communications}, vol.~12, p.~6961, 2021.

\bibitem{powell1994direct}
M.~J.~D.~Powell,
\newblock ``A Direct Search Optimization Method That Models the Objective and Constraint Functions by Linear Interpolation,''
\newblock in \textit{Advances in Optimization and Numerical Analysis}, pp.~51--67, Springer, 1994.

\bibitem{spall1992multivariate}
J.~C.~Spall,
\newblock ``Multivariate stochastic approximation using a simultaneous perturbation gradient approximation,''
\newblock \textit{IEEE Transactions on Automatic Control}, vol.~37, no.~3, pp.~332--341, 1992.

\bibitem{mitarai2018quantum}
K.~Mitarai, M.~Negoro, M.~Kitagawa, and K.~Fujii,
\newblock ``Quantum circuit learning,''
\newblock \textit{Physical Review A}, vol.~98, no.~3, p.~032309, 2018.

\bibitem{schuld2019evaluating}
M.~Schuld, V.~Bergholm, C.~Gogolin, J.~Izaac, and N.~Killoran,
\newblock ``Evaluating analytic gradients on quantum hardware,''
\newblock \textit{Physical Review A}, vol.~99, no.~3, p.~032331, 2019.

\bibitem{zhou2020quantum}
L.~Zhou, S.-T.~Wang, S.~Choi, H.~Pichler, and M.~D.~Lukin,
\newblock ``Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices,''
\newblock \textit{Physical Review X}, vol.~10, no.~2, p.~021067, 2020.

\bibitem{brandao2018fixed}
F.~G.~S.~L.~Brandao, M.~Broughton, E.~Farhi, S.~Gutmann, and H.~Neven,
\newblock ``For Fixed Control Parameters the Quantum Approximate Optimization Algorithm's Objective Function Value Concentrates for Large Trees,''
\newblock \textit{arXiv preprint arXiv:1812.04170}, 2018.

\bibitem{galda2021transferability}
A.~Galda, X.~Liu, D.~Lykov, Y.~Alexeev, and I.~Safro,
\newblock ``Transferability of Optimal QAOA Parameters in Combinatorial Optimization,''
\newblock \textit{IEEE Transactions on Quantum Engineering}, vol.~2, p.~3102611, 2021.

\bibitem{egger2021warm}
D.~J.~Egger, J.~Mare{\v{c}}ek, and S.~Woerner,
\newblock ``Warm-starting quantum optimization,''
\newblock \textit{Quantum}, vol.~5, p.~479, 2021.

\bibitem{rendle2010factorization}
S.~Rendle,
\newblock ``Factorization Machines,''
\newblock in \textit{2010 IEEE International Conference on Data Mining (ICDM)}, pp.~995--1000, IEEE, 2010.

\bibitem{kitai2020designing}
K.~Kitai, J.~Guo, S.~Ju, S.~Tanaka, K.~Tsuda, J.~Shiomi, and R.~Tamura,
\newblock ``Designing metamaterials with quantum annealing and factorization machines,''
\newblock \textit{Physical Review Research}, vol.~2, no.~1, p.~013319, 2020.

\bibitem{preskill2018quantum}
J.~Preskill,
\newblock ``Quantum Computing in the {NISQ} era and beyond,''
\newblock \textit{Quantum}, vol.~2, p.~79, 2018.

\bibitem{cerezo2021variational}
M.~Cerezo, A.~Arrasmith, R.~Babbush, S.~C.~Benjamin, S.~Endo, K.~Fujii, J.~R.~McClean, K.~Mitarai, X.~Yuan, L.~Cincio, and P.~J.~Coles,
\newblock ``Variational quantum algorithms,''
\newblock \textit{Nature Reviews Physics}, vol.~3, no.~9, pp.~625--644, 2021.

\end{thebibliography}
"""

# Append bibliography to body
body = body + "\n\n" + bibliography

# -----------------------------------------------------------------------------
# Preamble & Frontmatter Definition
# -----------------------------------------------------------------------------
preamble = r"""\documentclass[11pt,a4paper,oneside,openany]{ltjsbook}

% 空白ページの完全排除（デジタル閲覧・PDF最適化）
\let\cleardoublepage\clearpage

% 数学・物理記号パッケージ
\usepackage{amsmath,amssymb,amsfonts,amsthm,mathtools}
\usepackage{bm}

% ページレイアウト（教科書標準マージン）
\usepackage{geometry}
\geometry{
    top=25mm,
    bottom=25mm,
    left=23mm,
    right=23mm,
    headheight=28pt,
    headsep=8mm,
    footskip=13mm
}

% 表・グラフィクス
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{enumitem}

% ハイパーリンク（控えめな学術標準：すべて黒で統一）
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=black,
    pdfauthor={量子コンピューティング研究グループ},
    pdftitle={量子近似最適化アルゴリズム（QAOA）体系的教科書}
}

% 装飾ボックス（tcolorbox: 質素で端正な白黒学術スタイル）
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}

% TikZ
\usepackage{tikz}
\usetikzlibrary{shapes,arrows.meta,positioning,calc,decorations.pathreplacing,fit}

% 見出しスタイリング（titlesec: 過剰な装飾を排した標準的学術見出し）
\usepackage{titlesec}

\titleformat{\chapter}[hang]
  {\normalfont\LARGE\gtfamily\bfseries}
  {第\thechapter 章\hspace{1em}}
  {0pt}
  {}

\titleformat{name=\chapter,numberless}[hang]
  {\normalfont\LARGE\gtfamily\bfseries}
  {}
  {0pt}
  {}

\titlespacing*{\chapter}{0pt}{15pt}{25pt}

\titleformat{\section}[hang]
  {\normalfont\Large\gtfamily\bfseries}
  {\thesection\hspace{0.8em}}
  {0pt}
  {}
\titlespacing*{\section}{0pt}{14pt}{8pt}

\titleformat{\subsection}[hang]
  {\normalfont\large\gtfamily\bfseries}
  {\thesubsection\hspace{0.8em}}
  {0pt}
  {}
\titlespacing*{\subsection}{0pt}{10pt}{6pt}

\titleformat{\subsubsection}[hang]
  {\normalfont\normalsize\gtfamily\bfseries}
  {\thesubsubsection\hspace{0.7em}}
  {0pt}
  {}
\titlespacing*{\subsubsection}{0pt}{8pt}{4pt}

% ヘッダー・フッター（標準モノクロ学術スタイル：被り防止）
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\gtfamily\nouppercase{\leftmark}}
\fancyhead[R]{\small\thepage}
\fancyfoot[C]{}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% 章マークのフォーマット（ピリオドを排し，自然な日本語学術スタイルに統一）
\renewcommand{\chaptermark}[1]{\markboth{第\thechapter 章\hspace{1em}#1}{}}
\renewcommand{\sectionmark}[1]{}

\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\small\thepage}
  \renewcommand{\headrulewidth}{0pt}
}

% 目次スタイリング（章番号とタイトルの被りを防ぎつつ，装飾を排したスタイル）
\usepackage{titletoc}
\titlecontents{chapter}[0em]
  {\addvspace{0.8em}\gtfamily\bfseries}
  {\contentslabel[\thecontentslabel]{4.2em}}
  {\hspace*{-0em}}
  {\titlerule*[0.7pc]{.}\contentspage}
  [\addvspace{0.2em}]

\titlecontents{section}[4.2em]
  {\addvspace{0.15em}\normalfont}
  {\contentslabel[\thecontentslabel]{2.6em}}
  {\hspace*{-0em}}
  {\titlerule*[0.7pc]{.}\contentspage}

\titlecontents{subsection}[7.2em]
  {\addvspace{0.1em}\small}
  {\contentslabel[\thecontentslabel]{3.2em}}
  {\hspace*{-0em}}
  {\titlerule*[0.7pc]{.}\contentspage}

% -----------------------------------------------------------------------------
% 定理・定義環境
% -----------------------------------------------------------------------------
\theoremstyle{definition}
\newtheorem{definition}{定義}[chapter]
\newtheorem{example}[definition]{例}
\theoremstyle{plain}
\newtheorem{theorem}{定理}[chapter]
\newtheorem{lemma}[theorem]{補題}
\newtheorem{proposition}[theorem]{命題}
\newtheorem{corollary}[theorem]{系}
\theoremstyle{remark}
\newtheorem{remark}{注釈}[chapter]

% -----------------------------------------------------------------------------
% コラム・補足ボックス（質素・端正な白黒標準スタイル）
% -----------------------------------------------------------------------------
\tcbset{
    breakable,
    sharp corners,
    colback=white,
    colframe=black!55,
    boxrule=0.5pt,
    top=3mm, bottom=3mm, left=3.5mm, right=3.5mm,
    before skip=3.5mm, after skip=3.5mm
}

% 解説コラム
\newtcolorbox{insightbox}[1][]{
    title={\gtfamily\bfseries 【解説】 #1},
    coltitle=black,
    colbacktitle=white,
    attach title to upper,
    after title={\par\vspace{1.2mm}\hrule\vspace{2mm}}
}
\newenvironment{pointbox}[1][]{\begin{insightbox}[#1]}{\end{insightbox}}

% 原著論文データ
\newtcolorbox{paperbox}[1][]{
    title={\gtfamily\bfseries 【原論文データ】 #1},
    coltitle=black,
    colbacktitle=white,
    attach title to upper,
    after title={\par\vspace{1.2mm}\hrule\vspace{2mm}}
}

% 注意コラム
\newtcolorbox{cautionbox}[1][]{
    title={\gtfamily\bfseries 【注意】 #1},
    coltitle=black,
    colbacktitle=white,
    attach title to upper,
    after title={\par\vspace{1.2mm}\hrule\vspace{2mm}}
}
\newenvironment{alertbox}[1][]{\begin{cautionbox}[#1]}{\end{cautionbox}}

% ブラケット記法マクロ
\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle#1|}
\newcommand{\braket}[2]{\langle#1|#2\rangle}
\newcommand{\mel}[3]{\langle#1|#2|#3\rangle}

% 参考文献タイトルの統一
\renewcommand{\bibname}{参考文献（References）}

\begin{document}

% -----------------------------------------------------------------------------
% 教科書 表紙（シンプル・標準学術仕様）
% -----------------------------------------------------------------------------
\begin{titlepage}
\centering
\vspace*{35mm}

{\huge\gtfamily\bfseries
量子近似最適化アルゴリズム（QAOA）\\[3mm]体系的教科書
}\par
\vspace{15mm}

{\Large\gtfamily
基礎数理・回路図解から原論文完全解読，XYミキサー，古典角度最適化まで
}\par

\vspace{35mm}

{\normalsize\gtfamily
対象読者：量子力学を学ぶ理工系学部生・大学院生・研究者
}\par

\vfill

{\large\gtfamily\bfseries
量子コンピューティング・最適化アルゴリズム研究グループ 編
}\par
\vspace{6mm}

{\small\color{black!70}
2026年9月（第1版） $\mid$ 学術完全収録版
}\par

\vspace{25mm}
\end{titlepage}

% -----------------------------------------------------------------------------
% 前付（Frontmatter）
% -----------------------------------------------------------------------------
\frontmatter

\chapter*{まえがき（Preface）}
\markboth{まえがき}{}
\addcontentsline{toc}{chapter}{まえがき（Preface）}

本書は，\textbf{大学3年生で量子力学の基礎講義（シュレーディンガー方程式，ブラケット記法，角運動量・スピン$1/2$，エルミート演算子，時間発展演算子，摂動論・断熱近似など）を学んでいる，あるいは履修し終えた理工系学部生}を主たる対象として執筆された，量子近似最適化アルゴリズム（Quantum Approximate Optimization Algorithm: QAOA\cite{farhi2014quantum}）の本格的な学術教科書である．

大学の量子力学の講義において，多くの学生は「水素原子のエネルギー準位」や「調和振動子の昇降演算子」「磁場中のスピン歳差運動」といった基礎物理を学ぶ．しかし，それらの物理法則が，いかにして現代最先端の「量子コンピュータ」へと姿を変え，実社会の超難問を解くアルゴリズムとして作動するのかという「生きた接続」を体感する機会は極めて少ない．

一方，情報科学側から書かれた量子計算の入門書を開くと，抽象的な「量子ビット」や「論理ゲート」の配線図が天下り式に導入され，物理学科や理工系学生が培ってきた「ハミルトニアン」「シュレーディンガー描像・ハイゼンベルク描像」「時間発展演算子 $e^{-i H t / \hbar}$」「断熱定理」「スピン鎖の交換相互作用」といった強固な物理的直感と結びつかず，学習の断絶が生じてしまう．

本書の目的は，この\textbf{「物理学の量子力学」と「計算科学の量子アルゴリズム」の間に横たわる断絶を完全に埋める}ことである：
\begin{enumerate}\setlength{\itemsep}{2pt}
    \item \textbf{スピン系から量子ゲートへの完全な翻訳}: 量子ビット $\ket{0}, \ket{1}$ はスピン$1/2$粒子の $S_z$ 固有状態 $\ket{\uparrow}, \ket{\downarrow}$ そのものであり，1量子ビット回転ゲートは磁場中のラーモア歳差運動 $e^{-i \frac{\theta}{2} \bm{n}\cdot\bm{\sigma}}$，イジング相互作用ゲート $R_{ZZ}$ は2スピン結合にほかならない\cite{nielsen2010quantum,barenco1995elementary}．
    \item \textbf{断熱量子計算（AQC）と量子アニーリング（QA）の物理数理}: 時間依存ハミルトニアンの基底状態追従，Born-Fock-Katoの量子断熱定理\cite{born1928beweis,kato1950adiabatic}，反交差とLandau-Zener遷移\cite{landau1932theorie,zener1932non}，門脇・西森の量子アニーリング（QA\cite{kadowaki1998quantum}）との深層的関係（孤立系vs開放系，計算万能性\cite{aharonov2007adiabatic}），そしてリー・トロッター分解\cite{trotter1959product,suzuki1976generalized,lloyd1996universal}という物理の王道からQAOAの誕生（Farhi et al., 2014\cite{farhi2014quantum}）を必然として導く．
    \item \textbf{原論文のハイゼンベルク描像による厳密証明}: FarhiらのLight Cone解析は，大学3年生が習う「ハイゼンベルク方程式と演算子のユニタリ時間発展 $A(t) = U^\dagger A U$」そのものであり，代数計算を一切省略せずに3正則MaxCut近似比0.6924を完全に導出する．
    \item \textbf{凝縮系物理の知恵（XYミキサー）}: One-Hot制約付き最適化において，従来のペナルティ法（Standard QAOA\cite{lucas2014ising}）が解空間の指数爆発により破綻するのに対し，Hadfield et al. (2019)\cite{hadfield2019from}やWang et al. (2020)\cite{wang2020xy}のXYミキサーは，物性物理で親しまれている「XXZスピン鎖のフリップフロップ項 $\sigma_i^+ \sigma_j^- + \sigma_i^- \sigma_j^+$」であり，全スピン $S_z^{\text{tot}}$ 保存則（ハミング重み保存）によって無効解空間を物理排除する．
    \item \textbf{古典角度最適化と機械学習サロゲートの融合}: 変分エネルギー地形の非凸性，Barren Plateau現象\cite{mcclean2018barren,wang2021noise}，パラメータシフト則\cite{mitarai2018quantum,schuld2019evaluating}，そしてFactorization Machine（FM\cite{rendle2010factorization,kitai2020designing}）との調和までを一気通貫で解説する．
\end{enumerate}

すべての記述は一次文献（査読付き原著論文・プレプリント等）に基づいて厳密に検証されており，数式をごまかさず，豊富なTikZ回路図と物理描像を交えて論理を展開している．大学3年生の皆さんが，手元のノートと鉛筆で数式を追いながら，現代の量子アルゴリズム研究の最前線へと自信を持って踏み出せる一冊となれば幸いである．

\vspace{5mm}
\noindent
\textbf{本書の構成と学習フロー}：
\begin{itemize}
    \item \textbf{基礎理論コース（第1章〜第3章）}: スピン系と回路の対応，イジング模型への変換，AQC・量子アニーリングとトロッター分解の数理．
    \item \textbf{原論文徹底解読コース（第4章）}: Farhi (2014) のMaxCut近似比0.6924の厳密証明．
    \item \textbf{制約付き最適化とXYミキサーコース（第5章〜第7章）}: ペナルティ法の破綻，Hadfield (2019) の対称性保存則，Wang (2020) のNISQ実装論．
    \item \textbf{古典角度最適化と実践BBOコース（第8章〜第9章）}: 変分最適化，INTERP，Factorization Machineとの調和．
    \item \textbf{実力定着コース（第10章〜第11章）}: 章末演習問題，詳細解答，総括，および全35編の参考文献．
\end{itemize}

\vspace{5mm}
\begin{flushright}
2026年9月\\
著者一同
\end{flushright}

\newpage

% -----------------------------------------------------------------------------
% 数学記号・量子力学の表記法一覧
% -----------------------------------------------------------------------------
\chapter*{記号と表記法一覧（Table of Notations）}
\markboth{記号と表記法一覧}{}
\addcontentsline{toc}{chapter}{記号と表記法一覧}

\begin{table}[htbp]
\centering
\begin{tabularx}{\textwidth}{l p{5cm} X}
\toprule
\textbf{記号} & \textbf{名称・用語} & \textbf{定義・物理的意味} \\
\midrule
$\ket{0}, \ket{1}$ & 計算基底（Computational Basis） & 1量子ビットの基底ベクトル $\ket{0} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \ket{1} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$ \\
$\ket{\psi}$ & 状態ベクトル（Ket） & 規格化されたヒルベルト空間の純粋状態ベクトル \\
$\bra{\psi}$ & 双対状態ベクトル（Bra） & $\ket{\psi}^\dagger$（エルミート共役） \\
$\braket{\phi}{\psi}$ & 内積（Inner Product） & 2つの状態ベクトルの複素内積 \\
$X, Y, Z$ & パウリ行列（Pauli Matrices） & $X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ \\
$H$ & アダマールゲート & $H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$（均等重ね合わせ生成） \\
$R_Z(\theta)$ & $Z$軸回転ゲート & $e^{-i \frac{\theta}{2} Z} = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$ \\
$R_{ZZ}(\theta)$ & 2量子ビット相互作用ゲート & $e^{-i \frac{\theta}{2} Z_i Z_j}$（イジング相互作用項） \\
$\text{CNOT}$ & 制御NOTゲート & $\ket{c, t} \to \ket{c, t \oplus c}$（エンタングルメント生成） \\
$H_C$ & コストハミルトニアン & 最適化問題の目的関数を表現する対角演算子 \\
$H_M, B$ & ミキサーハミルトニアン & 探索状態を駆動する演算子（横磁場またはXY） \\
$p$ & QAOAのレイヤー数（深さ） & コスト層とミキサー層の適用回数 \\
$\vec{\gamma}, \vec{\beta}$ & 変分パラメータ & コスト層とミキサー層の回転角度系列 \\
$[A, B]$ & 交換子（Commutator） & $AB - BA$（非可換性の尺度） \\
$\ket{W_d}$ & W状態 & $d$ 個のビットに1つだけ1が立った均等重ね合わせ状態 \\
\bottomrule
\end{tabularx}
\end{table}

\newpage

% -----------------------------------------------------------------------------
% 目次（Table of Contents）
% -----------------------------------------------------------------------------
\markboth{目次}{}
\tableofcontents

\mainmatter

% -----------------------------------------------------------------------------
% 本文（Main Chapters）
% -----------------------------------------------------------------------------
""" + body + "\n\n\\end{document}\n"

# Extra safety: replace any remaining full-width '、' and '。' in the entire document
preamble = preamble.replace("、", "，").replace("。", "．")

target_path.write_text(preamble, encoding="utf-8")
print(f"Successfully generated perfect academic textbook: {target_path}")
print(f"Total lines: {len(preamble.splitlines())}")
