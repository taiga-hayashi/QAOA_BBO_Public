#!/usr/bin/env python3
"""
Convert qaoa_tutorial.tex into a high-end academic textbook design.
"""

import re
from pathlib import Path

source_path = Path("/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/qaoa_tutorial/qaoa_tutorial.tex.bak")
target_path = Path("/Users/hayashitaiga/Library/CloudStorage/GoogleDrive-taiga.hayashi@gmail.com/My Drive/Intern_Fujitsu/qaoa_tutorial/qaoa_tutorial.tex")

content = source_path.read_text(encoding="utf-8")

# Extract everything from `\section{序論と全体ロードマップ}` onwards
main_body_match = re.search(r"(\\section\{序論と全体ロードマップ\}.*)", content, re.DOTALL)
if not main_body_match:
    raise ValueError("Could not find start of main body")

raw_body = main_body_match.group(1)
# Remove \end{document} from the end if present
raw_body = re.sub(r"\\end\{document\}\s*$", "", raw_body.strip())

# Shift section levels:
# \section -> \chapter
# \subsection -> \section
# \subsubsection -> \subsection
# \paragraph -> \subsubsection

# To prevent accidental double substitution, we do a multi-pass or token replacement
body = raw_body
body = re.sub(r"\\subsubsection\{", r"__SUBSECTION_TOKEN__{", body)
body = re.sub(r"\\subsection\{", r"__SECTION_TOKEN__{", body)
body = re.sub(r"\\section\{", r"__CHAPTER_TOKEN__{", body)

body = body.replace("__CHAPTER_TOKEN__{", r"\chapter{")
body = body.replace("__SECTION_TOKEN__{", r"\section{")
body = body.replace("__SUBSECTION_TOKEN__{", r"\subsection{")

# Add Chapter goals to each chapter
chapter_goals = {
    "序論と全体ロードマップ": [
        "組合せ最適化問題におけるNP困難性と計算爆発の背景を理解する",
        "QAOAの歴史的変遷（Farhi原論文 $\\to$ Hadfield XYミキサー $\\to$ FM連携）の全体像をつかむ",
        "本書の各章の関連性と効果的な学習手順を把握する"
    ],
    "初学者のための量子回路超入門：基本ゲート・回路図の読み方": [
        "量子ビット（Qubit）と重ね合わせ状態の幾何学的・代数的表現を習得する",
        "1量子ビット基本ゲート（$X, H, R_Z$）と2量子ビットCNOTゲートの作用を理解する",
        "イジング相互作用 $R_{ZZ}(\\theta)$ をCNOTと $R_Z$ でゲート分解するメカニズムを視覚的に把握する"
    ],
    "数学的基礎：組合せ最適化・イジング模型・断熱量子計算": [
        "QUBO（二次二値最適化）から物理のスピン・イジング模型への厳密な変数変換を修得する",
        "断熱量子計算（AQC）の基本原理、量子断熱定理、および必要時間の公式を導出する",
        "反交差とランダウ・ツェナー遷移による一次相転移の壁と、トロッター分解の数学（BCH公式）を理解する",
        "トロッター化AQCの挫折からQAOAへの歴史的大逆転パラダイムシフトを認識する"
    ],
    "QAOA原論文（Farhi et al., 2014）の完全徹底解読": [
        "Farhiアンサンツ回路（コスト層とミキサー層の交互作用）の数理構造を解明する",
        "量子・古典ハイブリッド最適化ループ（VQA）の作動機序を理解する",
        "単調非減少性と断熱極限における厳密収束性の定理を把握する",
        "局所因果律（Light Cone）に基づき、3正則グラフMaxCutの近似比 0.6924 を厳密導出する"
    ],
    "制約付き最適化の壁：ペナルティ法の破綻メカニズム": [
        "現実の産業最適化に不可避なOne-Hot制約（等式制約）の定式化を理解する",
        "Standard QAOA（ペナルティ法）における有効解比率の指数関数的消滅（次元の呪い）を定量解析する",
        "ペナルティ重み $\\lambda$ のトレードオフと局所解トラップによるアルゴリズム的破綻を解明する"
    ],
    "XYミキサーとQAO Ansatz（Hadfield et al., 2019）の完全解読": [
        "Hadfieldらの「Quantum Alternating Operator Ansatz」の基本設計指針を理解する",
        "XYミキサー演算子 $\\frac{1}{2}(X_i X_j + Y_i Y_j)$ によるスワップ（フリップフロップ）作用を導出する",
        "ハミング重み保存則 $[H_M, \\sum Z_i] = 0$ の代数的一致と、許容部分空間への完全閉じ込めを証明する",
        "W状態による制約充足初期化と、探索空間の劇的圧縮効果を定量比較する"
    ],
    "XYミキサーの数理解析とNISQ実装論（Wang et al., 2020）": [
        "XYミキサーの2パラメータ周期性（$2\\pi$ 対称性）と探索空間のコンパクト性を理解する",
        "完全グラフ型相互作用とリング型（1次元最近接）トポロジーの回路深さを比較する",
        "奇偶（Odd-Even）スワップネットワークによるNISQフレンドリーな回路コンパイルを修得する"
    ],
    "QAOAにおける古典の角度最適化：理論・オプティマイザ・最新戦略": [
        "変分エネルギー地形の3大ボトルネック（非凸性、ショットノイズ、Barren Plateau）を分析する",
        "オプティマイザ（COBYLA vs SPSA vs 勾配法/パラメータシフト則）の得失を比較する",
        "断熱経路の滑らかさを活かした補間ヒューリスティック（INTERP）を修得する",
        "BBOイテレーション間での角度転移学習（Warm-Start）の理論と実践を理解する"
    ],
    "Factorization Machine（FM）サロゲートモデルとの完全調和": [
        "Factorization Machineの2次相互作用構造とQUBOの完全な1対1代数対応を理解する",
        "One-Hot変数に対するFMとXY-QAOAの対称性の調和を把握する",
        "ペナルティ完全ゼロによる自律型ブラックボックス最適化（Self-Driving Lab）パイプラインを俯瞰する"
    ],
    "理解度確認演習問題と解答": [
        "パウリ交換関係の厳密証明を通じて代数計算力を定着させる",
        "2量子ビットXYユニタリの時間発展行列表現をテイラー展開から導出する",
        "W状態と一様状態の制約充足確率の差異を定量計算し、XYミキサーの優位性を確認する"
    ],
    "まとめ": [
        "本書で学んだAQC、QAOA、XYミキサー、古典角度最適化、FM-BBOの全体系を総括する",
        "今後のNISQからFTQCへの展望と実践的研究開発の指針を整理する"
    ]
}

# Insert chaptergoal boxes right after each chapter
def insert_chapter_goals(text):
    for ch_title, goals in chapter_goals.items():
        pattern = re.compile(r"(\\chapter\{" + re.escape(ch_title) + r"\})")
        goals_tex = "\\begin{chaptergoal}\n" + "\n".join([f"    \\item {g}" for g in goals]) + "\n\\end{chaptergoal}\n"
        text = pattern.sub(r"\1\n\n" + goals_tex, text, count=1)
    return text

body = insert_chapter_goals(body)

# Replace alertbox with cautionbox, pointbox with insightbox, paperbox with paperbox
# (or define matching environments)

preamble = r"""\documentclass[11pt,a4paper,openany]{ltjsbook}

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
    headheight=18pt,
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

% ハイパーリンク
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue!75!black,
    citecolor=green!50!black,
    urlcolor=magenta!75!black,
    pdfauthor={量子コンピューティング研究グループ},
    pdftitle={量子近似最適化アルゴリズム（QAOA）体系的教科書}
}

% 装飾ボックス（tcolorbox）
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}

% TikZ
\usepackage{tikz}
\usetikzlibrary{shapes,arrows.meta,positioning,calc,decorations.pathreplacing,fit}

% 見出しスタイリング（titlesec）
\usepackage{titlesec}

% ヘッダー・フッター（fancyhdr）
\usepackage{fancyhdr}

% 目次スタイリング
\usepackage{titletoc}

% -----------------------------------------------------------------------------
% カラーパレット定義（学術教科書プレミアムパレット）
% -----------------------------------------------------------------------------
\definecolor{bookprimary}{RGB}{20, 42, 75}      % ディープネイビー (主色)
\definecolor{booksecondary}{RGB}{0, 125, 145}   % ディープシアン (副色)
\definecolor{bookaccent}{RGB}{125, 30, 65}      % プラムボルドー (定理・強調)
\definecolor{bookgold}{RGB}{195, 120, 10}       % アンバーゴールド (注意・警告)
\definecolor{bookgreen}{RGB}{20, 105, 55}       % フォレストグリーン (例題・成功)
\definecolor{bookbg}{RGB}{248, 250, 252}        % 薄いスレートグレー背景
\definecolor{bookborder}{RGB}{220, 226, 235}    % 境界線グレー
\definecolor{booktext}{RGB}{35, 40, 45}         % 濃墨テキスト

% -----------------------------------------------------------------------------
% 章・節・小節見出しデザイン
% -----------------------------------------------------------------------------
\titleformat{\chapter}[display]
  {\normalfont\gtfamily\bfseries\color{bookprimary}}
  {\raggedleft\fontsize{48}{48}\selectfont\color{bookprimary!22}\textbf{Chapter \thechapter}}
  {-26pt}
  {\huge\raggedright\color{bookprimary}}
  [\vspace{2mm}{\color{bookprimary}\rule{\textwidth}{1.8pt}}\vspace{1mm}]

\titleformat{\section}
  {\normalfont\Large\gtfamily\bfseries\color{bookprimary}}
  {\llap{\color{booksecondary}\rule[-2pt]{4pt}{1.25em}\hspace{7pt}}\thesection}
  {0.8em}
  {}
  [{\color{bookborder}\rule{\textwidth}{0.4pt}}]

\titleformat{\subsection}
  {\normalfont\large\gtfamily\bfseries\color{booksecondary!85!black}}
  {\thesubsection}
  {0.8em}
  {}

\titleformat{\subsubsection}
  {\normalfont\normalsize\gtfamily\bfseries\color{bookprimary}}
  {\thesubsubsection}
  {0.7em}
  {}

% -----------------------------------------------------------------------------
% ヘッダー・フッターデザイン
% -----------------------------------------------------------------------------
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE]{\small\gtfamily\color{bookprimary}\textbf{\leftmark}}
\fancyhead[RO]{\small\gtfamily\color{booksecondary!85!black}\textbf{\rightmark}}
\fancyfoot[C]{\small\color{booktext}― \thepage ―}
\renewcommand{\headrulewidth}{0.5pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{bookprimary!40}\leaders\hrule height \headrulewidth\hfill}}
\renewcommand{\footrulewidth}{0pt}

\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\small\color{booktext}― \thepage ―}
  \renewcommand{\headrulewidth}{0pt}
}

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
% 教科書用 tcolorbox コンポーネント群
% -----------------------------------------------------------------------------
\tcbset{
    enhanced,
    breakable,
    boxrule=0.8pt,
    arc=1.5mm
}

% 章冒頭の到達目標ボックス
\newtcolorbox{chaptergoal}{
    colback=bookprimary!3!white,
    colframe=bookprimary!75!black,
    title={★ 本章の学習到達目標},
    fonttitle=\gtfamily\bfseries\small,
    coltitle=white,
    arc=2mm,
    boxrule=1.0pt,
    top=3mm, bottom=3mm, left=4mm, right=4mm,
    before skip=4mm, after skip=6mm
}

% 直感的理解・物理コラム（旧pointbox）
\newtcolorbox{insightbox}[1][]{
    colback=booksecondary!4!white,
    colframe=booksecondary!80!black,
    title={💡 #1},
    fonttitle=\gtfamily\bfseries,
    coltitle=white,
    arc=1.5mm,
    top=2.5mm, bottom=2.5mm, left=3.5mm, right=3.5mm,
    before skip=3mm, after skip=3mm
}
\let\pointbox\insightbox
\let\endpointbox\endinsightbox

% 原著論文ハイライトボックス
\newtcolorbox{paperbox}[1][]{
    colback=purple!3!white,
    colframe=purple!75!black,
    title={📖 #1},
    fonttitle=\gtfamily\bfseries,
    coltitle=white,
    arc=1.5mm,
    top=2.5mm, bottom=2.5mm, left=3.5mm, right=3.5mm,
    before skip=3mm, after skip=3mm
}

% 注意・落とし穴ボックス（旧alertbox）
\newtcolorbox{cautionbox}[1][]{
    colback=bookgold!4!white,
    colframe=bookgold!90!black,
    title={⚠️ #1},
    fonttitle=\gtfamily\bfseries,
    coltitle=white,
    arc=1.5mm,
    top=2.5mm, bottom=2.5mm, left=3.5mm, right=3.5mm,
    before skip=3mm, after skip=3mm
}
\let\alertbox\cautionbox
\let\endalertbox\endcautionbox

% 定義ボックス
\newtcolorbox{defbox}[1][]{
    colback=bookprimary!3!white,
    colframe=bookprimary!85!black,
    title={【定義】 #1},
    fonttitle=\gtfamily\bfseries,
    coltitle=white,
    arc=1mm,
    top=2.5mm, bottom=2.5mm, left=3.5mm, right=3.5mm,
    before skip=3mm, after skip=3mm
}

% 定理ボックス
\newtcolorbox{thmbox}[1][]{
    colback=bookaccent!3!white,
    colframe=bookaccent!85!black,
    title={【定理】 #1},
    fonttitle=\gtfamily\bfseries,
    coltitle=white,
    arc=1mm,
    top=2.5mm, bottom=2.5mm, left=3.5mm, right=3.5mm,
    before skip=3mm, after skip=3mm
}

% 例題ボックス
\newtcolorbox{examplebox}[1][]{
    colback=bookgreen!3!white,
    colframe=bookgreen!80!black,
    title={📝 例題: #1},
    fonttitle=\gtfamily\bfseries,
    coltitle=white,
    arc=1.5mm,
    top=2.5mm, bottom=2.5mm, left=3.5mm, right=3.5mm,
    before skip=3mm, after skip=3mm
}

% ブラケット記法マクロ
\newcommand{\ket}[1]{|#1\rangle}
\newcommand{\bra}[1]{\langle#1|}
\newcommand{\braket}[2]{\langle#1|#2\rangle}
\newcommand{\mel}[3]{\langle#1|#2|#3\rangle}

\begin{document}

% -----------------------------------------------------------------------------
% 教科書 表紙（Book Cover）
% -----------------------------------------------------------------------------
\begin{titlepage}
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
        発行: 2026年最新版（第1版） $\mid$ 全20ページ完全収録版
    };

    \draw[line width=1.5pt, bookprimary!30] ([yshift=30mm, xshift=25mm]current page.south west) -- ([yshift=30mm, xshift=-25mm]current page.south east);
\end{tikzpicture}
\end{titlepage}

% -----------------------------------------------------------------------------
% 前付（Frontmatter）
% -----------------------------------------------------------------------------
\frontmatter

\chapter*{まえがき（Preface）}
\addcontentsline{toc}{chapter}{まえがき（Preface）}

量子近似最適化アルゴリズム（Quantum Approximate Optimization Algorithm: QAOA）は、誤り耐性量子コンピュータ（FTQC）が実現する前段階である、現代のNISQ（Noisy Intermediate-Scale Quantum）時代において、最も産業的実用性に近い変分量子アルゴリズム（VQA）の一つとして絶大な注目を集めている。

しかし、初学者がQAOAを本格的に学ぼうとする際、以下のような幾重もの巨大な学習の断絶に直面する：
\begin{enumerate}\setlength{\itemsep}{2pt}
    \item \textbf{量子回路と数式の解離}: 量子力学のブラケット表記やパウリ演算子の代数と、実際の量子プロセッサ上で稼働する「量子ゲート配線図」の結びつきが見えにくい。
    \item \textbf{歴史的文脈の欠落}: なぜFarhiらはQAOAを思いついたのか？その直接の母体である「断熱量子計算（AQC）」の物理や、それを離散化する「トロッター分解」の数学的限界を知らないと、QAOAがなぜ「変分パラメータ化」へと舵を切ったのかという本質が理解できない。
    \item \textbf{原論文の難解さ}: Farhiら（2014）の原著論文における「局所因果律（Light Cone）」を用いたMaxCut近似比0.6924の厳密導出は、代数計算が省略されており独力での再現が極めて困難である。
    \item \textbf{実問題での破綻}: MaxCutのようなトイ問題から、実産業のOne-Hot制約付き最適化に進んだ途端、従来のペナルティ法（Standard QAOA）が「解空間の指数爆発」により完全沈黙する。なぜXYミキサー（Hadfield et al., 2019 / Wang et al., 2020）が必要不可欠なのかの必然性が教科書に書かれていない。
    \item \textbf{古典角度最適化の現実}: 理論だけでなく、古典オプティマイザ（COBYLA vs SPSA）の挙動、Barren Plateau問題、補間・転移学習戦略といった実装上の死活問題が抜け落ちている。
\end{enumerate}

本書は、これらの断絶を完全に埋めるべく、\textbf{「数式を省略せず、直感的なTikZ回路図を豊富に交え、原論文の数理から最新のBBO実装論までを一本の美しい論理の糸で紡ぎ出す」}ことを目指して執筆された体系的教科書である。

\vspace{5mm}
\noindent
\textbf{本書の構成と学習フロー}：
\begin{itemize}
    \item \textbf{基礎理論コース（第1章〜第3章）}: 回路図の読み方、イジング模型への変換、AQCとトロッター分解の数理。
    \item \textbf{原論文徹底解読コース（第4章）}: Farhi (2014) のMaxCut近似比0.6924の厳密証明。
    \item \textbf{制約付き最適化とXYミキサーコース（第5章〜第7章）}: ペナルティ法の破綻、Hadfield (2019) の対称性保存則、Wang (2020) のNISQ実装論。
    \item \textbf{古典角度最適化と実践BBOコース（第8章〜第9章）}: 変分最適化、INTERP、Factorization Machineとの調和。
    \item \textbf{実力定着コース（第10章〜第11章）}: 章末演習問題、詳細解答、総括。
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
{
\hypersetup{linkcolor=bookprimary}
\tableofcontents
}

\mainmatter

% -----------------------------------------------------------------------------
% 本文（Main Chapters）
% -----------------------------------------------------------------------------
""" + body + "\n\n\\end{document}\n"

target_path.write_text(preamble, encoding="utf-8")
print("Successfully generated textbook LaTeX file:", target_path)
