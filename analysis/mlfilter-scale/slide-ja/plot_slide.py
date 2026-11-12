"""Slide versions of the result figures for docs/slide/slide_ja.pptx (2026-11-12).

Drawn at the size they are placed on the slide (inches = slide inches), Japanese labels,
>= 13 pt text, one fixed colour per method across all figures. Reads only csv/ in this folder.
usage: python plot_slide.py   -> png/ and pdf/ (stems slide_*)
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
CSV = HERE / "csv"
# proportional Japanese gothic (MS UI Gothic is not loadable from its .ttc by matplotlib; MS Gothic is monospaced)
FAMILY = "Hiragino Sans" if any(f.name == "Hiragino Sans" for f in font_manager.fontManager.ttflist) else "YuGothic"
plt.rcParams.update({
    "font.family": FAMILY, "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 13, "mathtext.fontset": "stix", "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "axes.unicode_minus": False, "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})

# fixed method styles (same meaning on every slide)
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a78d6", "#e8682f", "#1a9e6e", "#7b4fb3", "#8c8c8c"
M = {
    "XY-FMQAOA": dict(color=BLUE, ls="-", lw=3.0, label="XY-FMQAOA（$\\lambda=0$）"),
    "alpha1": dict(color="black", ls="--", lw=2.0, label="SA，$\\alpha=1$（調整済み）"),
    "alpha100": dict(color=ORANGE, ls="-", lw=2.0, label="SA，$\\alpha=100$（大きすぎ）"),
    "alpha0.3": dict(color=GREEN, ls="-.", lw=2.0, label="SA，$\\alpha=0.3$（小さすぎ）"),
    "CatSA": dict(color=PURPLE, ls=":", lw=2.4, label="カテゴリ型 SA"),
    "RandomFM": dict(color=GRAY, ls="--", lw=1.6, label="ランダム提案 + FM 選択"),
}
ALPHAS = [0.3, 1, 3, 10, 30, 100, 1000]
ATICK = ["0.3", "1", "3", "10", "30", "100", "1000"]


def read(name):
    with open(CSV / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    for ext in ("png", "pdf"):
        (HERE / ext).mkdir(exist_ok=True)
        fig.savefig(HERE / ext / f"{stem}.{ext}", bbox_inches=None, facecolor="white")
    plt.close(fig)
    from PIL import Image   # opaque RGB PNG: PowerPoint's PDF export greys out RGBA images
    png = HERE / "png" / f"{stem}.png"
    Image.open(png).convert("RGB").save(png)


def alpha_axis(ax):
    ax.set_xscale("log"); ax.set_xticks(ALPHAS); ax.set_xticklabels(ATICK); ax.minorticks_off(); ax.set_xlim(0.22, 1400)


def summary():
    d = {}
    for r in read("solver_summary.csv"):
        d[(int(r["layers"]), r["sampler"], r["metric"])] = (float(r["median"]), float(r["q25"]), float(r["q75"]))
    return d


def fig_sweetspot():
    """Fixed FM: (a) capture rate of the unseen FM minimum, (b) sample FM percentile and feasible fraction, vs alpha."""
    d = summary()
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.15), sharex=True)
    shades = {6: ("#b0b0b0", "o"), 7: ("#606060", "s"), 8: ("black", "^")}
    for ax in axes:
        ax.axvspan(0.7, 4.3, color="#e6f0fb", lw=0, zorder=0)
        alpha_axis(ax)
    for L, (c, mk) in shades.items():
        cap = [d.get((L, f"SA-a{a:g}", "capture"), (np.nan,))[0] for a in ALPHAS]
        axes[0].plot(ALPHAS, cap, color=c, marker=mk, ms=7, lw=1.8, label=f"$L={L}$")
        pct = [d.get((L, f"SA-a{a:g}", "sample_fm_pct"), (np.nan,))[0] for a in ALPHAS]
        fea = [d.get((L, f"SA-a{a:g}", "feasible"), (np.nan,))[0] for a in ALPHAS]
        axes[1].plot(ALPHAS, pct, color=c, marker=mk, ms=7, lw=1.8)
        axes[1].plot(ALPHAS, fea, color=c, marker=mk, ms=6, lw=1.4, ls=":", mfc="white")
    axes[0].set_ylabel("FM 最良解の\n発見率"); axes[0].set_ylim(-0.05, 1.12); axes[0].set_yticks([0, 0.5, 1])
    axes[0].legend(loc="upper right", ncol=3, handlelength=1.6, columnspacing=1.0, fontsize=12)
    axes[1].axhline(0.5, color=GRAY, ls="--", lw=1.2)
    axes[1].text(1400, 0.53, "ランダム（0.5）", ha="right", va="bottom", fontsize=12, color="#555555")
    axes[1].text(40, 0.04, "実線：FM 百分位（0 = 最良）", fontsize=12, va="bottom", ha="center")
    axes[1].text(14, 0.80, "点線：制約を満たす割合", fontsize=12, va="center")
    axes[1].set_ylabel("FM 百分位 /\n制約を満たす割合"); axes[1].set_ylim(-0.05, 1.12); axes[1].set_yticks([0, 0.5, 1])
    axes[1].set_xlabel("ペナルティ関数係数 $\\alpha$（$\\lambda=\\alpha S$）")
    for ax, t in zip(axes, "ab"):
        ax.text(-0.02, 1.02, f"({t})", transform=ax.transAxes, ha="right", va="bottom", fontsize=13)
    fig.tight_layout(h_pad=0.6)
    save(fig, "slide_sweetspot")


def fig_depth():
    """Fixed FM: normalized FM gap of the selected candidate vs XY depth p, with SA references, L = 6, 7, 8."""
    d = summary()
    fig, axes = plt.subplots(1, 3, figsize=(7.9, 4.6), sharey=True)
    refs = [(1, "black", "-", "SA $\\alpha=1$"), (3, "#555555", "--", "SA $\\alpha=3$"), (10, "#555555", "-.", "SA $\\alpha=10$"),
            (100, ORANGE, ":", "SA $\\alpha=100$")]
    ncand = {6: "4,096", 7: "16,384", 8: "65,536"}
    for ax, L in zip(axes, (6, 7, 8)):
        for a, c, ls, lab in refs:
            ax.axhline(d[(L, f"SA-a{a}", "norm_gap")][0], color=c, ls=ls, lw=1.6, label=lab)
        ax.axhline(d[(L, "RandomFM", "norm_gap")][0], color=GRAY, ls="--", lw=1.2, label="ランダム提案")
        p = np.arange(1, 7)
        med = np.array([d[(L, f"XY-p{k}-cvar10", "norm_gap")][0] for k in p])
        lo = np.array([d[(L, f"XY-p{k}-cvar10", "norm_gap")][1] for k in p])
        hi = np.array([d[(L, f"XY-p{k}-cvar10", "norm_gap")][2] for k in p])
        ax.fill_between(p, lo, hi, color=BLUE, alpha=0.18, lw=0)
        ax.plot(p, med, color=BLUE, lw=3.0, marker="o", ms=7, label="XY-FMQAOA")
        ax.set_xticks(p); ax.set_xlim(0.6, 6.4); ax.set_ylim(-0.003, 0.09)
        ax.set_title(f"$L={L}$（{ncand[L]} 候補）", fontsize=13)
        ax.set_xlabel("XY-QAOA の深さ $p$")
    axes[0].set_ylabel("FM ギャップ（小さいほど良い）")
    h, l = axes[0].get_legend_handles_labels()
    order = [5, 0, 1, 2, 3, 4]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", ncol=3, fontsize=12, handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=0.6)
    save(fig, "slide_depth")


def fig_loop_alpha():
    """Closed loop (L = 7, 15 seeds): best rank after 100 evaluations vs alpha."""
    rows = read("loop_alpha_sweep.csv")
    by = {}
    for r in rows:
        by.setdefault(r["condition"], []).append(int(r["best_rank"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    med = [np.median(by[f"alpha{a:g}"]) for a in ALPHAS]
    q1 = [np.percentile(by[f"alpha{a:g}"], 25) for a in ALPHAS]
    q3 = [np.percentile(by[f"alpha{a:g}"], 75) for a in ALPHAS]
    ax.fill_between(ALPHAS, q1, q3, color="#d9d9d9", lw=0, label="SA の IQR")
    ax.plot(ALPHAS, med, color="black", marker="o", ms=8, lw=2.0, label="SA（中央値）")
    ax.axhline(np.median(by["XY-FMQAOA"]), color=BLUE, lw=3.0, label="XY-FMQAOA（中央値）")
    ax.axhline(np.median(by["RandomFM"]), color=GRAY, ls="--", lw=1.6, label="ランダム提案 + FM 選択")
    ax.set_yscale("log"); ax.set_ylim(0.75, 1500); ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(["1", "3", "10", "30", "100"])
    ax.minorticks_off(); alpha_axis(ax)
    ax.set_xlabel("ペナルティ関数係数 $\\alpha$"); ax.set_ylabel("100 回評価後の最良解の順位\n（1 = 最適解）")
    reach = {a: sum(v == 1 for v in by[f"alpha{a:g}"]) for a in ALPHAS}
    for a in ALPHAS:
        ax.text(a, 850, f"{reach[a]}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.axhline(420, color="#bbbbbb", lw=0.8)
    ax.text(0.25, 1000, "", fontsize=1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=12, labelspacing=0.3)
    ax.set_title("数字：最適解に到達した Seed 数（15 中）", fontsize=13, loc="left")
    fig.tight_layout()
    save(fig, "slide_loop_alpha")


def fig_confirm2():
    """Second confirmation (L = 7, 30 seeds): median best rank and fraction of seeds that found the optimum vs evaluations."""
    rows = read("confirm2_curves.csv")
    conds = ["XY-FMQAOA", "alpha1", "CatSA", "alpha100", "alpha0.3"]
    E = np.arange(40, 101)
    traj = {}
    for r in rows:
        if r["condition"] in conds:
            traj.setdefault((r["condition"], r["seed"]), {})[int(r["evaluations"])] = int(r["best_rank"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    for c in conds:
        mats = []
        for (cc, s), t in traj.items():
            if cc != c:
                continue
            v, last = [], None
            for e in E:
                last = t.get(e, last)
                v.append(last)
            mats.append(v)
        A = np.array(mats, dtype=float)
        st = {k: M[c][k] for k in ("color", "ls", "lw")}
        axes[0].plot(E, np.median(A, axis=0), **st, label=M[c]["label"])
        axes[1].plot(E, (A == 1).mean(axis=0), **st)
    axes[0].set_yscale("log"); axes[0].set_ylim(0.8, 400); axes[0].set_yticks([1, 3, 10, 30, 100, 300]); axes[0].set_yticklabels(["1", "3", "10", "30", "100", "300"])
    axes[0].minorticks_off(); axes[0].set_ylabel("最良解の順位（中央値）"); axes[0].set_xlabel("評価回数")
    axes[1].set_ylim(-0.02, 1.02); axes[1].set_ylabel("最適解に到達した割合"); axes[1].set_xlabel("評価回数")
    for ax, t in zip(axes, "ab"):
        ax.set_xlim(40, 100)
        ax.text(-0.13, 1.0, f"({t})", transform=ax.transAxes, ha="right", va="bottom", fontsize=13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=13, handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(w_pad=2.5, rect=(0, 0.17, 1, 1))
    save(fig, "slide_confirm2")


if __name__ == "__main__":
    fig_sweetspot(); fig_depth(); fig_loop_alpha(); fig_confirm2()
    print("font:", FAMILY)
