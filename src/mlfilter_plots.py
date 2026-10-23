"""Re-plot the six mlfilter studies from the saved summary files (no recomputation).

Shared by analysis/<study>/replot/plot.py and plot.ipynb. Inputs are read from
data/runs/<study>/<config_id>/<run_id>/raw/ after checking their SHA-256 against the
run's manifest.json. The original audit (re-running Neal / FM / QAOA) is NOT repeated;
only figures and aggregate tables are regenerated from the stored summary values.
Needs numpy and matplotlib only.
"""
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WS = Path(__file__).resolve().parents[1]
STUDIES = ["mlfilter", "mlfilter-fixedfm", "mlfilter-diag", "mlfilter-xydepth", "mlfilter-penalty", "mlfilter-pen30"]
CATEGORY = "replot"
matplotlib.rcParams["svg.hashsalt"] = "mlfilter"  # reruns give identical SVG ids


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(study, ws=WS):
    """Locate the single imported run, load config / summary and verify input checksums."""
    ws = Path(ws)
    runs = sorted((ws / "data/runs" / study).glob("c-*/r-*"))
    if len(runs) != 1:
        raise RuntimeError(f"expected exactly one run for {study}, found {len(runs)}")
    run = runs[0]
    manifest = {r["path"]: r["sha256"] for r in json.loads((run / "manifest.json").read_text(encoding="utf-8"))["files"]}
    ctx = dict(ws=ws, study=study, run=run, raw=run / "raw", out=ws / "analysis" / study / CATEGORY, checked=[])
    ctx["cfg"] = json.loads((ws / "experiments" / study / "config.json").read_text(encoding="utf-8"))
    names = {"mlfilter": ["raw/data/lookup_table.npz", "raw/data/best_spectrum.npz", "raw/json/summary.json"]}.get(
        study, ["raw/json/summary.json"] + (["raw/json/trajectories.json"] if study in ("mlfilter-penalty", "mlfilter-pen30") else []))
    for rel in names:
        actual, expected = sha256(run / rel), manifest[rel]
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {run / rel}")
        ctx["checked"].append({"path": str(Path(run / rel).relative_to(ws)), "sha256": actual})
    if study != "mlfilter":
        ctx["summary"] = json.loads((run / "raw/json/summary.json").read_text(encoding="utf-8"))
        assert ctx["summary"]["status"] == "completed"
    if study in ("mlfilter-penalty", "mlfilter-pen30"):
        ctx["traj"] = json.loads((run / "raw/json/trajectories.json").read_text(encoding="utf-8"))
    return ctx


def pct(values):
    return np.percentile(values, [25, 50, 75])


def scatter_iqr(ax, rows, methods, key, label, color_of, marker_of, xticklabels, ylim01, legend_kw, n_jitter=5, s=20, alpha=.55, capsize=3, rotation=35):
    for i, method in enumerate(methods):
        values = [r[key] for r in rows if r["method"] == method]
        q1, med, q3 = pct(values)
        color, marker = color_of(i), marker_of(i)
        ax.scatter(i + np.linspace(-.1, .1, n_jitter), values, color=color, alpha=alpha, s=s)
        ax.errorbar(i, med, yerr=[[med - q1], [q3 - med]], fmt=marker, color=color, capsize=capsize, label=xticklabels[i][1])
    ax.set_xticks(range(len(methods)), [t[0] for t in xticklabels], rotation=rotation, ha="right")
    ax.set_ylabel(label)
    ax.grid(alpha=.2)
    ax.legend(**legend_kw)
    if ylim01:
        ax.set_ylim(-.03, 1.05)


# ---------------------------------------------------------------- per-study figure factories
def figs_mlfilter(c):
    spec = np.load(c["raw"] / "data/best_spectrum.npz")
    fom = np.load(c["raw"] / "data/lookup_table.npz")["fom"]
    assert fom.shape == (4096,)

    def spectra(ax):
        ax.plot(spec["wavelength_nm"], spec["T"], label="Best structure T", color="#2ca02c")
        ax.plot(spec["wavelength_nm"], spec["R"], label="Best structure R", color="#1f77b4", alpha=.7)
        ax.plot(spec["wavelength_nm"], spec["ideal_T"], label="Target T", color="black", linestyle="--")
        ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Power fraction"); ax.set_ylim(-.03, 1.05); ax.legend(); ax.grid(alpha=.2)

    def ecdf(ax):
        ax.plot(np.sort(fom), np.arange(1, 4097) / 4096, label="All 4096 structures", color="#1f77b4")
        ax.set_xlabel("FOM (dimensionless, lower is better)"); ax.set_ylabel("Fraction of structures"); ax.legend(); ax.grid(alpha=.2)

    def overview():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained"); spectra(axes[0]); ecdf(axes[1]); return fig

    def single(draw):
        def f():
            fig, ax = plt.subplots(figsize=(7, 5), layout="constrained"); draw(ax); return fig
        return f
    return [("filter_enumeration_overview", overview), ("best_spectrum", single(spectra)), ("objective_ecdf", single(ecdf))], 300


def _method_figs(c, methods, keys, labels_markers, overview_name, ovsize, singlesize, ylim_keys, color_of, marker_of, legend_kw, **style):
    rows = c["summary"]["per_model"]
    tl = labels_markers

    def draw(ax, key, label):
        scatter_iqr(ax, rows, methods, key, label, color_of, marker_of, tl, key in ylim_keys, legend_kw, **style)

    def overview():
        fig, axes = plt.subplots(2, 2, figsize=ovsize, layout="constrained")
        for ax, (k, l) in zip(axes.flat, keys):
            draw(ax, k, l)
        return fig

    def single(k, l):
        def f():
            fig, ax = plt.subplots(figsize=singlesize, layout="constrained"); draw(ax, k, l); return fig
        return f
    return [(overview_name, overview)] + [(k, single(k, l)) for k, l in keys], 300


def figs_fixedfm(c):
    methods = c["cfg"]["methods"] + c["cfg"]["diagnostic_variants"]
    assert methods == ["Adaptive-FMQA", "LargePenalty-FMQA", "Penalty-FMQAOA", "XY-FMQAOA", "XY-CVaR10", "Categorical-SA-unseen-best"]
    short = ["Adaptive SA", "Large SA100", "X-QAOA", "XY-QAOA", "XY-CVaR10", "CatSA unseen"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c", "#006400", "#9467bd"]
    markers = ["o", "s", "^", "D", "D", "o"]
    keys = [("capture", "Unseen FM-min capture fraction"), ("gap", "Selected FM gap score (FOM units)"),
            ("feasible", "Raw feasible fraction"), ("candidate_rate", "Unseen candidate acquisition fraction")]
    return _method_figs(c, methods, keys, [(s, s) for s in short], "fixed_fm_sampler_overview", (14, 10), (9, 6),
                        {"capture", "feasible", "candidate_rate"}, colors.__getitem__, markers.__getitem__, dict(fontsize=7, ncol=2),
                        s=22, alpha=.6, capsize=4, rotation=25)


def _diag_methods(c):
    return [m for obj in c["cfg"]["objectives"] for m in (f"{obj}-49", f"{obj}-same-1024", f"{obj}-wide-1024")]


def figs_diag(c):
    methods = _diag_methods(c)
    keys = [("ideal_capture100", "Ideal FM-min capture probability / 100 shots"), ("observed_capture", "Observed unseen FM-min capture fraction"),
            ("gap", "Selected FM gap score (FOM units)"), ("seen_probability", "Probability on evaluated candidates")]
    return _method_figs(c, methods, keys, [(m, m) for m in methods], "p1_sampler_diagnosis_overview", (18, 12), (12, 6),
                        {"ideal_capture100", "observed_capture", "seen_probability"},
                        lambda i: ["#2ca02c", "#006400", "#9467bd"][i // 3], lambda i: ["o", "s", "D"][i % 3], dict(fontsize=6, ncol=3))


def figs_xydepth(c):
    methods = [f"{obj}-p{d}-1024" for obj in c["cfg"]["objectives"] for d in (1, 2, 3)]
    keys = [("ideal_capture100", "Ideal unseen FM-min capture /100 shots"), ("observed_capture", "Observed unseen FM-min capture fraction"),
            ("gap", "Selected FM gap score (FOM units)"), ("search_seconds", "Angle search wall time (s)")]
    return _method_figs(c, methods, keys, [(m, m) for m in methods], "xy_depth_overview", (18, 12), (12, 6),
                        {"ideal_capture100", "observed_capture"},
                        lambda i: ["#2ca02c", "#006400", "#9467bd"][i // 3], lambda i: ["o", "s", "D"][i % 3], dict(fontsize=6, ncol=3))


def figs_penalty(c):
    P, S = c["cfg"], c["summary"]
    fixed, bbo, hist = S["fixed_per_model"], S["BBO_per_run"], c["traj"]
    assert len(fixed) == 50 and len(bbo) == 55
    alphas, conditions, seeds = P["alphas"], P["conditions"], P["seeds"]
    colors = plt.get_cmap("viridis")(np.linspace(.05, .9, len(alphas)))

    def curve(ax, rows, key, label, adaptive=False):
        med, q1, q3 = [], [], []
        for a in alphas:
            rr = [r[key] for r in rows if r.get("alpha") == a and not r.get("adaptive", False)]
            lo, m, hi = pct(rr); q1.append(lo); med.append(m); q3.append(hi)
        ax.plot(alphas, med, "o-", color="#ff7f0e", label="Fixed alpha median"); ax.fill_between(alphas, q1, q3, color="#ff7f0e", alpha=.2, label="Fixed alpha IQR")
        if adaptive:
            rr = [r[key] for r in rows if r.get("adaptive")]
            lo, m, hi = pct(rr); ax.axhline(m, color="#1f77b4", label="Adaptive median"); ax.axhspan(lo, hi, color="#1f77b4", alpha=.12, label="Adaptive IQR")
        ax.set_xscale("log"); ax.set_xlabel("Normalized penalty alpha (lambda = S alpha)"); ax.set_ylabel(label); ax.legend(fontsize=8); ax.grid(alpha=.2)
        if key in ("capture", "feasible"):
            ax.set_ylim(-.03, 1.05)

    panels = [(fixed, "capture", "Fixed-FM unseen minimum capture", False), (fixed, "feasible", "Fixed-FM raw feasible fraction", False),
              (bbo, "final_regret", "BBO final normalized regret", True), (bbo, "gap", "BBO median selected FM gap score", True)]

    def overview():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
        for ax, args in zip(axes.flat, panels):
            curve(ax, *args)
        return fig

    def single(args):
        def f():
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained"); curve(ax, *args); return fig
        return f

    def by_cycle():
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        for i, cond in enumerate(conditions):
            arr = np.array([hist[f"{s}:{cond['id']}"]["best_by_cycle"] for s in seeds]); lo, med, hi = np.percentile(arr, [25, 50, 75], axis=0)
            color = "#1f77b4" if cond["adaptive"] else colors[i]
            ax.plot(range(61), med, color=color, label=cond["id"]); ax.fill_between(range(61), lo, hi, color=color, alpha=.07)
        ax.set_xlabel("SA proposal cycle (after 40 initial evaluations)"); ax.set_ylabel("Best optical FOM"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
        return fig

    def by_eval(seed):
        def f():
            fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
            for i, cond in enumerate(conditions):
                h = np.array(hist[f"{seed}:{cond['id']}"]["best_by_evaluations"]); color = "#1f77b4" if cond["adaptive"] else colors[i]
                ax.plot(h[:, 0], h[:, 1], color=color, label=cond["id"]); ax.scatter(h[-1, 0], h[-1, 1], color=color, s=15)
            ax.set_xlabel("Actual black-box evaluations"); ax.set_ylabel("Best optical FOM"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.2)
            return fig
        return f
    figs = [("FMQA_penalty_overview", overview)]
    figs += [(n, single(a)) for n, a in zip(["fixed_capture", "fixed_feasible", "BBO_final_regret", "BBO_FM_gap"], panels)]
    figs += [("BBO_by_proposal_cycle", by_cycle)] + [(f"BBO_true_evaluations_seed{s}", by_eval(s)) for s in seeds]
    return figs, 300


def figs_pen30(c):
    S, P = c["summary"], c["cfg"]
    rows, hist, opt = S["per_run"], c["traj"], S["true_optimum"]
    methods = ["alpha1", "alpha3", "alpha100", "Random"]
    assert len(rows) == 120
    styles = {"alpha1": ("#0072B2", "-", "o"), "alpha3": ("#009E73", "--", "D"), "alpha100": ("#E69F00", "-.", "s"), "Random": ("#555555", ":", "^")}

    def draw_hit(ax):
        budgets = np.arange(40, 101)
        for m in methods:
            rr = [r for r in rows if r["method"] == m]
            counts = np.array([sum(r["firsthit"] is not None and r["firsthit"] <= n for r in rr) for n in budgets]); color, ls, _ = styles[m]
            ph = counts / 30; z = 1.959963984540054; center = (ph + z * z / 60) / (1 + z * z / 30); half = z * np.sqrt(ph * (1 - ph) / 30 + z * z / (4 * 900)) / (1 + z * z / 30)
            ax.step(budgets, ph, where="post", color=color, ls=ls, label=m); ax.fill_between(budgets, center - half, center + half, step="post", color=color, alpha=.09)
        ax.set(xlabel="True evaluations (including 40 initial)", ylabel="Observed optimum-hit fraction (n=30)", ylim=(-.03, 1.03)); ax.legend(fontsize=9); ax.grid(alpha=.2)

    def draw_final(ax):
        data = [[r["final_best"] for r in rows if r["method"] == m] for m in methods]
        ax.boxplot(data, tick_labels=methods, showfliers=False)
        for i, (m, values) in enumerate(zip(methods, data), 1):
            jitter = np.random.default_rng(1016).uniform(-.13, .13, len(values)); ax.scatter(i + jitter, values, s=15, color=styles[m][0], alpha=.65, label=m)
        ax.axhline(opt, color="black", ls=":", label="Exact optimum"); ax.set_ylabel("Final best optical FOM"); ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8)

    def overview():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained"); draw_hit(axes[0]); draw_final(axes[1]); return fig

    def single(draw):
        def f():
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained"); draw(ax); return fig
        return f

    def convergence():
        end = min(r["actual_evaluations"] for r in rows)
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for m in methods:
            values = np.array([hist[f"{s}:{m}"]["by_evaluations"][39:end] for s in P["seeds"]]); lo, med, hi = np.percentile(values, [25, 50, 75], axis=0)
            color, ls, _ = styles[m]; ax.plot(range(40, end + 1), med, color=color, ls=ls, label=m); ax.fill_between(range(40, end + 1), lo, hi, color=color, alpha=.12)
        ax.set(xlabel="True evaluations (common observed range)", ylabel="Best optical FOM; median and IQR"); ax.legend(); ax.grid(alpha=.2)
        return fig
    def with_font(f):  # original script set font.size=11 for this study only
        def g():
            with plt.rc_context({"font.size": 11}):
                return f()
        return g
    return [(n, with_font(f)) for n, f in [("validation30_overview", overview), ("optimum_hit_by_evaluations", single(draw_hit)),
            ("final_FOM", single(draw_final)), ("convergence_common_observed", convergence)]], 220


FIGS = {"mlfilter": figs_mlfilter, "mlfilter-fixedfm": figs_fixedfm, "mlfilter-diag": figs_diag, "mlfilter-xydepth": figs_xydepth,
        "mlfilter-penalty": figs_penalty, "mlfilter-pen30": figs_pen30}


# ---------------------------------------------------------------- tables
def _agg(rows, group, metrics):
    out = []
    for g in dict.fromkeys(r[group] for r in rows):
        rr = [r for r in rows if r[group] == g]
        rec = {group: g, "n": len(rr)}
        for k in metrics:
            vals = [r[k] for r in rr if r.get(k) is not None]
            if vals:
                q1, m, q3 = pct(vals); rec.update({f"{k}_q25": q1, f"{k}_median": m, f"{k}_q75": q3})
        out.append(rec)
    return out


def tables(c):
    s = c["study"]
    if s == "mlfilter":
        fom = np.load(c["raw"] / "data/lookup_table.npz")["fom"]
        spec = np.load(c["raw"] / "data/best_spectrum.npz")
        return {"fom_ecdf.csv": [{"rank": i + 1, "fom": float(v), "fraction": (i + 1) / 4096} for i, v in enumerate(np.sort(fom))],
                "best_spectrum.csv": [{"wavelength_nm": float(w), "T": float(t), "R": float(r), "target_T": float(i)}
                                      for w, t, r, i in zip(spec["wavelength_nm"], spec["T"], spec["R"], spec["ideal_T"])]}
    S = c["summary"]
    if s in ("mlfilter-fixedfm", "mlfilter-diag", "mlfilter-xydepth"):
        rows = S["per_model"]; keys = [k for k in rows[0] if k not in ("seed", "method")]
        return {"per_model_rows.csv": rows, "method_median_iqr.csv": _agg(rows, "method", keys)}
    if s == "mlfilter-penalty":
        f, b = S["fixed_per_model"], S["BBO_per_run"]
        return {"fixed_fm_rows.csv": f, "bbo_run_rows.csv": b,
                "fixed_fm_by_alpha.csv": _agg(f, "alpha", ["capture", "gap", "feasible", "candidate_rate"]),
                "bbo_by_condition.csv": _agg(b, "condition", ["final_best", "final_regret", "firsthit_score101", "actual_evaluations", "feasible", "gap"])}
    rows = S["per_run"]
    return {"per_run_rows.csv": rows, "method_median_iqr.csv": _agg(rows, "method", ["final_best", "final_regret", "capped_firsthit101", "actual_evaluations", "seconds"])}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------- run / compare
def compare_with_original(c, names):
    """Pixel-level difference between the redrawn PNGs and the copied original PNGs."""
    rows = []
    for n in names:
        new, old = c["out"] / "png" / f"{n}.png", c["ws"] / "analysis" / c["study"] / "overview/png" / f"{n}.png"
        rec = {"figure": n, "original_exists": old.exists()}
        if old.exists():
            a, b = plt.imread(new), plt.imread(old)
            rec.update(same_shape=a.shape == b.shape, new_shape="x".join(map(str, a.shape)), original_shape="x".join(map(str, b.shape)))
            if a.shape == b.shape:
                d = np.abs(a.astype(float) - b.astype(float)); rec.update(mean_abs_diff=float(d.mean()), max_abs_diff=float(d.max()), differing_pixel_fraction=float((d.max(axis=-1) > 1e-6).mean()))
        rows.append(rec)
    return rows


def run(study, ws=WS):
    c = load(study, ws)
    figs, dpi = FIGS[study](c)
    out = c["out"]
    for sub in ("png", "pdf", "svg", "csv"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    files = []
    for name, make in figs:
        fig = make()
        for ext in ("png", "pdf", "svg"):
            meta = {"png": {"Software": None}, "pdf": {"CreationDate": None, "Creator": None, "Producer": None}, "svg": {"Date": None, "Creator": None}}[ext]
            fig.savefig(out / ext / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", metadata=meta)
        plt.close(fig)
    for fname, rows in tables(c).items():
        write_csv(out / "csv" / fname, rows)
    write_csv(out / "csv/original_figure_comparison.csv", compare_with_original(c, [n for n, _ in figs]))
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.parent.name in ("png", "pdf", "svg", "csv"):
            files.append({"path": str(p.relative_to(out)), "size": p.stat().st_size, "sha256": sha256(p)})
    import numpy, matplotlib as mpl  # noqa: E401
    env = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(), "numpy": numpy.__version__, "matplotlib": mpl.__version__}
    (out / "run_record.json").write_text(json.dumps({"study": study, "inputs_verified": c["checked"], "environment": env, "dpi": dpi, "outputs": files}, ensure_ascii=False, indent=1), encoding="utf-8")
    return c, [n for n, _ in figs], files


if __name__ == "__main__":
    for st in (sys.argv[1:] or STUDIES):
        _, names, files = run(st)
        print(st, len(names), "figures,", len(files), "files")
