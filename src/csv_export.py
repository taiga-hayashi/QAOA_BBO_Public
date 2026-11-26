"""Aggregate raw per-target outputs (`outputs/<mode>/runs/<target>/`) into `results/csv/<target>.csv`."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .optics import fom_to_ranks

ROOT = Path(__file__).resolve().parents[1]


def resolve_out_dir(out_dir: Path | None, *, smoke: bool) -> Path:
    if out_dir is not None:
        return out_dir
    return (ROOT / "outputs" / "smoke" / "csv") if smoke else (ROOT / "results" / "csv")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fieldnames})
    tmp.replace(path)
    print(f"[csv] {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {len(rows)} rows", flush=True)


def _load_completed(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("status") == "completed" else None


def _load_truth(run_dir: Path, layers: int) -> tuple[np.ndarray, np.ndarray] | None:
    npz_path = run_dir / "truth" / f"L{layers}_truth.npz"
    if not npz_path.exists():
        return None
    fom = np.load(npz_path)["fom"]
    return fom, fom_to_ranks(fom)


def _rank_stats(values: list[int]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_seeds": int(len(arr)),
        "best_rank_median": float(np.median(arr)),
        "best_rank_q25": float(np.percentile(arr, 25)),
        "best_rank_q75": float(np.percentile(arr, 75)),
        "n_optimum": int(np.sum(arr == 1)),
        "best_rank_mean": float(np.mean(arr)),
        "best_rank_se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
    }


def _padded_traj(values: np.ndarray, budget: int) -> np.ndarray:
    acc = np.minimum.accumulate(values[:budget])
    if len(acc) < budget:
        acc = np.pad(acc, (0, budget - len(acc)), mode="edge")
    return acc


def _collect_closed_loop_ranks(tc: dict, tdir: Path, rank_table: np.ndarray) -> dict[str, dict[int, int]]:
    budget = int(tc["budget"])
    ranks_by_cond: dict[str, dict[int, int]] = {}
    for cond in tc["conditions"]:
        cid = cond["id"]
        for seed in tc["seeds"]:
            doc = _load_completed(tdir / f"{cid}_seed{seed}.json")
            if doc is not None:
                ids = np.asarray(doc["final_ids"][:budget], dtype=np.int64)
                ranks_by_cond.setdefault(cid, {})[int(seed)] = int(rank_table[ids].min())
    return ranks_by_cond


def _export_l678_fom_curves(tc: dict, tdir: Path, run_dir: Path, csv_dir: Path) -> None:
    rows: list[dict] = []
    for lc in tc["layers_configs"]:
        layers = int(lc["layers"])
        truth_pair = _load_truth(run_dir, layers)
        if truth_pair is None:
            continue
        fom_table, ranks_table = truth_pair
        global_opt = float(fom_table.min())
        budget = int(lc["budget"])
        init_pts = int(lc["initial_points"])

        for cid in sorted(c["id"] for c in lc["conditions"]):
            fom_trajs, rank_trajs = [], []
            for seed in lc["seeds"]:
                doc = _load_completed(tdir / f"L{layers}_{cid}_seed{seed}.json")
                if doc is None:
                    continue
                ids = np.asarray(doc["final_ids"], dtype=np.int64)
                fom_trajs.append(_padded_traj(fom_table[ids], budget))
                rank_trajs.append(_padded_traj(ranks_table[ids], budget))
            if not fom_trajs:
                continue
            fom_mat = np.vstack(fom_trajs)
            rank_mat = np.vstack(rank_trajs)
            n_seeds = len(fom_trajs)
            for evals in range(init_pts, budget + 1):
                col_f = fom_mat[:, evals - 1]
                col_r = rank_mat[:, evals - 1]
                rows.append({
                    "layers": layers,
                    "condition": cid,
                    "cycle": evals - init_pts,
                    "evaluations": evals,
                    "n_seeds": n_seeds,
                    "best_fom_median": float(np.median(col_f)),
                    "best_fom_q25": float(np.percentile(col_f, 25)),
                    "best_fom_q75": float(np.percentile(col_f, 75)),
                    "n_optimum": int(np.sum(col_r == 1)),
                    "global_optimum_fom": global_opt,
                })

    if not rows:
        print(f"[csv] skip {tc['csv']} (no completed runs)", flush=True)
        return
    _write_csv(
        csv_dir / tc["csv"],
        ["layers", "condition", "cycle", "evaluations", "n_seeds", "best_fom_median", "best_fom_q25", "best_fom_q75", "n_optimum", "global_optimum_fom"],
        rows,
    )


def _export_l7_alpha_sweep(tc: dict, tdir: Path, run_dir: Path, csv_dir: Path) -> None:
    truth_pair = _load_truth(run_dir, int(tc["layers"]))
    if truth_pair is None or not tdir.exists():
        print(f"[csv] skip {tc['csv']} (no completed runs)", flush=True)
        return
    ranks_by_cond = _collect_closed_loop_ranks(tc, tdir, truth_pair[1])
    rows = []
    for cond in tc["conditions"]:
        cid = cond["id"]
        if cid not in ranks_by_cond:
            continue
        alpha_val = f"{cond['alpha']:g}" if "alpha" in cond else None
        rows.append({"condition": cid, "alpha": alpha_val, **_rank_stats(list(ranks_by_cond[cid].values()))})
    if rows:
        _write_csv(
            csv_dir / tc["csv"],
            ["condition", "alpha", "n_seeds", "best_rank_median", "best_rank_q25", "best_rank_q75", "n_optimum", "best_rank_mean", "best_rank_se"],
            rows,
        )


def _export_l7_paired_ranks(tc: dict, tdir: Path, run_dir: Path, csv_dir: Path) -> None:
    truth_pair = _load_truth(run_dir, int(tc["layers"]))
    if truth_pair is None or not tdir.exists():
        print(f"[csv] skip {tc['csv']} (no completed runs)", flush=True)
        return
    ranks_by_cond = _collect_closed_loop_ranks(tc, tdir, truth_pair[1])
    xy_ranks = ranks_by_cond.get("XY-FMQAOA", {})
    rows = []
    for comp in tc.get("comparators", ["alpha0.3", "alpha100", "alpha1"]):
        comp_ranks = ranks_by_cond.get(comp, {})
        for seed in sorted(set(comp_ranks) & set(xy_ranks)):
            cr, xr = comp_ranks[seed], xy_ranks[seed]
            outcome = "xy_better" if xr < cr else ("tie" if xr == cr else "xy_worse")
            rows.append({
                "comparator": comp, "seed": seed,
                "comparator_rank": cr, "xy_rank": xr, "outcome": outcome,
            })
    if rows:
        _write_csv(csv_dir / tc["csv"], ["comparator", "seed", "comparator_rank", "xy_rank", "outcome"], rows)


def _export_l7_schedule_retune(tc: dict, tdir: Path, run_dir: Path, csv_dir: Path) -> None:
    truth_pair = _load_truth(run_dir, int(tc["layers"]))
    if truth_pair is None or not tdir.exists():
        print(f"[csv] skip {tc['csv']} (no completed runs)", flush=True)
        return
    ranks_by_cond = _collect_closed_loop_ranks(tc, tdir, truth_pair[1])
    rows = []
    for cond in tc["conditions"]:
        cid = cond["id"]
        for seed in tc["seeds"]:
            if seed in ranks_by_cond.get(cid, {}):
                rows.append({"condition": cid, "seed": seed, "best_rank": ranks_by_cond[cid][seed]})
    if rows:
        _write_csv(csv_dir / tc["csv"], ["condition", "seed", "best_rank"], rows)


def _export_explore_5seeds(tc: dict, tdir: Path, run_dir: Path, csv_dir: Path) -> None:
    if not tdir.exists():
        print(f"[csv] skip {tc['csv']} (no completed runs)", flush=True)
        return
    budget = int(tc["budget"])
    init_pts = int(tc["initial_points"])
    rows = []
    for layers in tc["layers_list"]:
        truth_pair = _load_truth(run_dir, int(layers))
        if truth_pair is None:
            continue
        _, ranks_table = truth_pair
        for cond in tc["conditions"]:
            cid = cond["id"]
            trajs = []
            for seed in tc["seeds"]:
                doc = _load_completed(tdir / f"L{layers}_{cid}_seed{seed}.json")
                if doc is not None:
                    ids = np.asarray(doc["final_ids"], dtype=np.int64)
                    trajs.append(_padded_traj(ranks_table[ids], budget))
            if not trajs:
                continue
            mat = np.vstack(trajs)
            n_seeds = len(trajs)
            for evals in range(init_pts, budget + 1):
                col = mat[:, evals - 1]
                rows.append({
                    "layers": layers, "condition": cid, "evaluations": evals,
                    "n_seeds": n_seeds, "best_rank_median": float(np.median(col)),
                    "frac_optimum": float(np.mean(col == 1)),
                })
    if rows:
        _write_csv(csv_dir / tc["csv"], ["layers", "condition", "evaluations", "n_seeds", "best_rank_median", "frac_optimum"], rows)


def _collect_fixed_fm_means(tc: dict, tdir: Path) -> dict[tuple[int, str], list[dict]]:
    per_fm_means: dict[tuple[int, str], list[dict]] = {}
    metrics = ("capture", "norm_gap", "feasible", "sample_fm_pct")
    for layers in tc["layers_list"]:
        for seed in tc["seeds"]:
            doc = _load_completed(tdir / f"L{layers}_seed{seed}.json")
            if doc is None:
                continue
            by_sampler: dict[str, list[dict]] = {}
            for row in doc["batches"]:
                by_sampler.setdefault(row["sampler"], []).append(row)
            for smp, rows in by_sampler.items():
                agg_row: dict = {"seed": seed}
                for k in metrics:
                    vals = [float(r[k]) for r in rows if r.get(k) is not None]
                    agg_row[k] = float(np.mean(vals)) if vals else None
                per_fm_means.setdefault((int(layers), smp), []).append(agg_row)
    return per_fm_means


def _fm_stats(
    per_fm_means: dict[tuple[int, str], list[dict]],
    layers: int,
    sampler: str,
    key: str,
    *,
    require_all: bool = False,
) -> tuple[float | None, float | None, float | None, float | None, float | None, int]:
    rows = per_fm_means.get((layers, sampler), [])
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals or (require_all and len(vals) < len(rows)):
        return None, None, None, None, None, 0
    arr = np.asarray(vals, dtype=np.float64)
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75)), float(np.mean(arr)), se, len(arr)


def _export_fixedfm_alpha_sweep(tc: dict, tdir: Path, _run_dir: Path, csv_dir: Path) -> None:
    means = _collect_fixed_fm_means(tc, tdir)
    rows = []
    for layers in tc["layers_list"]:
        for alpha in tc["sa_alphas"]:
            smp = f"SA-auto-a{alpha:g}"
            cap, _, _, cap_mean, _, n_cap = _fm_stats(means, layers, smp, "capture")
            pct, _, _, pct_mean, _, _ = _fm_stats(means, layers, smp, "sample_fm_pct", require_all=True)
            fea, _, _, fea_mean, _, _ = _fm_stats(means, layers, smp, "feasible")
            if n_cap:
                rows.append({
                    "layers": layers, "alpha": f"{alpha:g}",
                    "capture_median": cap, "sample_fm_pct_median": pct, "feasible_median": fea, "n_fm": n_cap,
                    "capture_mean": cap_mean, "sample_fm_pct_mean": pct_mean, "feasible_mean": fea_mean,
                })
    if rows:
        _write_csv(
            csv_dir / tc["csv"],
            ["layers", "alpha", "capture_median", "sample_fm_pct_median", "feasible_median", "n_fm", "capture_mean", "sample_fm_pct_mean", "feasible_mean"],
            rows,
        )


def _export_fixedfm_depth_sweep(tc: dict, tdir: Path, _run_dir: Path, csv_dir: Path) -> None:
    means = _collect_fixed_fm_means(tc, tdir)
    xy_cfg = tc["xy"]
    max_depth = int(xy_cfg["depth"])
    obj = xy_cfg.get("objective", "cvar10")
    rows = []
    for layers in tc["layers_list"]:
        for p in range(1, max_depth + 1):
            med, q25, q75, mean, se, n_fm = _fm_stats(means, layers, f"XY-p{p}-{obj}", "norm_gap")
            if n_fm:
                rows.append({
                    "layers": layers, "method": "XY-FMQAOA", "p": p,
                    "norm_gap_median": med, "norm_gap_q25": q25, "norm_gap_q75": q75,
                    "norm_gap_mean": mean, "norm_gap_se": se,
                })
        for alpha in tc["sa_alphas"]:
            med, q25, q75, mean, se, n_fm = _fm_stats(means, layers, f"SA-auto-a{alpha:g}", "norm_gap")
            if n_fm:
                rows.append({
                    "layers": layers, "method": f"SA-a{alpha:g}", "p": None,
                    "norm_gap_median": med, "norm_gap_q25": q25, "norm_gap_q75": q75,
                    "norm_gap_mean": mean, "norm_gap_se": se,
                })
        med, q25, q75, mean, se, n_fm = _fm_stats(means, layers, "RandomFM", "norm_gap")
        if n_fm:
            rows.append({
                "layers": layers, "method": "RandomFM", "p": None,
                "norm_gap_median": med, "norm_gap_q25": q25, "norm_gap_q75": q75,
                "norm_gap_mean": mean, "norm_gap_se": se,
            })
    if rows:
        _write_csv(
            csv_dir / tc["csv"],
            ["layers", "method", "p", "norm_gap_median", "norm_gap_q25", "norm_gap_q75", "norm_gap_mean", "norm_gap_se"],
            rows,
        )


def _export_fixedfm_percentile_hist(tc: dict, tdir: Path, _run_dir: Path, csv_dir: Path) -> None:
    layers = int(tc["layers_list"][0])
    xy_cfg = tc["xy"]
    xy_label = f"XY-p{int(xy_cfg['depth'])}-{xy_cfg.get('objective', 'cvar10')}"
    target_samplers = (*[f"SA-auto-a{a:g}" for a in tc["sa_alphas"]], xy_label, "RandomFM")

    bins = np.linspace(0.0, 1.0, 21)
    bin_width = float(bins[1] - bins[0])
    pooled: dict[str, list[np.ndarray]] = {}
    for seed in tc["seeds"]:
        npz_path = tdir / f"L{layers}_seed{seed}_pct.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            for smp in target_samplers:
                if smp in data.files:
                    pooled.setdefault(smp, []).append(data[smp])

    rows = []
    for smp in target_samplers:
        if smp not in pooled:
            continue
        counts, _ = np.histogram(np.concatenate(pooled[smp]), bins=bins)
        total = float(counts.sum())
        dens = (counts / total / bin_width) if total > 0 else np.zeros_like(counts, dtype=float)
        for i in range(len(counts)):
            rows.append({
                "layers": layers, "sampler": smp,
                "bin_lo": float(bins[i]), "bin_hi": round(float(bins[i + 1]), 10),
                "count_pooled": int(counts[i]), "density": float(dens[i]),
            })
    if rows:
        _write_csv(csv_dir / tc["csv"], ["layers", "sampler", "bin_lo", "bin_hi", "count_pooled", "density"], rows)


def _export_fixedfm_schedule_retune(tc: dict, tdir: Path, _run_dir: Path, csv_dir: Path) -> None:
    means = _collect_fixed_fm_means(tc, tdir)
    xy_cfg = tc["xy"]
    xy_label = f"XY-p{int(xy_cfg['depth'])}-{xy_cfg.get('objective', 'cvar10')}"
    rows = []
    for layers in tc["layers_list"]:
        for sch in tc["schedules"]:
            for alpha in tc["sa_alphas"]:
                med, _, _, _, _, n_fm = _fm_stats(means, layers, f"SA-{sch}-a{alpha:g}", "sample_fm_pct", require_all=True)
                if (layers, f"SA-{sch}-a{alpha:g}") in means:
                    rows.append({
                        "layers": layers, "series": f"SA-{sch}", "alpha": f"{alpha:g}",
                        "sample_fm_pct_median": med, "n_fm": n_fm,
                    })
        for smp in ("RandomFM", xy_label):
            med, _, _, _, _, n_fm = _fm_stats(means, layers, smp, "sample_fm_pct", require_all=True)
            if n_fm:
                rows.append({
                    "layers": layers, "series": smp, "alpha": None,
                    "sample_fm_pct_median": med, "n_fm": n_fm,
                })
    if rows:
        _write_csv(csv_dir / tc["csv"], ["layers", "series", "alpha", "sample_fm_pct_median", "n_fm"], rows)


def _export_reads_sweep(tc: dict, tdir: Path, _run_dir: Path, csv_dir: Path) -> None:
    L = int(tc["layers"])
    by_key: dict[tuple[int, str], list[dict]] = {}
    for seed in tc["seeds"]:
        doc = _load_completed(tdir / f"L{L}_seed{seed}.json")
        if doc is None:
            continue
        for rec in doc["records"]:
            by_key.setdefault((int(rec["reads"]), rec["method"]), []).append(rec)
    if not by_key:
        print(f"[csv] skip {tc['csv']} (no completed runs)", flush=True)
        return

    rows = []
    for reads in tc["reads_list"]:
        for method in ("SA_a100", "RandomFM"):
            recs = by_key.get((int(reads), method), [])
            if not recs:
                continue
            g = np.asarray([r["norm_gap"] for r in recs], dtype=np.float64)
            p_vals = [r["sample_fm_pct"] for r in recs if r.get("sample_fm_pct") is not None]
            p = np.asarray(p_vals, dtype=np.float64) if len(p_vals) == len(recs) else None
            rows.append({
                "reads": int(reads), "method": method,
                "gap_median": f"{np.median(g):.6f}", "gap_q25": f"{np.percentile(g, 25):.6f}", "gap_q75": f"{np.percentile(g, 75):.6f}",
                "pct_median": None if p is None else f"{np.median(p):.6f}",
                "pct_q25": None if p is None else f"{np.percentile(p, 25):.6f}",
                "pct_q75": None if p is None else f"{np.percentile(p, 75):.6f}",
            })
    _write_csv(csv_dir / tc["csv"], ["reads", "method", "gap_median", "gap_q25", "gap_q75", "pct_median", "pct_q25", "pct_q75"], rows)


_EXPORTERS = {
    "closedloop_l678_fom_curves": _export_l678_fom_curves,
    "closedloop_l7_alpha_sweep": _export_l7_alpha_sweep,
    "closedloop_l7_paired_ranks": _export_l7_paired_ranks,
    "closedloop_l7_schedule_retune": _export_l7_schedule_retune,
    "closedloop_l678_explore_5seeds": _export_explore_5seeds,
    "fixedfm_sa_alpha_sweep": _export_fixedfm_alpha_sweep,
    "fixedfm_xy_depth_sweep": _export_fixedfm_depth_sweep,
    "fixedfm_l7_percentile_hist": _export_fixedfm_percentile_hist,
    "fixedfm_sa_schedule_retune": _export_fixedfm_schedule_retune,
    "fixedfm_l7_sa_reads_sweep": _export_reads_sweep,
}


def export_target(
    target_name: str,
    cfg: dict,
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    smoke: bool = False,
) -> None:
    """Export the CSV file for one target from `run_dir/<target_name>/`."""
    if target_name not in _EXPORTERS:
        raise ValueError(f"Unknown CSV export target: {target_name}")
    csv_dir = resolve_out_dir(out_dir, smoke=smoke)
    tc = cfg["targets"][target_name]
    _EXPORTERS[target_name](tc, run_dir / target_name, run_dir, csv_dir)
