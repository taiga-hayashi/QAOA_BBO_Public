"""Main entry point for reproducing the XY-FMQAOA vs Penalty SA experiments.

Module structure:
    - optics.py  : Multilayer optical filter TMM evaluator and exhaustive truth table
    - fm.py      : Factorization Machine (FM) surrogate training, QUBO conversion, candidate selection
    - xy_qaoa.py : XY-QAOA solver (OpenQARP `qarpx` + exact NumPy fallback, CVaR + INTERP/Powell)
    - sa.py      : Penalty SA (fixed / adaptive / schedule retuning) and random baselines
    - main.py    : Experiment runner (fixed-FM, reads sweep, and closed-loop BBO)

Examples:
    # Fast end-to-end smoke check (~30-60s)
    python main.py --suite all --smoke --workers 4

    # Run individual suites (resumable per-seed checkpoints)
    python main.py --suite truth
    python main.py --suite fixed_fm --workers 8 --backend auto
    python main.py --suite reads_sweep
    python main.py --suite confirm1 --workers 8 --backend auto
    python main.py --suite l6_30seeds --workers 8 --backend auto
    python main.py --suite l7_30seeds --workers 8 --backend auto
    python main.py --suite l8_30seeds --workers 8 --backend auto
    python main.py --suite explore_L678 --workers 8 --backend auto
"""
from __future__ import annotations

import os

for _env_key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")

import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from fm import bits_to_ids, fit_fm, initial_ids, model_hash, predict_all, select_candidate
from optics import atomic_write_json, ensure_truth_table, labels_for, onehot, utc_now
from sa import AdaptivePenaltyTracker, random_search_trajectory, sample_random_feasible, sample_sa_auto, sample_sa_schedule
from xy_qaoa import angle_search, create_xy_backend, interp_ladder_search

ROOT = Path(__file__).resolve().parent
CLOSED_LOOP_SUITES = ("confirm1", "l6_30seeds", "l7_30seeds", "l8_30seeds", "explore_L678")


# ---------------------------------------------------------------------------
# 1. Fixed-FM evaluation (penalty alpha sweep, XY depth, schedule retuning)
# ---------------------------------------------------------------------------

class FrozenFMContext:
    """Shared FM rank and percentile statistics across samplers on one frozen FM."""

    def __init__(self, pred: np.ndarray, seen: list[int], layers: int) -> None:
        self.pred = pred
        self.layers = layers
        self.seen = set(seen)
        mask = np.ones(len(pred), dtype=bool)
        mask[list(seen)] = False
        self.unseen_min = float(pred[mask].min())
        self.unseen_max = float(pred[mask].max())

        order = np.argsort(pred, kind="stable")
        pct = np.empty(len(pred), dtype=np.float64)
        pct[order] = np.arange(len(pred)) / (len(pred) - 1)
        self.pct = pct

        un = np.flatnonzero(mask)
        un_order = un[np.argsort(pred[un], kind="stable")]
        self.unseen_rank = np.full(len(pred), -1, dtype=np.int64)
        self.unseen_rank[un_order] = np.arange(len(un))
        self.n_unseen = len(un)

    def summarize_batch(self, raw: np.ndarray, x_all: np.ndarray) -> tuple[dict, list[float]]:
        chosen, filters = select_candidate(raw, self.pred, x_all, self.seen, self.layers)
        valid_ids = [i for i in bits_to_ids(raw, self.layers) if i is not None]
        span = max(self.unseen_max - self.unseen_min, 1e-300)
        pct_samples = self.pct[valid_ids].tolist() if valid_ids else []
        metrics = {
            "selected_id": chosen,
            "capture": bool(chosen is not None and abs(self.pred[chosen] - self.unseen_min) < 1e-8),
            "norm_gap": 1.0 if chosen is None else float((self.pred[chosen] - self.unseen_min) / span),
            "fm_rank_pct": (
                1.0 if chosen is None else float(self.unseen_rank[chosen] / max(self.n_unseen - 1, 1))
            ),
            "feasible": filters["raw_feasible_rate"],
            "unique_unseen": filters["unique_unseen_count"],
            "sample_fm_pct": float(np.mean(pct_samples)) if pct_samples else None,
        }
        return metrics, pct_samples


def run_fixed_fm_task(task: dict) -> str:
    out_json = Path(task["out_json"])
    out_npz = Path(task["out_npz"])
    if out_json.exists() and out_npz.exists():
        try:
            if json.loads(out_json.read_text(encoding="utf-8")).get("status") == "completed":
                return f"skip {out_json.name}"
        except Exception:
            pass

    torch.set_num_threads(1)
    layers = int(task["layers"])
    seed = int(task["seed"])
    truth = np.load(task["truth_npz"])["fom"]
    x_all = onehot(labels_for(layers))
    init_ids = initial_ids(seed, layers, count=int(task["initial_points"]))
    ys = [float(truth[i]) for i in init_ids]

    sur_cfg = task["surrogate_cfg"]
    m, qd, q, b, s = fit_fm(
        x_all[init_ids],
        ys,
        seed,
        rank=sur_cfg["rank"],
        epochs=sur_cfg["epochs"],
        learning_rate=sur_cfg["learning_rate"],
        optimizer_name=sur_cfg["optimizer"],
        weight_decay=sur_cfg["weight_decay"],
    )
    pred = predict_all(x_all, q, b)
    ctx = FrozenFMContext(pred, init_ids, layers)

    num_reads = int(task["sa_cfg"]["num_reads"])
    num_sweeps = int(task["sa_cfg"]["num_sweeps"])
    reps = int(task["reps"])
    batches: list[dict] = []
    dists: dict[str, list[float]] = {}

    def _record(label: str, raw: np.ndarray, extra: dict) -> None:
        metrics, pct_vals = ctx.summarize_batch(raw, x_all)
        batches.append({"sampler": label, **extra, **metrics})
        dists.setdefault(label, []).extend(pct_vals)

    for sch in task["schedules"]:
        for alpha in task["sa_alphas"]:
            for rep in range(reps):
                t0 = time.perf_counter()
                raw, sm = sample_sa_schedule(
                    qd, b, s, float(alpha), layers, seed * 100000 + rep,
                    schedule=sch, num_reads=num_reads, num_sweeps=num_sweeps,
                )
                _record(
                    f"SA-{sch}-a{alpha:g}",
                    raw,
                    {
                        "alpha": alpha, "schedule": sch, "rep": rep,
                        "seconds": time.perf_counter() - t0,
                        "beta_range": sm["beta_range"], "num_sweeps": sm["num_sweeps"],
                    },
                )

    for rep in range(reps):
        raw, _ = sample_random_feasible(layers, seed * 100000 + rep, num_reads=num_reads)
        _record("RandomFM", raw, {"rep": rep})

    xy_cfg = task["xy_cfg"]
    depth = int(xy_cfg["depth"])
    objective = xy_cfg.get("objective", "cvar10")
    evals_per_depth = int(xy_cfg["evals_per_depth"])
    t_xy = time.perf_counter()
    backend = create_xy_backend(q / max(s, 1e-30), [4] * layers, task["xy_backend"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, 2020, depth]))
    best_angles, best_obj, prob, total_evals = interp_ladder_search(
        backend, pred, s, depth, objective, evals_per_depth, rng
    )
    xy_search_seconds = time.perf_counter() - t_xy

    cdf = np.cumsum(prob)
    xy_label = "XY-p4-cvar10"
    for rep in range(reps):
        draws = np.minimum(
            np.searchsorted(cdf, np.random.default_rng(seed * 100000 + rep).random(num_reads) * cdf[-1], side="left"),
            len(prob) - 1,
        )
        _record(xy_label, x_all[draws], {"rep": rep, "configured_depth": depth})

    summary = {
        "status": "completed",
        "layers": layers,
        "seed": seed,
        "model_sha256": model_hash(m),
        "S": s,
        "xy_search": {
            "label": xy_label,
            "configured_depth": depth,
            "objective": objective,
            "evals_per_depth": evals_per_depth,
            "total_evaluations": total_evals,
            "best_objective": best_obj,
            "best_angles": best_angles.tolist(),
            "backend": type(backend).__name__,
            "seconds": xy_search_seconds,
        },
        "batches": batches,
        "completed_utc": utc_now(),
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = out_npz.with_name(out_npz.stem + "_tmp.npz")
    np.savez_compressed(tmp_npz, **{k: np.asarray(v, dtype=np.float32) for k, v in dists.items()})
    tmp_npz.replace(out_npz)
    atomic_write_json(out_json, summary)
    return f"done {out_json.name}"


# ---------------------------------------------------------------------------
# 2. Closed-loop BBO (FMQA / XY-FMQAOA / Penalty SA / Random)
# ---------------------------------------------------------------------------

def run_closed_loop_task(task: dict) -> str:
    out_path = Path(task["out_path"])
    if out_path.exists():
        try:
            if json.loads(out_path.read_text(encoding="utf-8")).get("status") == "completed":
                return f"skip {out_path.name}"
        except Exception:
            pass

    torch.set_num_threads(1)
    seed = int(task["seed"])
    condition = task["condition"]
    layers = int(task["layers"])
    budget = int(task["budget"])
    initial_points = int(task["initial_points"])
    cycle_cap = int(task["cycle_cap"])
    truth = np.load(task["truth_npz"])["fom"]
    sur_cfg = task["surrogate_cfg"]
    sa_cfg = task["sa_cfg"]
    ad_cfg = task["adaptive_cfg"]
    kind = condition["kind"]

    if kind == "random":
        res = random_search_trajectory(
            seed, layers, truth, budget,
            initial_points=initial_points,
            random_stream_offset=int(task["random_stream_offset"]),
        )
    else:
        x_all = onehot(labels_for(layers))
        ids = initial_ids(seed, layers, count=initial_points)
        ys = [float(truth[i]) for i in ids]
        tracker = AdaptivePenaltyTracker(**ad_cfg)
        events: list[dict] = []
        angles: np.ndarray | None = None
        start = time.perf_counter()

        for cycle in range(1, cycle_cap + 1):
            if len(ids) >= budget:
                break
            model_seed = seed + 100000 * (cycle - 1)
            t_train = time.perf_counter()
            m, qd, q, b, s = fit_fm(
                x_all[ids], ys, model_seed,
                rank=sur_cfg["rank"], epochs=sur_cfg["epochs"],
                learning_rate=sur_cfg["learning_rate"],
                optimizer_name=sur_cfg["optimizer"],
                weight_decay=sur_cfg["weight_decay"],
            )
            training_seconds = time.perf_counter() - t_train
            pred = predict_all(x_all, q, b)
            sampler_seed = seed * 100000 + (cycle - 1) * 1000

            t_samp = time.perf_counter()
            extra_meta: dict = {}
            if kind == "randomfm":
                raw, extra_meta = sample_random_feasible(layers, sampler_seed, num_reads=sa_cfg["num_reads"])
            elif kind in ("sa", "adaptive"):
                alpha = tracker.get_current_alpha() if kind == "adaptive" else float(condition["alpha"])
                raw, extra_meta = sample_sa_schedule(
                    qd, b, s, alpha, layers, sampler_seed,
                    schedule=condition.get("schedule", "auto"),
                    num_reads=sa_cfg["num_reads"], num_sweeps=sa_cfg["num_sweeps"],
                )
            elif kind == "xy":
                depth = int(condition["depth"])
                objective = condition.get("objective", "cvar10")
                backend = create_xy_backend(q / max(s, 1e-30), [4] * layers, task["xy_backend"])
                rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 7, 2026]))
                if angles is None:
                    angles, obj, prob, evals = interp_ladder_search(
                        backend, pred, s, depth, objective, int(condition["first_evals"]), rng
                    )
                else:
                    angles, obj, prob, evals = angle_search(
                        backend, pred, s, depth, objective, [angles],
                        int(condition["cycle_evals"]), rng,
                        n_global=1, n_local=1, local_maxfev=int(condition["local_maxfev"]),
                    )
                cdf = np.cumsum(prob)
                draws = np.minimum(
                    np.searchsorted(cdf, np.random.default_rng(sampler_seed).random(sa_cfg["num_reads"]) * cdf[-1], side="left"),
                    len(prob) - 1,
                )
                raw = x_all[draws]
                extra_meta = {
                    "sampler": "XY-FMQAOA",
                    "backend": type(backend).__name__,
                    "depth": depth,
                    "objective": objective,
                    "angles": angles.tolist(),
                    "angle_objective": obj,
                    "angle_evaluations": evals,
                    "sampler_seed": sampler_seed,
                }
            else:
                raise ValueError(f"Unsupported condition kind: {kind}")

            sampling_seconds = time.perf_counter() - t_samp
            seen = set(ids)
            chosen, filters = select_candidate(raw, pred, x_all, seen, layers)
            unseen_mask = np.ones(len(pred), dtype=bool)
            unseen_mask[ids] = False
            unseen_min = float(pred[unseen_mask].min())

            event = {
                "cycle": cycle,
                "model_seed": model_seed,
                "model_sha256": model_hash(m),
                "S_fm": s,
                **extra_meta,
                **filters,
                "selected_id": chosen,
                "selected_value": None,
                "selected_FM_gap": None if chosen is None else float(pred[chosen] - unseen_min),
                "unseen_FM_min_hit": bool(chosen is not None and abs(pred[chosen] - unseen_min) < 1e-8),
                "evaluations_before": len(ids),
                "training_seconds": training_seconds,
                "sampling_seconds": sampling_seconds,
            }
            if kind == "adaptive":
                event["next_alpha"] = tracker.update(filters["raw_feasible_rate"])
            if chosen is not None:
                val = float(truth[chosen])
                ids.append(chosen)
                ys.append(val)
                event["selected_value"] = val
            event["evaluations_after"] = len(ids)
            event["best_so_far"] = float(min(ys))
            events.append(event)

        res = {
            "seed": seed,
            "condition": condition,
            "layers": layers,
            "budget": budget,
            "cycle_cap": cycle_cap,
            "initial_ids": ids[:initial_points],
            "final_ids": ids,
            "final_values": ys,
            "actual_evaluations": len(ids),
            "final_best": float(min(ys)),
            "events": events,
            "total_seconds": time.perf_counter() - start,
            "stopping_reason": "evaluation_cap" if len(ids) >= budget else "proposal_cycle_cap",
        }

    res["status"] = "completed"
    res["completed_utc"] = utc_now()
    atomic_write_json(out_path, res)
    return f"done {out_path.name} (best={res['final_best']:.4f}, evals={res['actual_evaluations']})"


# ---------------------------------------------------------------------------
# 3. Fixed-FM SA reads sweep (reads = 100..10,000)
# ---------------------------------------------------------------------------

def run_suite_reads_sweep(cfg: dict, run_dir: Path) -> None:
    sc = cfg["suites"]["reads_sweep"]
    L = int(sc["layers"])
    n_cand = 4**L
    reads_list = [int(r) for r in sc["reads_list"]]
    alpha = float(sc["alpha"])
    num_sweeps = int(sc["num_sweeps"])
    out_csv = run_dir / "reads_sweep" / "fixedfm_l7_sa_reads_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fom_all = np.load(run_dir / "truth" / f"L{L}_truth.npz")["fom"]
    X_all = onehot(labels_for(L)).astype(np.float32)

    sa_gaps = {r: [] for r in reads_list}
    rd_gaps = {r: [] for r in reads_list}
    sa_pcts = {r: [] for r in reads_list}
    rd_pcts = {r: [] for r in reads_list}

    t0 = time.perf_counter()
    for seed in sc["seeds"]:
        init = initial_ids(seed, L, count=int(sc["initial_points"]))
        _, qd, q_mat, bias, s_val = fit_fm(
            X_all[init], fom_all[init], seed=seed,
            rank=int(cfg["surrogate"]["rank"]),
            epochs=int(cfg["surrogate"]["epochs"]),
            learning_rate=float(cfg["surrogate"]["learning_rate"]),
            weight_decay=float(cfg["surrogate"]["weight_decay"]),
        )
        pred = predict_all(X_all, q_mat, bias)
        pmin, pmax = float(pred.min()), float(pred.max())
        span = max(pmax - pmin, 1e-12)
        order = np.argsort(pred, kind="stable")
        pct_of = np.empty(n_cand, dtype=np.float64)
        pct_of[order] = np.arange(n_cand, dtype=np.float64) / (n_cand - 1)

        raw_sa, _ = sample_sa_auto(
            qd, bias, s_val, alpha=alpha, layers=L,
            seed=seed + 1000, num_reads=max(reads_list), num_sweeps=num_sweeps,
        )
        rand_cids = np.random.default_rng(seed + 2000).integers(0, n_cand, size=max(reads_list))

        for r in reads_list:
            c_sa = np.array([i for i in bits_to_ids(raw_sa[:r], L) if i is not None], dtype=np.int64)
            c_rd = rand_cids[:r]
            sa_gaps[r].append((float(pred[c_sa].min()) - pmin) / span)
            rd_gaps[r].append((float(pred[c_rd].min()) - pmin) / span)
            sa_pcts[r].append(float(np.median(pct_of[c_sa])))
            rd_pcts[r].append(float(np.median(pct_of[c_rd])))
        print(f"[reads_sweep] seed={seed} done", flush=True)

    lines = ["reads,method,gap_median,gap_q25,gap_q75,pct_median,pct_q25,pct_q75"]
    for r in reads_list:
        for name, g, p in [("SA_a100", sa_gaps[r], sa_pcts[r]), ("RandomFM", rd_gaps[r], rd_pcts[r])]:
            lines.append(
                f"{r},{name},{np.median(g):.6f},{np.percentile(g, 25):.6f},{np.percentile(g, 75):.6f},"
                f"{np.median(p):.6f},{np.percentile(p, 25):.6f},{np.percentile(p, 75):.6f}"
            )
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[reads_sweep] Wrote {out_csv} ({time.perf_counter() - t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# 4. CLI orchestration
# ---------------------------------------------------------------------------

def load_effective_config(smoke: bool) -> dict:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if not smoke:
        return cfg
    cfg = copy.deepcopy(cfg)
    sm = cfg["smoke"]
    cfg["surrogate"].update(sm["surrogate"])
    cfg["sa"].update(sm["sa"])
    cfg["suites"]["fixed_fm"]["seeds"] = sm["fixed_fm"]["seeds"]
    cfg["suites"]["fixed_fm"]["reps"] = sm["fixed_fm"]["reps"]
    cfg["suites"]["fixed_fm"]["xy"].update(sm["fixed_fm"]["xy"])
    cfg["suites"]["reads_sweep"].update(sm["reads_sweep"])
    ov = sm["closed_loop_override"]
    for sname in CLOSED_LOOP_SUITES:
        sc = cfg["suites"][sname]
        sc["seeds"] = sc["seeds"][: ov["seeds_count"]]
        sc["budget"] = ov["budget"]
        sc["initial_points"] = ov["initial_points"]
        sc["proposal_cycle_cap"] = ov["proposal_cycle_cap"]
        for cond in sc["conditions"]:
            if cond["kind"] == "xy":
                cond.update(ov["xy_override"])
    return cfg


def _execute_tasks(fn, tasks: list[dict], workers: int, label: str) -> None:
    if not tasks:
        return
    print(f"[{label}] Running {len(tasks)} tasks (workers={workers})...", flush=True)
    t0 = time.perf_counter()
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            print(f"  [{i}/{len(tasks)}] {fn(task)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(fn, t): t for t in tasks}
            for done_cnt, fut in enumerate(as_completed(futs), 1):
                print(f"  [{done_cnt}/{len(tasks)}] {fut.result()}", flush=True)
    print(f"[{label}] Completed in {time.perf_counter() - t0:.1f}s.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["all", "truth", "fixed_fm", "reads_sweep", *CLOSED_LOOP_SUITES], default="all")
    ap.add_argument("--backend", choices=["auto", "qarpx", "numpy"], default="auto")
    ap.add_argument("--tmm-backend", choices=["numpy", "tmm_fast"], default="numpy")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()

    mode_tag = "smoke" if args.smoke else "full"
    run_dir = args.run_dir or (ROOT / "outputs" / mode_tag / "runs")
    cfg = load_effective_config(args.smoke)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "pipeline_metadata.json",
        {
            "study": cfg["study"],
            "mode": mode_tag,
            "xy_backend_requested": args.backend,
            "workers": args.workers,
            "updated_utc": utc_now(),
            "effective_config": cfg,
        },
    )

    for L in cfg["optics"]["layers_list"]:
        t0 = time.perf_counter()
        fom, _ = ensure_truth_table(L, run_dir, tmm_backend=args.tmm_backend)
        print(f"[truth] L={L} ready ({len(fom)} states, min FOM={fom.min():.6f}, {time.perf_counter() - t0:.2f}s)", flush=True)

    if args.suite in ("all", "fixed_fm"):
        sc = cfg["suites"]["fixed_fm"]
        out_sub = run_dir / "fixed_fm"
        out_sub.mkdir(parents=True, exist_ok=True)
        tasks = [
            {
                "layers": L, "seed": seed,
                "truth_npz": str(run_dir / "truth" / f"L{L}_truth.npz"),
                "initial_points": sc["initial_points"], "reps": sc["reps"],
                "schedules": sc["schedules"], "sa_alphas": sc["sa_alphas"],
                "xy_cfg": sc["xy"], "surrogate_cfg": cfg["surrogate"],
                "sa_cfg": cfg["sa"], "xy_backend": args.backend,
                "out_json": str(out_sub / f"L{L}_seed{seed}.json"),
                "out_npz": str(out_sub / f"L{L}_seed{seed}_pct.npz"),
            }
            for L in sc["layers_list"] for seed in sc["seeds"]
        ]
        _execute_tasks(run_fixed_fm_task, tasks, args.workers, "fixed_fm")

    if args.suite in ("all", "reads_sweep"):
        run_suite_reads_sweep(cfg, run_dir)

    for sname in CLOSED_LOOP_SUITES:
        if args.suite in ("all", sname):
            sc = cfg["suites"][sname]
            out_sub = run_dir / sname
            out_sub.mkdir(parents=True, exist_ok=True)
            layers_list = sc["layers_list"] if "layers_list" in sc else [sc["layers"]]
            multi_layer = "layers_list" in sc
            tasks = [
                {
                    "layers": L, "seed": seed, "condition": cond,
                    "truth_npz": str(run_dir / "truth" / f"L{L}_truth.npz"),
                    "budget": sc["budget"], "initial_points": sc["initial_points"],
                    "cycle_cap": sc["proposal_cycle_cap"],
                    "surrogate_cfg": cfg["surrogate"], "sa_cfg": cfg["sa"],
                    "adaptive_cfg": cfg["adaptive"], "xy_backend": args.backend,
                    "random_stream_offset": cfg["random_stream_offset"],
                    "out_path": str(out_sub / (f"L{L}_{cond['id']}_seed{seed}.json" if multi_layer else f"{cond['id']}_seed{seed}.json")),
                }
                for L in layers_list for cond in sc["conditions"] for seed in sc["seeds"]
            ]
            _execute_tasks(run_closed_loop_task, tasks, args.workers, sname)


if __name__ == "__main__":
    main()
