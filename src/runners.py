"""Experiment configuration, per-target task runners, and CLI helpers."""
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

from .fm import bits_to_ids, fit_fm, initial_ids, model_hash, predict_all, select_candidate
from .optics import atomic_write_json, ensure_truth_table, labels_for, onehot, utc_now
from .sa import (
    AdaptivePenaltyTracker,
    random_search_trajectory,
    sample_categorical_sa,
    sample_random_feasible,
    sample_sa_auto,
    sample_sa_schedule,
)
from .xy_qaoa import angle_search, create_xy_backend, interp_ladder_search

ROOT = Path(__file__).resolve().parents[1]


def default_workers() -> int:
    return min(8, os.cpu_count() or 1)


def default_run_dir(smoke: bool, run_dir: Path | None = None) -> Path:
    mode_tag = "smoke" if smoke else "full"
    return run_dir or (ROOT / "outputs" / mode_tag / "runs")


def _merged(base: dict, override: dict | None) -> dict:
    return {**base, **override} if override else dict(base)


def _apply_closed_loop_smoke(block: dict, ov: dict) -> None:
    block["seeds"] = block["seeds"][: ov["seeds_count"]]
    block["budget"] = ov["budget"]
    block["initial_points"] = ov["initial_points"]
    block["proposal_cycle_cap"] = ov["proposal_cycle_cap"]
    for cond in block["conditions"]:
        if cond["kind"] == "xy":
            cond.update(ov["xy_override"])


def load_effective_config(smoke: bool, config_path: Path | None = None) -> dict:
    cfg = json.loads((config_path or (ROOT / "config.json")).read_text(encoding="utf-8"))
    if not smoke:
        return cfg
    cfg = copy.deepcopy(cfg)
    sm = cfg["smoke"]
    cfg["surrogate"].update(sm["surrogate"])
    cfg["sa"].update(sm["sa"])
    ov = sm["closed_loop"]

    for tname, tc in cfg["targets"].items():
        if tname == "closedloop_l678_fom_curves":
            for lc in tc["layers_configs"]:
                _apply_closed_loop_smoke(lc, ov)
        elif tc.get("kind") == "closed_loop":
            _apply_closed_loop_smoke(tc, ov)
        elif tc.get("kind") == "fixed_fm":
            tc["seeds"] = tc["seeds"][: sm["fixed_fm"]["seeds_count"]]
            tc["reps"] = sm["fixed_fm"]["reps"]
            if tc.get("xy"):
                tc["xy"]["evals_per_depth"] = sm["fixed_fm"]["xy"]["evals_per_depth"]
                tc["xy"]["depth"] = min(int(tc["xy"]["depth"]), int(sm["fixed_fm"]["xy"]["depth"]))
        elif tc.get("kind") == "reads_sweep":
            tc["seeds"] = tc["seeds"][: sm["reads_sweep"]["seeds_count"]]
            tc["num_sweeps"] = sm["reads_sweep"]["num_sweeps"]
            tc["reads_list"] = sm["reads_sweep"]["reads_list"]
    return cfg


def targets_in_group(cfg: dict, group: str) -> list[str]:
    return [name for name, tc in cfg["targets"].items() if tc.get("group") == group]


def target_layers(tc: dict, *, layers_filter: list[int] | None = None) -> list[int]:
    if "layers_configs" in tc:
        layers = [int(lc["layers"]) for lc in tc["layers_configs"]]
        return [L for L in layers if layers_filter is None or L in layers_filter]
    if "layers_list" in tc:
        return [int(L) for L in tc["layers_list"]]
    return [int(tc["layers"])]


def prepare_run_dir(
    run_dir: Path | None,
    *,
    smoke: bool,
    cfg: dict,
    workers: int = 1,
) -> Path:
    resolved = default_run_dir(smoke, run_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        resolved / "pipeline_metadata.json",
        {
            "study": cfg["study"],
            "mode": "smoke" if smoke else "full",
            "xy_backend": "qarpx",
            "workers": workers,
            "updated_utc": utc_now(),
            "effective_config": cfg,
        },
    )
    return resolved


def prepare_truth_tables(
    cfg: dict,
    run_dir: Path,
    *,
    layers_list: list[int] | None = None,
    tmm_backend: str = "numpy",
) -> None:
    targets = layers_list if layers_list is not None else cfg["optics"]["layers_list"]
    for layers in targets:
        t0 = time.perf_counter()
        fom, _ = ensure_truth_table(int(layers), run_dir, tmm_backend=tmm_backend)
        print(
            f"[truth] L={layers} ready ({len(fom)} states, min FOM={fom.min():.6f}, {time.perf_counter() - t0:.2f}s)",
            flush=True,
        )


def execute_tasks(fn, tasks: list[dict], workers: int, label: str) -> None:
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


def parse_cli_args(
    description: str,
    target_choices: list[str],
    *,
    include_layers_filter: bool = False,
) -> argparse.Namespace:
    """Parse common CLI arguments for per-purpose run scripts."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--target", choices=target_choices, default="all", help="CSV target to compute (default: all in this script)")
    if include_layers_filter:
        ap.add_argument("--layers", type=int, nargs="*", default=None, help="Optional layer filter (e.g. --layers 6 7)")
    ap.add_argument("--config", type=Path, default=None, help="Custom config JSON (default: config.json)")
    ap.add_argument("--smoke", action="store_true", help="Run shrunk quick check into outputs/smoke/")
    ap.add_argument("--workers", type=int, default=default_workers(), help="Parallel workers (default: auto up to 8)")
    ap.add_argument("--tmm-backend", choices=["numpy", "tmm_fast"], default="numpy")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="Output CSV directory")
    return ap.parse_args()


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


def _is_completed(*paths: Path) -> bool:
    json_path = paths[0]
    if not all(p.exists() for p in paths):
        return False
    try:
        return json.loads(json_path.read_text(encoding="utf-8")).get("status") == "completed"
    except Exception:
        return False


def _sample_from_prob(prob: np.ndarray, x_all: np.ndarray, seed: int, num_reads: int) -> np.ndarray:
    cdf = np.cumsum(prob)
    draws = np.minimum(
        np.searchsorted(cdf, np.random.default_rng(seed).random(num_reads) * cdf[-1], side="left"),
        len(prob) - 1,
    )
    return x_all[draws]


def _run_fixed_fm_classical(
    task: dict,
    x_all: np.ndarray,
    qd: dict,
    b: float,
    s: float,
    record,
) -> None:
    layers = int(task["layers"])
    seed = int(task["seed"])
    num_reads = int(task["sa_cfg"]["num_reads"])
    num_sweeps = int(task["sa_cfg"]["num_sweeps"])
    reps = int(task["reps"])

    for sch in task["schedules"]:
        for alpha in task["sa_alphas"]:
            for rep in range(reps):
                t0 = time.perf_counter()
                raw, sm = sample_sa_schedule(
                    qd, b, s, float(alpha), layers, seed * 100000 + rep,
                    schedule=sch, num_reads=num_reads, num_sweeps=num_sweeps,
                )
                record(
                    f"SA-{sch}-a{alpha:g}",
                    raw,
                    {
                        "alpha": alpha, "schedule": sch, "rep": rep,
                        "seconds": time.perf_counter() - t0,
                        "beta_range": sm["beta_range"], "num_sweeps": sm["num_sweeps"],
                    },
                )

    if task.get("include_random_fm", True):
        for rep in range(reps):
            raw, _ = sample_random_feasible(layers, seed * 100000 + rep, num_reads=num_reads)
            record("RandomFM", raw, {"rep": rep})


def _run_fixed_fm_xy(
    task: dict,
    x_all: np.ndarray,
    pred: np.ndarray,
    q: np.ndarray,
    s: float,
    record,
) -> tuple[dict | None, list[dict]]:
    xy_cfg = task.get("xy_cfg")
    if not xy_cfg:
        return None, []
    layers = int(task["layers"])
    seed = int(task["seed"])
    num_reads = int(task["sa_cfg"]["num_reads"])
    reps = int(task["reps"])
    depth = int(xy_cfg["depth"])
    objective = xy_cfg.get("objective", "cvar10")
    evals_per_depth = int(xy_cfg["evals_per_depth"])

    t_xy = time.perf_counter()
    backend = create_xy_backend(q / max(s, 1e-30), [4] * layers)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, 2020, depth]))
    xy_history: list[dict] = []
    best_angles, best_obj, _, total_evals = interp_ladder_search(
        backend, pred, s, depth, objective, evals_per_depth, rng, history=xy_history
    )
    xy_search_seconds = time.perf_counter() - t_xy

    xy_searches = []
    for entry in xy_history:
        d = int(entry["depth"])
        xy_label = f"XY-p{d}-{objective}"
        for rep in range(reps):
            raw = _sample_from_prob(entry["prob"], x_all, seed * 100000 + rep, num_reads)
            record(xy_label, raw, {"rep": rep, "configured_depth": d})
        xy_searches.append({
            "label": xy_label,
            "configured_depth": d,
            "objective": objective,
            "evals_per_depth": evals_per_depth,
            "evaluations": entry["evals"],
            "best_objective": entry["objective"],
            "best_angles": entry["angles"],
            "backend": type(backend).__name__,
            "seconds": entry["seconds"],
        })

    final_search = {
        "label": f"XY-p{depth}-{objective}",
        "configured_depth": depth,
        "objective": objective,
        "evals_per_depth": evals_per_depth,
        "total_evaluations": total_evals,
        "best_objective": best_obj,
        "best_angles": best_angles.tolist(),
        "backend": type(backend).__name__,
        "seconds": xy_search_seconds,
    }
    return final_search, xy_searches


def run_fixed_fm_task(task: dict) -> str:
    """Evaluate configured samplers on one frozen FM."""
    out_json = Path(task["out_json"])
    out_npz = Path(task["out_npz"])
    if _is_completed(out_json, out_npz):
        return f"skip {out_json.name}"

    torch.set_num_threads(1)
    layers = int(task["layers"])
    seed = int(task["seed"])
    truth = np.load(task["truth_npz"])["fom"]
    x_all = onehot(labels_for(layers))
    init_ids = initial_ids(seed, layers, count=int(task["initial_points"]))
    ys = [float(truth[i]) for i in init_ids]

    sur_cfg = task["surrogate_cfg"]
    m, qd, q, b, s = fit_fm(
        x_all[init_ids], ys, seed,
        rank=sur_cfg["rank"], epochs=sur_cfg["epochs"],
        learning_rate=sur_cfg["learning_rate"],
        optimizer_name=sur_cfg["optimizer"],
        weight_decay=sur_cfg["weight_decay"],
    )
    pred = predict_all(x_all, q, b)
    ctx = FrozenFMContext(pred, init_ids, layers)

    batches: list[dict] = []
    dists: dict[str, list[float]] = {}

    def _record(label: str, raw: np.ndarray, extra: dict) -> None:
        metrics, pct_vals = ctx.summarize_batch(raw, x_all)
        batches.append({"sampler": label, **extra, **metrics})
        dists.setdefault(label, []).extend(pct_vals)

    _run_fixed_fm_classical(task, x_all, qd, b, s, _record)
    xy_search, xy_searches = _run_fixed_fm_xy(task, x_all, pred, q, s, _record)

    summary = {
        "status": "completed",
        "layers": layers,
        "seed": seed,
        "model_sha256": model_hash(m),
        "S": s,
        "xy_search": xy_search,
        "xy_searches": xy_searches,
        "batches": batches,
        "completed_utc": utc_now(),
    }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = out_npz.with_name(out_npz.stem + "_tmp.npz")
    np.savez_compressed(tmp_npz, **{k: np.asarray(v, dtype=np.float32) for k, v in dists.items()})
    tmp_npz.replace(out_npz)
    atomic_write_json(out_json, summary)
    return f"done {out_json.name}"


def run_reads_sweep_task(task: dict) -> str:
    """Run SA reads sweep (100..10,000) vs RandomFM on one L=7 frozen FM."""
    out_json = Path(task["out_json"])
    if _is_completed(out_json):
        return f"skip {out_json.name}"

    torch.set_num_threads(1)
    layers = int(task["layers"])
    seed = int(task["seed"])
    n_cand = 4**layers
    reads_list = [int(r) for r in task["reads_list"]]
    alpha = float(task["alpha"])
    num_sweeps = int(task["num_sweeps"])
    sur_cfg = task["surrogate_cfg"]

    fom_all = np.load(task["truth_npz"])["fom"]
    x_all = onehot(labels_for(layers)).astype(np.float32)
    init = initial_ids(seed, layers, count=int(task["initial_points"]))
    _, qd, q_mat, bias, s_val = fit_fm(
        x_all[init], fom_all[init], seed=seed,
        rank=int(sur_cfg["rank"]), epochs=int(sur_cfg["epochs"]),
        learning_rate=float(sur_cfg["learning_rate"]),
        weight_decay=float(sur_cfg["weight_decay"]),
        optimizer_name=sur_cfg.get("optimizer", "AdamW"),
    )
    pred = predict_all(x_all, q_mat, bias)
    pmin, pmax = float(pred.min()), float(pred.max())
    span = max(pmax - pmin, 1e-12)
    order = np.argsort(pred, kind="stable")
    pct_of = np.empty(n_cand, dtype=np.float64)
    pct_of[order] = np.arange(n_cand, dtype=np.float64) / (n_cand - 1)

    t0 = time.perf_counter()
    raw_sa, _ = sample_sa_auto(
        qd, bias, s_val, alpha=alpha, layers=layers,
        seed=seed + 1000, num_reads=max(reads_list), num_sweeps=num_sweeps,
    )
    rand_cids = np.random.default_rng(seed + 2000).integers(0, n_cand, size=max(reads_list))

    records = []
    for r in reads_list:
        c_sa = np.array([i for i in bits_to_ids(raw_sa[:r], layers) if i is not None], dtype=np.int64)
        c_rd = rand_cids[:r]
        records.append({
            "reads": r, "method": "SA_a100",
            "norm_gap": 1.0 if len(c_sa) == 0 else (float(pred[c_sa].min()) - pmin) / span,
            "sample_fm_pct": None if len(c_sa) == 0 else float(np.median(pct_of[c_sa])),
            "feasible_count": int(len(c_sa)),
        })
        records.append({
            "reads": r, "method": "RandomFM",
            "norm_gap": (float(pred[c_rd].min()) - pmin) / span,
            "sample_fm_pct": float(np.median(pct_of[c_rd])),
            "feasible_count": int(len(c_rd)),
        })

    atomic_write_json(
        out_json,
        {
            "status": "completed", "layers": layers, "seed": seed,
            "alpha": alpha, "num_sweeps": num_sweeps, "records": records,
            "elapsed_seconds": time.perf_counter() - t0, "completed_utc": utc_now(),
        },
    )
    return f"done {out_json.name}"


def _propose_batch(
    task: dict,
    *,
    condition: dict,
    x_all: np.ndarray,
    pred: np.ndarray,
    qd: dict,
    q: np.ndarray,
    b: float,
    s: float,
    tracker: AdaptivePenaltyTracker,
    angles: np.ndarray | None,
    cycle: int,
    sampler_seed: int,
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    layers = int(task["layers"])
    seed = int(task["seed"])
    sa_cfg = task["sa_cfg"]
    kind = condition["kind"]

    if kind == "randomfm":
        raw, meta = sample_random_feasible(layers, sampler_seed, num_reads=sa_cfg["num_reads"])
        return raw, meta, angles
    if kind == "catsa":
        raw, meta = sample_categorical_sa(
            pred, layers, sampler_seed,
            num_reads=sa_cfg["num_reads"], num_sweeps=sa_cfg["num_sweeps"],
        )
        return raw, meta, angles
    if kind in ("sa", "adaptive"):
        alpha = tracker.get_current_alpha() if kind == "adaptive" else float(condition["alpha"])
        raw, meta = sample_sa_schedule(
            qd, b, s, alpha, layers, sampler_seed,
            schedule=condition.get("schedule", "auto"),
            num_reads=sa_cfg["num_reads"], num_sweeps=sa_cfg["num_sweeps"],
        )
        return raw, meta, angles
    if kind == "xy":
        depth = int(condition["depth"])
        objective = condition.get("objective", "cvar10")
        backend = create_xy_backend(q / max(s, 1e-30), [4] * layers)
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
        raw = _sample_from_prob(prob, x_all, sampler_seed, sa_cfg["num_reads"])
        meta = {
            "sampler": "XY-FMQAOA", "backend": type(backend).__name__,
            "depth": depth, "objective": objective, "angles": angles.tolist(),
            "angle_objective": obj, "angle_evaluations": evals, "sampler_seed": sampler_seed,
        }
        return raw, meta, angles
    raise ValueError(f"Unsupported condition kind: {kind}")


def run_closed_loop_task(task: dict) -> str:
    """Run one closed-loop BBO trajectory for a given condition and seed."""
    out_path = Path(task["out_path"])
    if _is_completed(out_path):
        return f"skip {out_path.name}"

    torch.set_num_threads(1)
    seed = int(task["seed"])
    condition = task["condition"]
    layers = int(task["layers"])
    budget = int(task["budget"])
    initial_points = int(task["initial_points"])
    cycle_cap = int(task["cycle_cap"])
    truth = np.load(task["truth_npz"])["fom"]
    sur_cfg = task["surrogate_cfg"]
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
        tracker = AdaptivePenaltyTracker(**task["adaptive_cfg"])
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
            raw, extra_meta, angles = _propose_batch(
                task,
                condition=condition, x_all=x_all, pred=pred,
                qd=qd, q=q, b=b, s=s, tracker=tracker,
                angles=angles, cycle=cycle, sampler_seed=sampler_seed,
            )
            sampling_seconds = time.perf_counter() - t_samp

            seen = set(ids)
            chosen, filters = select_candidate(raw, pred, x_all, seen, layers)
            unseen_mask = np.ones(len(pred), dtype=bool)
            unseen_mask[ids] = False
            unseen_min = float(pred[unseen_mask].min())

            event = {
                "cycle": cycle, "model_seed": model_seed, "model_sha256": model_hash(m), "S_fm": s,
                **extra_meta, **filters, "selected_id": chosen, "selected_value": None,
                "selected_FM_gap": None if chosen is None else float(pred[chosen] - unseen_min),
                "unseen_FM_min_hit": bool(chosen is not None and abs(pred[chosen] - unseen_min) < 1e-8),
                "evaluations_before": len(ids), "training_seconds": training_seconds, "sampling_seconds": sampling_seconds,
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
            "seed": seed, "condition": condition, "layers": layers, "budget": budget,
            "cycle_cap": cycle_cap, "initial_ids": ids[:initial_points], "final_ids": ids,
            "final_values": ys, "actual_evaluations": len(ids), "final_best": float(min(ys)),
            "events": events, "total_seconds": time.perf_counter() - start,
            "stopping_reason": "evaluation_cap" if len(ids) >= budget else "proposal_cycle_cap",
        }

    res["status"] = "completed"
    res["completed_utc"] = utc_now()
    atomic_write_json(out_path, res)
    return f"done {out_path.name} (best={res['final_best']:.4f}, evals={res['actual_evaluations']})"


def run_target(
    target_name: str,
    cfg: dict,
    run_dir: Path,
    *,
    workers: int = 1,
    layers_filter: list[int] | None = None,
) -> None:
    """Execute one CSV target configured under `cfg['targets'][target_name]`."""
    tc = cfg["targets"][target_name]
    out_sub = run_dir / target_name
    out_sub.mkdir(parents=True, exist_ok=True)
    sur_cfg = _merged(cfg["surrogate"], tc.get("surrogate"))
    sa_cfg = _merged(cfg["sa"], tc.get("sa"))
    ad_cfg = _merged(cfg["adaptive"], tc.get("adaptive"))

    if "layers_configs" in tc:
        tasks = [
            {
                "layers": int(lc["layers"]), "seed": seed, "condition": cond,
                "truth_npz": str(run_dir / "truth" / f"L{int(lc['layers'])}_truth.npz"),
                "budget": lc["budget"], "initial_points": lc["initial_points"],
                "cycle_cap": lc["proposal_cycle_cap"],
                "surrogate_cfg": _merged(sur_cfg, lc.get("surrogate")),
                "sa_cfg": _merged(sa_cfg, lc.get("sa")),
                "adaptive_cfg": _merged(ad_cfg, lc.get("adaptive")),
                "random_stream_offset": cfg["random_stream_offset"],
                "out_path": str(out_sub / f"L{int(lc['layers'])}_{cond['id']}_seed{seed}.json"),
            }
            for lc in tc["layers_configs"]
            if layers_filter is None or int(lc["layers"]) in layers_filter
            for cond in lc["conditions"]
            for seed in lc["seeds"]
        ]
        execute_tasks(run_closed_loop_task, tasks, workers, target_name)
        return

    kind = tc["kind"]
    if kind == "closed_loop":
        layers_list = tc["layers_list"] if "layers_list" in tc else [tc["layers"]]
        multi_layer = "layers_list" in tc
        tasks = [
            {
                "layers": L, "seed": seed, "condition": cond,
                "truth_npz": str(run_dir / "truth" / f"L{L}_truth.npz"),
                "budget": tc["budget"], "initial_points": tc["initial_points"],
                "cycle_cap": tc["proposal_cycle_cap"],
                "surrogate_cfg": sur_cfg, "sa_cfg": sa_cfg, "adaptive_cfg": ad_cfg,
                "random_stream_offset": cfg["random_stream_offset"],
                "out_path": str(out_sub / (f"L{L}_{cond['id']}_seed{seed}.json" if multi_layer else f"{cond['id']}_seed{seed}.json")),
            }
            for L in layers_list for cond in tc["conditions"] for seed in tc["seeds"]
        ]
        execute_tasks(run_closed_loop_task, tasks, workers, target_name)
    elif kind == "fixed_fm":
        tasks = [
            {
                "layers": L, "seed": seed,
                "truth_npz": str(run_dir / "truth" / f"L{L}_truth.npz"),
                "initial_points": tc["initial_points"], "reps": tc["reps"],
                "schedules": tc["schedules"], "sa_alphas": tc["sa_alphas"],
                "include_random_fm": tc.get("include_random_fm", True),
                "xy_cfg": tc.get("xy"), "surrogate_cfg": sur_cfg, "sa_cfg": sa_cfg,
                "out_json": str(out_sub / f"L{L}_seed{seed}.json"),
                "out_npz": str(out_sub / f"L{L}_seed{seed}_pct.npz"),
            }
            for L in tc["layers_list"] for seed in tc["seeds"]
        ]
        execute_tasks(run_fixed_fm_task, tasks, workers, target_name)
    elif kind == "reads_sweep":
        L = int(tc["layers"])
        tasks = [
            {
                "layers": L, "seed": seed,
                "truth_npz": str(run_dir / "truth" / f"L{L}_truth.npz"),
                "initial_points": tc["initial_points"], "alpha": tc["alpha"],
                "num_sweeps": tc["num_sweeps"], "reads_list": tc["reads_list"],
                "surrogate_cfg": sur_cfg,
                "out_json": str(out_sub / f"L{L}_seed{seed}.json"),
            }
            for seed in tc["seeds"]
        ]
        execute_tasks(run_reads_sweep_task, tasks, workers, target_name)
    else:
        raise ValueError(f"Unknown target kind: {kind}")
