"""Entry point for running modular FMQA / FMQAOA on registered categorical problems (SSH-server ready)."""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.stats import wilcoxon

from config import CODE_DIR, load_config, parse_custom_sa_spec, parse_custom_xy_qaoa, parse_custom_standard_qaoa
from evaluator import CategoricalTableProblem, load_problem
from fm_model import model_sha256
from fm_target_scaling import train_scaled_fm, qubo_original_scale
from qaoa_sampler import sample_standard_qaoa, sample_xy_qaoa
from qubo import fm_to_qubo, qubo_to_ising_scale
from sa_sampler import sample_sa
from shared_numerics import AdaptivePenaltyTracker


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _step_seed(seed: int, step: int, method_idx: int) -> int:
    seq = np.random.SeedSequence([int(seed), int(step), int(method_idx), 20261016])
    return int(seq.generate_state(1)[0] % (2**31 - 1))


def method_penalty_alpha(method: str, cfg: Dict[str, Any]) -> float:
    """Return the normalized penalty coefficient alpha for a given method name."""
    if method == "Adaptive-FMQA":
        return float(cfg["penalty"].get("adaptive_initial_alpha", 1.0))
    if method in ("Exact-FM", "Exact-FMQA"):
        return 0.0
    sa_spec = parse_custom_sa_spec(method, cfg["sa"], cfg["penalty"])
    if sa_spec is not None:
        return float(sa_spec[0])
    pen_cfg = cfg["penalty"]
    if method == "Penalty-FMQAOA" or method.startswith("Penalty-FMQAOA("):
        return float(pen_cfg["standard_qaoa_alpha"])
    if method == "XY-FMQAOA" or method.startswith("XY-FMQAOA("):
        return 0.0
    raise ValueError(f"Unknown method '{method}'")


def run_single_step_sampler(
    method: str,
    method_idx: int,
    q_dict: Dict[Any, float],
    q_mat: np.ndarray,
    offset: float,
    ising_scale: float,
    problem: CategoricalTableProblem,
    cfg: Dict[str, Any],
    seed: int,
    step: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Dispatch one sampling step to Exact-FM, SA, or QAOA according to the method name."""
    if method in ("Exact-FM", "Exact-FMQA"):
        t0 = time.perf_counter()
        fm_preds = np.einsum("bi,ij,bj->b", problem.patterns, q_mat, problem.patterns) + offset
        order = np.argsort(fm_preds, kind="stable")
        raw_bits = problem.patterns[order].copy()
        meta = {
            "sampler": "Exact-Feasible-Enumeration",
            "penalty_alpha": 0.0,
            "penalty_lambda": 0.0,
            "num_reads": 0,
            "elapsed_sec": float(time.perf_counter() - t0),
        }
        return raw_bits, meta

    s_rng = _step_seed(seed, step, method_idx)
    pen_cfg = cfg["penalty"]

    sa_spec = parse_custom_sa_spec(method, cfg["sa"], pen_cfg)
    if method == "Adaptive-FMQA":
        sa_spec = (float(pen_cfg.get("adaptive_current_alpha", pen_cfg.get("adaptive_initial_alpha", 1.0))), cfg["sa"])
    if sa_spec is not None:
        alpha, sa_cfg_custom = sa_spec
        lam = alpha * ising_scale
        return sample_sa(
            q_dict=q_dict,
            group_sizes=problem.group_sizes,
            penalty_lambda=lam,
            penalty_alpha=alpha,
            offset=offset,
            sa_cfg=sa_cfg_custom,
            seed=s_rng,
        )

    custom_standard_cfg = parse_custom_standard_qaoa(method, cfg["qaoa"])
    if method == "Penalty-FMQAOA" or custom_standard_cfg is not None:
        alpha = float(pen_cfg["standard_qaoa_alpha"])
        lam = alpha * ising_scale
        return sample_standard_qaoa(
            q_mat=q_mat,
            group_sizes=problem.group_sizes,
            valid_basis_indices=problem.valid_basis_indices,
            penalty_lambda=lam,
            penalty_alpha=alpha,
            ising_scale=ising_scale,
            qaoa_cfg=custom_standard_cfg if custom_standard_cfg is not None else cfg["qaoa"],
            seed=s_rng,
        )

    custom_xy_cfg = parse_custom_xy_qaoa(method, cfg["qaoa"])
    if method == "XY-FMQAOA" or custom_xy_cfg is not None:
        if float(pen_cfg["xy_qaoa_lambda"]) != 0.0:
            raise ValueError("XY-FMQAOA penalty lambda must be 0.0")
        q_cfg = custom_xy_cfg if custom_xy_cfg is not None else cfg["qaoa"]
        return sample_xy_qaoa(
            q_mat=q_mat,
            group_sizes=problem.group_sizes,
            feasible_patterns=problem.patterns,
            valid_basis_indices=problem.valid_basis_indices,
            ising_scale=ising_scale,
            qaoa_cfg=q_cfg,
            seed=s_rng,
        )

    raise ValueError(
        f"Unknown method '{method}'. Expected one of: "
        "Exact-FM, TunedPenalty-FMQA, LargePenalty-FMQA, Penalty-FMQAOA, XY-FMQAOA, FMQA(...), or XY-FMQAOA(...)."
    )


def run_seed(
    seed: int,
    problem: CategoricalTableProblem,
    cfg: Dict[str, Any],
    existing_rec: Dict[str, Any] | None = None,
    checkpoint_path: Path | None = None,
) -> Dict[str, Any]:
    """Run all configured methods on a single seed under strict initial-data fairness."""
    config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    n_init = int(cfg["problem"]["initial_samples"])
    n_cycles = 1 if cfg.get("mode") == "fixed_fm" else int(cfg["problem"]["bbo_cycles"])
    init_ids = problem.initial_dataset(seed=seed, n_initial=n_init)
    init_y = [problem.evaluate_id(i) for i in init_ids]
    init_best = float(min(init_y))
    init_sha = hashlib.sha256(json.dumps(init_ids).encode("utf-8")).hexdigest()

    existing_methods: Dict[str, Any] = {}
    if (
        existing_rec is not None
        and existing_rec.get("config_sha256") == config_sha
        and existing_rec.get("dataset_sha256") == problem.data_sha256
        and existing_rec.get("initial_samples") == n_init
        and existing_rec.get("bbo_cycles") == n_cycles
        and existing_rec.get("initial_sha256") == init_sha
    ):
        existing_methods = dict(existing_rec.get("methods", {}))

    seed_record: Dict[str, Any] = {
        "problem_id": problem.problem_id,
        "dataset_sha256": problem.data_sha256,
        "config_sha256": config_sha,
        "seed": int(seed),
        "mode": cfg.get("mode", "bbo"),
        "initial_samples": n_init,
        "bbo_cycles": n_cycles,
        "initial_candidate_ids": init_ids,
        "initial_sha256": init_sha,
        "initial_best_objective": init_best,
        "initial_normalized_regret": problem.normalized_regret(init_best),
        "methods": dict(existing_methods),
    }

    for m_idx, method in enumerate(cfg["methods"]):
        previous = existing_methods.get(method, {})
        if previous and previous.get("status", "completed") == "completed":
            continue
        t_method = time.perf_counter()
        evaluated_ids = list(init_ids)
        evaluated_y = list(init_y)
        seen_set = set(evaluated_ids)
        best_so_far = init_best
        alpha_val = method_penalty_alpha(method, cfg)
        if method in ("Exact-FM", "Exact-FMQA"):
            sampler_family = "Exact"
        elif "QAOA" in method:
            sampler_family = "QAOA"
        else:
            sampler_family = "SA"

        history: List[Dict[str, Any]] = list(previous.get("history", []))
        first_model_hash: str | None = previous.get("step0_fm_sha256")
        tracker = AdaptivePenaltyTracker(initial_alpha=cfg["penalty"].get("adaptive_initial_alpha", 1.0))
        for h in history:
            if h["chosen_candidate_id"] is not None:
                cid = int(h["chosen_candidate_id"])
                evaluated_ids.append(cid)
                evaluated_y.append(problem.evaluate_id(cid))
                seen_set.add(cid)
            if method == "Adaptive-FMQA":
                tracker.update(h["raw_feasible_rate"])
        if history:
            best_so_far = float(history[-1]["best_objective_so_far"])
        prior_elapsed = float(previous.get("total_elapsed_sec", 0.0))

        for step in range(len(history), n_cycles):
            x_train = problem.patterns[evaluated_ids].astype(np.float32)
            # Step-0 uses identical FM training seed across all methods for fairness
            fm_seed = int(seed) if step == 0 else int(seed) + step * 10007
            model, target_transform = train_scaled_fm(
                x_train, evaluated_y, problem.n_bits, cfg["fm"], fm_seed,
                standardize=cfg["fm"].get("target_standardization", False),
            )
            if step == 0:
                first_model_hash = model_sha256(model)

            q_dict, q_mat, offset = fm_to_qubo(model)
            # Restore original objective units before selection,penalties and diagnostics.
            q_mat, offset = qubo_original_scale(q_mat, offset, target_transform)
            q_dict = {key: float(value) * target_transform["scale"] for key, value in q_dict.items()}
            ising_scale = qubo_to_ising_scale(q_mat)
            fm_preds = np.einsum("bi,ij,bj->b", problem.patterns, q_mat, problem.patterns) + offset

            step_cfg = cfg
            if method == "Adaptive-FMQA":
                step_cfg = copy.deepcopy(cfg)
                step_cfg["penalty"]["adaptive_current_alpha"] = tracker.get_current_alpha()
            raw_bits, sampler_meta = run_single_step_sampler(
                method=method,
                method_idx=m_idx,
                q_dict=q_dict,
                q_mat=q_mat,
                offset=offset,
                ising_scale=ising_scale,
                problem=problem,
                cfg=step_cfg,
                seed=seed,
                step=step,
            )
            chosen_id, sel_stats = problem.select_unseen_candidate(
                raw_bits=raw_bits,
                fm_predictions=fm_preds,
                seen_ids=seen_set,
            )

            if chosen_id is not None:
                y_new = problem.evaluate_id(chosen_id)
                evaluated_ids.append(chosen_id)
                evaluated_y.append(y_new)
                seen_set.add(chosen_id)
                best_so_far = min(best_so_far, y_new)
            else:
                y_new = None

            history.append({
                "step": step + 1,
                "ising_scale_S": float(ising_scale),
                "fm_sha256": model_sha256(model),
                "fm_training_seed": fm_seed,
                "target_transform": target_transform,
                "chosen_candidate_id": chosen_id,
                "chosen_true_objective": y_new,
                "best_objective_so_far": float(best_so_far),
                "normalized_regret_so_far": problem.normalized_regret(best_so_far),
                "total_bb_evaluations": len(evaluated_ids),
                **sel_stats,
                "sampler": sampler_meta,
            })

            if method == "Adaptive-FMQA":
                history[-1]["next_alpha"] = tracker.update(sel_stats["raw_feasible_rate"])
            seed_record["methods"][method] = {
                "status": "running", "history": history,
                "step0_fm_sha256": first_model_hash,
                "total_elapsed_sec": prior_elapsed + time.perf_counter() - t_method,
            }
            if checkpoint_path is not None:
                model_dir = checkpoint_path.parent.parent / "checkpoints"
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / f"seed{seed}_method{m_idx}_latest.pt"
                tmp_model = model_path.with_suffix(".pt.tmp")
                torch.save({"state_dict": model.state_dict(), "training_ids": evaluated_ids[:len(x_train)],
                            "training_seed": fm_seed, "model_sha256": model_sha256(model),
                            "target_transform": target_transform,
                            "proposal": step + 1}, tmp_model)
                tmp_model.replace(model_path)
                _save_json(checkpoint_path, seed_record)
            print(f"Seed {seed} {method}: proposal {step + 1}/{n_cycles}, evaluations {len(evaluated_ids)}, best FOM {best_so_far:.6g}", flush=True)

        defined_gaps = [h["selected_fm_gap"] for h in history if h["selected_fm_gap"] is not None]
        mean_fm_gap = float(np.mean(defined_gaps)) if defined_gaps else None
        defined_rhos = [h["unseen_spearman_rho"] for h in history if h.get("unseen_spearman_rho") is not None]
        mean_rho = float(np.mean(defined_rhos)) if defined_rhos else None
        final_rho = float(defined_rhos[-1]) if defined_rhos else None
        defined_ranks = [h["unseen_fm_min_true_rank"] for h in history if h.get("unseen_fm_min_true_rank") is not None]
        mean_true_rank = float(np.mean(defined_ranks)) if defined_ranks else None

        first_sm = history[0]["sampler"]
        sample_budget = int(first_sm.get("num_reads") or first_sm.get("shots") or 0)
        sa_sweeps = first_sm.get("num_sweeps")

        seed_record["methods"][method] = {
            "status": "completed",
            "sampler_family": sampler_family,
            "penalty_alpha": float(alpha_val),
            "sample_budget": sample_budget,
            "sa_sweeps": sa_sweeps,
            "step0_fm_sha256": first_model_hash,
            "final_best_objective": float(best_so_far),
            "final_normalized_regret": problem.normalized_regret(best_so_far),
            "total_bb_evaluations": len(evaluated_ids),
            "mean_raw_feasible_rate": float(np.mean([h["raw_feasible_rate"] for h in history])),
            "mean_fm_minimum_hit": float(np.mean([h["fm_minimum_hit"] for h in history])),
            "mean_selected_fm_gap": mean_fm_gap,
            "mean_unseen_spearman_rho": mean_rho,
            "final_unseen_spearman_rho": final_rho,
            "mean_fm_min_true_rank": mean_true_rank,
            "mean_unseen_success": float(np.mean([h["unseen_success"] for h in history])),
            "total_elapsed_sec": float(prior_elapsed + time.perf_counter() - t_method),
            "history": history,
        }
        if checkpoint_path is not None:
            _save_json(checkpoint_path, seed_record)
        print(
            f"[Seed {seed:4d}] {method:34s} (n={sample_budget:4d}, alpha={alpha_val:6.2f}) | Best={best_so_far:.4f} "
            f"| Regret={problem.normalized_regret(best_so_far):.4f} "
            f"| Feas={seed_record['methods'][method]['mean_raw_feasible_rate']:.3f} "
            f"| FMHit={seed_record['methods'][method]['mean_fm_minimum_hit']:.3f} "
            f"| Time={seed_record['methods'][method]['total_elapsed_sec']:.2f}s",
            flush=True,
        )

    ordered_methods: Dict[str, Any] = {}
    for m in cfg["methods"]:
        if m in seed_record["methods"]:
            ordered_methods[m] = seed_record["methods"][m]
    for m, d in seed_record["methods"].items():
        if m not in ordered_methods:
            ordered_methods[m] = d
    seed_record["methods"] = ordered_methods

    step0_hashes = {m: d["step0_fm_sha256"] for m, d in seed_record["methods"].items()}
    if len(set(step0_hashes.values())) != 1:
        raise RuntimeError(f"Fairness violation: step-0 FM hashes differ across methods: {step0_hashes}")

    return seed_record


def export_csv_tables(seed_records: List[Dict[str, Any]], csv_dir: Path) -> None:
    """Export flat CSV tables (step_history.csv and seed_summary.csv) for easy local notebook plotting."""
    csv_dir.mkdir(parents=True, exist_ok=True)

    step_fields = [
        "seed",
        "method",
        "sampler_family",
        "penalty_alpha",
        "sample_budget",
        "sa_sweeps",
        "step",
        "budget_index",
        "total_bb_evaluations",
        "best_objective_so_far",
        "normalized_regret_so_far",
        "chosen_candidate_id",
        "chosen_true_objective",
        "raw_feasible_rate",
        "fm_minimum_hit",
        "selected_fm_gap",
        "unseen_spearman_rho",
        "unseen_fm_min_true_rank",
        "unseen_success",
        "unique_unseen_count",
        "ising_scale_S",
        "penalty_lambda",
        "elapsed_sec",
    ]
    with (csv_dir / "step_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields)
        writer.writeheader()
        for rec in seed_records:
            s = rec["seed"]
            n_init = rec["initial_samples"]
            for method, mdata in rec["methods"].items():
                fam = mdata.get("sampler_family", "QAOA" if "QAOA" in method else "SA")
                alpha_m = mdata.get("penalty_alpha")
                n_bud = mdata.get("sample_budget")
                sw_m = mdata.get("sa_sweeps")
                writer.writerow({
                    "seed": s,
                    "method": method,
                    "sampler_family": fam,
                    "penalty_alpha": alpha_m,
                    "sample_budget": n_bud,
                    "sa_sweeps": sw_m,
                    "step": 0,
                    "budget_index": n_init,
                    "total_bb_evaluations": n_init,
                    "best_objective_so_far": rec["initial_best_objective"],
                    "normalized_regret_so_far": rec["initial_normalized_regret"],
                    "chosen_candidate_id": None,
                    "chosen_true_objective": None,
                    "raw_feasible_rate": None,
                    "fm_minimum_hit": None,
                    "selected_fm_gap": None,
                    "unseen_spearman_rho": None,
                    "unseen_fm_min_true_rank": None,
                    "unseen_success": None,
                    "unique_unseen_count": None,
                    "ising_scale_S": None,
                    "penalty_lambda": None,
                    "elapsed_sec": 0.0,
                })
                for h in mdata["history"]:
                    sm = h["sampler"]
                    writer.writerow({
                        "seed": s,
                        "method": method,
                        "sampler_family": fam,
                        "penalty_alpha": sm.get("penalty_alpha", alpha_m),
                        "sample_budget": n_bud,
                        "sa_sweeps": sw_m,
                        "step": h["step"],
                        "budget_index": n_init + h["step"],
                        "total_bb_evaluations": h["total_bb_evaluations"],
                        "best_objective_so_far": h["best_objective_so_far"],
                        "normalized_regret_so_far": h["normalized_regret_so_far"],
                        "chosen_candidate_id": h["chosen_candidate_id"],
                        "chosen_true_objective": h["chosen_true_objective"],
                        "raw_feasible_rate": h["raw_feasible_rate"],
                        "fm_minimum_hit": h["fm_minimum_hit"],
                        "selected_fm_gap": h["selected_fm_gap"],
                        "unseen_spearman_rho": h.get("unseen_spearman_rho"),
                        "unseen_fm_min_true_rank": h.get("unseen_fm_min_true_rank"),
                        "unseen_success": h["unseen_success"],
                        "unique_unseen_count": h["unique_unseen_count"],
                        "ising_scale_S": h["ising_scale_S"],
                        "penalty_lambda": sm.get("penalty_lambda"),
                        "elapsed_sec": sm.get("elapsed_sec"),
                    })

    summary_fields = [
        "seed",
        "method",
        "sampler_family",
        "penalty_alpha",
        "sample_budget",
        "sa_sweeps",
        "initial_best_objective",
        "initial_normalized_regret",
        "final_best_objective",
        "final_normalized_regret",
        "mean_raw_feasible_rate",
        "mean_fm_minimum_hit",
        "mean_selected_fm_gap",
        "mean_unseen_spearman_rho",
        "final_unseen_spearman_rho",
        "mean_fm_min_true_rank",
        "mean_unseen_success",
        "total_bb_evaluations",
        "total_elapsed_sec",
        "step0_fm_sha256",
    ]
    with (csv_dir / "seed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for rec in seed_records:
            for method, mdata in rec["methods"].items():
                writer.writerow({
                    "seed": rec["seed"],
                    "method": method,
                    "sampler_family": mdata.get("sampler_family", "QAOA" if "QAOA" in method else "SA"),
                    "penalty_alpha": mdata.get("penalty_alpha"),
                    "sample_budget": mdata.get("sample_budget"),
                    "sa_sweeps": mdata.get("sa_sweeps"),
                    "initial_best_objective": rec["initial_best_objective"],
                    "initial_normalized_regret": rec["initial_normalized_regret"],
                    "final_best_objective": mdata["final_best_objective"],
                    "final_normalized_regret": mdata["final_normalized_regret"],
                    "mean_raw_feasible_rate": mdata["mean_raw_feasible_rate"],
                    "mean_fm_minimum_hit": mdata["mean_fm_minimum_hit"],
                    "mean_selected_fm_gap": mdata.get("mean_selected_fm_gap"),
                    "mean_unseen_spearman_rho": mdata.get("mean_unseen_spearman_rho"),
                    "final_unseen_spearman_rho": mdata.get("final_unseen_spearman_rho"),
                    "mean_fm_min_true_rank": mdata.get("mean_fm_min_true_rank"),
                    "mean_unseen_success": mdata["mean_unseen_success"],
                    "total_bb_evaluations": mdata["total_bb_evaluations"],
                    "total_elapsed_sec": mdata["total_elapsed_sec"],
                    "step0_fm_sha256": mdata["step0_fm_sha256"],
                })


def _median_iqr(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"), "iqr": float("nan")}
    q25, q50, q75 = np.percentile(values, [25, 50, 75])
    return {
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def summarize_results(seed_records: List[Dict[str, Any]], methods: List[str]) -> Dict[str, Any]:
    """Compute Median [IQR] and paired Wilcoxon signed-rank tests across seeds."""
    metrics = [
        "final_best_objective",
        "final_normalized_regret",
        "mean_raw_feasible_rate",
        "mean_fm_minimum_hit",
        "mean_selected_fm_gap",
        "mean_unseen_spearman_rho",
        "final_unseen_spearman_rho",
        "mean_fm_min_true_rank",
        "mean_unseen_success",
        "total_elapsed_sec",
    ]
    by_method: Dict[str, Any] = {}
    for m in methods:
        first_m = seed_records[0]["methods"][m]
        by_method[m] = {
            "sampler_family": first_m.get("sampler_family"),
            "penalty_alpha": first_m.get("penalty_alpha"),
            "sample_budget": first_m.get("sample_budget"),
            "sa_sweeps": first_m.get("sa_sweeps"),
        }
        for k in metrics:
            vals = [float(r["methods"][m][k]) for r in seed_records if r["methods"][m].get(k) is not None]
            by_method[m][k] = {
                **_median_iqr(vals),
                "per_seed": {
                    str(r["seed"]): (float(r["methods"][m][k]) if r["methods"][m].get(k) is not None else None)
                    for r in seed_records
                },
            }

    ref_xy = None
    if "XY-FMQAOA" in methods:
        ref_xy = "XY-FMQAOA"
    elif "XY-FMQAOA(p=3,cvar,powell49)" in methods:
        ref_xy = "XY-FMQAOA(p=3,cvar,powell49)"
    else:
        xy_candidates = [m for m in methods if m.startswith("XY-FMQAOA")]
        if xy_candidates:
            ref_xy = xy_candidates[-1]

    paired_tests = []
    if ref_xy is not None and len(seed_records) >= 2:
        others = [m for m in methods if m != ref_xy]
        n_comp = max(1, len(others))
        for other in others:
            diffs = [
                float(r["methods"][ref_xy]["final_normalized_regret"])
                - float(r["methods"][other]["final_normalized_regret"])
                for r in seed_records
            ]
            if all(abs(d) < 1e-12 for d in diffs):
                p_raw = 1.0
            else:
                try:
                    _, p_raw = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
                    p_raw = float(p_raw)
                except ValueError:
                    p_raw = 1.0
            paired_tests.append({
                "pair": f"{ref_xy} vs {other}",
                "metric": "final_normalized_regret",
                "diff_median_iqr": _median_iqr(diffs),
                "p_value_raw": p_raw,
                "p_value_bonferroni": min(1.0, p_raw * n_comp),
            })

    return {"methods": by_method, "paired_tests": paired_tests}


def main(argv: List[str] | None = None) -> None:
    cfg = load_config(argv)
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = (CODE_DIR / out_dir).resolve()
    json_dir = out_dir / "json"
    csv_dir = out_dir / "csv"
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    problem = load_problem(cfg["problem"])

    meta_cfg = {
        **cfg,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "problem_id": problem.problem_id,
            "dataset_sha256": problem.data_sha256,
            "f_opt": problem.f_opt,
            "f_worst": problem.f_worst,
        },
    }
    (json_dir / "config_used.json").write_text(
        json.dumps(meta_cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    resume = bool(cfg.get("resume", True))
    expected_init = int(cfg["problem"]["initial_samples"])
    expected_cycles = 1 if cfg.get("mode") == "fixed_fm" else int(cfg["problem"]["bbo_cycles"])
    config_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    seed_records = []
    for seed in cfg["seeds"]:
        seed_path = json_dir / f"seed_{seed}.json"
        existing_rec = None
        if resume and seed_path.exists():
            existing_rec = json.loads(seed_path.read_text(encoding="utf-8"))
            if (
                existing_rec.get("config_sha256") == config_sha
                and existing_rec.get("dataset_sha256") == problem.data_sha256
                and all(existing_rec.get("methods", {}).get(m, {}).get("status", "missing") == "completed" for m in cfg["methods"])
                and existing_rec.get("initial_samples") == expected_init
                and existing_rec.get("bbo_cycles") == expected_cycles
            ):
                print(f"[Seed {int(seed):4d}] Loaded existing record from {seed_path.name} (resume=True)", flush=True)
                seed_records.append(existing_rec)
                export_csv_tables(seed_records, csv_dir)
                continue
        rec = run_seed(int(seed), problem, cfg, existing_rec=existing_rec, checkpoint_path=seed_path)
        seed_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        seed_records.append(rec)
        export_csv_tables(seed_records, csv_dir)

    summary = summarize_results(seed_records, cfg["methods"])
    (json_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== Summary Across {len(seed_records)} Seeds (Median [Q25, Q75]) ===")
    for m in cfg["methods"]:
        s = summary["methods"][m]
        reg = s["final_normalized_regret"]
        hit = s["mean_fm_minimum_hit"]
        feas = s["mean_raw_feasible_rate"]
        rho = s["mean_unseen_spearman_rho"]
        alpha_v = s["penalty_alpha"]
        n_b = s.get("sample_budget", 0)
        print(
            f"- {m:34s} (n={n_b:4d}, alpha={alpha_v:6.2f}) | Regret: {reg['median']:.4f} [{reg['q25']:.4f}, {reg['q75']:.4f}] "
            f"| FM-Hit: {hit['median']:.3f} [{hit['q25']:.3f}, {hit['q75']:.3f}] "
            f"| Feas: {feas['median']:.3f} | Rho: {rho['median']:.3f}"
        )
    print(f"\nSaved JSON & CSV results to: {out_dir}")


if __name__ == "__main__":
    main()
