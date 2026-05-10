#!/usr/bin/env python3
"""
62_run_fig5_design_time.py — Fig.5-style design-phase runtime/cost driver.

Generates or consumes multiple DNN tasksets, runs each requested algorithm on
each taskset as a separate call, and aggregates design-phase wall time and
profiling cost. Live mode is cache-first and can be globally capped; dry-run is
the default.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.integration.dnn_algorithm_runner import DNNAlgorithmResult, run_dnn_rta_algorithm
from src.integration.dnn_workload_generator import WorkloadConfig, generate_tasksets
from src.integration.live_budget import LiveProfileBudget
from src.integration.split_point_policy import get_enabled_boundaries


def _check_min_free_gb(min_free_gb: float | None) -> None:
    """Raise RuntimeError if free disk space on the REPO filesystem is below min_free_gb."""
    if min_free_gb is None:
        return
    import shutil as _shutil
    usage = _shutil.disk_usage(REPO)
    free_gb = usage.free / (1 << 30)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Disk guard: only {free_gb:.1f} GB free on {REPO} filesystem "
            f"(--min-free-gb {min_free_gb}).  "
            f"Run:  python scripts/22_clean_generated_artifacts.py --onnx --engines --yes"
        )


AlgorithmSpec = Tuple[str, str]

_ALGORITHM_SETS: Dict[str, List[str]] = {
    "main4": ["ss:tol-fb", "uni:tol-fb", "ss:opt", "uni:opt"],
    "full8": ["ss:opt", "ss:heu", "ss:tol", "ss:tol-fb",
              "uni:opt", "uni:heu", "uni:tol", "uni:tol-fb"],
    "ss_only": ["ss:opt", "ss:heu", "ss:tol", "ss:tol-fb"],
    "uni_only": ["uni:opt", "uni:heu", "uni:tol", "uni:tol-fb"],
}

# Accept paper-style labels (e.g. "SS-tol-fb") as aliases for canonical forms
_LABEL_ALIASES: Dict[str, str] = {
    "SS-opt": "ss:opt", "SS-heu": "ss:heu", "SS-tol": "ss:tol", "SS-tol-fb": "ss:tol-fb",
    "UNI-opt": "uni:opt", "UNI-heu": "uni:heu", "UNI-tol": "uni:tol", "UNI-tol-fb": "uni:tol-fb",
    "ss-opt": "ss:opt", "ss-heu": "ss:heu", "ss-tol": "ss:tol", "ss-tol-fb": "ss:tol-fb",
    "uni-opt": "uni:opt", "uni-heu": "uni:heu", "uni-tol": "uni:tol", "uni-tol-fb": "uni:tol-fb",
}

_KNOWN_ALGORITHMS: Dict[str, set] = {
    "ss": {"single", "max", "tol", "tol-fb", "heu", "heu-k", "opt", "opt-k"},
    "uni": {"single", "max", "tol", "tol-fb", "heu", "opt"},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Fig.5-style DNN design-phase runtime/cost experiments"
    )
    ap.add_argument("--models", nargs="+", default=["alexnet", "resnet18"])
    ap.add_argument("--num-tasks", type=int, default=8)
    ap.add_argument("--utilization", type=float, default=0.6)
    ap.add_argument("--num-tasksets", type=int, default=10)
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=["uni:opt", "uni:heu", "ss:tol-fb"],
        help=(
            "Algorithms as model:algorithm (e.g. uni:opt ss:tol-fb) or paper labels "
            "(e.g. SS-tol-fb UNI-opt). Overridden by --algorithm-set."
        ),
    )
    ap.add_argument(
        "--algorithm-set",
        choices=list(_ALGORITHM_SETS.keys()),
        default=None,
        help="Predefined algorithm set (overrides --algorithms). "
             "main4={SS-tol-fb,UNI-tol-fb,SS-opt,UNI-opt}, full8=all 8 variants.",
    )
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--wcet-metric", default="p99", choices=["p99", "mean"], dest="wcet_metric")
    ap.add_argument("--split-policy", default="five_points",
                    choices=["all", "paper_like", "stage", "five_points", "ten_points", "major_blocks"])
    ap.add_argument("--taskgen-mode", default="dnnsplitting", choices=["legacy", "dnnsplitting"])
    ap.add_argument("--utilization-basis", default="total", choices=["gpu", "total"])
    ap.add_argument("--num-cpus", type=int, default=1)
    ap.add_argument("--num-tasks-per-cpu", type=int, default=None)
    ap.add_argument("--period-min-ms", type=float, default=1.0)
    ap.add_argument("--period-max-ms", type=float, default=10000.0)
    ap.add_argument("--cpu-pre-min", type=float, default=0.5)
    ap.add_argument("--cpu-pre-max", type=float, default=2.0)
    ap.add_argument("--cpu-post-min", type=float, default=0.2)
    ap.add_argument("--cpu-post-max", type=float, default=1.0)
    ap.add_argument("--g-ratio-min", type=float, default=0.6)
    ap.add_argument("--g-ratio-max", type=float, default=1.0)
    ap.add_argument("--uunifast-cpu-utilization", action="store_false",
                    dest="uniform_cpu_utilization",
                    help="Use UUniFast instead of uniform CPU utilization split")
    ap.add_argument("--uniform-cpu-utilization", action="store_true", default=True)
    ap.add_argument("--uniform-task-utilization", action="store_true", default=False)
    ap.add_argument("--g-utilization-threshold", type=float, default=1.0)
    ap.add_argument("--number-of-inference-segments", type=int, default=1)
    ap.add_argument("--max-block-count", type=int, default=None)
    ap.add_argument("--per-splitting-overhead", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=200)
    ap.add_argument("--max-candidates", type=int, default=100000)
    ap.add_argument("--max-profiles", type=int, default=100000)
    ap.add_argument("--max-iterations", type=int, default=1000)
    ap.add_argument(
        "--allow-proactive-splitting",
        action="store_true",
        default=False,
        help="For paper-style OPT/HEU, continue split search even if no-split is schedulable.",
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--live", action="store_true", default=False,
                    help="Enable live TensorRT profiling; default is dry-run")
    ap.add_argument("--cache-only-live", action="store_true", default=False,
                    help="In live mode, skip uncached masks")
    ap.add_argument("--global-max-real-profiles", type=int, default=None,
                    help="Optional global cap on new selected-mask profiles")
    ap.add_argument("--stop-on-first-build", action="store_true", default=False)
    ap.add_argument(
        "--reset-eval-cache-for-run",
        action="store_true",
        default=False,
        help=(
            "Before running, delete selected-mask artifacts/evaluation JSONs for "
            "models in this run. Base dag_aligned_full artifacts are preserved."
        ),
    )
    ap.add_argument(
        "--progress-interval-sec",
        type=float,
        default=30.0,
        help="Minimum interval for extra progress summary lines",
    )
    ap.add_argument("--stop-after-tasksets", type=int, default=None,
                    help="Debug: only run the first K tasksets")
    ap.add_argument("--tasksets", nargs="*", default=[],
                    help="Existing taskset JSONs to copy/consume instead of generating")
    ap.add_argument("--taskset-dir", default=None,
                    help="Directory containing taskset_*.json files to consume")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default=str(REPO / "results" / "dnn_experiments"))
    ap.add_argument(
        "--allow-equal-wcet-fallback",
        action="store_true",
        default=False,
        dest="allow_equal_wcet_fallback",
        help="Development/CI only: fall back to equal-weight WCET/N when "
             "base chunk profiling data is missing (produces approximate results). "
             "Requires running scripts/21_profile_base_chunks.py for accurate results.",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        dest="min_free_gb",
        metavar="N",
        help="Abort before live build/profile if free disk space on REPO filesystem "
             "is below N GB.  Ignored in dry-run mode.",
    )
    return ap.parse_args()


def parse_algorithms(values: Iterable[str]) -> List[AlgorithmSpec]:
    out: List[AlgorithmSpec] = []
    for value in values:
        normalized = _LABEL_ALIASES.get(value, value)
        if ":" in normalized:
            model, algorithm = normalized.split(":", 1)
        else:
            model, algorithm = "ss", normalized
        model, algorithm = model.lower(), algorithm.lower()
        if model not in _KNOWN_ALGORITHMS:
            raise ValueError(
                f"Unknown RTA model {model!r} in {value!r}. Use 'ss' or 'uni'."
            )
        if algorithm not in _KNOWN_ALGORITHMS[model]:
            known = ", ".join(sorted(_KNOWN_ALGORITHMS[model]))
            raise ValueError(
                f"Unknown {model.upper()} algorithm {algorithm!r} in {value!r}. "
                f"Known: {known}"
            )
        out.append((model, algorithm))
    return out


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    mode = "live" if args.live else "dry"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"fig5_u{args.utilization:.2f}_{args.split_policy}_{mode}_{stamp}".replace(".", "p")


def collect_existing_tasksets(args: argparse.Namespace) -> List[Path]:
    paths = [Path(p) for p in args.tasksets]
    if args.taskset_dir:
        td = Path(args.taskset_dir)
        paths.extend(sorted(td.glob("taskset_*.json")))
    return paths


def prepare_tasksets(args: argparse.Namespace, out_dir: Path) -> List[Path]:
    taskset_root = out_dir / "generated_tasksets"
    taskset_root.mkdir(parents=True, exist_ok=True)
    existing = collect_existing_tasksets(args)
    if existing:
        copied: List[Path] = []
        input_dir = taskset_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for idx, source in enumerate(existing):
            source = source if source.is_absolute() else (REPO / source)
            target = input_dir / f"{idx:03d}_{source.name}"
            shutil.copy2(source, target)
            normalize_taskset(target, args)
            copied.append(target)
        return copied[: args.stop_after_tasksets] if args.stop_after_tasksets else copied

    cfg = WorkloadConfig(
        models=args.models,
        n_tasks=args.num_tasks,
        utilization=args.utilization,
        n_tasksets=args.num_tasksets,
        precision=args.precision,
        wcet_metric=args.wcet_metric,
        cpu_pre_range=(args.cpu_pre_min, args.cpu_pre_max),
        cpu_post_range=(args.cpu_post_min, args.cpu_post_max),
        period_min_ms=args.period_min_ms,
        period_max_ms=args.period_max_ms,
        cpu_id=0,
        seed=args.seed,
        utilization_basis=args.utilization_basis,
        taskgen_mode=args.taskgen_mode,
        num_cpus=args.num_cpus,
        tasks_per_cpu=args.num_tasks_per_cpu,
        g_ratio_range=(args.g_ratio_min, args.g_ratio_max),
        uniform_cpu_utilization=args.uniform_cpu_utilization,
        uniform_task_utilization=args.uniform_task_utilization,
        g_utilization_threshold=args.g_utilization_threshold,
        number_of_inference_segments=args.number_of_inference_segments,
        max_block_count=args.max_block_count,
        per_splitting_overhead=args.per_splitting_overhead,
        max_retries=args.max_retries,
    )
    paths = generate_tasksets(cfg, output_dir=taskset_root)
    return paths[: args.stop_after_tasksets] if args.stop_after_tasksets else paths


def normalize_taskset(path: Path, args: argparse.Namespace) -> None:
    raw = json.loads(path.read_text())
    raw.setdefault("name", path.stem)
    raw["precision"] = args.precision
    raw["wcet_metric"] = args.wcet_metric
    raw.setdefault("utilization_basis", args.utilization_basis)
    raw.setdefault("taskgen_mode", args.taskgen_mode)
    for index, task in enumerate(raw.get("tasks", [])):
        task["precision"] = args.precision
        task["wcet_metric"] = args.wcet_metric
        if "target_chunks" not in task and "initial_mask" not in task:
            task["target_chunks"] = 1
        if raw.get("taskgen_mode") != "dnnsplitting" and args.num_cpus > 0:
            task["cpu_id"] = index % args.num_cpus
    raw.get("tasks", []).sort(key=lambda t: float(t["deadline_ms"]))
    for priority, task in enumerate(raw.get("tasks", []), start=1):
        task["priority"] = priority
    path.write_text(json.dumps(raw, indent=2))


def initial_active_by_task(path: Path) -> Dict[str, int]:
    raw = json.loads(path.read_text())
    out: Dict[str, int] = {}
    for task in raw.get("tasks", []):
        if "initial_mask" in task:
            active = int(sum(int(x) for x in task["initial_mask"]))
        else:
            active = max(0, int(task.get("target_chunks", 1)) - 1)
        out[str(task["task_name"])] = active
    return out


def taskset_metrics(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text())
    tasks = raw.get("tasks", [])
    return {
        "task_count": len(tasks),
        "model_distribution": json.dumps(dict(Counter(str(t.get("model_name", "")).lower() for t in tasks))),
        "gpu_util": float(raw.get("_actual_gpu_utilization", 0.0) or 0.0),
        "cpu_util": float(raw.get("_actual_cpu_utilization", 0.0) or 0.0),
        "total_util": float(raw.get("_actual_total_utilization", raw.get("_utilization", 0.0)) or 0.0),
        "max_cpu_partition_util": max(
            (float(v) for v in (raw.get("_actual_cpu_partition_utilization", {}) or {}).values()),
            default=0.0,
        ),
    }


def models_in_tasksets(tasksets: List[Path]) -> List[str]:
    models = set()
    for path in tasksets:
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        for task in raw.get("tasks", []):
            model = str(task.get("model_name", "")).lower()
            if model:
                models.add(model)
    return sorted(models)


def reset_selected_mask_cache(models: List[str], precision: str, out_dir: Path) -> Dict[str, Any]:
    """
    Remove selected-mask artifacts for the requested models.

    This intentionally matches only names containing '<model>_mask_' and does
    not remove dag_aligned_full configs, base ONNX, base engines, or model
    registry artifacts.
    """
    deleted: List[str] = []
    patterns: List[Tuple[Path, str]] = []
    for model in models:
        patterns.extend([
            (REPO / "results" / "evaluations" / model, f"{model}_mask_*_{precision}.json"),
            (REPO / "results" / "evaluations" / model, f"{model}_mask_*_{precision}_cpp_raw.json"),
            (REPO / "results" / "table4", f"{model}_cpp_{model}_mask_*_{precision}.json"),
            (REPO / "artifacts" / "split_configs" / model, f"{model}_mask_*.json"),
            (REPO / "artifacts" / "onnx" / model, f"{model}_mask_*"),
            (REPO / "artifacts" / "engines" / model, f"{model}_mask_*"),
        ])

    for root, pattern in patterns:
        if not root.exists():
            continue
        for path in sorted(root.glob(pattern)):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                deleted.append(str(path.relative_to(REPO)))
            except FileNotFoundError:
                continue

    report = {
        "models": models,
        "precision": precision,
        "deleted_count": len(deleted),
        "deleted_paths": deleted,
        "note": (
            "Deleted selected-mask artifacts only. Base dag_aligned_full artifacts "
            "and global profiling DB file were preserved."
        ),
    }
    (out_dir / "cache_reset_report.json").write_text(json.dumps(report, indent=2))
    return report


def summarize_result(
    taskset_path: Path,
    model: str,
    algorithm: str,
    result: DNNAlgorithmResult,
    wall_clock_s: float,
) -> Dict[str, Any]:
    initial_active = initial_active_by_task(taskset_path)
    metrics = taskset_metrics(taskset_path)
    task_chunks: Dict[str, int] = {}
    task_active: Dict[str, int] = {}
    task_masks: Dict[str, List[int]] = {}
    task_variants: Dict[str, str] = {}
    disabled_active_total = 0
    split_count = 0

    for tr in result.task_results:
        final_mask = list(tr.final_mask)
        active = int(sum(final_mask))
        task_chunks[tr.task_name] = int(tr.final_k_chunks)
        task_active[tr.task_name] = active
        task_masks[tr.task_name] = final_mask
        task_variants[tr.task_name] = tr.variant_name
        if active != initial_active.get(tr.task_name, 0):
            split_count += 1
        enabled = set(get_enabled_boundaries(tr.model_name, result.policy_name, len(final_mask)))
        disabled_active_total += sum(
            1 for idx, bit in enumerate(final_mask) if bit and idx not in enabled
        )

    error_type = result.error_type or ""
    error_message = ""
    if result.error:
        error_message = str(result.error).splitlines()[0]
    single_schedulable = getattr(result, "single_schedulable", None)
    split_required = bool(single_schedulable is False and result.schedulable and split_count > 0)
    split_proactive = bool(single_schedulable is True and split_count > 0)

    return {
        "taskset": taskset_path.stem,
        "taskset_path": str(taskset_path.relative_to(REPO)),
        "algorithm_label": f"{model.upper()}-{algorithm}",
        "rta_model": model.upper(),
        "algorithm": algorithm,
        "schedulable": bool(result.schedulable),
        "analysis_error": bool(result.analysis_error or result.error),
        "error_type": error_type,
        "error_message": error_message,
        "unschedulable_reason": result.unschedulable_reason or "",
        "wall_clock_s": wall_clock_s,
        "optimization_runtime_s": float(result.duration_s),
        "algorithm_iterations": int(result.algorithm_iterations),
        "masks_evaluated": int(result.stats.masks_evaluated),
        "baseline_k1_hits": int(result.stats.baseline_k1_hits),
        "real_profiles": int(result.stats.real_profiles),
        "cache_hits": int(result.stats.cache_hits),
        "skipped_cache_misses": int(result.stats.skipped_cache_misses),
        "dry_run_evaluations": int(result.stats.dry_run_evaluations),
        "builds_triggered": int(result.stats.builds_triggered),
        "exports_triggered": int(result.stats.exports_triggered),
        "interval_cache_hits": int(result.stats.total_interval_cache_hits),
        "interval_cache_misses": int(result.stats.total_interval_cache_misses),
        "interval_onnx_cache_hits": int(result.stats.total_interval_onnx_cache_hits),
        "interval_onnx_cache_misses": int(result.stats.total_interval_onnx_cache_misses),
        "interval_engine_cache_hits": int(result.stats.total_interval_engine_cache_hits),
        "interval_engine_cache_misses": int(result.stats.total_interval_engine_cache_misses),
        "interval_engine_build_wall_s": float(result.stats.total_interval_engine_build_wall_s),
        "total_export_wall_s": float(result.stats.total_export_wall_s),
        "total_build_wall_s": float(result.stats.total_build_wall_s),
        "total_profile_wall_s": float(result.stats.total_profile_wall_s),
        "total_estimated_cold_s": float(result.stats.total_estimated_cold_s),
        "split_triggered": split_count > 0,
        "single_schedulable": single_schedulable,
        "split_required": split_required,
        "split_proactive": split_proactive,
        "early_stopped_no_split": bool(getattr(result, "early_stopped_no_split", False)),
        "split_task_count": split_count,
        "final_total_active_boundaries": sum(task_active.values()),
        "disabled_active_boundaries": disabled_active_total,
        "policy_violation": disabled_active_total > 0,
        "final_chunk_counts": json.dumps(task_chunks),
        "final_active_boundaries": json.dumps(task_active),
        "task_masks": json.dumps(task_masks),
        "task_variants": json.dumps(task_variants),
        **metrics,
    }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    labels = sorted({r["algorithm_label"] for r in rows})
    summary: List[Dict[str, Any]] = []
    for label in labels:
        items = [r for r in rows if r["algorithm_label"] == label]
        runtimes = [float(r["wall_clock_s"]) for r in items]
        opt_runtimes = [float(r["optimization_runtime_s"]) for r in items]
        n = len(items)
        summary.append({
            "algorithm": label,
            "tasksets": n,
            "schedulable_count": sum(1 for r in items if r["schedulable"]),
            "schedulability_ratio": (
                sum(1 for r in items if r["schedulable"]) / n if n else 0.0
            ),
            "split_triggered_count": sum(1 for r in items if r["split_triggered"]),
            "split_triggered_ratio": (
                sum(1 for r in items if r["split_triggered"]) / n if n else 0.0
            ),
            "split_required_count": sum(1 for r in items if r.get("split_required")),
            "split_required_ratio": (
                sum(1 for r in items if r.get("split_required")) / n if n else 0.0
            ),
            "split_proactive_count": sum(1 for r in items if r.get("split_proactive")),
            "split_proactive_ratio": (
                sum(1 for r in items if r.get("split_proactive")) / n if n else 0.0
            ),
            "single_schedulable_count": sum(1 for r in items if r.get("single_schedulable") is True),
            "analysis_error_count": sum(1 for r in items if r["analysis_error"]),
            "policy_violation_count": sum(1 for r in items if r["policy_violation"]),
            "mean_wall_clock_s": statistics.mean(runtimes) if runtimes else 0.0,
            "median_wall_clock_s": statistics.median(runtimes) if runtimes else 0.0,
            "min_wall_clock_s": min(runtimes) if runtimes else 0.0,
            "max_wall_clock_s": max(runtimes) if runtimes else 0.0,
            "p95_wall_clock_s": percentile(runtimes, 0.95),
            "mean_optimization_runtime_s": statistics.mean(opt_runtimes) if opt_runtimes else 0.0,
            "mean_masks_evaluated": avg(items, "masks_evaluated"),
            "mean_real_profiles": avg(items, "real_profiles"),
            "mean_cache_hits": avg(items, "cache_hits"),
            "mean_skipped_cache_misses": avg(items, "skipped_cache_misses"),
            "mean_builds_triggered": avg(items, "builds_triggered"),
            "mean_exports_triggered": avg(items, "exports_triggered"),
            "total_real_profiles": sum(int(r["real_profiles"]) for r in items),
            "total_cache_hits": sum(int(r["cache_hits"]) for r in items),
            "total_masks_evaluated": sum(int(r["masks_evaluated"]) for r in items),
            "total_baseline_k1_hits": sum(int(r.get("baseline_k1_hits", 0)) for r in items),
            "mean_baseline_k1_hits": avg(items, "baseline_k1_hits"),
            "total_interval_cache_hits": sum(int(r.get("interval_cache_hits", 0)) for r in items),
            "total_interval_cache_misses": sum(int(r.get("interval_cache_misses", 0)) for r in items),
            "total_interval_onnx_cache_hits": sum(int(r.get("interval_onnx_cache_hits", 0)) for r in items),
            "total_interval_engine_cache_hits": sum(int(r.get("interval_engine_cache_hits", 0)) for r in items),
            "mean_interval_engine_build_wall_s": avg(items, "interval_engine_build_wall_s"),
            "mean_total_export_wall_s": avg(items, "total_export_wall_s"),
            "mean_total_build_wall_s": avg(items, "total_build_wall_s"),
            "mean_total_profile_wall_s": avg(items, "total_profile_wall_s"),
            "mean_total_estimated_cold_s": avg(items, "total_estimated_cold_s"),
        })
    return summary


def avg(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(float(r.get(key, 0.0) or 0.0) for r in rows) / len(rows) if rows else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(out_dir: Path, args: argparse.Namespace, tasksets: List[Path],
                  rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]],
                  live_budget: Optional[LiveProfileBudget]) -> None:
    lines = [
        "# Fig.5 Design-Time Summary",
        "",
        "## Run config",
        "",
        f"- Mode: {'live/cache-first' if args.live else 'dry-run'}",
        f"- Models: {', '.join(args.models)}",
        f"- Algorithms: {', '.join(args.algorithms)}",
        f"- Tasksets: {len(tasksets)}",
        f"- Utilization: {args.utilization}",
        f"- Split policy: {args.split_policy}",
        f"- Taskgen mode: {args.taskgen_mode}",
        f"- Utilization basis: {args.utilization_basis}",
        f"- Max candidates: {args.max_candidates}",
        f"- Max profiles: {args.max_profiles}",
        f"- Warmup/iters: {args.warmup}/{args.iters}",
        f"- Reset selected-mask cache: {args.reset_eval_cache_for_run}",
        f"- Progress interval: {args.progress_interval_sec}s",
    ]
    if live_budget is not None:
        lines += [
            f"- cache_only_live: {live_budget.cache_only}",
            f"- global_max_real_profiles: {live_budget.global_max_real_profiles}",
            f"- global_profile_budget_used: {live_budget.used_real_profiles}",
            f"- skipped_cache_misses: {live_budget.skipped_cache_misses}",
        ]
    lines += [
        "",
        "## Design-time aggregate",
        "",
        format_table(
            summary_rows,
            [
                "algorithm",
                "tasksets",
                "mean_wall_clock_s",
                "median_wall_clock_s",
                "p95_wall_clock_s",
                "mean_masks_evaluated",
                "mean_real_profiles",
                "mean_cache_hits",
                "schedulability_ratio",
                "split_triggered_ratio",
                "analysis_error_count",
                "policy_violation_count",
            ],
        ),
        "",
        "## Warnings",
        "",
        "- Live runtime includes cache effects. Interpret `real_profiles` and `cache_hits` together with runtime.",
        "- A cache hit is real measured data reuse, but it does not include rebuild cost in the current run.",
        "- `wall_clock_s` is Python orchestration plus export/build/profile/cache overhead, not runtime-server inference latency.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_run_config(out_dir: Path, args: argparse.Namespace,
                     algorithms: List[AlgorithmSpec], tasksets: List[Path]) -> None:
    data = vars(args).copy()
    data["dry_run_effective"] = not args.live
    data["algorithm_specs"] = [f"{m}:{a}" for m, a in algorithms]
    data["tasksets"] = [str(p.relative_to(REPO)) for p in tasksets]
    (out_dir / "run_config.json").write_text(json.dumps(data, indent=2))


def main() -> int:
    args = parse_args()
    run_name = args.run_name or make_run_name(args)
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.algorithm_set is not None:
        args.algorithms = _ALGORITHM_SETS[args.algorithm_set]
    algorithms = parse_algorithms(args.algorithms)
    tasksets = prepare_tasksets(args, out_dir)
    if not tasksets:
        print("[error] no tasksets generated or supplied", file=sys.stderr)
        return 1

    reset_report: Optional[Dict[str, Any]] = None
    if args.reset_eval_cache_for_run:
        reset_models = models_in_tasksets(tasksets)
        reset_report = reset_selected_mask_cache(reset_models, args.precision, out_dir)
        print(
            f"[cache-reset] deleted {reset_report['deleted_count']} selected-mask "
            f"artifact(s) for models {reset_models}",
            flush=True,
        )

    if args.live:
        _check_min_free_gb(getattr(args, "min_free_gb", None))

    live_budget: Optional[LiveProfileBudget] = None
    if args.live:
        cache_only = args.cache_only_live or (
            args.global_max_real_profiles is not None
            and args.global_max_real_profiles == 0
        )
        live_budget = LiveProfileBudget(
            cache_only=cache_only,
            global_max_real_profiles=args.global_max_real_profiles,
            stop_on_first_build=args.stop_on_first_build,
        )

    write_run_config(out_dir, args, algorithms, tasksets)
    print(f"Run: {run_name}")
    print(f"Mode: {'live/cache-first' if args.live else 'dry-run'}")
    print(f"Tasksets: {len(tasksets)}")
    print(f"Algorithms: {[f'{m}:{a}' for m, a in algorithms]}")
    print(f"Output: {out_dir.relative_to(REPO)}")
    print(f"Reset selected-mask cache: {args.reset_eval_cache_for_run}")
    if live_budget is not None:
        print(f"Global max real profiles: {live_budget.global_max_real_profiles}")
        print(f"Cache-only live: {live_budget.cache_only}")

    rows: List[Dict[str, Any]] = []
    dry_run = not args.live
    run_stopped = False
    total_jobs = len(tasksets) * len(algorithms)
    job_idx = 0
    experiment_t0 = time.time()
    last_progress = experiment_t0
    cumulative = {
        "masks_evaluated": 0,
        "real_profiles": 0,
        "cache_hits": 0,
        "skipped_cache_misses": 0,
    }

    for taskset_idx, taskset_path in enumerate(tasksets, start=1):
        if run_stopped:
            break
        for model, algorithm in algorithms:
            job_idx += 1
            label = f"{model.upper()}-{algorithm}"
            elapsed = time.time() - experiment_t0
            print(
                f"[{job_idx}/{total_jobs}] taskset {taskset_idx}/{len(tasksets)} "
                f"{taskset_path.name} {label} start elapsed={elapsed:.1f}s "
                f"cum_masks={cumulative['masks_evaluated']} "
                f"cum_real={cumulative['real_profiles']} "
                f"cum_cache={cumulative['cache_hits']} "
                f"cum_skipped={cumulative['skipped_cache_misses']}",
                flush=True,
            )
            t0 = time.time()
            result = run_dnn_rta_algorithm(
                dnn_taskset_path=taskset_path,
                model=model,
                algorithm=algorithm,
                precision=args.precision,
                wcet_metric=args.wcet_metric,
                dry_run=dry_run,
                policy_name=args.split_policy,
                max_profiles=args.max_profiles,
                max_candidates=args.max_candidates,
                max_iterations=args.max_iterations,
                warmup=args.warmup,
                iters=args.iters,
                live_budget=live_budget,
                allow_proactive_splitting=args.allow_proactive_splitting,
                allow_equal_wcet_fallback=args.allow_equal_wcet_fallback,
            )
            wall = time.time() - t0
            row = summarize_result(taskset_path, model, algorithm, result, wall)
            rows.append(row)
            cumulative["masks_evaluated"] += int(result.stats.masks_evaluated)
            cumulative["real_profiles"] += int(result.stats.real_profiles)
            cumulative["cache_hits"] += int(result.stats.cache_hits)
            cumulative["skipped_cache_misses"] += int(result.stats.skipped_cache_misses)
            print(
                f"  {'SCHED' if result.schedulable else 'MISS '} "
                f"wall={wall:.2f}s masks={result.stats.masks_evaluated} "
                f"real={result.stats.real_profiles} cache={result.stats.cache_hits} "
                f"k1={result.stats.baseline_k1_hits} "
                f"skipped={result.stats.skipped_cache_misses} "
                f"split={row['split_triggered']}"
                f" elapsed={time.time() - experiment_t0:.1f}s",
                flush=True,
            )
            now = time.time()
            if now - last_progress >= args.progress_interval_sec:
                print(
                    f"[progress] elapsed={now - experiment_t0:.1f}s "
                    f"completed_jobs={job_idx}/{total_jobs} "
                    f"masks={cumulative['masks_evaluated']} "
                    f"real={cumulative['real_profiles']} "
                    f"cache={cumulative['cache_hits']} "
                    f"skipped={cumulative['skipped_cache_misses']}",
                    flush=True,
                )
                last_progress = now
            if live_budget is not None and live_budget.stopped:
                run_stopped = True
                break

    summary_rows = aggregate(rows)

    per_fields = [
        "taskset", "taskset_path", "algorithm_label", "rta_model", "algorithm",
        "schedulable", "analysis_error", "error_type", "error_message",
        "unschedulable_reason", "wall_clock_s", "optimization_runtime_s",
        "algorithm_iterations", "masks_evaluated", "baseline_k1_hits",
        "real_profiles", "cache_hits",
        "skipped_cache_misses", "dry_run_evaluations", "builds_triggered",
        "exports_triggered",
        "interval_cache_hits", "interval_cache_misses",
        "interval_onnx_cache_hits", "interval_onnx_cache_misses",
        "interval_engine_cache_hits", "interval_engine_cache_misses",
        "interval_engine_build_wall_s",
        "total_export_wall_s", "total_build_wall_s", "total_profile_wall_s",
        "total_estimated_cold_s",
        "split_triggered", "single_schedulable",
        "split_required", "split_proactive", "early_stopped_no_split",
        "split_task_count",
        "final_total_active_boundaries", "disabled_active_boundaries",
        "policy_violation", "task_count", "model_distribution", "gpu_util",
        "cpu_util", "total_util", "max_cpu_partition_util", "final_chunk_counts",
        "final_active_boundaries", "task_masks", "task_variants",
    ]
    summary_fields = [
        "algorithm", "tasksets", "schedulable_count", "schedulability_ratio",
        "split_triggered_count", "split_triggered_ratio", "split_required_count",
        "split_required_ratio", "split_proactive_count", "split_proactive_ratio",
        "single_schedulable_count", "analysis_error_count",
        "policy_violation_count", "mean_wall_clock_s", "median_wall_clock_s",
        "min_wall_clock_s", "max_wall_clock_s", "p95_wall_clock_s",
        "mean_optimization_runtime_s", "mean_masks_evaluated",
        "mean_baseline_k1_hits", "total_baseline_k1_hits",
        "mean_real_profiles", "mean_cache_hits", "mean_skipped_cache_misses",
        "mean_builds_triggered", "mean_exports_triggered", "total_real_profiles",
        "total_cache_hits", "total_masks_evaluated",
        "total_interval_cache_hits", "total_interval_cache_misses",
        "total_interval_onnx_cache_hits", "total_interval_engine_cache_hits",
        "mean_interval_engine_build_wall_s",
        "mean_total_export_wall_s", "mean_total_build_wall_s",
        "mean_total_profile_wall_s", "mean_total_estimated_cold_s",
    ]
    write_csv(out_dir / "per_taskset_algorithm_results.csv", rows, per_fields)
    write_csv(out_dir / "fig5_design_time_summary.csv", summary_rows, summary_fields)
    (out_dir / "all_results.json").write_text(json.dumps(rows, indent=2))
    write_summary(out_dir, args, tasksets, rows, summary_rows, live_budget)

    print(f"\nSaved: {(out_dir / 'per_taskset_algorithm_results.csv').relative_to(REPO)}")
    print(f"Saved: {(out_dir / 'fig5_design_time_summary.csv').relative_to(REPO)}")
    print(f"Saved: {(out_dir / 'summary.md').relative_to(REPO)}")
    return 0 if not any(r["analysis_error"] for r in rows) else 2


if __name__ == "__main__":
    sys.exit(main())
