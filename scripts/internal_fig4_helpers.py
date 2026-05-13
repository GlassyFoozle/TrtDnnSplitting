#!/usr/bin/env python3
"""
56_run_fig4_pilot.py — Small Fig.4-style DNN schedulability pilot.

This is a safe validation driver, not the final large experiment. By default it
generates small DNN-backed tasksets, runs selected UNI/SS algorithms in dry-run
mode, aggregates schedulability ratios, and reports whether any split masks
actually changed from their initial no-split configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.integration.dnn_algorithm_runner import DNNAlgorithmResult, run_dnn_rta_algorithm
from src.integration.dnn_workload_generator import (
    WorkloadConfig,
    generate_tasksets,
    _get_base_gpu_wcet_ms,
)
from src.integration.live_budget import LiveProfileBudget
from src.integration.split_point_policy import get_enabled_boundaries


AlgorithmSpec = Tuple[str, str]
TasksetEntry = Tuple[Optional[float], Path]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a small safe Fig.4-style DNN schedulability pilot"
    )
    ap.add_argument("--models", nargs="+", default=["alexnet", "resnet18", "vgg19"])
    ap.add_argument("--num-tasks", type=int, default=3)
    ap.add_argument("--utilizations", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--num-tasksets", type=int, default=5)
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=["ss:tol-fb", "ss:opt", "uni:tol-fb", "uni:opt"],
        help="Algorithms as model:algorithm, e.g. ss:tol-fb uni:opt",
    )
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--wcet-metric", default="max", choices=["max", "p99", "mean"], dest="wcet_metric")
    ap.add_argument("--cpu-pre-min", type=float, default=0.5)
    ap.add_argument("--cpu-pre-max", type=float, default=2.0)
    ap.add_argument("--cpu-post-min", type=float, default=0.2)
    ap.add_argument("--cpu-post-max", type=float, default=1.0)
    ap.add_argument("--num-cpus", type=int, default=1)
    ap.add_argument("--num-tasks-per-cpu", type=int, default=None)
    ap.add_argument("--period-min-ms", type=float, default=1.0)
    ap.add_argument("--period-max-ms", type=float, default=5000.0)
    ap.add_argument(
        "--utilization-basis",
        default="total",
        choices=["gpu", "total"],
        help="Use GPU-only or CPU+GPU cost when deriving generated task periods",
    )
    ap.add_argument(
        "--taskgen-mode",
        default="dnnsplitting",
        choices=["legacy", "dnnsplitting"],
        help="legacy range-based CPU sampling or DNNSplitting-compatible generation",
    )
    ap.add_argument(
        "--g-ratio-min",
        type=float,
        default=0.6,
        help="Minimum accepted actual G/(C+G) ratio in dnnsplitting taskgen mode",
    )
    ap.add_argument(
        "--g-ratio-max",
        type=float,
        default=1.0,
        help="Maximum accepted actual G/(C+G) ratio in dnnsplitting taskgen mode",
    )
    ap.add_argument(
        "--uniform-cpu-utilization",
        action="store_true",
        default=True,
        help="Split total utilization evenly across CPUs in dnnsplitting taskgen mode",
    )
    ap.add_argument(
        "--uunifast-cpu-utilization",
        action="store_false",
        dest="uniform_cpu_utilization",
        help="Use UUniFast across CPUs in dnnsplitting taskgen mode",
    )
    ap.add_argument(
        "--uniform-task-utilization",
        action="store_true",
        default=False,
        help="Split each CPU's utilization evenly across tasks in dnnsplitting mode",
    )
    ap.add_argument("--g-utilization-threshold", type=float, default=1.0)
    ap.add_argument("--number-of-inference-segments", type=int, default=1)
    ap.add_argument("--max-block-count", type=int, default=None)
    ap.add_argument("--per-splitting-overhead", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=200)
    ap.add_argument("--split-policy", default="stage", choices=["all", "paper_like", "stage", "five_points", "ten_points", "major_blocks"])
    ap.add_argument("--max-candidates", type=int, default=50)
    ap.add_argument("--max-profiles", type=int, default=10)
    ap.add_argument("--max-iterations", type=int, default=1000)
    ap.add_argument(
        "--allow-proactive-splitting",
        action="store_true",
        default=False,
        help="For paper-style OPT/HEU, continue split search even if no-split is schedulable.",
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run mode is the default. Kept as an explicit readable flag.",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Disable dry-run and allow cache-first real profiling/builds under caps.",
    )
    ap.add_argument(
        "--cache-only-live",
        action="store_true",
        default=False,
        dest="cache_only_live",
        help="In live mode: skip any mask not already in the evaluation cache.",
    )
    ap.add_argument(
        "--global-max-real-profiles",
        type=int,
        default=None,
        dest="global_max_real_profiles",
        help="Hard cap on new TRT profiles across the entire run (shared budget). "
             "0 means cache-only (no new builds). None (default) = unlimited.",
    )
    ap.add_argument(
        "--stop-on-first-build",
        action="store_true",
        default=False,
        dest="stop_on_first_build",
        help="Stop the run immediately if a cache miss would trigger a new build.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--output-dir",
        default=str(REPO / "results" / "dnn_experiments"),
        help="Base output directory; run output is written under <output-dir>/<run-name>",
    )
    ap.add_argument(
        "--tasksets",
        nargs="*",
        default=[],
        help="Optional existing taskset JSONs to consume instead of generating.",
    )
    ap.add_argument(
        "--taskset-dir",
        default=None,
        help="Optional directory of taskset_*.json files to consume instead of generating.",
    )
    ap.add_argument(
        "--deadline-scale",
        type=float,
        default=1.0,
        help=(
            "Validation-only: set D_i = T_i * scale after generation/copy. "
            "Use values below 1.0 to force tighter tasksets."
        ),
    )
    return ap.parse_args()


def parse_algorithm_specs(specs: Iterable[str]) -> List[AlgorithmSpec]:
    parsed: List[AlgorithmSpec] = []
    for spec in specs:
        if ":" in spec:
            model, algorithm = spec.split(":", 1)
        else:
            model, algorithm = "ss", spec
        parsed.append((model.lower(), algorithm.lower()))
    return parsed


def util_dir_name(utilization: float) -> str:
    return f"u{utilization:.2f}".replace(".", "p")


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    mode = "live" if args.live else "dry"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"fig4_pilot_n{args.num_tasks}_{mode}_{stamp}"


def collect_existing_tasksets(args: argparse.Namespace) -> List[Path]:
    paths = [Path(p) for p in args.tasksets]
    if args.taskset_dir:
        td = Path(args.taskset_dir)
        paths.extend(sorted(td.glob("taskset_*.json")))
    return paths


def prepare_tasksets(args: argparse.Namespace, out_dir: Path) -> List[TasksetEntry]:
    taskset_root = out_dir / "generated_tasksets"
    taskset_root.mkdir(parents=True, exist_ok=True)

    existing = collect_existing_tasksets(args)
    if existing:
        return copy_and_normalize_tasksets(existing, taskset_root, args)

    entries: List[TasksetEntry] = []
    for util_idx, utilization in enumerate(args.utilizations):
        util_dir = taskset_root / util_dir_name(utilization)
        cfg = WorkloadConfig(
            models=args.models,
            n_tasks=args.num_tasks,
            utilization=utilization,
            n_tasksets=args.num_tasksets,
            precision=args.precision,
            wcet_metric=args.wcet_metric,
            cpu_pre_range=(args.cpu_pre_min, args.cpu_pre_max),
            cpu_post_range=(args.cpu_post_min, args.cpu_post_max),
            period_min_ms=args.period_min_ms,
            period_max_ms=args.period_max_ms,
            cpu_id=0,
            seed=args.seed + util_idx,
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
        written = generate_tasksets(cfg, output_dir=util_dir)
        for idx, path in enumerate(written):
            normalize_taskset(path, args, utilization=utilization, taskset_idx=idx)
            entries.append((utilization, path))
    return entries


def copy_and_normalize_tasksets(
    paths: List[Path],
    taskset_root: Path,
    args: argparse.Namespace,
) -> List[TasksetEntry]:
    entries: List[TasksetEntry] = []
    input_dir = taskset_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for idx, source in enumerate(paths):
        source = source if source.is_absolute() else (REPO / source)
        target = input_dir / f"{idx:03d}_{source.name}"
        shutil.copy2(source, target)
        util = infer_utilization(target)
        normalize_taskset(target, args, utilization=util, taskset_idx=idx)
        entries.append((util, target))
    return entries


def infer_utilization(path: Path) -> Optional[float]:
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return None
    value = raw.get("_utilization") or raw.get("utilization")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_taskset(
    path: Path,
    args: argparse.Namespace,
    utilization: Optional[float],
    taskset_idx: int,
) -> None:
    """
    Normalize generated/copied tasksets for dnn_taskset_loader compatibility.

    The current workload generator emits per-task precision/wcet fields but not
    top-level precision/wcet fields or target_chunks. Script 56 keeps that fix
    local to its output copy so earlier scripts are not changed.
    """
    raw = json.loads(path.read_text())
    raw.setdefault("name", path.stem)
    raw.setdefault("description", "Generated/copied by scripts/56_run_fig4_pilot.py")
    raw["precision"] = args.precision
    raw["wcet_metric"] = args.wcet_metric
    raw.setdefault("taskgen_mode", args.taskgen_mode)
    if "utilization_basis" not in raw:
        legacy_notes = " ".join(str(t.get("notes", "")) for t in raw.get("tasks", []))
        raw["utilization_basis"] = (
            "gpu" if raw.get("_generated") and "basis=" not in legacy_notes
            else args.utilization_basis
        )
    raw["_fig4_pilot_normalized"] = True
    raw["_deadline_scale"] = args.deadline_scale
    if utilization is not None:
        raw["_utilization"] = utilization
    raw.setdefault("_taskset_idx", taskset_idx)

    tasks = raw.get("tasks", [])
    for index, task in enumerate(tasks):
        if "target_chunks" not in task and "initial_mask" not in task:
            task["target_chunks"] = 1
        task["precision"] = args.precision
        task["wcet_metric"] = args.wcet_metric
        if args.num_cpus > 0 and raw.get("taskgen_mode") != "dnnsplitting":
            task["cpu_id"] = index % args.num_cpus
        if args.deadline_scale <= 0:
            raise ValueError("--deadline-scale must be positive")
        if args.deadline_scale != 1.0:
            task["deadline_ms"] = round(float(task["period_ms"]) * args.deadline_scale, 6)
            note = task.get("notes", "")
            suffix = f" deadline_scale={args.deadline_scale:.4g}"
            task["notes"] = f"{note};{suffix}" if note else suffix.strip()

    tasks.sort(key=lambda t: float(t["deadline_ms"]))
    for priority, task in enumerate(tasks, start=1):
        task["priority"] = priority

    path.write_text(json.dumps(raw, indent=2))


def load_initial_masks(path: Path) -> Dict[str, List[int]]:
    raw = json.loads(path.read_text())
    masks: Dict[str, List[int]] = {}
    for task in raw.get("tasks", []):
        task_name = str(task.get("task_name", ""))
        if "initial_mask" in task:
            masks[task_name] = list(task["initial_mask"])
        elif int(task.get("target_chunks", 1)) <= 1:
            # Length is unknown without candidate-space loading; active count is
            # enough for split-trigger detection because this means all-zero.
            masks[task_name] = []
        else:
            # Target-K generated masks are expanded by the loader. We record the
            # requested active-boundary count and compare by active count later.
            k = int(task.get("target_chunks", 1))
            masks[task_name] = [1] * max(0, k - 1)
    return masks


def mask_changed(initial_mask: List[int], final_mask: List[int]) -> bool:
    if len(initial_mask) == len(final_mask):
        return list(initial_mask) != list(final_mask)
    return sum(initial_mask) != sum(final_mask)


def task_result_to_dict(task_result: Any) -> Dict[str, Any]:
    return {
        "task_name": task_result.task_name,
        "model_name": task_result.model_name,
        "cpu_id": task_result.cpu_id,
        "period_ms": task_result.period_ms,
        "deadline_ms": task_result.deadline_ms,
        "C_ms": task_result.C_ms,
        "G_ms": task_result.G_ms,
        "R_ms": task_result.R_ms,
        "slack_ms": task_result.slack_ms,
        "schedulable": task_result.schedulable,
        "final_mask": list(task_result.final_mask),
        "final_active_boundaries": int(sum(task_result.final_mask)),
        "final_k_chunks": task_result.final_k_chunks,
        "final_chunk_times_ms": list(task_result.final_chunk_times_ms),
        "max_G_block": task_result.max_G_block,
        "variant_name": task_result.variant_name,
        "profile_result_path": task_result.profile_result_path,
    }


def summarize_result(
    utilization: Optional[float],
    taskset_path: Path,
    model: str,
    algorithm: str,
    result: DNNAlgorithmResult,
    initial_masks: Dict[str, List[int]],
) -> Dict[str, Any]:
    task_details = [task_result_to_dict(tr) for tr in result.task_results]
    util_metrics = taskset_utilization_metrics(taskset_path, result.precision, result.wcet_metric)
    split_task_count = 0
    total_active = 0
    total_disabled_active = 0
    total_chunks = 0

    for detail in task_details:
        final_mask = detail["final_mask"]
        initial_mask = initial_masks.get(detail["task_name"], [])
        enabled = get_enabled_boundaries(
            detail["model_name"], result.policy_name, len(final_mask)
        )
        enabled_set = set(enabled)
        disabled_active = sum(
            1 for idx, bit in enumerate(final_mask) if bit and idx not in enabled_set
        )
        detail["initial_active_boundaries"] = int(sum(initial_mask))
        detail["disabled_active_boundaries"] = int(disabled_active)
        detail["split_triggered"] = mask_changed(initial_mask, final_mask)
        if detail["split_triggered"]:
            split_task_count += 1
        total_active += int(detail["final_active_boundaries"])
        total_disabled_active += int(disabled_active)
        total_chunks += int(detail["final_k_chunks"])

    error_type, error_message = split_error(result)
    task_chunk_counts = {
        d["task_name"]: d["final_k_chunks"] for d in task_details
    }
    task_masks = {d["task_name"]: d["final_mask"] for d in task_details}
    single_schedulable = getattr(result, "single_schedulable", None)
    split_required = bool(single_schedulable is False and result.schedulable and split_task_count > 0)
    split_proactive = bool(single_schedulable is True and split_task_count > 0)
    return {
        "utilization": utilization,
        "taskset": taskset_path.stem,
        "taskset_path": str(taskset_path.relative_to(REPO)),
        "rta_model": model.upper(),
        "algorithm": algorithm,
        "algorithm_label": f"{model.upper()}-{algorithm}",
        "schedulable": bool(result.schedulable),
        "error": result.error or "",
        "error_type": error_type,
        "error_message": error_message,
        "analysis_error": bool(getattr(result, "analysis_error", False) or result.error),
        "overload_reason": getattr(result, "unschedulable_reason", None) or "",
        "diagnostic_message": getattr(result, "diagnostic_message", None) or "",
        "duration_s": float(result.duration_s),
        "optimization_runtime_s": float(result.duration_s),
        "algorithm_iterations": int(result.algorithm_iterations),
        "masks_evaluated": int(result.stats.masks_evaluated),
        "cache_hits": int(result.stats.cache_hits),
        "real_profiles": int(result.stats.real_profiles),
        "skipped_cache_misses": int(getattr(result.stats, "skipped_cache_misses", 0)),
        "unique_masks_evaluated": int(getattr(result.stats, "unique_masks_evaluated", 0)),
        "unique_mask_cache_hits": int(getattr(result.stats, "unique_mask_cache_hits", 0)),
        "unique_skipped_masks": int(getattr(result.stats, "unique_skipped_masks", 0)),
        "interval_timing_cache_hits": int(getattr(result.stats, "interval_timing_cache_hits", 0)),
        "k_split_calls": int(getattr(result.stats, "k_split_calls", 0)),
        "k_split_cache_hits": int(getattr(result.stats, "k_split_cache_hits", 0)),
        "k_split_candidate_masks": int(getattr(result.stats, "k_split_candidate_masks", 0)),
        "k_split_candidate_chunk_profiles": int(getattr(result.stats, "k_split_candidate_chunk_profiles", 0)),
        "k_split_candidate_inference_runs": int(getattr(result.stats, "k_split_candidate_inference_runs", 0)),
        "early_stop_optimistic_checks": int(getattr(result.stats, "early_stop_optimistic_checks", 0)),
        "early_stop_optimistic_deadline_misses": int(getattr(result.stats, "early_stop_optimistic_deadline_misses", 0)),
        "dry_run_evaluations": int(result.stats.dry_run_evaluations),
        "builds_triggered": int(result.stats.builds_triggered),
        "exports_triggered": int(result.stats.exports_triggered),
        "task_count": len(task_details),
        "gpu_util": util_metrics["gpu_util"],
        "cpu_util": util_metrics["cpu_util"],
        "total_util": util_metrics["total_util"],
        "max_cpu_partition_util": util_metrics["max_cpu_partition_util"],
        "split_task_count": split_task_count,
        "any_split_triggered": split_task_count > 0,
        "split_triggered": split_task_count > 0,
        "single_schedulable": single_schedulable,
        "split_required": split_required,
        "split_proactive": split_proactive,
        "early_stopped_no_split": bool(getattr(result, "early_stopped_no_split", False)),
        "total_final_active_boundaries": total_active,
        "final_total_active_boundaries": total_active,
        "disabled_active_boundaries": total_disabled_active,
        "policy_violation": total_disabled_active > 0,
        "average_final_active_boundaries": (
            total_active / len(task_details) if task_details else 0.0
        ),
        "total_final_chunks": total_chunks,
        "average_final_chunks": total_chunks / len(task_details) if task_details else 0.0,
        "task_active_boundaries": json.dumps({
            d["task_name"]: d["final_active_boundaries"] for d in task_details
        }),
        "task_chunk_counts": json.dumps(task_chunk_counts),
        "final_chunk_counts": json.dumps(task_chunk_counts),
        "task_masks": json.dumps(task_masks),
        "task_details": task_details,
    }


def taskset_utilization_metrics(path: Path, precision: str, wcet_metric: str) -> Dict[str, float]:
    raw = json.loads(path.read_text())
    if all(k in raw for k in (
        "_actual_gpu_utilization",
        "_actual_cpu_utilization",
        "_actual_total_utilization",
    )):
        partitions = raw.get("_actual_cpu_partition_utilization", {}) or {}
        return {
            "gpu_util": float(raw.get("_actual_gpu_utilization", 0.0)),
            "cpu_util": float(raw.get("_actual_cpu_utilization", 0.0)),
            "total_util": float(raw.get("_actual_total_utilization", 0.0)),
            "max_cpu_partition_util": max((float(v) for v in partitions.values()), default=0.0),
        }

    gpu_util = 0.0
    cpu_util = 0.0
    per_cpu: Dict[str, float] = defaultdict(float)
    for task in raw.get("tasks", []):
        period = float(task["period_ms"])
        cpu = float(task.get("cpu_pre_ms", 0.0)) + float(task.get("cpu_post_ms", 0.0))
        gpu = _get_base_gpu_wcet_ms(
            task["model_name"],
            precision or raw.get("precision", "fp32"),
            wcet_metric or raw.get("wcet_metric", "max"),
        ) or 0.0
        gpu_util += gpu / period
        cpu_part = cpu / period
        cpu_util += cpu_part
        per_cpu[str(task.get("cpu_id", 0))] += cpu_part
    return {
        "gpu_util": gpu_util,
        "cpu_util": cpu_util,
        "total_util": gpu_util + cpu_util,
        "max_cpu_partition_util": max(per_cpu.values(), default=0.0),
    }


def split_error(result: DNNAlgorithmResult) -> Tuple[str, str]:
    if not result.error:
        return "", ""
    error_type = getattr(result, "error_type", None) or ""
    message = result.error
    if not error_type and ":" in message:
        maybe_type, rest = message.split(":", 1)
        if maybe_type.endswith("Error") or maybe_type.endswith("Exception"):
            error_type = maybe_type.strip()
            message = rest.strip()
    first_line = message.splitlines()[0] if message else ""
    return error_type, first_line


def aggregate(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        util_key = "unknown" if row["utilization"] is None else f"{float(row['utilization']):.2f}"
        grouped[(util_key, row["algorithm_label"])].append(row)

    ratio_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    for (util_key, label), items in sorted(grouped.items()):
        n = len(items)
        sched = sum(1 for r in items if r["schedulable"])
        split_sets = sum(1 for r in items if r["any_split_triggered"])
        ratio_rows.append({
            "utilization": util_key,
            "algorithm": label,
            "total_tasksets": n,
            "schedulable_count": sched,
            "unschedulable_count": n - sched,
            "schedulability_ratio": sched / n if n else 0.0,
            "analysis_error_count": sum(1 for r in items if r.get("analysis_error")),
            "avg_masks_evaluated": avg(items, "masks_evaluated"),
            "avg_real_profiles": avg(items, "real_profiles"),
            "avg_cache_hits": avg(items, "cache_hits"),
            "avg_skipped_cache_misses": avg(items, "skipped_cache_misses"),
            "avg_unique_masks_evaluated": avg(items, "unique_masks_evaluated"),
            "avg_unique_mask_cache_hits": avg(items, "unique_mask_cache_hits"),
            "avg_unique_skipped_masks": avg(items, "unique_skipped_masks"),
            "avg_interval_timing_cache_hits": avg(items, "interval_timing_cache_hits"),
            "avg_k_split_calls": avg(items, "k_split_calls"),
            "avg_k_split_cache_hits": avg(items, "k_split_cache_hits"),
            "avg_k_split_candidate_masks": avg(items, "k_split_candidate_masks"),
            "avg_k_split_candidate_chunk_profiles": avg(items, "k_split_candidate_chunk_profiles"),
            "avg_k_split_candidate_inference_runs": avg(items, "k_split_candidate_inference_runs"),
            "avg_early_stop_optimistic_checks": avg(items, "early_stop_optimistic_checks"),
            "avg_early_stop_optimistic_deadline_misses": avg(
                items, "early_stop_optimistic_deadline_misses"
            ),
            "avg_dry_run_evaluations": avg(items, "dry_run_evaluations"),
            "avg_final_active_boundaries": avg(items, "average_final_active_boundaries"),
            "disabled_active_boundaries": sum(
                int(r.get("disabled_active_boundaries", 0) or 0) for r in items
            ),
            "policy_violation_count": sum(1 for r in items if r.get("policy_violation")),
            "avg_gpu_util": avg(items, "gpu_util"),
            "avg_cpu_util": avg(items, "cpu_util"),
            "avg_total_util": avg(items, "total_util"),
            "max_cpu_partition_util": max(
                (float(r.get("max_cpu_partition_util", 0.0) or 0.0) for r in items),
                default=0.0,
            ),
            "split_triggered_tasksets": split_sets,
            "split_triggered_pct": split_sets / n if n else 0.0,
            "split_required_tasksets": sum(1 for r in items if r.get("split_required")),
            "split_required_pct": (
                sum(1 for r in items if r.get("split_required")) / n if n else 0.0
            ),
            "split_proactive_tasksets": sum(1 for r in items if r.get("split_proactive")),
            "split_proactive_pct": (
                sum(1 for r in items if r.get("split_proactive")) / n if n else 0.0
            ),
            "avg_duration_s": avg(items, "duration_s"),
            "avg_optimization_runtime_s": avg(items, "optimization_runtime_s"),
            "error_count": sum(1 for r in items if r.get("error")),
        })
        split_rows.append({
            "utilization": util_key,
            "algorithm": label,
            "tasksets": n,
            "tasksets_with_any_split": split_sets,
            "split_triggered_pct": split_sets / n if n else 0.0,
            "avg_split_tasks_per_taskset": avg(items, "split_task_count"),
            "avg_final_active_boundaries": avg(items, "average_final_active_boundaries"),
            "avg_final_chunks": avg(items, "average_final_chunks"),
            "disabled_active_boundaries": sum(
                int(r.get("disabled_active_boundaries", 0) or 0) for r in items
            ),
            "policy_violation_count": sum(1 for r in items if r.get("policy_violation")),
            "avg_masks_evaluated": avg(items, "masks_evaluated"),
            "avg_cache_hits": avg(items, "cache_hits"),
            "avg_real_profiles": avg(items, "real_profiles"),
            "avg_skipped_cache_misses": avg(items, "skipped_cache_misses"),
            "avg_k_split_calls": avg(items, "k_split_calls"),
            "avg_k_split_cache_hits": avg(items, "k_split_cache_hits"),
            "avg_k_split_candidate_masks": avg(items, "k_split_candidate_masks"),
            "avg_k_split_candidate_chunk_profiles": avg(items, "k_split_candidate_chunk_profiles"),
            "avg_k_split_candidate_inference_runs": avg(items, "k_split_candidate_inference_runs"),
            "avg_early_stop_optimistic_checks": avg(items, "early_stop_optimistic_checks"),
            "avg_early_stop_optimistic_deadline_misses": avg(
                items, "early_stop_optimistic_deadline_misses"
            ),
            "avg_dry_run_evaluations": avg(items, "dry_run_evaluations"),
        })

    return ratio_rows, split_rows


def avg(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key, 0.0) or 0.0) for r in rows) / len(rows)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames and rows:
        fieldnames = list(rows[0].keys())
    if not fieldnames:
        path.write_text("")
        return
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


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    tasksets: List[TasksetEntry],
    ratio_rows: List[Dict[str, Any]],
    split_rows: List[Dict[str, Any]],
    per_rows: List[Dict[str, Any]],
    live_budget: "Optional[LiveProfileBudget]" = None,
) -> None:
    any_split = any(r["any_split_triggered"] for r in per_rows)
    errors = [r for r in per_rows if r.get("error")]
    generated_by_util: Dict[str, int] = defaultdict(int)
    for util, _ in tasksets:
        util_key = "unknown" if util is None else f"{float(util):.2f}"
        generated_by_util[util_key] += 1
    using_existing = bool(collect_existing_tasksets(args))
    requested = len(tasksets) if using_existing else len(args.utilizations) * args.num_tasksets
    lines = [
        "# Fig.4 Pilot Summary",
        "",
        "## Run config",
        "",
        f"- Mode: {'live/cache-first' if args.live else 'dry-run'}",
        f"- Models: {', '.join(args.models)}",
        f"- Algorithms: {', '.join(args.algorithms)}",
        f"- Tasksets generated/copied: {len(tasksets)}",
        f"- Requested/generated tasksets: {requested}",
        f"- Utilizations: {', '.join(str(u) for u in args.utilizations)}",
        f"- Tasks per taskset: {args.num_tasks}",
        f"- Split policy: {args.split_policy}",
        f"- Max candidates: {args.max_candidates}",
        f"- Max profiles: {args.max_profiles}",
        f"- Deadline scale: {args.deadline_scale}",
        f"- Utilization basis: {args.utilization_basis}",
        f"- Task generation mode: {args.taskgen_mode}",
        f"- Number of CPUs: {args.num_cpus}",
        f"- G-ratio range: [{args.g_ratio_min}, {args.g_ratio_max}]",
    ]
    if live_budget is not None:
        lines += [
            f"- cache_only_live: {live_budget.cache_only}",
            f"- global_max_real_profiles: {live_budget.global_max_real_profiles}",
            f"- stop_on_first_build: {live_budget.stop_on_first_build}",
            f"- global_profile_budget_used: {live_budget.used_real_profiles}",
            f"- skipped_cache_misses_attempts: {live_budget.skipped_cache_misses}",
        ]
    lines += [
        "",
        "## Taskset counts",
        "",
        format_table(
            [
                {
                    "utilization": util_key,
                    "tasksets": count,
                }
                for util_key, count in sorted(generated_by_util.items())
            ],
            ["utilization", "tasksets"],
        ),
        "",
        "## Fig.4-style schedulability metrics",
        "",
        format_table(
            ratio_rows,
            [
                "utilization",
                "algorithm",
                "total_tasksets",
                "schedulable_count",
                "unschedulable_count",
                "schedulability_ratio",
                "analysis_error_count",
                "avg_total_util",
                "avg_masks_evaluated",
                "disabled_active_boundaries",
                "policy_violation_count",
                "avg_real_profiles",
                "avg_cache_hits",
                "avg_skipped_cache_misses",
                "avg_duration_s",
                "error_count",
            ],
        ),
        "",
        "## Fig.5-style optimization-cost metrics",
        "",
        format_table(
            ratio_rows,
            [
                "utilization",
                "algorithm",
                "avg_optimization_runtime_s",
                "avg_masks_evaluated",
                "avg_dry_run_evaluations",
                "avg_cache_hits",
                "avg_real_profiles",
                "avg_skipped_cache_misses",
                "analysis_error_count",
            ],
        ),
        "",
        "## Split activity",
        "",
        format_table(
            split_rows,
            [
                "utilization",
                "algorithm",
                "tasksets",
                "tasksets_with_any_split",
                "split_triggered_pct",
                "avg_split_tasks_per_taskset",
                "avg_final_active_boundaries",
                "avg_final_chunks",
                "disabled_active_boundaries",
                "policy_violation_count",
            ],
        ),
        "",
        "## Validation verdict",
        "",
        f"- Actual splitting observed: {'yes' if any_split else 'no'}",
        f"- Result rows with errors: {len(errors)}",
        "",
        "## Warnings and limitations",
        "",
        "- This pilot is dry-run by default and does not prove final TensorRT timing behavior.",
        "- `--deadline-scale` is validation-only and should not be used for paper Section VIII reproduction unless explicitly justified.",
        "- `stage` policy intentionally restricts the search space and is mainly useful for smoke testing.",
        "- Python orchestration time is reported only as algorithm-driver runtime, not runtime-server performance.",
    ]
    if len(tasksets) < requested and not using_existing:
        lines.append(
            "- Fewer tasksets than requested were generated; UUniFast samples can be rejected by the period range."
        )
    if errors:
        lines.extend(["", "## Errors", ""])
        for row in errors[:20]:
            err_prefix = f"{row.get('error_type')}: " if row.get("error_type") else ""
            lines.append(
                f"- {row['algorithm_label']} on {row['taskset']}: "
                f"{err_prefix}{row.get('error_message') or row.get('error')}"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_run_config(
    out_dir: Path,
    args: argparse.Namespace,
    run_name: str,
    algorithm_specs: List[AlgorithmSpec],
    tasksets: List[TasksetEntry],
) -> None:
    data = vars(args).copy()
    data["run_name"] = run_name
    data["dry_run_effective"] = not args.live
    data["algorithm_specs"] = [f"{m}:{a}" for m, a in algorithm_specs]
    data["tasksets"] = [
        {
            "utilization": util,
            "path": str(path.relative_to(REPO)),
        }
        for util, path in tasksets
    ]
    (out_dir / "run_config.json").write_text(json.dumps(data, indent=2))


def main() -> int:
    args = parse_args()
    dry_run = not args.live
    run_name = make_run_name(args)
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    algorithm_specs = parse_algorithm_specs(args.algorithms)

    # Build global live-profile budget (live mode only)
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

    print(f"Run: {run_name}")
    print(f"Mode: {'live/cache-first' if args.live else 'dry-run'}")
    if live_budget is not None:
        print(f"  cache_only_live          : {live_budget.cache_only}")
        print(f"  global_max_real_profiles : {live_budget.global_max_real_profiles}")
        print(f"  stop_on_first_build      : {live_budget.stop_on_first_build}")
    print(f"Algorithms: {[f'{m}:{a}' for m, a in algorithm_specs]}")
    print(f"Output: {out_dir.relative_to(REPO)}")

    tasksets = prepare_tasksets(args, out_dir)
    if not tasksets:
        print("[Error] No tasksets were generated or copied.", file=sys.stderr)
        return 1

    write_run_config(out_dir, args, run_name, algorithm_specs, tasksets)

    per_rows: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    run_stopped = False

    for util, taskset_path in tasksets:
        if run_stopped:
            break
        print(f"\nTaskset: {taskset_path.relative_to(REPO)}")
        initial_masks = load_initial_masks(taskset_path)
        for model, algorithm in algorithm_specs:
            if run_stopped:
                break
            label = f"{model.upper()}-{algorithm}"
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
            )
            row = summarize_result(util, taskset_path, model, algorithm, result, initial_masks)
            if live_budget is not None:
                row["skipped_cache_misses"] = int(result.stats.skipped_cache_misses)
            per_rows.append(row)
            all_results.append({
                **{k: v for k, v in row.items() if k != "task_details"},
                "stats": result.stats.to_dict(),
                "task_details": row["task_details"],
            })
            sched = "SCHED" if result.schedulable else "MISS"
            split = "split" if row["any_split_triggered"] else "no-split"
            skipped = result.stats.skipped_cache_misses
            error = " ERROR" if result.error else ""
            live_info = f" skipped={skipped:2d}" if live_budget is not None else ""
            print(
                f"  {label:16s} {sched:5s} {split:8s} "
                f"masks={result.stats.masks_evaluated:3d} "
                f"dry={result.stats.dry_run_evaluations:3d} "
                f"cache={result.stats.cache_hits:3d} "
                f"real={result.stats.real_profiles:3d}"
                f"{live_info}{error}"
            )
            if live_budget is not None and live_budget.stopped:
                print(
                    f"\n[live_budget] stop_on_first_build fired: "
                    f"{live_budget.stop_model}/{live_budget.stop_variant}"
                )
                run_stopped = True
                break

    ratio_rows, split_rows = aggregate(per_rows)

    per_fieldnames = [
        "utilization",
        "taskset",
        "taskset_path",
        "rta_model",
        "algorithm",
        "algorithm_label",
        "schedulable",
        "error",
        "error_type",
        "error_message",
        "analysis_error",
        "overload_reason",
        "diagnostic_message",
        "duration_s",
        "optimization_runtime_s",
        "algorithm_iterations",
        "masks_evaluated",
        "cache_hits",
        "real_profiles",
        "skipped_cache_misses",
        "k_split_calls",
        "k_split_cache_hits",
        "k_split_candidate_masks",
        "k_split_candidate_chunk_profiles",
        "k_split_candidate_inference_runs",
        "early_stop_optimistic_checks",
        "early_stop_optimistic_deadline_misses",
        "dry_run_evaluations",
        "builds_triggered",
        "exports_triggered",
        "task_count",
        "gpu_util",
        "cpu_util",
        "total_util",
        "max_cpu_partition_util",
        "split_task_count",
        "any_split_triggered",
        "split_triggered",
        "single_schedulable",
        "split_required",
        "split_proactive",
        "early_stopped_no_split",
        "total_final_active_boundaries",
        "final_total_active_boundaries",
        "disabled_active_boundaries",
        "policy_violation",
        "average_final_active_boundaries",
        "total_final_chunks",
        "average_final_chunks",
        "task_active_boundaries",
        "task_chunk_counts",
        "final_chunk_counts",
        "task_masks",
    ]
    ratio_fieldnames = [
        "utilization",
        "algorithm",
        "total_tasksets",
        "schedulable_count",
        "unschedulable_count",
        "schedulability_ratio",
        "analysis_error_count",
        "avg_masks_evaluated",
        "avg_real_profiles",
        "avg_cache_hits",
        "avg_skipped_cache_misses",
        "avg_k_split_calls",
        "avg_k_split_cache_hits",
        "avg_k_split_candidate_masks",
        "avg_k_split_candidate_chunk_profiles",
        "avg_k_split_candidate_inference_runs",
        "avg_early_stop_optimistic_checks",
        "avg_early_stop_optimistic_deadline_misses",
        "avg_dry_run_evaluations",
        "avg_final_active_boundaries",
        "disabled_active_boundaries",
        "policy_violation_count",
        "avg_gpu_util",
        "avg_cpu_util",
        "avg_total_util",
        "max_cpu_partition_util",
        "split_triggered_tasksets",
        "split_triggered_pct",
        "split_required_tasksets",
        "split_required_pct",
        "split_proactive_tasksets",
        "split_proactive_pct",
        "avg_duration_s",
        "avg_optimization_runtime_s",
        "error_count",
    ]
    split_fieldnames = [
        "utilization",
        "algorithm",
        "tasksets",
        "tasksets_with_any_split",
        "split_triggered_pct",
        "avg_split_tasks_per_taskset",
        "avg_final_active_boundaries",
        "avg_final_chunks",
        "disabled_active_boundaries",
        "policy_violation_count",
        "avg_masks_evaluated",
        "avg_cache_hits",
        "avg_real_profiles",
        "avg_skipped_cache_misses",
        "avg_k_split_calls",
        "avg_k_split_cache_hits",
        "avg_k_split_candidate_masks",
        "avg_k_split_candidate_chunk_profiles",
        "avg_k_split_candidate_inference_runs",
        "avg_early_stop_optimistic_checks",
        "avg_early_stop_optimistic_deadline_misses",
        "avg_dry_run_evaluations",
    ]

    write_csv(out_dir / "per_taskset_results.csv", per_rows, per_fieldnames)
    write_csv(out_dir / "schedulability_ratio.csv", ratio_rows, ratio_fieldnames)
    write_csv(out_dir / "split_activity.csv", split_rows, split_fieldnames)
    (out_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
    write_summary(out_dir, args, tasksets, ratio_rows, split_rows, per_rows, live_budget)

    print(f"\nSaved: {(out_dir / 'per_taskset_results.csv').relative_to(REPO)}")
    print(f"Saved: {(out_dir / 'schedulability_ratio.csv').relative_to(REPO)}")
    print(f"Saved: {(out_dir / 'split_activity.csv').relative_to(REPO)}")
    print(f"Saved: {(out_dir / 'summary.md').relative_to(REPO)}")

    any_split = any(row["any_split_triggered"] for row in per_rows)
    print(f"\nActual splitting observed: {'yes' if any_split else 'no'}")
    if live_budget is not None:
        print(f"Global profile budget used : {live_budget.used_real_profiles}")
        print(f"Skipped cache misses       : {live_budget.skipped_cache_misses}")
        if live_budget.stopped:
            print(
                f"Run stopped on first build : {live_budget.stop_model}/{live_budget.stop_variant}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
