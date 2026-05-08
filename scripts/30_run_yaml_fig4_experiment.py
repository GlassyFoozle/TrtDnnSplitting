#!/usr/bin/env python3
"""
65_run_yaml_fig4_experiment.py — YAML-driven Fig.4-style DNN experiment runner.

This script maps DNNSplitting/Bertogna-style YAML task-generation configs onto
the real-DNN TensorRTServer experiment pipeline.

Important real-DNN difference:
  DNNSplitting samples a synthetic period and derives synthetic G from U*T.
  Here, the DNN GPU WCET is fixed by real TensorRT profiling metadata. In the
  default mode, period_range is ignored and each task uses:

      G = real model GPU WCET
      total_exec = G / sampled_G_ratio
      CPU = total_exec - G
      T = D = total_exec / U_i

Use --no-ignore-period-range only to apply the YAML period_range as a validity
filter after this derivation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.integration.dnn_algorithm_runner import run_dnn_rta_algorithm
from src.integration.dnn_workload_generator import WorkloadConfig, generate_tasksets
from src.integration.live_budget import LiveProfileBudget


AlgorithmSpec = Tuple[str, str, str]
TasksetEntry = Tuple[float, Path]

_DEFAULT_ALGORITHMS: List[AlgorithmSpec] = [
    ("ss", "tol-fb", "SS_ours"),
    ("uni", "tol-fb", "UNI_ours"),
    ("ss", "opt", "SS_Buttazzo"),
    ("uni", "opt", "UNI_Buttazzo"),
]


def _load_fig4_helpers():
    """Load shared helper functions from the internal_fig4_helpers module."""
    helper_path = REPO / "scripts" / "internal_fig4_helpers.py"
    spec = importlib.util.spec_from_file_location("fig4_pilot_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FIG4 = _load_fig4_helpers()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run a YAML-driven Fig.4-style DNN schedulability experiment"
    )
    ap.add_argument("--config", required=True, help="DNNSplitting-style YAML config")
    ap.add_argument("--models", nargs="+", default=["alexnet", "resnet18", "vgg19"])
    ap.add_argument(
        "--split-policy",
        default="major_blocks",
        choices=["all", "paper_like", "stage", "five_points", "ten_points", "major_blocks"],
    )
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--wcet-metric", default="p99", choices=["p99", "mean"], dest="wcet_metric")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--live", action="store_true", default=False)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-candidates", type=int, default=10000)
    ap.add_argument("--max-profiles", type=int, default=500)
    ap.add_argument("--max-iterations", type=int, default=1000)
    ap.add_argument("--global-max-real-profiles", type=int, default=None)
    ap.add_argument("--cache-only-live", action="store_true", default=False)
    ap.add_argument("--stop-on-first-build", action="store_true", default=False)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--ignore-period-range",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Ignore YAML period_range when deriving real-DNN periods. "
            "Use --no-ignore-period-range to apply it as a validity filter."
        ),
    )
    ap.add_argument(
        "--num-tasksets-override",
        "--n-tasksets-override",
        dest="num_tasksets_override",
        type=int,
        default=None,
        help="Override YAML n_task_sets for smoke tests",
    )
    ap.add_argument(
        "--utilizations",
        nargs="*",
        type=float,
        default=None,
        help="Override YAML utilization_range/utilization_step",
    )
    ap.add_argument(
        "--allow-proactive-splitting",
        action="store_true",
        default=False,
        help="Allow OPT/HEU search even when no-split is already schedulable.",
    )
    ap.add_argument(
        "--output-dir",
        default=str(REPO / "results" / "dnn_experiments"),
        help="Base output directory; run output goes under <output-dir>/<run-name>",
    )
    return ap.parse_args()


def strip_comment(line: str) -> str:
    in_quote = False
    quote_char = ""
    for idx, ch in enumerate(line):
        if ch in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
        elif ch == "#" and not in_quote:
            return line[:idx]
    return line


def parse_simple_yaml(path: Path) -> Dict[str, Any]:
    """
    Parse the flat YAML files used by DNNSplitting configs.

    This deliberately avoids a PyYAML dependency for base-environment smoke
    checks. It supports key: scalar, booleans, and inline lists.
    """
    data: Dict[str, Any] = {}
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = strip_comment(raw_line).strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{lineno}: expected 'key: value', got {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = None
            continue
        low = value.lower()
        if low == "true":
            data[key] = True
        elif low == "false":
            data[key] = False
        else:
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value
    return data


def resolve_config_path(config: str) -> Path:
    requested = Path(config)
    if requested.exists():
        return requested
    fallback = REPO.parent / "DNNSplitting" / "overnight" / requested.name
    if fallback.exists():
        print(f"[warn] config not found at {requested}; using workspace copy {fallback}")
        return fallback
    raise FileNotFoundError(f"YAML config not found: {requested}")


def range_pair(data: Dict[str, Any], key: str, default: Tuple[float, float]) -> Tuple[float, float]:
    value = data.get(key, list(default))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    scalar = float(value)
    return scalar, scalar


def int_range_choice(data: Dict[str, Any], key: str, default: int) -> Tuple[int, str]:
    lo, hi = range_pair(data, key, (default, default))
    if int(lo) != int(hi):
        return int(lo), f"{key} is a range [{lo}, {hi}]; mapped to lower bound {int(lo)}"
    return int(lo), ""


def first_range_value(data: Dict[str, Any], key: str, default: float) -> Tuple[float, str]:
    lo, hi = range_pair(data, key, (default, default))
    if lo != hi:
        return lo, f"{key} is a range [{lo}, {hi}]; mapped to lower bound {lo}"
    return lo, ""


def utilization_list(data: Dict[str, Any], override: Optional[List[float]]) -> List[float]:
    if override:
        return [round(float(u), 10) for u in override]
    lo, hi = range_pair(data, "utilization_range", (0.7, 0.9))
    step = float(data.get("utilization_step", 0.05))
    if step <= 0:
        raise ValueError("utilization_step must be positive")
    values: List[float] = []
    cur = lo
    while cur <= hi + (step / 1000.0):
        values.append(round(cur, 10))
        cur += step
    return values


def make_run_name(args: argparse.Namespace, config_path: Path) -> str:
    if args.run_name:
        return args.run_name
    mode = "live" if args.live else "dry"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"yaml_fig4_{config_path.stem}_{mode}_{stamp}"


def util_dir_name(utilization: float) -> str:
    return f"u{utilization:.2f}".replace(".", "p")


def build_mapping(
    yaml_data: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    num_cpus, note = int_range_choice(yaml_data, "number_of_cpu_range", 1)
    if note:
        notes.append(note)
    tasks_per_cpu, note = int_range_choice(yaml_data, "number_of_tasks_per_cpu_range", 1)
    if note:
        notes.append(note)
    g_min, g_max = range_pair(yaml_data, "G_ratio_range", (0.6, 1.0))
    g_threshold, note = first_range_value(yaml_data, "G_utilization_threshold_range", 1.0)
    if note:
        notes.append(note)
    n_inference_segments, note = int_range_choice(
        yaml_data, "number_of_inference_segments_range", 1
    )
    if note:
        notes.append(note)
    max_block_count, note = int_range_choice(yaml_data, "max_block_count_range", 20)
    if note:
        notes.append(note)
    period_lo, period_hi = range_pair(yaml_data, "period_range", (1.0, 10000.0))
    n_tasksets = int(args.num_tasksets_override or yaml_data.get("n_task_sets", 1))
    utilizations = utilization_list(yaml_data, args.utilizations)

    if args.ignore_period_range:
        period_min_ms = 0.001
        period_max_ms = 1_000_000_000.0
        notes.append(
            "period_range is ignored by default for real-DNN generation; "
            "periods are derived from real GPU WCET, sampled G_ratio, and U_i."
        )
    else:
        period_min_ms = period_lo
        period_max_ms = period_hi
        notes.append(
            "period_range is applied as a validity filter after real-DNN period derivation."
        )

    mapping = {
        "num_cpus": num_cpus,
        "num_tasks_per_cpu": tasks_per_cpu,
        "num_tasks": num_cpus * tasks_per_cpu,
        "utilizations": utilizations,
        "num_tasksets_per_utilization": n_tasksets,
        "g_ratio_min": g_min,
        "g_ratio_max": g_max,
        "uniform_cpu_utilization": bool(yaml_data.get("uniform_cpu_utilization", True)),
        "uniform_task_utilization": bool(yaml_data.get("uniform_task_utilization", False)),
        "g_utilization_threshold": g_threshold,
        "number_of_inference_segments": n_inference_segments,
        "max_block_count": max_block_count,
        "per_splitting_overhead": float(yaml_data.get("per_splitting_overhead", 0.0)),
        "yaml_period_range": [period_lo, period_hi],
        "ignore_period_range": bool(args.ignore_period_range),
        "period_min_ms_used": period_min_ms,
        "period_max_ms_used": period_max_ms,
        "taskgen_mode": "dnnsplitting",
        "utilization_basis": "total",
    }
    return mapping, notes


def generate_yaml_tasksets(
    args: argparse.Namespace,
    mapping: Dict[str, Any],
    out_dir: Path,
) -> List[TasksetEntry]:
    root = out_dir / "generated_tasksets"
    root.mkdir(parents=True, exist_ok=True)
    entries: List[TasksetEntry] = []
    for util_idx, utilization in enumerate(mapping["utilizations"]):
        util_dir = root / util_dir_name(float(utilization))
        cfg = WorkloadConfig(
            models=args.models,
            n_tasks=int(mapping["num_tasks"]),
            utilization=float(utilization),
            n_tasksets=int(mapping["num_tasksets_per_utilization"]),
            precision=args.precision,
            wcet_metric=args.wcet_metric,
            period_min_ms=float(mapping["period_min_ms_used"]),
            period_max_ms=float(mapping["period_max_ms_used"]),
            seed=args.seed + util_idx,
            utilization_basis="total",
            taskgen_mode="dnnsplitting",
            num_cpus=int(mapping["num_cpus"]),
            tasks_per_cpu=int(mapping["num_tasks_per_cpu"]),
            g_ratio_range=(float(mapping["g_ratio_min"]), float(mapping["g_ratio_max"])),
            uniform_cpu_utilization=bool(mapping["uniform_cpu_utilization"]),
            uniform_task_utilization=bool(mapping["uniform_task_utilization"]),
            g_utilization_threshold=float(mapping["g_utilization_threshold"]),
            number_of_inference_segments=int(mapping["number_of_inference_segments"]),
            max_block_count=int(mapping["max_block_count"]),
            per_splitting_overhead=float(mapping["per_splitting_overhead"]),
            max_retries=500,
        )
        paths = generate_tasksets(cfg, output_dir=util_dir)
        for path in paths:
            entries.append((float(utilization), path))
    return entries


def load_initial_masks(path: Path) -> Dict[str, List[int]]:
    return _FIG4.load_initial_masks(path)


def summarize_result(
    utilization: float,
    taskset_path: Path,
    rta_model: str,
    algorithm: str,
    algorithm_label: str,
    result: Any,
    initial_masks: Dict[str, List[int]],
) -> Dict[str, Any]:
    row = _FIG4.summarize_result(
        utilization, taskset_path, rta_model, algorithm, result, initial_masks
    )
    row["algorithm_label"] = algorithm_label
    row["algorithm"] = algorithm_label
    row["algorithm_impl"] = f"{rta_model}:{algorithm}"
    row.update(taskset_diagnostics(taskset_path))
    return row


def taskset_diagnostics(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text())
    g_info = raw.get("_actual_g_ratio") or {}
    period_info = raw.get("_actual_period_ms") or {}
    model_dist = raw.get("_model_distribution")
    if not model_dist:
        model_dist = defaultdict(int)
        for task in raw.get("tasks", []):
            model_dist[str(task.get("model_name", "unknown"))] += 1
        model_dist = dict(model_dist)
    return {
        "actual_g_ratio_min": float(g_info.get("min", 0.0) or 0.0),
        "actual_g_ratio_max": float(g_info.get("max", 0.0) or 0.0),
        "actual_g_ratio_avg": float(g_info.get("avg", 0.0) or 0.0),
        "actual_period_min_ms": float(period_info.get("min", 0.0) or 0.0),
        "actual_period_max_ms": float(period_info.get("max", 0.0) or 0.0),
        "model_distribution": json.dumps(model_dist, sort_keys=True),
    }


def avg(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key, 0.0) or 0.0) for r in rows) / len(rows)


def aggregate(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        util_key = f"{float(row['utilization']):.2f}"
        grouped[(util_key, row["algorithm_label"])].append(row)

    ratio_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    for (util_key, label), items in sorted(grouped.items()):
        n = len(items)
        sched = sum(1 for r in items if r.get("schedulable"))
        split_sets = sum(1 for r in items if r.get("any_split_triggered"))
        ratio_rows.append({
            "utilization": util_key,
            "algorithm": label,
            "total_tasksets": n,
            "schedulable_count": sched,
            "unschedulable_count": n - sched,
            "schedulability_ratio": sched / n if n else 0.0,
            "analysis_error_count": sum(1 for r in items if r.get("analysis_error")),
            "error_count": sum(1 for r in items if r.get("error")),
            "policy_violation_count": sum(1 for r in items if r.get("policy_violation")),
            "disabled_active_boundaries": sum(
                int(r.get("disabled_active_boundaries", 0) or 0) for r in items
            ),
            "avg_gpu_util": avg(items, "gpu_util"),
            "avg_cpu_util": avg(items, "cpu_util"),
            "avg_total_util": avg(items, "total_util"),
            "avg_actual_g_ratio": avg(items, "actual_g_ratio_avg"),
            "min_actual_g_ratio": min(
                (float(r.get("actual_g_ratio_min", 0.0) or 0.0) for r in items),
                default=0.0,
            ),
            "max_actual_g_ratio": max(
                (float(r.get("actual_g_ratio_max", 0.0) or 0.0) for r in items),
                default=0.0,
            ),
            "avg_masks_evaluated": avg(items, "masks_evaluated"),
            "avg_dry_run_evaluations": avg(items, "dry_run_evaluations"),
            "avg_cache_hits": avg(items, "cache_hits"),
            "avg_real_profiles": avg(items, "real_profiles"),
            "avg_skipped_cache_misses": avg(items, "skipped_cache_misses"),
            "avg_optimization_runtime_s": avg(items, "optimization_runtime_s"),
            "split_triggered_tasksets": split_sets,
            "split_triggered_pct": split_sets / n if n else 0.0,
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
            "avg_dry_run_evaluations": avg(items, "dry_run_evaluations"),
            "avg_cache_hits": avg(items, "cache_hits"),
            "avg_real_profiles": avg(items, "real_profiles"),
        })
    return ratio_rows, split_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    return _FIG4.format_table(rows, columns)


def taskset_diag_by_util(tasksets: List[TasksetEntry]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Path]] = defaultdict(list)
    for util, path in tasksets:
        grouped[f"{float(util):.2f}"].append(path)
    rows: List[Dict[str, Any]] = []
    for util_key, paths in sorted(grouped.items()):
        diag_rows = []
        model_counts: Dict[str, int] = defaultdict(int)
        for path in paths:
            raw = json.loads(path.read_text())
            diag = {
                **taskset_diagnostics(path),
                "gpu_util": float(raw.get("_actual_gpu_utilization", 0.0) or 0.0),
                "cpu_util": float(raw.get("_actual_cpu_utilization", 0.0) or 0.0),
                "total_util": float(raw.get("_actual_total_utilization", 0.0) or 0.0),
            }
            diag_rows.append(diag)
            for model, count in json.loads(diag["model_distribution"]).items():
                model_counts[model] += int(count)
        rows.append({
            "utilization": util_key,
            "tasksets": len(paths),
            "avg_gpu_util": avg(diag_rows, "gpu_util"),
            "avg_cpu_util": avg(diag_rows, "cpu_util"),
            "avg_total_util": avg(diag_rows, "total_util"),
            "avg_actual_g_ratio": avg(diag_rows, "actual_g_ratio_avg"),
            "min_actual_g_ratio": min(
                (float(r["actual_g_ratio_min"]) for r in diag_rows), default=0.0
            ),
            "max_actual_g_ratio": max(
                (float(r["actual_g_ratio_max"]) for r in diag_rows), default=0.0
            ),
            "period_min_ms": min(
                (float(r["actual_period_min_ms"]) for r in diag_rows), default=0.0
            ),
            "period_max_ms": max(
                (float(r["actual_period_max_ms"]) for r in diag_rows), default=0.0
            ),
            "model_distribution": json.dumps(dict(sorted(model_counts.items()))),
        })
    return rows


def write_yaml_mapping_report(
    out_dir: Path,
    config_path: Path,
    yaml_data: Dict[str, Any],
    mapping: Dict[str, Any],
    mapping_notes: List[str],
    tasksets: List[TasksetEntry],
) -> None:
    diag_rows = taskset_diag_by_util(tasksets)
    lines = [
        "# YAML Mapping Report",
        "",
        f"- YAML config: `{config_path}`",
        f"- Tasksets generated: {len(tasksets)}",
        "",
        "## Period Range Handling",
        "",
        (
            "The YAML `period_range` is ignored in the default real-DNN mode. "
            "Unlike synthetic DNNSplitting, this framework fixes GPU execution "
            "from TensorRT DNN WCET metadata, samples `G_ratio`, derives "
            "`total_exec = G / G_ratio`, and then derives `T = D = total_exec / U_i`."
        ),
        "",
        "Use `--no-ignore-period-range` only when the YAML period range should be "
        "applied as a validity filter after real-DNN period derivation.",
        "",
        "## Original YAML Values",
        "",
        "```json",
        json.dumps(yaml_data, indent=2, sort_keys=True),
        "```",
        "",
        "## Mapped Generator Values",
        "",
        "```json",
        json.dumps(mapping, indent=2, sort_keys=True),
        "```",
        "",
        "## Mapping Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in mapping_notes)
    lines += [
        "",
        "## Actual Generated Diagnostics",
        "",
        format_table(
            diag_rows,
            [
                "utilization",
                "tasksets",
                "avg_gpu_util",
                "avg_cpu_util",
                "avg_total_util",
                "avg_actual_g_ratio",
                "min_actual_g_ratio",
                "max_actual_g_ratio",
                "period_min_ms",
                "period_max_ms",
                "model_distribution",
            ],
        ),
    ]
    (out_dir / "yaml_mapping_report.md").write_text("\n".join(lines) + "\n")


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    config_path: Path,
    mapping: Dict[str, Any],
    tasksets: List[TasksetEntry],
    ratio_rows: List[Dict[str, Any]],
    split_rows: List[Dict[str, Any]],
    per_rows: List[Dict[str, Any]],
    live_budget: Optional[LiveProfileBudget],
) -> None:
    errors = [r for r in per_rows if r.get("error")]
    any_split = any(r.get("any_split_triggered") for r in per_rows)
    lines = [
        "# YAML Fig.4 Experiment Summary",
        "",
        "## Run Config",
        "",
        f"- YAML config: `{config_path}`",
        f"- Mode: {'live/cache-first' if args.live else 'dry-run'}",
        f"- Models: {', '.join(args.models)}",
        f"- Split policy: {args.split_policy}",
        f"- Algorithms: SS_ours, UNI_ours, SS_Buttazzo, UNI_Buttazzo",
        f"- Tasksets generated: {len(tasksets)}",
        f"- Utilizations: {', '.join(str(u) for u in mapping['utilizations'])}",
        f"- Tasksets per utilization: {mapping['num_tasksets_per_utilization']}",
        f"- CPUs: {mapping['num_cpus']}",
        f"- Tasks per CPU: {mapping['num_tasks_per_cpu']}",
        f"- G-ratio range: [{mapping['g_ratio_min']}, {mapping['g_ratio_max']}]",
        f"- Ignore period_range: {mapping['ignore_period_range']}",
        f"- Max candidates: {args.max_candidates}",
        f"- Max profiles: {args.max_profiles}",
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
        "## Schedulability Ratio",
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
                "policy_violation_count",
                "avg_total_util",
                "avg_actual_g_ratio",
                "avg_masks_evaluated",
                "avg_dry_run_evaluations",
                "avg_real_profiles",
                "avg_cache_hits",
            ],
        ),
        "",
        "## Split Activity",
        "",
        format_table(
            split_rows,
            [
                "utilization",
                "algorithm",
                "tasksets",
                "tasksets_with_any_split",
                "split_triggered_pct",
                "avg_final_active_boundaries",
                "disabled_active_boundaries",
                "policy_violation_count",
            ],
        ),
        "",
        "## Validation",
        "",
        f"- Actual splitting observed: {'yes' if any_split else 'no'}",
        f"- Result rows with errors: {len(errors)}",
        f"- Policy violations: {sum(1 for r in per_rows if r.get('policy_violation'))}",
        "",
        "## Notes",
        "",
        "- Dry-run mode uses existing chunk timing metadata and does not build/profile new TensorRT engines.",
        "- Live mode is cache-first and uses the existing global live profile budget controls.",
        "- `period_range` is not the task-generation driver in the default real-DNN mapping.",
    ]
    if errors:
        lines += ["", "## Errors", ""]
        for row in errors[:20]:
            prefix = f"{row.get('error_type')}: " if row.get("error_type") else ""
            lines.append(
                f"- {row['algorithm_label']} on {row['taskset']}: "
                f"{prefix}{row.get('error_message') or row.get('error')}"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_run_config(
    out_dir: Path,
    args: argparse.Namespace,
    config_path: Path,
    yaml_data: Dict[str, Any],
    mapping: Dict[str, Any],
    tasksets: List[TasksetEntry],
) -> None:
    data = vars(args).copy()
    data["config"] = str(config_path)
    data["dry_run_effective"] = not args.live
    data["yaml_values"] = yaml_data
    data["mapped_values"] = mapping
    data["algorithms"] = [f"{m}:{a}:{label}" for m, a, label in _DEFAULT_ALGORITHMS]
    data["tasksets"] = [
        {"utilization": util, "path": str(path.relative_to(REPO))}
        for util, path in tasksets
    ]
    (out_dir / "run_config.json").write_text(json.dumps(data, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    dry_run = not args.live
    config_path = resolve_config_path(args.config)
    yaml_data = parse_simple_yaml(config_path)
    mapping, mapping_notes = build_mapping(yaml_data, args)
    run_name = make_run_name(args, config_path)
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Run: {run_name}", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Mode: {'live/cache-first' if args.live else 'dry-run'}", flush=True)
    print(f"Models: {args.models}", flush=True)
    print(f"Split policy: {args.split_policy}", flush=True)
    print(f"Utilizations: {mapping['utilizations']}", flush=True)
    print(f"Tasksets per U: {mapping['num_tasksets_per_utilization']}", flush=True)
    print(f"Output: {out_dir.relative_to(REPO)}", flush=True)

    tasksets = generate_yaml_tasksets(args, mapping, out_dir)
    if not tasksets:
        print("[error] no tasksets generated", file=sys.stderr)
        return 1

    write_run_config(out_dir, args, config_path, yaml_data, mapping, tasksets)
    write_yaml_mapping_report(out_dir, config_path, yaml_data, mapping, mapping_notes, tasksets)

    per_rows: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    start = time.time()

    for taskset_idx, (util, taskset_path) in enumerate(tasksets, start=1):
        initial_masks = load_initial_masks(taskset_path)
        print(
            f"\n[{taskset_idx}/{len(tasksets)}] {taskset_path.relative_to(REPO)}",
            flush=True,
        )
        for rta_model, algorithm, label in _DEFAULT_ALGORITHMS:
            result = run_dnn_rta_algorithm(
                dnn_taskset_path=taskset_path,
                model=rta_model,
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
            row = summarize_result(
                util, taskset_path, rta_model, algorithm, label, result, initial_masks
            )
            per_rows.append(row)
            all_results.append({
                **{k: v for k, v in row.items() if k != "task_details"},
                "stats": result.stats.to_dict(),
                "task_details": row["task_details"],
            })
            sched = "SCHED" if result.schedulable else "MISS"
            split = "split" if row["any_split_triggered"] else "no-split"
            error = " ERROR" if result.error else ""
            print(
                f"  {label:14s} {sched:5s} {split:8s} "
                f"masks={result.stats.masks_evaluated:4d} "
                f"dry={result.stats.dry_run_evaluations:4d} "
                f"cache={result.stats.cache_hits:3d} "
                f"real={result.stats.real_profiles:3d}{error}",
                flush=True,
            )
            if live_budget is not None and live_budget.stopped:
                print(
                    f"[live_budget] stopped on first build: "
                    f"{live_budget.stop_model}/{live_budget.stop_variant}",
                    flush=True,
                )
                break
        if live_budget is not None and live_budget.stopped:
            break

    ratio_rows, split_rows = aggregate(per_rows)

    per_fields = [
        "utilization", "taskset", "taskset_path", "algorithm", "algorithm_label",
        "algorithm_impl", "rta_model", "schedulable", "analysis_error", "error_type",
        "error_message", "overload_reason", "duration_s", "optimization_runtime_s",
        "masks_evaluated", "dry_run_evaluations", "real_profiles", "cache_hits",
        "skipped_cache_misses", "gpu_util", "cpu_util", "total_util",
        "max_cpu_partition_util", "actual_g_ratio_min", "actual_g_ratio_max",
        "actual_g_ratio_avg", "actual_period_min_ms", "actual_period_max_ms",
        "model_distribution", "split_triggered", "split_task_count",
        "single_schedulable", "split_required", "split_proactive",
        "early_stopped_no_split", "final_total_active_boundaries",
        "disabled_active_boundaries", "policy_violation", "average_final_chunks",
        "task_chunk_counts", "task_masks",
    ]
    ratio_fields = [
        "utilization", "algorithm", "total_tasksets", "schedulable_count",
        "unschedulable_count", "schedulability_ratio", "analysis_error_count",
        "error_count", "policy_violation_count", "disabled_active_boundaries",
        "avg_gpu_util", "avg_cpu_util", "avg_total_util", "avg_actual_g_ratio",
        "min_actual_g_ratio", "max_actual_g_ratio", "avg_masks_evaluated",
        "avg_dry_run_evaluations", "avg_cache_hits", "avg_real_profiles",
        "avg_skipped_cache_misses", "avg_optimization_runtime_s",
        "split_triggered_tasksets", "split_triggered_pct",
    ]
    split_fields = [
        "utilization", "algorithm", "tasksets", "tasksets_with_any_split",
        "split_triggered_pct", "avg_split_tasks_per_taskset",
        "avg_final_active_boundaries", "avg_final_chunks",
        "disabled_active_boundaries", "policy_violation_count",
        "avg_masks_evaluated", "avg_dry_run_evaluations", "avg_cache_hits",
        "avg_real_profiles",
    ]

    write_csv(out_dir / "per_taskset_results.csv", per_rows, per_fields)
    write_csv(out_dir / "schedulability_ratio.csv", ratio_rows, ratio_fields)
    write_csv(out_dir / "split_activity.csv", split_rows, split_fields)
    (out_dir / "all_results.json").write_text(json.dumps(all_results, indent=2))
    write_summary(
        out_dir, args, config_path, mapping, tasksets, ratio_rows, split_rows,
        per_rows, live_budget,
    )

    elapsed = time.time() - start
    print(f"\nSaved: {(out_dir / 'schedulability_ratio.csv').relative_to(REPO)}")
    print(f"Saved: {(out_dir / 'yaml_mapping_report.md').relative_to(REPO)}")
    print(f"Saved: {(out_dir / 'summary.md').relative_to(REPO)}")
    print(f"Elapsed algorithm time: {elapsed:.2f}s")
    print(f"Result rows with errors: {sum(1 for r in per_rows if r.get('error'))}")
    print(f"Policy violations: {sum(1 for r in per_rows if r.get('policy_violation'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
