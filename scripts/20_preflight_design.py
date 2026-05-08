#!/usr/bin/env python3
"""
61_preflight_live_design.py — Generate/inspect one live design-phase taskset.

This script does not build engines or profile TensorRT. It prints the taskset
that should be passed to script 56, the model distribution, split-policy enabled
boundary counts, and a conservative candidate-count estimate for the requested
algorithms.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.integration.dnn_workload_generator import WorkloadConfig, generate_tasksets
from src.integration.split_point_policy import get_enabled_boundaries
from src.optimization.candidate_space import load_candidate_space


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Preflight one uncapped live design-phase experiment without profiling"
    )
    ap.add_argument("--taskset", default=None, help="Existing taskset JSON to inspect")
    ap.add_argument("--models", nargs="+", default=["alexnet", "resnet18"])
    ap.add_argument("--num-tasks", type=int, default=8)
    ap.add_argument("--utilization", type=float, default=0.6)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--wcet-metric", default="p99", choices=["p99", "mean"], dest="wcet_metric")
    ap.add_argument("--num-cpus", type=int, default=1)
    ap.add_argument("--num-tasks-per-cpu", type=int, default=None)
    ap.add_argument("--period-min-ms", type=float, default=1.0)
    ap.add_argument("--period-max-ms", type=float, default=10000.0)
    ap.add_argument("--utilization-basis", default="total", choices=["gpu", "total"])
    ap.add_argument("--taskgen-mode", default="dnnsplitting", choices=["legacy", "dnnsplitting"])
    ap.add_argument("--g-ratio-min", type=float, default=0.6)
    ap.add_argument("--g-ratio-max", type=float, default=1.0)
    ap.add_argument("--g-utilization-threshold", type=float, default=1.0)
    ap.add_argument("--number-of-inference-segments", type=int, default=1)
    ap.add_argument("--max-block-count", type=int, default=None)
    ap.add_argument("--per-splitting-overhead", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=200)
    ap.add_argument("--split-policy", default="five_points")
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=["uni:opt", "uni:heu", "ss:tol-fb"],
        help="Algorithms to estimate, e.g. uni:opt uni:heu ss:tol-fb",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--output-dir",
        default=str(REPO / "results" / "dnn_experiments"),
        help="Base output directory when generating a taskset",
    )
    return ap.parse_args()


def parse_algorithms(values: Iterable[str]) -> List[Tuple[str, str]]:
    out = []
    for value in values:
        if ":" in value:
            model, alg = value.split(":", 1)
        else:
            model, alg = "ss", value
        out.append((model.lower(), alg.lower()))
    return out


def generate_or_load_taskset(args: argparse.Namespace) -> Path:
    if args.taskset:
        path = Path(args.taskset)
        return path if path.is_absolute() else REPO / path

    run_name = args.run_name or f"live_design_preflight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir) / run_name / "generated_tasksets"
    cfg = WorkloadConfig(
        models=args.models,
        n_tasks=args.num_tasks,
        utilization=args.utilization,
        n_tasksets=1,
        precision=args.precision,
        wcet_metric=args.wcet_metric,
        period_min_ms=args.period_min_ms,
        period_max_ms=args.period_max_ms,
        cpu_id=0,
        seed=args.seed,
        utilization_basis=args.utilization_basis,
        taskgen_mode=args.taskgen_mode,
        num_cpus=args.num_cpus,
        tasks_per_cpu=args.num_tasks_per_cpu,
        g_ratio_range=(args.g_ratio_min, args.g_ratio_max),
        g_utilization_threshold=args.g_utilization_threshold,
        number_of_inference_segments=args.number_of_inference_segments,
        max_block_count=args.max_block_count,
        per_splitting_overhead=args.per_splitting_overhead,
        max_retries=args.max_retries,
    )
    paths = generate_tasksets(cfg, output_dir=out_dir)
    if not paths:
        raise RuntimeError(
            "No taskset generated. Try widening --period-min-ms/--period-max-ms "
            "or --g-ratio-min/--g-ratio-max."
        )
    return paths[0]


def policy_counts(models: Iterable[str], policy: str, precision: str) -> Dict[str, dict]:
    out = {}
    for model in sorted(set(m.lower() for m in models)):
        cs = load_candidate_space(model, precision)
        enabled = get_enabled_boundaries(model, policy, cs.boundary_count)
        out[model] = {
            "chunks": cs.candidate_count,
            "boundaries": cs.boundary_count,
            "enabled": enabled,
            "enabled_count": len(enabled),
            "subset_universe": 2 ** len(enabled),
        }
    return out


def estimate_search(raw: dict, counts: Dict[str, dict], algorithms: List[Tuple[str, str]]) -> List[dict]:
    rows = []
    task_models = [str(t["model_name"]).lower() for t in raw.get("tasks", [])]
    for rta_model, algorithm in algorithms:
        total = 0
        notes = ""
        for model in task_models:
            m = counts[model]["enabled_count"]
            if algorithm == "opt":
                total += 2 ** m
                notes = "BFS subset universe over enabled boundaries"
            elif algorithm == "heu":
                total += 1 + (m * (m + 1) // 2)
                notes = "Worst-case greedy one-additional-split probes"
            elif algorithm in ("tol", "tol-fb", "heu-k", "opt-k"):
                total += m + 1
                notes = "Approximate single K ladder; fallback/iterations can repeat"
            else:
                total += 1
                notes = "No split search or unknown algorithm"
        rows.append({
            "algorithm": f"{rta_model}:{algorithm}",
            "estimated_candidate_evaluations": total,
            "notes": notes,
        })
    return rows


def fmt_table(rows: List[dict], columns: List[str]) -> str:
    if not rows:
        return ""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    lines = ["  " + "  ".join(c.ljust(widths[c]) for c in columns)]
    lines.append("  " + "  ".join("-" * widths[c] for c in columns))
    for row in rows:
        lines.append("  " + "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    taskset_path = generate_or_load_taskset(args)
    raw = json.loads(taskset_path.read_text())
    algorithms = parse_algorithms(args.algorithms)

    task_models = [str(t["model_name"]).lower() for t in raw.get("tasks", [])]
    counts = policy_counts(task_models or args.models, args.split_policy, args.precision)
    estimates = estimate_search(raw, counts, algorithms)

    print("Live design-phase preflight")
    print(f"Taskset: {taskset_path}")
    print(f"Tasks: {len(raw.get('tasks', []))}")
    print(f"Precision/metric: {raw.get('precision', args.precision)}/{raw.get('wcet_metric', args.wcet_metric)}")
    print(f"Taskgen mode: {raw.get('taskgen_mode', args.taskgen_mode)}")
    print(f"Utilization basis: {raw.get('utilization_basis', args.utilization_basis)}")
    print(f"Requested utilization: {raw.get('_utilization', args.utilization)}")
    print(f"Actual total utilization: {raw.get('_actual_total_utilization', 'unknown')}")
    print(f"Actual GPU utilization: {raw.get('_actual_gpu_utilization', 'unknown')}")
    print(f"Actual CPU utilization: {raw.get('_actual_cpu_utilization', 'unknown')}")
    print(f"CPU partition utilization: {raw.get('_actual_cpu_partition_utilization', {})}")
    print(f"Model distribution: {dict(Counter(task_models))}")

    policy_rows = []
    for model, info in counts.items():
        policy_rows.append({
            "model": model,
            "base_chunks": info["chunks"],
            "boundaries": info["boundaries"],
            "enabled_count": info["enabled_count"],
            "enabled_indices": info["enabled"],
            "subset_universe": info["subset_universe"],
        })
    print("\nSplit policy:")
    print(f"  policy: {args.split_policy}")
    print(fmt_table(policy_rows, ["model", "base_chunks", "boundaries", "enabled_count", "enabled_indices", "subset_universe"]))

    print("\nEstimated search size for this taskset:")
    print(fmt_table(estimates, ["algorithm", "estimated_candidate_evaluations", "notes"]))

    print("\nUse this taskset path in the live commands:")
    try:
        print(f"  {taskset_path.relative_to(REPO)}")
    except ValueError:
        print(f"  {taskset_path}")

    report_path = taskset_path.parent / "preflight_summary.json"
    report_path.write_text(json.dumps({
        "taskset_path": str(taskset_path),
        "taskset": raw,
        "policy": args.split_policy,
        "policy_counts": counts,
        "algorithm_estimates": estimates,
    }, indent=2))
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
