"""
53_run_dnn_split_algorithm.py — Run a DNN-aware splitting algorithm on a taskset.

Usage:
  conda run -n trt python scripts/53_run_dnn_split_algorithm.py \\
      --taskset configs/dnn_tasksets/mixed_two_dnn_demo.json \\
      --model ss \\
      --algorithm tol-fb \\
      --precision fp32 \\
      --wcet-metric p99 \\
      --use-cpp

Examples:
  # No-split baseline
  python scripts/53_run_dnn_split_algorithm.py \\
      --taskset configs/dnn_tasksets/mixed_two_dnn_demo.json --model ss --algorithm single

  # Max-split baseline (dry-run = no actual engine builds)
  python scripts/53_run_dnn_split_algorithm.py \\
      --taskset configs/dnn_tasksets/mixed_two_dnn_demo.json --model ss --algorithm max --dry-run

  # Tolerance-fit splitting
  python scripts/53_run_dnn_split_algorithm.py \\
      --taskset configs/dnn_tasksets/mixed_two_dnn_demo.json --model ss --algorithm tol --dry-run

  # Tolerance + fallback (the key algorithm)
  python scripts/53_run_dnn_split_algorithm.py \\
      --taskset configs/dnn_tasksets/mixed_two_dnn_demo.json --model ss --algorithm tol-fb --dry-run

  # UNI model
  python scripts/53_run_dnn_split_algorithm.py \\
      --taskset configs/dnn_tasksets/mixed_two_dnn_demo.json --model uni --algorithm tol-fb --dry-run

Output:
  results/dnn_algorithms/<run_name>/
    algorithm_result.json
    algorithm_report.md
    per_task_final_masks.csv
    profile_events.csv
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.integration import run_dnn_rta_algorithm


def main():
    ap = argparse.ArgumentParser(
        description="Run DNN-aware splitting algorithm on a taskset"
    )
    ap.add_argument("--taskset", required=True,
                    help="Path to DNN taskset JSON (absolute or relative to configs/dnn_tasksets/)")
    ap.add_argument("--model", default="ss", choices=["ss", "uni"],
                    help="Task model: ss or uni (default: ss)")
    ap.add_argument("--algorithm", default="tol-fb",
                    choices=["single", "max", "tol", "tol-fb",
                             "heu", "heu-k", "opt", "opt-k"],
                    help="Algorithm (default: tol-fb). "
                         "heu/opt = paper-style; heu-k/opt-k = K-balanced approximation")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--wcet-metric", default="p99", choices=["p99", "mean"],
                    dest="wcet_metric",
                    help="Per-chunk timing column to use for RTA (default: p99)")
    ap.add_argument("--use-cpp", action="store_true", default=True,
                    help="Use C++ table4_runner for profiling (default: True)")
    ap.add_argument("--no-cpp", dest="use_cpp", action="store_false",
                    help="Use Python TRT fallback profiler")
    ap.add_argument("--force-profile", action="store_true", default=False,
                    help="Re-profile even if cache hit exists")
    ap.add_argument("--dry-run", action="store_true", default=False,
                    help="Skip TRT engine builds; use estimated timing from base chunks")
    ap.add_argument("--max-iterations", type=int, default=1000,
                    help="Max algorithm iterations before giving up (default: 1000)")
    ap.add_argument("--exact-opt-max-boundaries", type=int, default=0,
                    help="For opt-k: if boundary_count <= N, enumerate all 2^N masks exactly (default: 0 = off)")
    ap.add_argument("--split-policy", default="all",
                    choices=["all", "paper_like", "stage", "five_points", "ten_points", "major_blocks"],
                    dest="split_policy",
                    help="Split-point policy for paper-style heu/opt (default: all)")
    ap.add_argument("--max-profiles", type=int, default=500,
                    dest="max_profiles",
                    help="Budget for paper-style OPT/HEU per task (default: 500)")
    ap.add_argument("--max-candidates", type=int, default=10000,
                    dest="max_candidates",
                    help="Max BFS candidates for paper-style OPT (default: 10000)")
    ap.add_argument("--allow-proactive-splitting", action="store_true", default=False,
                    help="For paper-style OPT/HEU, keep searching even if no-split is schedulable.")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--output-dir", default=None,
                    help="Override output directory (default: results/dnn_algorithms/<run_name>/)")
    args = ap.parse_args()

    taskset_path = args.taskset

    run_name = (
        f"{Path(taskset_path).stem}"
        f"_{args.model.upper()}"
        f"_{args.algorithm}"
        f"_{args.split_policy}"
        f"_{'dry' if args.dry_run else 'live'}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = REPO / "results" / "dnn_algorithms" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Taskset: {taskset_path}")
    print(f"Model: {args.model.upper()}  Algorithm: {args.algorithm}  "
          f"Precision: {args.precision}  Metric: {args.wcet_metric}")
    print(f"dry_run: {args.dry_run}  force_profile: {args.force_profile}")
    print(f"Output: {out_dir}")
    print()

    result = run_dnn_rta_algorithm(
        dnn_taskset_path=taskset_path,
        model=args.model,
        algorithm=args.algorithm,
        precision=args.precision,
        wcet_metric=args.wcet_metric,
        use_cpp=args.use_cpp,
        force_profile=args.force_profile,
        dry_run=args.dry_run,
        max_iterations=args.max_iterations,
        exact_opt_max_boundaries=args.exact_opt_max_boundaries,
        warmup=args.warmup,
        iters=args.iters,
        policy_name=args.split_policy,
        max_profiles=args.max_profiles,
        max_candidates=args.max_candidates,
        allow_proactive_splitting=args.allow_proactive_splitting,
    )

    print(result.summary())

    if result.error:
        print(f"\n[Error] {result.error}", file=sys.stderr)

    # Save algorithm_result.json
    result_json = {
        "run_name": run_name,
        "rta_model": result.rta_model,
        "algorithm": result.algorithm,
        "policy_name": result.policy_name,
        "taskset_path": result.taskset_path,
        "precision": result.precision,
        "wcet_metric": result.wcet_metric,
        "dry_run": result.dry_run,
        "schedulable": result.schedulable,
        "error_type": getattr(result, "error_type", None),
        "error": result.error,
        "unschedulable_reason": getattr(result, "unschedulable_reason", None),
        "diagnostic_message": getattr(result, "diagnostic_message", None),
        "analysis_error": getattr(result, "analysis_error", False),
        "single_schedulable": getattr(result, "single_schedulable", None),
        "early_stopped_no_split": getattr(result, "early_stopped_no_split", False),
        "duration_s": result.duration_s,
        "algorithm_iterations": result.algorithm_iterations,
        "stats": result.stats.to_dict(),
        "task_results": [
            {
                "task_name": tr.task_name,
                "model_name": tr.model_name,
                "cpu_id": tr.cpu_id,
                "period_ms": tr.period_ms,
                "deadline_ms": tr.deadline_ms,
                "C_ms": tr.C_ms,
                "G_ms": tr.G_ms,
                "R_ms": tr.R_ms,
                "slack_ms": tr.slack_ms,
                "schedulable": tr.schedulable,
                "m": tr.m,
                "max_G_block": tr.max_G_block,
                "B_high": tr.B_high,
                "B_low": tr.B_low,
                "I": tr.I,
                "final_k_chunks": tr.final_k_chunks,
                "final_chunk_times_ms": tr.final_chunk_times_ms,
                "final_mask": tr.final_mask,
                "variant_name": tr.variant_name,
                "profile_result_path": tr.profile_result_path,
            }
            for tr in result.task_results
        ],
    }
    json_path = out_dir / "algorithm_result.json"
    json_path.write_text(json.dumps(result_json, indent=2))
    print(f"\nSaved: {json_path.relative_to(REPO)}")

    # Save per_task_final_masks.csv
    masks_csv = out_dir / "per_task_final_masks.csv"
    with masks_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_name", "model_name", "final_k_chunks",
                    "max_G_block_ms", "schedulable", "variant_name", "final_mask"])
        for tr in result.task_results:
            w.writerow([
                tr.task_name, tr.model_name, tr.final_k_chunks,
                f"{tr.max_G_block:.6f}", tr.schedulable,
                tr.variant_name,
                "".join(str(b) for b in tr.final_mask),
            ])
    print(f"Saved: {masks_csv.relative_to(REPO)}")

    # Save algorithm_report.md
    _write_report(out_dir, run_name, result, args)
    print(f"Saved: {(out_dir / 'algorithm_report.md').relative_to(REPO)}")

    verdict = "SCHEDULABLE" if result.schedulable else "NOT SCHEDULABLE"
    print(f"\n→ {verdict}")

    return 0 if result.schedulable else 1


def _write_report(out_dir: Path, run_name: str, result, args):
    lines = [
        f"# DNN Algorithm Run Report",
        f"",
        f"**Run**: `{run_name}`  ",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Model**: {result.rta_model}-{result.algorithm}  ",
        f"**Taskset**: `{result.taskset_path}`  ",
        f"**Precision**: {result.precision}  **WCET metric**: {result.wcet_metric}  "
        f"**dry_run**: {result.dry_run}  ",
        f"",
        f"## Result",
        f"",
        f"**{'SCHEDULABLE' if result.schedulable else 'NOT SCHEDULABLE'}**",
        f"",
        f"- Duration: {result.duration_s:.2f}s",
        f"- Algorithm iterations: {result.algorithm_iterations}",
        f"- Unschedulable reason: {getattr(result, 'unschedulable_reason', None) or ''}",
        f"- Diagnostic: {getattr(result, 'diagnostic_message', None) or ''}",
        f"- Analysis error: {getattr(result, 'analysis_error', False)}",
        f"",
        f"## Profiling Statistics",
        f"",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| masks_evaluated | {result.stats.masks_evaluated} |",
        f"| cache_hits | {result.stats.cache_hits} |",
        f"| real_profiles | {result.stats.real_profiles} |",
        f"| builds_triggered | {result.stats.builds_triggered} |",
        f"| dry_run_evaluations | {result.stats.dry_run_evaluations} |",
        f"",
        f"## Per-Task Results",
        f"",
        f"| Task | R (ms) | D (ms) | Slack (ms) | C (ms) | G (ms) | K | MaxBlk | B_hi | I | Sched? |",
        f"|------|--------|--------|------------|--------|--------|---|--------|------|---|--------|",
    ]
    for tr in result.task_results:
        flag = "✓" if tr.schedulable else "✗"
        lines.append(
            f"| {tr.task_name} | {tr.R_ms:.4f} | {tr.deadline_ms:.4f} | "
            f"{tr.slack_ms:.4f} | {tr.C_ms:.4f} | {tr.G_ms:.4f} | "
            f"{tr.final_k_chunks} | {tr.max_G_block:.4f} | "
            f"{tr.B_high:.4f} | {tr.I:.4f} | {flag} |"
        )
    lines += [
        f"",
        f"## Final Masks",
        f"",
    ]
    for tr in result.task_results:
        lines.append(f"**{tr.task_name}** (K={tr.final_k_chunks}):  ")
        lines.append(f"  mask: `{''.join(str(b) for b in tr.final_mask)}`  ")
        if tr.final_chunk_times_ms:
            times_str = ", ".join(f"{t:.4f}" for t in tr.final_chunk_times_ms)
            lines.append(f"  chunk_times: [{times_str}] ms  ")
        if tr.variant_name:
            lines.append(f"  variant: `{tr.variant_name}`  ")
        lines.append("")

    if result.error:
        lines += [f"## Error", f"", f"```", f"{result.error}", f"```", f""]

    (out_dir / "algorithm_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
