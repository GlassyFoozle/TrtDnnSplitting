#!/usr/bin/env python3
"""
15_compare_k1_timing_semantics.py — Diagnose K=1 timing consistency between
task generation and RTA analysis in dry-run mode.

Prints, for each model:
  - Whether profiling data is available (results/table4/)
  - The G value used by task generation (_DRY_RUN_BASE_WCET_MS fallback or cache)
  - The G value seen by RTA analysis after K=1 patching (sum of base_chunk_times_ms)
  - Whether generation G == analysis G (consistent) or not (inconsistent)

An inconsistency (analysis G=0 while generation G>0) causes all tasksets to
appear trivially schedulable with K=1 in dry-run mode, so no splitting is
ever triggered. This script is for diagnosis without running TensorRT.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODELS = ["alexnet", "resnet18", "vgg19"]
PRECISION = "fp32"

_DRY_RUN_BASE_WCET_MS = {"alexnet": 1.754, "resnet18": 1.037, "vgg19": 7.562}


def main() -> int:
    from src.optimization.candidate_space import load_candidate_space
    from src.integration.dnn_workload_generator import _get_base_gpu_wcet_ms

    print("=" * 72)
    print("K=1 timing semantics: task generation vs. RTA analysis")
    print("=" * 72)
    print()
    print(f"Repo: {REPO}")
    print()

    has_table4 = {}
    for model in MODELS:
        p = REPO / "results" / "table4" / f"{model}_cpp_dag_aligned_full_{PRECISION}.json"
        has_table4[model] = p.exists()

    header = f"{'Model':<12}  {'Table4':6}  {'Gen G (ms)':>12}  {'RTA G (ms)':>12}  "
    header += f"{'N':>4}  {'G/N (ms)':>10}  {'Consistent':>10}"
    print(header)
    print("-" * 80)

    all_consistent = True
    for model in MODELS:
        cs = load_candidate_space(model, PRECISION)
        gen_g = _get_base_gpu_wcet_ms(model, PRECISION, "p99")
        rta_g = sum(cs.chunk_gpu_p99_ms)
        n = cs.candidate_count
        g_per_chunk = rta_g / n if n > 0 else 0.0
        consistent = abs(gen_g - rta_g) < 0.001 if gen_g is not None else False
        if not consistent:
            all_consistent = False
        table4 = "YES" if has_table4[model] else "no"
        status = "OK" if consistent else "MISMATCH"
        print(
            f"{model:<12}  {table4:<6}  {gen_g or 0.0:>12.4f}  {rta_g:>12.4f}  "
            f"{n:>4}  {g_per_chunk:>10.6f}  {status:>10}"
        )

    print()
    if all_consistent:
        print("RESULT: Generation and analysis G are consistent.")
        print("        Dry-run schedulability experiment will be meaningful.")
    else:
        print("RESULT: INCONSISTENCY DETECTED.")
        print("        Generation uses non-zero G for periods, but analysis sees G=0.")
        print("        This causes all tasksets to appear trivially schedulable at K=1.")
        print()
        print("CAUSE:  results/table4/<model>_cpp_dag_aligned_full_fp32.json is missing.")
        print("        candidate_space.py fell back to all-zero chunk times before")
        print("        the fix added in PR 'Diagnose Fig4 K1 timing semantics'.")
        print()
        print("FIX:    The fix in candidate_space.py adds Priority 3: when timing is")
        print("        all-zero, distribute _DRY_RUN_BASE_WCET_MS equally across N chunks.")
        print("        After the fix, Gen G == RTA G and experiments are meaningful.")

    print()
    print("Per-model chunk timing detail (RTA analysis view after candidate_space load):")
    print()
    for model in MODELS:
        cs = load_candidate_space(model, PRECISION)
        total_mean = sum(cs.chunk_gpu_mean_ms)
        total_p99 = sum(cs.chunk_gpu_p99_ms)
        print(f"  {model}  N={cs.candidate_count}  has_timing={cs.has_timing}")
        print(f"    sum_mean={total_mean:.4f}ms  sum_p99={total_p99:.4f}ms")
        print(f"    per-chunk mean: [{', '.join(f'{t:.4f}' for t in cs.chunk_gpu_mean_ms[:5])}{'...' if cs.candidate_count > 5 else ''}]")
        print()

    return 0 if all_consistent else 1


if __name__ == "__main__":
    sys.exit(main())
