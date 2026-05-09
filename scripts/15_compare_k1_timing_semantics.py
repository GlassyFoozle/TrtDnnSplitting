#!/usr/bin/env python3
"""
15_compare_k1_timing_semantics.py — Verify K=1 timing consistency between
task generation and RTA analysis.

Prints, for each model:
  - Whether profiling data is available (results/table4/)
  - The G value used by task generation (from profiling cache, or fallback)
  - The G value seen by RTA analysis (sum of per-chunk p99 from candidate_space)
  - Source of each G value
  - Whether generation G == analysis G (consistent) or not

Consistent means both sources use the same profiled data. After running
scripts/21_profile_base_chunks.py, both should reflect real Jetson measurements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODELS = ["alexnet", "resnet18", "vgg19"]
PRECISION = "fp32"

_FALLBACK_WCET_MS = {"alexnet": 1.754, "resnet18": 1.037, "vgg19": 7.562}


def _gen_g_source(model: str, precision: str) -> str:
    """Return a human-readable string describing where _get_base_gpu_wcet_ms got its value."""
    from src.optimization.profiling_db import ProfilingDB

    cache_path = REPO / "results" / "optimization" / ".profiling_cache.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            key = f"{model.lower()}|dag_aligned_full|{precision}"
            entry = data.get("entries", {}).get(key)
            if entry:
                p99 = entry.get("per_chunk_gpu_p99_ms")
                if p99:
                    return "profiling cache (p99)"
                mean = entry.get("per_chunk_gpu_mean_ms")
                if mean:
                    return "profiling cache (mean)"
        except Exception:
            pass

    eval_dir = REPO / "results" / "evaluations" / model.lower()
    if eval_dir.exists():
        for f in sorted(eval_dir.glob("*.json")):
            try:
                d = json.loads(f.read_text())
                if d.get("mask") and sum(d["mask"]) == 0:
                    return f"evaluation JSON ({f.name})"
            except Exception:
                continue

    return "hardcoded fallback (_DRY_RUN_BASE_WCET_MS)"


def main() -> int:
    from src.optimization.candidate_space import load_candidate_space
    from src.integration.dnn_workload_generator import _get_base_gpu_wcet_ms

    print("=" * 76)
    print("K=1 timing semantics: task generation vs. RTA analysis")
    print("=" * 76)
    print()
    print(f"Repo: {REPO}")
    print()

    has_table4 = {}
    for model in MODELS:
        p = REPO / "results" / "table4" / f"{model}_cpp_dag_aligned_full_{PRECISION}.json"
        has_table4[model] = p.exists()

    header = (f"{'Model':<12}  {'Table4':6}  {'Gen G (ms)':>12}  "
              f"{'RTA G (ms)':>12}  {'N':>4}  {'Consistent':>10}")
    print(header)
    print("-" * 64)

    all_consistent = True
    rows = []
    for model in MODELS:
        cs = load_candidate_space(model, PRECISION)
        gen_g = _get_base_gpu_wcet_ms(model, PRECISION, "p99")
        rta_g = sum(cs.chunk_gpu_p99_ms)
        n = cs.candidate_count
        consistent = gen_g is not None and abs(gen_g - rta_g) < 0.001
        if not consistent:
            all_consistent = False
        table4 = "YES" if has_table4[model] else "no"
        status = "OK" if consistent else "MISMATCH"
        gen_source = _gen_g_source(model, PRECISION)
        print(
            f"{model:<12}  {table4:<6}  {gen_g or 0.0:>12.4f}  {rta_g:>12.4f}  "
            f"{n:>4}  {status:>10}"
        )
        rows.append((model, gen_g, rta_g, gen_source, consistent))

    print()

    # Per-model source detail
    for model, gen_g, rta_g, gen_source, consistent in rows:
        print(f"  {model}:  Gen G source = {gen_source}")
        if not consistent:
            print(f"    WARNING: Gen G={gen_g:.4f}ms  RTA G={rta_g:.4f}ms  diff={abs((gen_g or 0)-rta_g):.4f}ms")

    print()

    if all_consistent:
        print("RESULT: Generation and analysis G are consistent.")
        print("        Schedulability experiments will use real profiled WCET values.")
    else:
        print("RESULT: INCONSISTENCY DETECTED.")
        print()
        # Diagnose the specific failure mode
        cache_path = REPO / "results" / "optimization" / ".profiling_cache.json"
        table4_missing = [m for m in MODELS if not has_table4[m]]
        cache_missing = not cache_path.exists()

        if table4_missing:
            print(f"CAUSE:  results/table4/ JSON missing for: {table4_missing}")
            print(f"FIX:    Run: conda run -n trt python scripts/21_profile_base_chunks.py")
            print(f"              --models {' '.join(table4_missing)} --precision {PRECISION}")
        elif cache_missing:
            print("CAUSE:  Profiling cache (.profiling_cache.json) does not exist.")
            print("        Task generator is using hardcoded fallback WCET values.")
            print("FIX:    Re-run scripts/21_profile_base_chunks.py to rebuild the cache.")
        else:
            print("CAUSE:  Profiling cache exists but task generator is not reading real values.")
            print("        Check that results/optimization/.profiling_cache.json is non-empty")
            print("        and has entries for the affected models.")
            print()
            print("FIX:    Re-run scripts/21_profile_base_chunks.py (or manually re-import")
            print("        results/table4/ JSONs into the profiling cache).")

    print()
    print("Per-model chunk timing detail (RTA analysis view):")
    print()
    for model in MODELS:
        cs = load_candidate_space(model, PRECISION)
        total_mean = sum(cs.chunk_gpu_mean_ms)
        total_p99 = sum(cs.chunk_gpu_p99_ms)
        print(f"  {model}  N={cs.candidate_count}  has_timing={cs.has_timing}")
        print(f"    sum_mean={total_mean:.4f}ms  sum_p99={total_p99:.4f}ms")
        print(f"    per-chunk p99: [{', '.join(f'{t:.4f}' for t in cs.chunk_gpu_p99_ms[:5])}"
              f"{'...' if cs.candidate_count > 5 else ''}]")
        print()

    return 0 if all_consistent else 1


if __name__ == "__main__":
    sys.exit(main())
