#!/usr/bin/env python3
"""
15_compare_k1_timing_semantics.py — Inspect K=1 timing used by workload
generation.

Prints, for each model:
  - Whether profiling data is available (results/table4/)
  - The K=1 G value used by task generation
  - The dag_aligned_full chunk-sum value used as split candidate metadata
  - Source of each G value

Generation must use measured all-zero K=1 timing when available. The
dag_aligned_full sum is reported only to show how different fully split metadata
can be from actual no-split execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODELS = ["alexnet", "resnet18", "vgg19"]
PRECISION = "fp32"
WCET_METRIC = "max"

def _gen_g_source(model: str, precision: str) -> str:
    """Return a human-readable string describing where _get_base_gpu_wcet_ms got its value."""
    from src.optimization.config_evaluator import mask_to_variant_name
    from src.integration.dnn_workload_generator import _get_base_chunk_count

    n_base = _get_base_chunk_count(model)
    if n_base:
        mask = [0] * (n_base - 1)
        variant = mask_to_variant_name(model, mask)
        eval_json = REPO / "results" / "evaluations" / model / f"{variant}_{precision}.json"
        if eval_json.exists():
            return f"measured K=1 evaluation JSON ({eval_json.name})"

    return "dry-run reference fallback (_DRY_RUN_BASE_WCET_MS)"


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

    header = (f"{'Model':<12}  {'Table4':6}  {'Gen K1 G':>12}  "
              f"{'DagFull sum':>12}  {'N':>4}  {'Delta':>12}")
    print(header)
    print("-" * 68)

    all_has_gen = True
    rows = []
    for model in MODELS:
        cs = load_candidate_space(model, PRECISION)
        gen_g = _get_base_gpu_wcet_ms(model, PRECISION, WCET_METRIC)
        dag_sum = sum(cs.chunk_gpu_max_ms)
        n = cs.candidate_count
        if gen_g is None:
            all_has_gen = False
        table4 = "YES" if has_table4[model] else "no"
        delta = (dag_sum - gen_g) if gen_g is not None else 0.0
        gen_source = _gen_g_source(model, PRECISION)
        print(
            f"{model:<12}  {table4:<6}  {gen_g or 0.0:>12.4f}  {dag_sum:>12.4f}  "
            f"{n:>4}  {delta:>12.4f}"
        )
        rows.append((model, gen_g, dag_sum, gen_source))

    print()

    # Per-model source detail
    for model, gen_g, dag_sum, gen_source in rows:
        print(f"  {model}:  Gen G source = {gen_source}")
        if gen_g is not None and abs(gen_g - dag_sum) > 0.001:
            print(
                f"    note: K1={gen_g:.4f}ms differs from dag_aligned_full sum="
                f"{dag_sum:.4f}ms by {abs(gen_g - dag_sum):.4f}ms"
            )

    print()

    if all_has_gen:
        print("RESULT: Workload generation has a K=1 timing source for every model.")
    else:
        print("RESULT: K=1 TIMING MISSING.")
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
        total_max = sum(cs.chunk_gpu_max_ms)
        print(f"  {model}  N={cs.candidate_count}  has_timing={cs.has_timing}")
        print(f"    sum_mean={total_mean:.4f}ms  sum_max={total_max:.4f}ms")
        print(f"    per-chunk max: [{', '.join(f'{t:.4f}' for t in cs.chunk_gpu_max_ms[:5])}"
              f"{'...' if cs.candidate_count > 5 else ''}]")
        print()

    return 0 if all_has_gen else 1


if __name__ == "__main__":
    sys.exit(main())
