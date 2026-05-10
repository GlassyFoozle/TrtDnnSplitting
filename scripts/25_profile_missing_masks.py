"""
25_profile_missing_masks.py — Profile the top-N missing masks from a cache-only run.

Reads skipped_masks_detail from a cache-only run's all_results.json, ranks the
truly-missing masks by how often they were skipped (frequency across records),
and runs live evaluate_mask for each up to --max-real-profiles.

This warms exactly the masks that are blocking cache-only coverage, instead of
re-running the full Fig.4 search.

Usage
-----
  # Dry run — show which masks would be profiled
  python scripts/25_profile_missing_masks.py --run-dir results/dnn_experiments/<name> --dry-run

  # Profile top 50 missing masks (all models)
  python scripts/25_profile_missing_masks.py --run-dir results/dnn_experiments/<name> --top-n 50

  # Limit to specific models
  python scripts/25_profile_missing_masks.py --run-dir results/dnn_experiments/<name> \\
      --top-n 50 --models vgg19 alexnet

  # With disk and budget guards
  python scripts/25_profile_missing_masks.py --run-dir results/dnn_experiments/<name> \\
      --top-n 50 --min-free-gb 20 --max-real-profiles 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def parse_args():
    ap = argparse.ArgumentParser(
        description="Profile missing masks from a cache-only run to warm the interval cache"
    )
    ap.add_argument("--run-dir", required=True,
                    help="Path to cache-only run directory (contains all_results.json)")
    ap.add_argument("--top-n", type=int, default=50,
                    help="Profile the top N missing masks ranked by skip frequency")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Limit to these models (default: all models in run)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="Abort if free disk space drops below this threshold (GB)")
    ap.add_argument("--max-real-profiles", type=int, default=None,
                    help="Hard cap on total live profile runs (default: unlimited)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show which masks would be profiled without executing")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    return ap.parse_args()


def _check_min_free_gb(min_free_gb):
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


def _compute_merge_groups(mask: list) -> list:
    groups, current = [], [0]
    for i, bit in enumerate(mask):
        if bit == 1:
            groups.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    groups.append(current)
    return groups


def _interval_has_gpu_timing(model: str, group: list, precision: str) -> bool:
    t_path = REPO / "artifacts" / "chunk_cache" / model / f"int_{group[0]}_{group[-1]}" / "timing.json"
    if not t_path.exists():
        return False
    try:
        t = json.loads(t_path.read_text())
    except Exception:
        return False
    return bool(t.get(f"gpu_mean_ms_{precision}") and t.get(f"gpu_p99_ms_{precision}"))


def _is_mask_cached(model: str, mask: list, precision: str) -> bool:
    import hashlib
    mask_str = "".join(str(b) for b in mask)
    h = hashlib.sha256(mask_str.encode()).hexdigest()[:8]
    k = sum(mask) + 1
    variant = f"{model}_mask_{h}_k{k}"
    p = REPO / "results" / "evaluations" / model / f"{variant}_{precision}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    if d.get("error"):
        return False
    chunk_times = d.get("per_chunk_gpu_mean_ms")
    return bool(chunk_times and len(chunk_times) == k)


def _can_assemble(model: str, mask: list, precision: str) -> bool:
    groups = _compute_merge_groups(mask)
    return all(_interval_has_gpu_timing(model, grp, precision) for grp in groups)


def collect_skipped_candidate_masks(records: list, allowed_models=None) -> dict:
    """
    Build {model: {mask_tuple: skip_count}} from skipped_masks_detail across all records.
    skip_count = number of records that included this mask in their skipped set.
    """
    freq: dict = defaultdict(lambda: defaultdict(int))
    for rec in records:
        detail = rec.get("stats", {}).get("skipped_masks_detail", {})
        for model, masks in detail.items():
            if allowed_models and model not in allowed_models:
                continue
            for mask in masks:
                freq[model][tuple(mask)] += 1
    return {m: dict(d) for m, d in freq.items()}


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    precision = args.precision

    p = run_dir / "all_results.json"
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    records = json.loads(p.read_text())

    allowed_models = set(args.models) if args.models else None
    freq_by_model = collect_skipped_candidate_masks(records, allowed_models)

    if not freq_by_model:
        print("No skipped_masks_detail found in this run.")
        print("Either no masks were skipped, the run pre-dates commit 4c62e41,")
        print("or no masks matched the specified --models filter.")
        return

    # Collect truly-missing masks (not in eval JSON, not assembleable)
    # and rank by skip frequency
    candidates: list[tuple[int, str, list]] = []  # (freq, model, mask)
    already_cached = 0
    already_assembleable = 0

    for model, mask_freq in freq_by_model.items():
        for mask_tuple, count in mask_freq.items():
            mask = list(mask_tuple)
            if _is_mask_cached(model, mask, precision):
                already_cached += 1
                continue
            if _can_assemble(model, mask, precision):
                already_assembleable += 1
                continue
            candidates.append((count, model, mask))

    # Sort by frequency descending, then by K ascending (faster to profile first)
    candidates.sort(key=lambda x: (-x[0], sum(x[2])))

    print(f"Run dir: {run_dir}")
    print(f"Precision: {precision}")
    print(f"Skipped masks already in eval JSON: {already_cached}")
    print(f"Skipped masks now assembleable:     {already_assembleable}")
    print(f"Truly missing masks:                {len(candidates)}")
    print(f"Will profile:                       {min(len(candidates), args.top_n)}")
    print()

    top_candidates = candidates[: args.top_n]

    if not top_candidates:
        print("Nothing to profile — all skipped masks are now cached or assembleable.")
        return

    if args.dry_run:
        print(f"[DRY-RUN] Would profile {len(top_candidates)} masks:")
        for i, (count, model, mask) in enumerate(top_candidates):
            k = sum(mask) + 1
            mask_str = "".join(str(b) for b in mask)
            print(f"  {i+1:3d}. {model}  K={k}  freq={count}  mask={mask_str}")
        return

    # Disk guard
    _check_min_free_gb(args.min_free_gb)

    from src.optimization.config_evaluator import evaluate_mask

    profiled = 0
    failed = 0
    cached_hits = 0

    for i, (count, model, mask) in enumerate(top_candidates):
        if args.max_real_profiles is not None and profiled >= args.max_real_profiles:
            print(f"\nBudget reached ({args.max_real_profiles} real profiles). Stopping.")
            break

        # Re-check disk before each profile (long-running)
        _check_min_free_gb(args.min_free_gb)

        k = sum(mask) + 1
        mask_str = "".join(str(b) for b in mask)
        print(f"\n[{i+1}/{len(top_candidates)}] {model}  K={k}  freq={count}  mask={mask_str}")

        result = evaluate_mask(
            model_name=model,
            mask=mask,
            precision=precision,
            warmup=args.warmup,
            iters=args.iters,
        )

        if result.cache_hit:
            cached_hits += 1
            print(f"  → cache hit (was already cached when we got here)")
        elif result.error:
            failed += 1
            print(f"  → ERROR: {result.error}")
        elif result.ok():
            profiled += 1
            mean_ms = result.per_chunk_gpu_mean_ms or []
            print(f"  → OK  chunk_times_mean={[f'{t:.2f}' for t in mean_ms]}")
        else:
            failed += 1
            print(f"  → FAILED (no timing)")

    print(f"\nDone.")
    print(f"  Profiled:    {profiled}")
    print(f"  Cache hits:  {cached_hits}")
    print(f"  Failed:      {failed}")
    print()
    print("Run scripts/24_backfill_interval_timings.py to update interval GPU timing.")
    print("Then re-run scripts/23_inspect_cache_coverage.py to verify coverage.")


if __name__ == "__main__":
    main()
