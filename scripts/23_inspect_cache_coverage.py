"""
23_inspect_cache_coverage.py — Diagnose interval cache coverage for a run directory.

Given a run directory (from scripts 30 or 40), reports:
  - Unique candidate masks encountered by algorithms
  - Exact mask-level eval JSON hits
  - Interval-assembled hits (masks that could be served from interval timing)
  - Truly missing masks (no eval JSON AND incomplete interval timing)
  - Interval timing coverage per model (total intervals, with/without GPU timing)
  - Top missing intervals required by skipped masks

Usage
-----
  python scripts/23_inspect_cache_coverage.py --run-dir results/dnn_experiments/<name>
  python scripts/23_inspect_cache_coverage.py --run-dir results/dnn_experiments/<name> --precision fp16
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
    ap = argparse.ArgumentParser(description="Inspect interval cache coverage for a run")
    ap.add_argument("--run-dir", required=True, help="Path to run directory (contains all_results.json)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--top-missing-intervals", type=int, default=20,
                    help="Show top N missing intervals by frequency")
    return ap.parse_args()


def load_run_results(run_dir: Path) -> list:
    p = run_dir / "all_results.json"
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(p.read_text())


def _compute_merge_groups(mask: list) -> list:
    """Pure-Python merge group computation (no torch needed)."""
    n_chunks = len(mask) + 1
    groups, current = [], [0]
    for i, bit in enumerate(mask):
        if bit == 1:
            groups.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    groups.append(current)
    return groups


def _interval_dir(model: str, group: list) -> Path:
    return REPO / "artifacts" / "chunk_cache" / model / f"int_{group[0]}_{group[-1]}"


def _interval_has_gpu_timing(model: str, group: list, precision: str) -> bool:
    t_path = _interval_dir(model, group) / "timing.json"
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
    if not chunk_times:
        return False
    return len(chunk_times) == k


def _can_assemble(model: str, mask: list, precision: str) -> bool:
    groups = _compute_merge_groups(mask)
    return all(_interval_has_gpu_timing(model, grp, precision) for grp in groups)


def interval_timing_coverage(precision: str) -> dict:
    """Return per-model dict of (total, with_gpu, without_gpu)."""
    cache = REPO / "artifacts" / "chunk_cache"
    result = {}
    for model_dir in sorted(cache.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        total = 0
        has_gpu = 0
        missing = []
        for int_dir in sorted(model_dir.iterdir()):
            if not int_dir.is_dir():
                continue
            total += 1
            t_path = int_dir / "timing.json"
            if t_path.exists():
                t = json.loads(t_path.read_text())
                if t.get(f"gpu_mean_ms_{precision}") and t.get(f"gpu_p99_ms_{precision}"):
                    has_gpu += 1
                    continue
            missing.append(int_dir.name)
        result[model] = {"total": total, "has_gpu": has_gpu, "missing_dirs": missing}
    return result


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    precision = args.precision

    print(f"Run dir: {run_dir}")
    print(f"Precision: {precision}")
    print()

    # ── 1. Interval timing coverage ───────────────────────────────────────────
    coverage = interval_timing_coverage(precision)
    print("=== Interval Timing Coverage ===")
    for model, info in coverage.items():
        total, has_gpu = info["total"], info["has_gpu"]
        print(f"  {model}: {has_gpu}/{total} intervals with GPU timing  "
              f"({total - has_gpu} missing)")
    print()

    # ── 2. Load masks from run ────────────────────────────────────────────────
    records = load_run_results(run_dir)

    # Collect all unique (model, mask) pairs from task_masks fields
    encountered: dict[str, set] = defaultdict(set)  # model → set of mask tuples
    for rec in records:
        task_masks_raw = rec.get("task_masks", {})
        # task_masks may be a JSON string
        if isinstance(task_masks_raw, str):
            try:
                task_masks = json.loads(task_masks_raw)
            except Exception:
                task_masks = {}
        else:
            task_masks = task_masks_raw or {}
        if isinstance(task_masks, dict):
            for task_name, mask in task_masks.items():
                if isinstance(mask, list) and len(mask) > 0:
                    model_dist_raw = rec.get("model_distribution", "{}")
                    try:
                        dist = json.loads(model_dist_raw) if isinstance(model_dist_raw, str) else (model_dist_raw or {})
                    except Exception:
                        dist = {}
                    if len(dist) == 1:
                        model = next(iter(dist))
                        encountered[model].add(tuple(mask))

    # If model couldn't be extracted from task_masks, try task_details
    if not any(encountered.values()):
        for rec in records:
            task_details = rec.get("task_details", [])
            for td in task_details:
                model = td.get("model_name", "")
                mask = td.get("final_mask", [])
                if model and mask:
                    encountered[model].add(tuple(mask))

    print("=== Mask Coverage by Model ===")
    total_exact_hits = 0
    total_assembleable = 0
    total_truly_missing = 0
    missing_interval_freq: dict[str, dict] = defaultdict(lambda: defaultdict(int))

    for model in sorted(encountered):
        masks = encountered[model]
        if not masks:
            continue
        exact_hits = 0
        assembleable = 0
        truly_missing = 0
        for mask_tuple in masks:
            mask = list(mask_tuple)
            if _is_mask_cached(model, mask, precision):
                exact_hits += 1
            elif _can_assemble(model, mask, precision):
                assembleable += 1
            else:
                truly_missing += 1
                # Track which intervals are missing
                groups = _compute_merge_groups(mask)
                for grp in groups:
                    if not _interval_has_gpu_timing(model, grp, precision):
                        key = f"int_{grp[0]}_{grp[-1]}"
                        missing_interval_freq[model][key] += 1

        total_exact_hits += exact_hits
        total_assembleable += assembleable
        total_truly_missing += truly_missing
        print(f"\n  {model}:  {len(masks)} unique masks encountered")
        print(f"    exact eval JSON hits:  {exact_hits}")
        print(f"    interval-assembleable: {assembleable}")
        print(f"    truly missing:         {truly_missing}")

    print(f"\n  TOTAL:  {total_exact_hits + total_assembleable + total_truly_missing} unique masks")
    print(f"    exact hits:            {total_exact_hits}")
    print(f"    interval-assembleable: {total_assembleable}")
    print(f"    truly missing:         {total_truly_missing}")

    # ── 3. Top missing intervals ──────────────────────────────────────────────
    if truly_missing > 0 or any(missing_interval_freq.values()):
        print(f"\n=== Top Missing Intervals (by mask frequency) ===")
        for model in sorted(missing_interval_freq):
            freq = missing_interval_freq[model]
            top = sorted(freq.items(), key=lambda x: -x[1])[:args.top_missing_intervals]
            print(f"\n  {model}:")
            for int_name, count in top:
                print(f"    {int_name}: needed by {count} unique masks")

    # ── 4. Skipped mask stats from all_results.json ───────────────────────────
    total_attempts  = sum(r.get("stats", {}).get("skipped_cache_misses", 0) for r in records)
    unique_skipped  = sum(r.get("stats", {}).get("unique_skipped_masks", 0) for r in records)
    intv_hits       = sum(r.get("stats", {}).get("interval_timing_cache_hits", 0) for r in records)
    cache_hits_sum  = sum(r.get("stats", {}).get("cache_hits", 0) for r in records)
    print(f"\n=== ProfilingStats Summary (aggregated from stats keys) ===")
    print(f"  skipped_cache_misses_attempts: {total_attempts}")
    print(f"  unique_skipped_masks:          {unique_skipped}")
    print(f"  interval_timing_cache_hits:    {intv_hits}")
    print(f"  total cache_hits:              {cache_hits_sum}")

    print()
    if truly_missing == 0 and total_assembleable > 0:
        print("ACTION: Run scripts/24_backfill_interval_timings.py then re-run cache-only.")
    elif truly_missing == 0:
        print("ACTION: All masks are covered. No backfill needed.")
    else:
        print(f"ACTION: {total_assembleable} masks can be covered by backfill.")
        print(f"        {truly_missing} masks still need live profiling.")
        print("        Run scripts/24_backfill_interval_timings.py first.")


if __name__ == "__main__":
    main()
