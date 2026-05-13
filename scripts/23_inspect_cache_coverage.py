"""
23_inspect_cache_coverage.py — Diagnose interval cache coverage for a run directory.

Given a run directory (from scripts 30 or 40), reports three sections:

  A) ProfilingStats Summary
     Aggregated counters from all result records: cache_hits, skipped_cache_misses,
     interval_timing_cache_hits, unique_skipped_masks, etc.

  B) Final-Mask Coverage
     Coverage for the final selected mask per task (from task_masks in all_results.json).
     Exact eval-JSON hits, interval-assembleable, and truly missing.

  C) Search-Space Candidate Coverage
     Coverage for ALL masks explored during algorithm search (from skipped_masks_detail
     in stats, populated since commit 4c62e41).  Shows per-model breakdown with
     required intervals, which have timing, and which are missing.
     Also reports top missing intervals by frequency across all candidate masks.

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
    ap.add_argument("--top-missing-masks", type=int, default=10,
                    help="Show details for top N truly-missing candidate masks per model")
    return ap.parse_args()


def load_run_results(run_dir: Path) -> list:
    p = run_dir / "all_results.json"
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(p.read_text())


def _compute_merge_groups(mask: list) -> list:
    """Pure-Python merge group computation (no torch needed)."""
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
    return bool(
        t.get(f"gpu_mean_ms_{precision}")
        and t.get(f"gpu_p99_ms_{precision}")
        and t.get(f"gpu_max_ms_{precision}")
    )


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


def _missing_interval_keys(model: str, mask: list, precision: str) -> list:
    groups = _compute_merge_groups(mask)
    return [
        f"int_{grp[0]}_{grp[-1]}"
        for grp in groups
        if not _interval_has_gpu_timing(model, grp, precision)
    ]


def interval_timing_coverage(precision: str) -> dict:
    """Return per-model dict of (total, with_gpu, without_gpu)."""
    cache = REPO / "artifacts" / "chunk_cache"
    result = {}
    if not cache.exists():
        return result
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
                try:
                    t = json.loads(t_path.read_text())
                    if (
                        t.get(f"gpu_mean_ms_{precision}")
                        and t.get(f"gpu_p99_ms_{precision}")
                        and t.get(f"gpu_max_ms_{precision}")
                    ):
                        has_gpu += 1
                        continue
                except Exception:
                    pass
            missing.append(int_dir.name)
        result[model] = {"total": total, "has_gpu": has_gpu, "missing_dirs": missing}
    return result


def _collect_final_masks(records: list) -> dict:
    """Collect final selected masks from task_masks field (one per task row)."""
    encountered: dict = defaultdict(set)
    for rec in records:
        task_masks_raw = rec.get("task_masks", {})
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
    # Fallback: task_details
    if not any(encountered.values()):
        for rec in records:
            for td in rec.get("task_details", []):
                model = td.get("model_name", "")
                mask = td.get("final_mask", [])
                if model and mask:
                    encountered[model].add(tuple(mask))
    return encountered


def _collect_skipped_candidate_masks(records: list) -> dict:
    """
    Collect all unique (model, mask) pairs skipped during algorithm search.

    Reads skipped_masks_detail from stats (populated since commit 4c62e41).
    Returns {model: set of mask tuples}.
    """
    skipped: dict = defaultdict(set)
    for rec in records:
        detail = rec.get("stats", {}).get("skipped_masks_detail", {})
        for model, masks in detail.items():
            for mask in masks:
                skipped[model].add(tuple(mask))
    return skipped


def _analyze_masks(model: str, masks: set, precision: str):
    """Return (exact_hits, assembleable, truly_missing, missing_interval_freq)."""
    exact_hits = 0
    assembleable = 0
    truly_missing = 0
    missing_freq: dict = defaultdict(int)
    for mask_tuple in masks:
        mask = list(mask_tuple)
        if _is_mask_cached(model, mask, precision):
            exact_hits += 1
        elif _can_assemble(model, mask, precision):
            assembleable += 1
        else:
            truly_missing += 1
            for key in _missing_interval_keys(model, mask, precision):
                missing_freq[key] += 1
    return exact_hits, assembleable, truly_missing, dict(missing_freq)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    precision = args.precision

    print(f"Run dir: {run_dir}")
    print(f"Precision: {precision}")
    print()

    # ── Interval timing coverage ─────────────────────────────────────────────
    coverage = interval_timing_coverage(precision)
    print("=== Interval Timing Coverage ===")
    for model, info in coverage.items():
        total, has_gpu = info["total"], info["has_gpu"]
        print(f"  {model}: {has_gpu}/{total} intervals with GPU timing  "
              f"({total - has_gpu} missing)")
    print()

    records = load_run_results(run_dir)

    # ────────────────────────────────────────────────────────────────────────
    # SECTION A: ProfilingStats Summary
    # ────────────────────────────────────────────────────────────────────────
    total_attempts   = sum(r.get("stats", {}).get("skipped_cache_misses", 0) for r in records)
    unique_skipped   = sum(r.get("stats", {}).get("unique_skipped_masks", 0) for r in records)
    intv_hits        = sum(r.get("stats", {}).get("interval_timing_cache_hits", 0) for r in records)
    cache_hits_sum   = sum(r.get("stats", {}).get("cache_hits", 0) for r in records)
    unique_eval      = sum(r.get("stats", {}).get("unique_masks_evaluated", 0) for r in records)
    unique_cache_hit = sum(r.get("stats", {}).get("unique_mask_cache_hits", 0) for r in records)

    print("=== Section A: ProfilingStats Summary (aggregated across all records) ===")
    print(f"  total cache_hits (eval JSON + interval):  {cache_hits_sum}")
    print(f"  interval_timing_cache_hits:               {intv_hits}")
    print(f"  unique_masks_evaluated:                   {unique_eval}")
    print(f"  unique_mask_cache_hits:                   {unique_cache_hit}")
    print(f"  skipped_cache_misses_attempts:            {total_attempts}")
    print(f"    (attempt-count; same mask × every algorithm attempt that hits it)")
    print(f"  unique_skipped_masks:                     {unique_skipped}")
    print(f"    (SUM of per-record unique counts; same mask can appear in many records)")
    print(f"    (Section C below shows the true global unique count from skipped_masks_detail)")

    if intv_hits == 0 and unique_cache_hit > 0 and unique_eval > 0:
        pct = 100 * unique_cache_hit // unique_eval if unique_eval else 0
        print()
        print(f"  NOTE: interval_timing_cache_hits=0 is expected here.")
        print(f"  {unique_cache_hit}/{unique_eval} ({pct}%) unique evaluated masks were served")
        print(f"  directly from eval JSON cache (exact hits), so interval assembly was never needed.")
        print(f"  Interval assembly is only the fallback for masks NOT in the eval JSON cache that")
        print(f"  are also blocked by the live budget.  The Section C candidate analysis below shows")
        print(f"  the truly unique skipped masks and which intervals they need.")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # SECTION B: Final-Mask Coverage
    # ────────────────────────────────────────────────────────────────────────
    final_masks = _collect_final_masks(records)

    print("=== Section B: Final-Mask Coverage (task_masks from result rows) ===")
    print("    The mask actually chosen as the final split for each scheduled task.")
    print()
    b_total_exact = 0
    b_total_assemble = 0
    b_total_missing = 0
    b_missing_freq_by_model: dict = {}

    for model in sorted(final_masks):
        masks = final_masks[model]
        if not masks:
            continue
        exact, assemble, missing, miss_freq = _analyze_masks(model, masks, precision)
        b_total_exact += exact
        b_total_assemble += assemble
        b_total_missing += missing
        b_missing_freq_by_model[model] = miss_freq
        print(f"  {model}: {len(masks)} unique final masks")
        print(f"    exact eval JSON hits:  {exact}")
        print(f"    interval-assembleable: {assemble}")
        print(f"    truly missing:         {missing}")

    print()
    print(f"  TOTAL: {b_total_exact + b_total_assemble + b_total_missing} unique final masks")
    print(f"    exact hits:            {b_total_exact}")
    print(f"    interval-assembleable: {b_total_assemble}")
    print(f"    truly missing:         {b_total_missing}")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # SECTION C: Search-Space Candidate Coverage
    # ────────────────────────────────────────────────────────────────────────
    skipped_masks = _collect_skipped_candidate_masks(records)
    has_candidate_data = any(skipped_masks.values())

    print("=== Section C: Search-Space Candidate Coverage (skipped_masks_detail) ===")
    if not has_candidate_data:
        print("  No skipped_masks_detail found in this run's records.")
        print("  (Run was produced before commit 4c62e41, or no masks were skipped.)")
        print()
        c_total_missing = 0
        c_total_assemble = 0
    else:
        print("  All masks explored during algorithm search that were skipped.")
        print()
        c_total_exact = 0
        c_total_assemble = 0
        c_total_missing = 0
        c_missing_freq_by_model: dict = {}

        for model in sorted(skipped_masks):
            masks = skipped_masks[model]
            if not masks:
                continue
            exact, assemble, missing, miss_freq = _analyze_masks(model, masks, precision)
            c_total_exact += exact
            c_total_assemble += assemble
            c_total_missing += missing
            c_missing_freq_by_model[model] = miss_freq
            print(f"  {model}: {len(masks)} unique candidate masks skipped during search")
            print(f"    now in eval JSON (cached after the fact): {exact}")
            print(f"    interval-assembleable (all intervals ok):  {assemble}")
            print(f"    truly missing (need live profiling):       {missing}")

        print()
        print(f"  TOTAL: {sum(len(v) for v in skipped_masks.values())} unique skipped candidate masks")
        print(f"    now in eval JSON:       {c_total_exact}")
        print(f"    interval-assembleable:  {c_total_assemble}")
        print(f"    truly missing:          {c_total_missing}")

        if c_total_missing > 0:
            print(f"\n  --- Truly-Missing Candidate Masks (top {args.top_missing_masks} per model, easiest first) ---")
            for model in sorted(skipped_masks):
                masks = skipped_masks[model]
                truly_missing_list = []
                for mask_tuple in masks:
                    mask = list(mask_tuple)
                    if not _is_mask_cached(model, mask, precision) and not _can_assemble(model, mask, precision):
                        miss_ivs = _missing_interval_keys(model, mask, precision)
                        total_ivs = len(_compute_merge_groups(mask))
                        truly_missing_list.append((mask, miss_ivs, total_ivs))
                if not truly_missing_list:
                    continue
                # Sort by number of missing intervals ascending (easiest to backfill first)
                truly_missing_list.sort(key=lambda x: len(x[1]))
                print(f"\n  {model}: {len(truly_missing_list)} truly-missing candidate masks")
                for mask, miss_ivs, total_ivs in truly_missing_list[:args.top_missing_masks]:
                    k = sum(mask) + 1
                    mask_str = "".join(str(b) for b in mask)
                    print(f"    K={k}  mask={mask_str}  intervals={total_ivs}  missing_intervals={miss_ivs}")

        print(f"\n=== Top Missing Intervals (candidate search-space, top {args.top_missing_intervals}) ===")
        if not any(c_missing_freq_by_model.get(m, {}) for m in c_missing_freq_by_model):
            print("  None — all candidate masks are either cached or interval-assembleable.")
        else:
            for model in sorted(c_missing_freq_by_model):
                freq = c_missing_freq_by_model[model]
                if not freq:
                    continue
                top = sorted(freq.items(), key=lambda x: -x[1])[:args.top_missing_intervals]
                print(f"\n  {model}:")
                for int_name, count in top:
                    print(f"    {int_name}: needed by {count} unique candidate masks")

    if b_total_missing > 0:
        print(f"\n=== Top Missing Intervals (final masks only) ===")
        for model in sorted(b_missing_freq_by_model):
            freq = b_missing_freq_by_model[model]
            if not freq:
                continue
            top = sorted(freq.items(), key=lambda x: -x[1])[:args.top_missing_intervals]
            print(f"\n  {model}:")
            for int_name, count in top:
                print(f"    {int_name}: needed by {count} final masks")
    print()

    # ────────────────────────────────────────────────────────────────────────
    # ACTION
    # ────────────────────────────────────────────────────────────────────────
    print("=== ACTION ===")
    can_backfill_final = b_total_assemble > 0
    can_backfill_cand  = c_total_assemble > 0 if has_candidate_data else False

    if b_total_missing == 0 and c_total_missing == 0 and not (can_backfill_final or can_backfill_cand):
        print("  All masks are covered. No backfill or profiling needed.")
    else:
        if can_backfill_final or can_backfill_cand:
            print("  Run scripts/24_backfill_interval_timings.py to populate missing interval GPU timing.")

        if b_total_missing > 0:
            print(f"\n  Final masks needing live profiling: {b_total_missing}")
            print(f"    (selected final split masks with no eval JSON and incomplete intervals)")

        if has_candidate_data and c_total_missing > 0:
            print(f"\n  Candidate (search-space) masks needing live profiling: {c_total_missing}")
            print(f"    (explored during search; no eval JSON and incomplete interval coverage)")
            print(f"    Run scripts/25_profile_missing_masks.py --run-dir {run_dir} to warm these.")
        elif unique_skipped > 0 and not has_candidate_data:
            print(f"\n  Unique skipped masks (from stats): {unique_skipped}")
            print(f"    Detailed per-mask breakdown unavailable (old run format).")
            print(f"    Re-run cache-only to populate skipped_masks_detail.")
    print()


if __name__ == "__main__":
    main()
