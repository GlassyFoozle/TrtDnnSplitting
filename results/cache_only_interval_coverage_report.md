# Cache-Only Interval Coverage Report

**Date:** 2026-05-10  
**Run analysed:** `fig4_main4_cacheonly_postbackfill` (n=5, main4, major_blocks, fp32)  
**Prior run:** `fig4_main4_live_n5_cacheonly_interval_check` (showed `interval_timing_cache_hits=0`)

---

## Summary of Findings

Two bugs blocked interval-cache assembly in cache-only mode:

| # | Root cause | Fix |
|---|-----------|-----|
| 1 | All 69 interval `timing.json` files lacked GPU timing fields — backfill never ran | `scripts/24_backfill_interval_timings.py` (one-shot) |
| 2 | `evaluate_mask()` imported `selective_split` (which imports torch) before the eval-JSON cache check — caused `ModuleNotFoundError` even for cached masks | Hoisted torch-free early cache check to step 0 in `evaluate_mask()` |

---

## Root Cause 1: Interval timing.json files had no GPU fields

### What happened

After the live n=5 run, `artifacts/chunk_cache/<model>/int_<s>_<e>/timing.json` files existed for all 69 intervals, but contained only build timing (`build_wall_s`, etc.) — no `gpu_mean_ms_fp32` or `gpu_p99_ms_fp32`.

`_backfill_interval_gpu_timing()` (called inside the live profiling pipeline) was responsible for writing these fields, but it only runs _during_ a live profile. The cache-only re-run never called it.

As a result, `can_assemble_from_intervals()` returned False for all masks, so `interval_timing_cache_hits` was always 0.

### Fix

Created `scripts/24_backfill_interval_timings.py` — a one-shot script that reads all existing `results/evaluations/**/*_fp32.json` eval JSONs, extracts `per_chunk_gpu_mean_ms` / `per_chunk_gpu_p99_ms` per chunk group, and writes `gpu_mean_ms_fp32` / `gpu_p99_ms_fp32` back to the matching `timing.json` files.

**Result after backfill:**
```
alexnet:  9/9  intervals with GPU timing  (0 missing)
resnet18: 7/7  intervals with GPU timing  (0 missing)
vgg19:   53/53 intervals with GPU timing  (0 missing)
```

---

## Root Cause 2: `evaluate_mask()` torch import before cache check

### What happened

`evaluate_mask()` had this structure at line 765:
```python
from src.splitting.selective_split import (   # ← torch imported here
    load_base_config, parse_boundary_mask, ...
)
...
# ── 3. Cache check  (line 796)
if not force and not dry_run:
    cached = _load_cached_result(...)
```

`selective_split.py` imports `torch` at module level. Even for masks that ARE in the eval JSON cache (and would return at step 3), the torch import ran first and raised `ModuleNotFoundError` in environments where torch is not installed.

The masks that triggered this were ones where `is_mask_cached()` returned True — causing `mask_applicator.py` to call `_eval_mask` directly, bypassing the live-budget skip path.

### Fix

Added a torch-free early cache check (step 0) at the very top of `evaluate_mask()`, before any `selective_split` import:

```python
# ── 0. Early cache check (torch-free) ────────────────────────
if not force and not dry_run and isinstance(mask, list) and variant_name is None:
    _early_variant = mask_to_variant_name(model_name, list(mask))
    _early_cached = _load_cached_result(model_name, _early_variant, precision)
    if _early_cached is not None and _early_cached.ok():
        print(f"  → cache hit (early)")
        return _early_cached
```

`mask_to_variant_name` and `_load_cached_result` use only `hashlib` and `json` — no torch.

---

## Results After Both Fixes

Run: `fig4_main4_cacheonly_postbackfill` (n=5, same config)

```
skipped_cache_misses_attempts:  4087   (attempt-count, same mask × algorithms)
unique_masks_evaluated:         4109
unique_mask_cache_hits:         4109   (100% of evaluated unique masks hit eval JSON)
unique_skipped_masks:           2483   (need live profiling — unexplored search space)
interval_timing_cache_hits:      347   (served from interval timing assembly)
total cache_hits:              11003
```

All 4109 unique masks that were evaluated returned from the eval JSON cache. An additional 347 evaluations were served via interval timing assembly (masks not in eval JSON but whose intervals all had GPU timing after backfill).

### Mask coverage from final results only (script 23)

```
alexnet:  3 unique final masks  →  1 exact hit,  2 truly missing
resnet18: 2 unique final masks  →  1 exact hit,  1 truly missing
vgg19:   15 unique final masks  → 13 exact hits, 2 truly missing
TOTAL:   20 unique final masks  → 15 exact hits, 5 truly missing
```

The 5 truly missing final masks correspond to wide-interval K=1 baseline spans (e.g. `int_0_21`, `int_0_13`) that were never profiled in the original live run.

The 2483 unique_skipped_masks are masks explored by algorithms during search that have no eval JSON and no interval coverage — they require live profiling.

---

## Interval Timing Coverage (post-backfill)

All 69 intervals populated:
```
alexnet:   9/9  with GPU timing
resnet18:  7/7  with GPU timing
vgg19:    53/53 with GPU timing
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/optimization/config_evaluator.py` | Added step 0 early cache check before selective_split import |
| `scripts/24_backfill_interval_timings.py` | Created — one-shot backfill of interval GPU timing from eval JSONs |
| `scripts/23_inspect_cache_coverage.py` | Created — diagnostic for interval cache coverage |
| `src/integration/dnn_algorithm_runner.py` | New counters: `unique_masks_evaluated`, `unique_mask_cache_hits`, `unique_skipped_masks`, `interval_timing_cache_hits`; `builds_triggered` gated on `not result.cache_hit` |
| `scripts/30_run_yaml_fig4_experiment.py` | New counters wired into summary/CSV/aggregation; `--min-free-gb` disk guard |
| `scripts/internal_fig4_helpers.py` | New counter fields in summarize_result, aggregate, write_summary |

---

## Action Items

- **No further backfill needed** — all 69 intervals are populated.
- **5 truly missing final masks** could be recovered by a targeted live run with those 3 models.
- **2483 unique_skipped_masks** represent unexplored search-space masks — acceptable for a cache-only run; they require live profiling budget to resolve.
- The `scripts/24_backfill_interval_timings.py` script should be re-run after any future live profiling batch to keep interval timing.json files current.
