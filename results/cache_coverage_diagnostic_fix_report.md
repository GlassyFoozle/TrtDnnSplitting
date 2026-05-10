# Cache Coverage Diagnostic Fix Report

**Date:** 2026-05-10  
**Commit:** clarify-cache-coverage-diagnostics

---

## Summary

After a second live cache-warming run (round2, cap50), `scripts/23_inspect_cache_coverage.py`
reported confusing numbers:
- `interval_timing_cache_hits: 0` — apparently regressed from 347 to 0
- `unique_skipped_masks: 1968` (Section A) vs 20 unique final masks (Section B)
- ACTION message said "1 mask still needs live profiling" while ProfilingStats showed 1968 skipped

This report documents the root causes, all fixes, and the updated diagnostic output.

---

## Issue 1: interval_timing_cache_hits=0 is correct, not a regression

### Root cause

`interval_timing_cache_hits=0` is the **expected and correct** value when all evaluated
masks are served from the eval JSON cache directly (exact hits).

The interval assembly path in `mask_applicator.py` is only reached when:
1. `is_mask_cached(model, mask)` returns **False** (mask not in eval JSON), AND
2. `check_before_real_eval()` returns a skip reason (cache-only mode)

After round2_cap50 warmed more eval JSONs, all 4326 evaluated unique masks now hit
`is_mask_cached` → True, so the code never enters the interval assembly block.
`interval_timing_cache_hits` is only incremented there → stays 0.

The postbackfill run had 347 interval_timing_cache_hits because at that time, 347 masks
were NOT in the eval JSON cache but were assembleable from intervals.  After round2
profiled those masks, they were promoted to exact eval JSON hits.  This is correct behavior.

### Fix

Added a clear NOTE to Section A of script 23 explaining when `interval_timing_cache_hits=0`
is expected, with the eval JSON hit percentage.

---

## Issue 2: Section A unique_skipped_masks is a SUM, not a global unique count

### Root cause

`unique_skipped_masks` in `all_results.json` stats is a per-record set-based count.
Script 30 SUMS this across all 100 records.  If the same mask is skipped in 6 records,
it contributes 6 to the sum.  The actual global unique count is much lower.

For the round2_cap50 cache-only run:
- `unique_skipped_masks: 1968` (sum of per-record unique counts, ~100 records × avg ~20)
- True global unique skipped masks: **316** (254 alexnet + 62 resnet18)

vgg19's explored mask space is fully covered by the eval JSON cache — zero vgg19 skips.

### Fix

Clarified Section A label: "(SUM of per-record unique counts; same mask can appear in
many records)".  Added a note pointing to Section C for the true global unique count.

---

## Issue 3: Script 23 lacked search-space candidate diagnostics (Section C)

### Root cause

Script 23 only analyzed `task_masks` (final selected masks) from `all_results.json`.
This gives at most ~20 unique masks.  The 316 (or 1968-counted) candidate masks
explored during algorithm search were invisible.

### Fix

Added `skipped_masks_detail` to `ProfilingStats`:
- New private field `_skipped_by_model: dict` in `ProfilingStats`
- When a mask is added to `_seen_skipped`, also record `(model, mask)` in `_skipped_by_model`
- Requires `model_name: str = ""` field on `MaskApplicationResult`, set on the skip return path in `mask_applicator.py`
- Serialized in `to_dict()` as `"skipped_masks_detail": {model: [[mask], ...]}`

Script 23 now has **Section C** that:
- Reads `skipped_masks_detail` from all records and builds the union
- For each skipped candidate mask: classifies as now-cached / interval-assembleable / truly missing
- Lists truly-missing candidate masks per model with K, mask bits, required intervals, missing intervals
- Reports top missing intervals by frequency across ALL candidate masks

---

## Issue 4: ACTION message was inaccurate

### Root cause

The ACTION section only checked `truly_missing` from the final-mask analysis (Section B),
which was 1–4 masks.  It ignored the potentially hundreds of candidate masks from Section C.

### Fix

ACTION section now distinguishes:
- **Final masks needing profiling** (Section B truly missing)
- **Candidate masks needing profiling** (Section C truly missing)
- Correctly references `scripts/25_profile_missing_masks.py` for candidate warming

---

## New Script: scripts/25_profile_missing_masks.py

Created to enable targeted warming of the masks blocking cache-only coverage.

Usage:
```bash
# Dry-run: show what would be profiled
python scripts/25_profile_missing_masks.py --run-dir results/dnn_experiments/<name> --dry-run

# Profile top 50 missing candidate masks
python scripts/25_profile_missing_masks.py --run-dir results/dnn_experiments/<name> --top-n 50

# With guards
python scripts/25_profile_missing_masks.py --run-dir results/dnn_experiments/<name> \
    --top-n 50 --min-free-gb 20 --max-real-profiles 100
```

Features:
- Reads `skipped_masks_detail` from a cache-only run
- Counts skip frequency per mask across all records (how often it blocked an algorithm)
- Skips masks already cached or assembleable
- Ranks by frequency descending, K ascending (fastest to profile first)
- Respects `--min-free-gb` disk guard and `--max-real-profiles` budget cap
- After profiling, calls `scripts/24_backfill_interval_timings.py` and re-runs script 23

---

## Current State (post-fix, fresh cache-only run)

Run: `fig4_main4_cacheonly_v3_skippeddetail` (n=5, seed 42, major_blocks, fp32)

**Section A:**
```
total cache_hits:              11485
interval_timing_cache_hits:    0  (expected: all evaluated masks hit eval JSON)
unique_masks_evaluated:        4306
unique_mask_cache_hits:        4306  (100%)
unique_skipped_masks (sum):    1968
```

**Section B (final masks):**
```
TOTAL: 20 unique final masks
  exact hits:            16
  interval-assembleable: 0
  truly missing:         4   (K=1 wide-span baselines: int_0_21, int_0_13, int_0_45)
```

**Section C (candidate masks):**
```
TOTAL: 316 unique skipped candidate masks
  now in eval JSON:       0
  interval-assembleable:  0
  truly missing:          316
    alexnet: 254  (top missing interval: int_0_5, needed by 64 masks)
    resnet18: 62  (top missing interval: int_12_13, needed by 16 masks)
```

To warm the 316 missing candidate masks:
```bash
python scripts/25_profile_missing_masks.py \
    --run-dir results/dnn_experiments/fig4_main4_cacheonly_v3_skippeddetail \
    --top-n 316 --min-free-gb 20
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/integration/mask_applicator.py` | Add `model_name: str = ""` to `MaskApplicationResult`; set on skip return |
| `src/integration/dnn_algorithm_runner.py` | Add `_skipped_by_model` dict; populate on skip; serialize as `skipped_masks_detail` |
| `scripts/23_inspect_cache_coverage.py` | Three-section rewrite: A (stats), B (final masks), C (search-space candidates); fix ACTION |
| `scripts/25_profile_missing_masks.py` | New — targeted profiling of missing candidate masks |
