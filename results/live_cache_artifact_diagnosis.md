# Live Cache Artifact Diagnosis

**Date:** 2026-05-10

## Observed after live n=5 main4 run

```
artifacts/chunk_cache  =  25 GB   (interval-level: ONNX + engines + timing)
artifacts/onnx         =  53 GB   (variant-specific ONNX copies)
artifacts/engines      =  53 GB   (variant-specific engine copies)
```

Total repo growth: ~131 GB.  
Cache-only rerun still showed `skipped_cache_misses=15389`.  
`results/evaluations/` has 100 eval JSONs (tiny; the real bottleneck is artifacts).

---

## Root Cause: Two-level redundancy in artifact layout

### Problem 1 — Every variant gets its own ONNX + engine copies

`make_selected_split_config()` in `selective_split.py` generates paths like:
```
chunks[i]["onnx"]         = "artifacts/onnx/<model>/<variant>/chunk{i}.onnx"
chunks[i]["engine_fp32"]  = "artifacts/engines/<model>/<variant>/chunk{i}_fp32.engine"
```

The evaluation pipeline then:

1. **On interval ONNX cache HIT**: `shutil.copy2(interval_cache/chunk.onnx, variant_dir/chunk0.onnx)`
2. **On interval ONNX cache MISS**: exports to `variant_dir/chunk0.onnx`, then `shutil.copy2(variant_dir → interval_cache)`
3. Same pattern for engines.

Result: every unique mask gets its own copy of every ONNX and engine, even when that
interval has been seen before. With K=7 chunks per mask and hundreds of masks, this
multiplies disk usage by the number of unique masks.

### Problem 2 — Interval cache exists but is not the source of truth

The C++ table4_runner reads the split config JSON to find engine paths. Because the
config points to `artifacts/engines/<model>/<variant>/`, the runner uses the variant
copy, not the interval cache. The interval cache is used only as a copy source for new
variants — it does not replace the variant dirs.

### Problem 3 — Interval timing.json has no GPU timing

`artifacts/chunk_cache/<model>/int_<start>_<end>/timing.json` stores:
```json
{"model": "...", "source_chunk_ids": [...], "export_wall_s": ..., "build_fp32_wall_s": ...}
```
It does **not** store `gpu_mean_ms` or `gpu_p99_ms` from the profiling step.

Without per-interval GPU timing, cache-only mode cannot assemble a mask result from
interval data; it can only serve masks that have an exact eval JSON in
`results/evaluations/`. This is why cache-only reruns still show `skipped_cache_misses`
— for any mask without an exact eval JSON, the system skips rather than assembles
from cached interval timings.

### Problem 4 — skipped_cache_misses counts attempts, not unique masks

`ProfilingStats.skipped_cache_misses` is incremented once per
`evaluate_and_apply_mask()` call that hits the budget gate. If the same mask is
evaluated by multiple taskset×algorithm combinations, it is counted multiple times.
There is no `unique_skipped_masks` counter.

---

## Fix Plan

### B — Path consolidation (eliminates 106 GB duplication for new runs)

Change `make_selected_split_config()` so chunk paths point directly to the interval
cache:
```
chunks[i]["onnx"]        → "artifacts/chunk_cache/<model>/int_<start>_<end>/chunk.onnx"
chunks[i]["engine_fp32"] → "artifacts/chunk_cache/<model>/int_<start>_<end>/chunk_fp32.engine"
```

The interval cache now IS the canonical storage. Variants that share an interval
reference the same file. No copies, no duplication.

The C++ table4_runner reads these paths from the config and will use the interval cache
files directly.

### C — GPU timing in interval timing.json

After profiling, map `per_chunk_gpu_mean_ms[i]` / `per_chunk_gpu_p99_ms[i]` back to
each interval and write to `timing.json`:
```json
{
  "model": "resnet18",
  "source_chunk_ids": [0, 1, 2, 3],
  "export_wall_s": 0.31,
  "build_fp32_wall_s": 8.93,
  "gpu_mean_ms": 0.1448,
  "gpu_p99_ms": 0.1513,
  "precision": "fp32"
}
```

Existing eval JSONs can be backfilled via a one-time utility.

### D — Cache-only assembly from interval timing

Add `can_assemble_from_intervals(model, mask, precision)` — returns True if all
required interval timing.json files have `gpu_mean_ms` + `gpu_p99_ms`.

When a cache-only request would be skipped, try assembly first:
- Load per-interval GPU timing
- Construct `EvaluationResult` with `per_chunk_gpu_p99_ms` from intervals
- Write eval JSON → next hit will use the eval JSON directly

### E — Unique counters

Add `unique_skipped_masks` (set-based) vs `skipped_cache_misses` (attempt count).
Document the distinction. Add `interval_timing_cache_hits`.

### F — Cleanup script

`scripts/22_clean_generated_artifacts.py` removes:
- `artifacts/onnx/<model>/` (old variant copies — safe to delete since interval cache is canonical)
- `artifacts/engines/<model>/` (same)
- `artifacts/logs/` (build logs)
- Never deletes: interval cache, evaluations, table4, profiling cache

### G — Disk guard

`--min-free-gb N` on scripts 30 and 40. Checked before each live build/profile.
Raises `RuntimeError` with cleanup instructions if free space < N GB.

---

## File Impact

| File | Change |
|------|--------|
| `src/splitting/selective_split.py` | `make_selected_split_config()`: chunk paths → interval cache |
| `src/optimization/config_evaluator.py` | Remove `shutil.copy2()`; export/build directly to interval paths; GPU timing backfill; interval assembly functions |
| `src/integration/mask_applicator.py` | Try interval assembly before returning skipped |
| `src/integration/dnn_algorithm_runner.py` | Add unique counters to ProfilingStats |
| `scripts/22_clean_generated_artifacts.py` | New cleanup script |
| `scripts/30_run_yaml_fig4_experiment.py` | Add `--min-free-gb` |
| `scripts/40_run_fig5_design_time.py` | Add `--min-free-gb` |
