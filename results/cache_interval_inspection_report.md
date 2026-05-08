# Cache / Interval Inspection Report

**Date:** 2026-05-08

---

## Files Inspected

| File | Lines | Purpose |
|------|-------|---------|
| `src/optimization/config_evaluator.py` | 581 | Mask → ONNX → engine → profile pipeline |
| `src/optimization/compiler.py` | 134 | Subprocess wrappers for export + build |
| `src/optimization/profiling_db.py` | 175 | Flat JSON cache for timing results |
| `src/splitting/selective_split.py` | 331 | Config generation; `compute_merge_groups` |
| `src/export/onnx_exporter.py` | 78 | `export_module()` — pure Python ONNX export |
| `scripts/internal_export_selected_split.py` | 98 | Subprocess driver for whole-variant ONNX export |
| `scripts/internal_build_selected_engines.sh` | 95 | Subprocess driver for `trtexec` per variant |
| `cpp_runtime/src/main_table4.cpp` | 151 | C++ table4_runner; produces per-chunk timing JSON |

---

## Existing Cache Layers

### Layer 1 — Mask-level eval JSON

**Path:** `results/evaluations/{model}/{model}_mask_{sha256_8}_{precision}.json`

**Key:** `{model}_mask_{sha256_8}_k{n_chunks}` derived from `mask_to_variant_name(model, mask)` where sha256 hashes `"".join(str(b) for b in mask)`.

**Check function:** `is_mask_cached()` — parses JSON, verifies no error, checks `per_chunk_gpu_mean_ms` has length == `sum(mask)+1`.

**Semantics:** A complete `EvaluationResult` for this exact mask. If present, `evaluate_mask()` returns immediately.

### Layer 2 — ProfilingDB

**Path:** `results/optimization/.profiling_cache.json`

**Key:** `{model}|{variant}|{precision}`

**Semantics:** Flat timing dict imported from C++ table4_runner result JSONs. Secondary lookup path; the eval JSON (layer 1) is the primary cache.

### Layer 3 — C++ raw result JSON

**Path:** `results/table4/{model}_cpp_{variant}_{precision}.json`

**Semantics:** Raw output of `table4_runner`. Imported into ProfilingDB on load.

---

## Artifact Paths (all variant-specific, no sharing today)

| Artifact | Path pattern |
|----------|-------------|
| Split config JSON | `artifacts/split_configs/{model}/{variant}.json` |
| ONNX (per chunk) | `artifacts/onnx/{model}/{variant}/chunk{i}.onnx` |
| Engine fp32 | `artifacts/engines/{model}/{variant}/chunk{i}_fp32.engine` |
| Engine fp16 | `artifacts/engines/{model}/{variant}/chunk{i}_fp16.engine` |

Where `variant = {model}_mask_{sha256_8}_k{n_chunks}`.

---

## Interval Identity

An **interval** is a contiguous range of base chunks merged into a single chunk by the mask.

```
mask = [1, 0, 0, 1, 0, 0, ...]      ← boundary bits
groups (from compute_merge_groups):
  group 0: [0]                        ← base chunk 0 alone
  group 1: [1, 2, 3]                  ← base chunks 1-3 merged
  group 2: [4, 5, ...]                ← remainder
```

Each group is identified by `source_chunk_ids = [start, start+1, ..., end]`.

**Stable interval key:** `(model, source_chunk_ids_tuple)` — equivalent to `(model, start_idx, end_idx)` since base chunks are always contiguous within a group.

**Input shape is fixed per group:** The input to group `i` is the output of base chunk `source_chunk_ids[0] - 1`, which is determined solely by the model and `start_idx`. No shape ambiguity.

---

## The Sharing Gap

Two masks with the same merged interval produce:
- **Identical** ONNX data (`export_module(build_merged_module(base_modules, in_shape), ...)`)
- **Identical** TRT engines (same ONNX + same `trtexec` flags)

But they store artifacts at **variant-specific paths** — no deduplication exists.

**Example** — alexnet with 14 base chunks:
```
Mask A = [1,0,0,...,0]       K=2: groups=[[0], [1..13]]
Mask B = [1,0,0,...,1,0]     K=3: groups=[[0], [1..12], [13]]
Mask C = [0,...,0,1,0]       K=2: groups=[[0..12], [13]]
```

- `Mask B` shares interval `[0]` with `Mask A` and `[13]` with `Mask C`.
- Currently: both ONNX/engine files for those intervals are built twice.
- With an interval cache: `Mask B` would reuse two of its three chunks from cache.

---

## Interval Cache Design (implemented)

**Cache root:** `artifacts/chunk_cache/{model}/int_{start}_{end}/`

| File | Contents |
|------|----------|
| `chunk.onnx` | ONNX for this merged interval |
| `chunk_{precision}.engine` | TRT engine for this interval + precision |
| `timing.json` | `export_wall_s`, `build_fp32_wall_s`, `build_fp16_wall_s` |

**Population:** On first export/build of an interval, the ONNX/engine are copied to the cache directory.

**Usage:** Before exporting/building a chunk, check if `artifacts/chunk_cache/{model}/int_{start}_{end}/chunk.onnx` exists. If yes, copy to variant-specific path and skip export.

**Export step:** Per-chunk with inline Python (`export_module()` called directly, no subprocess). Model weights are loaded once for all missing chunks in a variant.

**Build step (engines):** All-or-nothing interval cache check:
- If all engines found in cache → copy to variant paths, skip `build_engines()`
- If any missing → run `build_engines()` (whole variant), then populate interval cache

**Rationale for asymmetry:** ONNX export via `export_module()` is a pure Python call; per-chunk export saves subprocess overhead. Engine build requires `trtexec`; doing it per-chunk adds complexity with marginal gain for the typical K=2/K=3 case.

---

## Timing Accounting Added

New fields on `EvaluationResult`:

| Field | Type | Meaning |
|-------|------|---------|
| `export_wall_s` | float | Actual wall time for ONNX export (0.0 if all cache hits) |
| `build_wall_s` | float | Actual wall time for engine build (0.0 if all cache hits) |
| `profile_wall_s` | float | Wall time for C++ / Python TRT profiling |
| `interval_cache_hits` | int | Chunk ONNXes/engines reused from interval cache |
| `interval_cache_misses` | int | Chunk ONNXes/engines built fresh |
| `estimated_cold_export_s` | float | Sum of per-interval export times (cold-cache estimate) |
| `estimated_cold_build_s` | float | Sum of per-interval build times (cold-cache estimate) |
| `estimated_cold_total_s` | float | export + build + profile (full cold-cache estimate) |

`estimated_cold_total_s` allows Fig.5 to report the design-time cost that would have been paid on a cold cache, even when the actual run reused cached intervals.

---

## Progress Printing

`scripts/40_run_fig5_design_time.py`:
- Per-job print at start and end of each `(taskset, algorithm)` pair with `flush=True`
- `--progress-interval-sec` (default 30s) for extra summary lines
- **No changes needed**

`scripts/30_run_yaml_fig4_experiment.py`:
- Per-utilization and per-taskset print lines with `flush=True`
- **No changes needed**

---

## Key Invariants

1. **Mask-level cache takes priority:** `evaluate_mask()` returns early on a mask cache hit — no interval cache lookup needed.
2. **Interval cache is additive:** Removing `artifacts/chunk_cache/` does not break correctness; it only removes the acceleration layer.
3. **Backward-compatible:** Existing eval JSONs are unaffected. New `interval_*` / `*_wall_s` fields default to `0` / `None` when absent.
4. **K=1 masks never reach the interval cache:** The K=1 shortcut in `mask_applicator.py` returns before `evaluate_mask()` is called.
