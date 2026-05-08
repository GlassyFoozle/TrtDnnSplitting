# Fig.5 K=1 Accounting & Interval Cache Fix — Diagnosis Report

**Date:** 2026-05-08  
**Scope:** K=1 baseline accounting bug, UNI/SS consistency, per-chunk interval engine cache,
script 40 algorithm CLI.

---

## 1. Root Cause: K=1 Counted as `real_profiles`

### Symptom

When running `40_run_fig5_design_time.py` in dry-run mode with 8 tasks:

```
SS-tol-fb: SCHED wall=0.01s masks=8 real=8 cache=0 split=False
SS-opt:    SCHED wall=0.01s masks=8 real=8 cache=0 split=False
UNI-tol-fb: masks=0 real=0
UNI-opt:    masks=0 real=0
```

`real=8` for SS algorithms while `real=0` for UNI was a red flag: in dry-run, `real_profiles`
should always be 0 because no TRT engine is ever built or profiled.

### Cause

`_run_ss_single` and `_run_ss_tol_fb` call `apply_no_split_mask()` for each task as a K=1
initialization step. `evaluate_and_apply_mask` with an all-zero mask short-circuits at the
K=1 baseline path, returning `MaskApplicationResult(cache_hit=False, dry_run=True)`.

`ProfilingStats.update()` checked `result.dry_run` first, so K=1 dry-run results were counted
as `dry_run_evaluations`. But in **live mode** with all-zero base times (fresh clone, no
preflight), the K=1 error path returned `MaskApplicationResult(success=False, error="K=1
baseline unavailable", cache_hit=False, dry_run=False)` — without any flag distinguishing it
from a regular live profile. `ProfilingStats.update()` fell through to `real_profiles += 1`.

### Fix

Added `is_k1_baseline: bool = False` to `MaskApplicationResult`. Both K=1 return paths in
`evaluate_and_apply_mask` now set `is_k1_baseline=True`. `ProfilingStats.update()` checks
this flag first and routes to `baseline_k1_hits`, never to `real_profiles`, `cache_hits`, or
`dry_run_evaluations`. New `ProfilingStats.baseline_k1_hits` counter tracks these separately.

**Expected output after fix:**
```
SS-tol-fb: SCHED wall=0.01s masks=8 real=0 cache=0 k1=8 split=False
SS-opt:    SCHED wall=0.01s masks=8 real=0 cache=0 k1=8 split=False
UNI-tol-fb: masks=8 real=0 cache=0 k1=8
UNI-opt:    masks=8 real=0 cache=0 k1=8
```

---

## 2. UNI / SS K=1 Accounting Inconsistency

### Cause

`_run_ss_single` and `_run_ss_tol_fb` call `apply_no_split_mask` per task.
`_run_uni_single`, `_run_uni_tol`, and `_run_uni_tol_fb` did NOT, so they reported
`masks=0` even though they implicitly start from the no-split state.

This was partially a historical regression: `_run_uni_tol_fb` once had a K=1 init loop
that was reverted in Session 2 to avoid a G=0 `IndexError` in `convert_UNI_to_SS`.
That IndexError was fixed in Session 1 (`task.py` c_list sizing fix).

### Fix

Added explicit K=1 init loops (identical to `_run_ss_tol_fb`) to:
- `_run_uni_single` — before `convert_task_SS_to_UNI`
- `_run_uni_tol` — before `convert_task_SS_to_UNI`
- `_run_uni_tol_fb` — before `convert_task_SS_to_UNI`

UNI heu/opt go through `_paper_no_split_gate_uni` → `_run_uni_single`, which now includes K=1
init, so they are also consistent.

---

## 3. Per-Chunk Interval Engine Cache (All-or-Nothing → Per-Chunk)

### Previous Behavior

`_check_interval_engine_cache()` copied per-chunk engines from interval cache. But if any
engine was missing, `build_engines()` was called for the **whole variant** — rebuilding all K
chunks even if K−1 were already cached.

### Fix

Replaced `_check_interval_engine_cache + build_engines` with `_build_engines_with_interval_cache`:
- Iterates each chunk independently
- Cache hit → copies from `artifacts/chunk_cache/{model}/int_{start}_{end}/chunk_{prec}.engine`
- Cache miss → calls `build_single_engine()` (new function in `compiler.py`) per-chunk via
  direct trtexec invocation, then populates interval cache + writes `build_{precision}_wall_s`
  to `timing.json`
- Fails fast on any per-chunk build failure

New `build_single_engine(onnx_path, engine_path, precision, dry_run)` in `compiler.py`:
- Finds trtexec via `$TRTEXEC` env var (default `/usr/src/tensorrt/bin/trtexec`)
- Returns `(success: bool, wall_s: float)` for cold-cache estimation
- Logs per-build output to `artifacts/logs/{stem}_{precision}.log`

### New accounting fields

| Field | Where | Meaning |
|-------|-------|---------|
| `interval_onnx_cache_hits/misses` | `EvaluationResult`, `MaskApplicationResult`, `ProfilingStats` | ONNX export cache hits/misses |
| `interval_engine_cache_hits/misses` | Same | Engine build cache hits/misses |
| `interval_engine_build_wall_s` | Same | Actual wall time for per-chunk engine builds |

`interval_cache_hits/misses` (combined) are preserved for backward compatibility.

---

## 4. Script 40 Algorithm CLI Improvements

### Added `--algorithm-set`

```bash
python scripts/40_run_fig5_design_time.py \
    --algorithm-set full8 \
    --models alexnet resnet18 vgg19 \
    --num-tasksets 50
```

Sets: `main4`, `full8`, `ss_only`, `uni_only`.

### Label Aliases

Paper-style labels accepted:

| Paper label | Canonical |
|-------------|-----------|
| `SS-tol-fb` | `ss:tol-fb` |
| `UNI-opt`   | `uni:opt` |
| `SS-heu`    | `ss:heu` |
| … | … |

### Validation

Unknown model or algorithm raises `ValueError` with a clear message listing known algorithms.

### Progress Output

Added `k1=` to per-taskset summary line:
```
  SCHED wall=0.05s masks=8 real=0 cache=0 k1=8 skipped=0 split=False
```

---

## 5. Fig.4 Dry-Run Near-1 Schedulability

The near-1 schedulability observed in Fig.4 dry-run is **expected behavior** under the
following conditions:

- Dry-run uses `dag_aligned_full` base chunk times from the profiling cache or
  `_DRY_RUN_BASE_WCET_MS` fallback (alexnet=1.754ms, resnet18=1.037ms, vgg19=7.562ms).
- With `--utilization 0.6` to `1.0`, total GPU utilization = 0.6–1.0. At util=0.6 with
  periods randomly drawn from [1ms, 10000ms], most tasks have very long periods and the
  GPU block is tiny relative to the period. SS/UNI RTA naturally finds these schedulable
  without splitting.
- The `_detect_rta_overload` gate fires before any algorithm runs when the no-split
  taskset is trivially schedulable. This is logged as `SCHED no-split masks=0`.

Near-1 schedulability becomes a concern only if base WCET values are incorrect. Verification:
```bash
conda run -n trt python -c "
from src.optimization.profiling_db import ProfilingDB
from pathlib import Path
db = ProfilingDB(Path('results/optimization/.profiling_cache.json'))
db.import_all_cpp_results(Path('.'))
for m in ['alexnet','resnet18','vgg19']:
    print(m, db.get(m, 'dag_aligned_full', 'fp32'))
"
```

---

## 6. Test Coverage Added

| File | Tests | What they cover |
|------|-------|----------------|
| `tests/test_k1_accounting.py` | 9 | K=1 flag, ProfilingStats routing, no TRT call |
| `tests/test_fig5_cli.py` | 12 | Label aliases, algorithm sets, per-chunk engine cache |

Total tests: 46 (was 25).

---

## 7. Files Changed

| File | Change |
|------|--------|
| `src/integration/mask_applicator.py` | `is_k1_baseline`, split interval fields |
| `src/integration/dnn_algorithm_runner.py` | `baseline_k1_hits`, UNI K=1 inits, split interval accounting |
| `src/optimization/config_evaluator.py` | Split interval fields, `_build_engines_with_interval_cache` |
| `src/optimization/compiler.py` | `build_single_engine()` |
| `scripts/40_run_fig5_design_time.py` | `--algorithm-set`, aliases, `k1=` progress, new CSV columns |
| `tests/test_k1_accounting.py` | New (9 tests) |
| `tests/test_fig5_cli.py` | New (12 tests) |
