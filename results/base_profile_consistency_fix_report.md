# Base Profile Consistency Fix

**Date:** 2026-05-09

## Symptom

After running `scripts/21_profile_base_chunks.py`:

```
[warn] Cache import failed (non-fatal): 'ProfilingDB' object has no attribute 'upsert'
```

`scripts/15_compare_k1_timing_semantics.py` reported that generation G used
hardcoded fallback values while RTA analysis G used real table4 values:

```
alexnet   Gen=1.7540  RTA=1.7699  MISMATCH
resnet18  Gen=1.0370  RTA=1.0580  MISMATCH
vgg19     Gen=7.5620  RTA=7.5604  MISMATCH
```

Task periods were therefore derived from approximate WCET values rather than
real Jetson measurements, slightly shifting the utilization distribution.

## Root Cause

`scripts/21_profile_base_chunks.py::_import_to_cache()` called two methods
that do not exist on `ProfilingDB`:

```python
db.upsert(...)   # no such method — correct is db.put() or db.import_from_cpp_result()
db.save()        # no such method — ProfilingDB auto-flushes on every put()
```

`ProfilingDB` already had `import_from_cpp_result(path)` which reads the exact
table4 JSON format produced by `cpp_runtime/build/table4_runner`. The wrong
method names prevented the cache from being populated.

With an empty cache, `_get_base_gpu_wcet_ms()` in `dnn_workload_generator.py`
fell through all three real-data priorities and returned the hardcoded
`_DRY_RUN_BASE_WCET_MS` fallback (alexnet=1.754, resnet18=1.037, vgg19=7.562).

`load_candidate_space()` was unaffected because it reads from `results/table4/`
directly (Priority 2) and never needs the cache.

## Fixes

### Fix 1 — `scripts/21_profile_base_chunks.py`

Replaced the broken `_import_to_cache()` body with a call to the existing,
correct API:

```python
# Before (broken):
db.upsert(model, variant, precision, {...})
db.save()

# After (correct):
if db.import_from_cpp_result(p):
    imported += 1
```

`import_from_cpp_result()` reads model/variant/precision/chunks from the JSON,
calls `put()` (which auto-flushes), and returns `True` on success.

### Fix 2 — `scripts/15_compare_k1_timing_semantics.py`

- Added `_gen_g_source()` helper that reports where `_get_base_gpu_wcet_ms()`
  got its value (profiling cache, evaluation JSON, or hardcoded fallback)
- Replaced stale "analysis sees G=0" / "fallback added in PR" diagnostic text
  with accurate cause-and-fix descriptions covering: missing table4 JSON,
  missing cache, or cache populated but unreadable
- Tolerance check tightened to 0.001 ms

### Fix 3 — Populated cache

Ran `import_from_cpp_result()` against the three existing table4 JSONs to
populate `results/optimization/.profiling_cache.json`.

## Verification

### script 15 after fix

```
alexnet   YES  1.7699  1.7699  22   OK    (profiling cache p99)
resnet18  YES  1.0580  1.0580  14   OK    (profiling cache p99)
vgg19     YES  7.5604  7.5604  46   OK    (profiling cache p99)

RESULT: Generation and analysis G are consistent.
        Schedulability experiments will use real profiled WCET values.
```

### Smoke test (no --allow-equal-wcet-fallback)

```
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --algorithm-set main4 \
    --num-tasksets-override 2 \
    --dry-run \
    --run-name fig4_profiled_smoke
```

```
Result rows with errors: 0
Policy violations: 0
```

### pytest: 53 passed

## Files Changed

| File | Change |
|------|--------|
| `scripts/21_profile_base_chunks.py` | `_import_to_cache()`: use `import_from_cpp_result()` instead of `upsert()`/`save()` |
| `scripts/15_compare_k1_timing_semantics.py` | Added source reporting; updated stale diagnostic messages |
| `results/optimization/.profiling_cache.json` | Populated with real table4 p99 data for alexnet, resnet18, vgg19 |
