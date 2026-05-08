# Fig.4 K=1 Timing Semantics Diagnosis

**Date:** 2026-05-08  
**Scope:** Root cause of near-1.0 dry-run schedulability, comparison with prototype,
fix in `candidate_space.py`, and validation of post-fix behaviour.

---

## 1. Symptom

Running `scripts/30_run_yaml_fig4_experiment.py` in dry-run mode on a fresh clone
(no preflight profiling) produced near-1.0 schedulability for all algorithms at all
utilizations, with zero dry-run evaluations and no splitting triggered:

```
[1/50] …/taskset_000.json
  SS-tol-fb      SCHED no-split masks=  8 dry=  0 cache=0 real=0
  UNI-tol-fb     SCHED no-split masks=  8 dry=  0 cache=0 real=0
  SS-opt         SCHED no-split masks=  8 dry=  0 cache=0 real=0
  UNI-opt        SCHED no-split masks=  8 dry=  0 cache=0 real=0
```

`masks=8, k1=8, dry=0` means all eight evaluations were K=1 baseline
initialisations — no K>1 masks were ever tried.

Compare with the prototype (`TensorRTServer/trt_split_runtime_baseline/`) at U=0.90:

| Algorithm | Sched ratio | avg_masks | split_triggered_pct |
|-----------|-------------|-----------|---------------------|
| SS_ours   | 0.66        | 21.44     | 96%                 |
| UNI_ours  | 0.52        | 26.36     | 96%                 |

---

## 2. Root Cause

### 2a. Timing source asymmetry

Task **generation** (`dnn_workload_generator.py`) uses `_get_base_gpu_wcet_ms()` which
falls back to `_DRY_RUN_BASE_WCET_MS` (non-zero Jetson reference values) when no
profiling data is present:

```python
_DRY_RUN_BASE_WCET_MS = {
    "alexnet":  1.754,   # p99 FP32 ms, Jetson AGX Orin
    "resnet18": 1.037,
    "vgg19":    7.562,
}
```

Periods are derived as `T = G / u_i` (or via G_ratio for dnnsplitting mode),
so tasks get realistic periods even in a fresh clone.

Task **analysis** (`load_candidate_space()` in `candidate_space.py`) loaded chunk
times from two sources:
1. ProfilingDB object (provided by live pipeline)
2. `results/table4/<model>_cpp_dag_aligned_full_<precision>.json`

On a fresh clone, neither source is available. `per_chunk_means` and `per_chunk_p99`
both defaulted to `[0.0] * N`. This means:

```
base_chunk_times_ms = [0.0, 0.0, …, 0.0]   (N zeros)
```

### 2b. K=1 patching with zero base times

`apply_no_split_mask()` calls `evaluate_and_apply_mask(mask=[0]*N-1)`. The K=1 branch
calls `_apply_mask_to_chunk_times([0.0]*N, [0]*N-1)` → `chunk_times = [0.0]`. Then
`_patch_seg_task` sets:

```python
seg.G_block_list = [0.0]
seg_task.G = 0.0
seg_task.max_G_block = 0.0
```

With G=0 for all tasks, every taskset is trivially schedulable at K=1 — the RTA
recurrence has no GPU interference term. `_run_ss_tol_fb` and `_run_uni_tol_fb` exit
the main loop immediately after K=1 init without attempting any splits.

### 2c. Why the prototype was unaffected

The prototype (`TensorRTServer/trt_split_runtime_baseline/`) had real Jetson profiling
files committed under `results/table4/`:

| Model    | sum_p99 (ms) | Matches _DRY_RUN_BASE_WCET_MS? |
|----------|-------------|-------------------------------|
| alexnet  | 1.7535       | ≈ 1.754 ✓                     |
| resnet18 | 1.0375       | ≈ 1.037 ✓                     |
| vgg19    | 7.5617       | ≈ 7.562 ✓                     |

`_DRY_RUN_BASE_WCET_MS` was derived from these exact measurements (p99 column,
rounded). The final repo (`TrtDnnSplitting/`) does not include these files, so it
fell through to the all-zero default.

---

## 3. Fix

Added **Priority 3** in `load_candidate_space()` (`src/optimization/candidate_space.py`):

```python
# Priority 3: dry-run fallback — equal per-chunk allocation from known Jetson reference
# values. Without this, fresh-clone dry-run sees G=0 while task generation used
# _DRY_RUN_BASE_WCET_MS for periods — an inconsistency that makes every taskset
# trivially schedulable at K=1 with no splitting triggered.
if all(t == 0.0 for t in per_chunk_means):
    _DRY_RUN_WCET_MS = {"alexnet": 1.754, "resnet18": 1.037, "vgg19": 7.562}
    wcet = _DRY_RUN_WCET_MS.get(model_name.lower())
    if wcet is not None:
        per_chunk_means = [wcet / n] * n
        per_chunk_p99 = [wcet / n] * n
```

This makes `sum(base_chunk_times_ms) == _DRY_RUN_BASE_WCET_MS[model]` — the same
value used for period derivation. Generation G == analysis G in dry-run mode.

**Limitation**: the equal-weight distribution (`WCET/N` per chunk) is an approximation.
Real profiling shows non-uniform per-chunk times, which matters when the optimal split
placement depends on which chunks are heavier. For `major_blocks` policy this is
partially mitigated by restricting boundaries to architecturally meaningful locations.
Live profiling (`scripts/20_preflight_design.py`) remains the correct path for
paper-accurate results.

---

## 4. Validation

### 4a. Timing consistency check

Running `scripts/15_compare_k1_timing_semantics.py` after the fix:

```
Model         Table4    Gen G (ms)    RTA G (ms)     N    G/N (ms)  Consistent
alexnet       no            1.7540        1.7540    22    0.079727          OK
resnet18      no            1.0370        1.0370    14    0.074071          OK
vgg19         no            7.5620        7.5620    46    0.164391          OK

RESULT: Generation and analysis G are consistent.
```

### 4b. Smoke test: 10 tasksets at U=0.90

```
[1/10] …/taskset_000.json
  SS-tol-fb      SCHED split    masks=  11 dry=   3 cache=  0 real=  0
  UNI-tol-fb     SCHED split    masks=  11 dry=   3 cache=  0 real=  0
  SS-opt         MISS  no-split masks=  21 dry=   5 cache=  0 real=  0
  UNI-opt        MISS  split    masks=  26 dry=  15 cache=  0 real=  0
…
```

Schedulability ratios at U=0.90 over 10 tasksets:

| Algorithm  | Sched ratio | split_triggered_pct | avg_masks |
|------------|-------------|---------------------|-----------|
| SS-tol-fb  | 1.00        | 100%                | 16.1      |
| UNI-tol-fb | 0.70        | 100%                | 15.3      |
| SS-opt     | 0.30        | 50%                 | 443.9     |
| UNI-opt    | 0.00        | 60%                 | 244.5     |

Qualitatively matches the prototype direction (SS > UNI; tol-fb > opt in this dry-run
approximation). The OPT algorithms search hundreds of candidates, confirming the split
search is now engaged.

### 4c. Tests

All 46 tests pass unchanged after the fix (the two tests that explicitly set
`base_chunk_times_ms = [0.0]*N` to test the K=1 live-mode error path are unaffected
since they override the loaded values directly).

---

## 5. Known Limitations of Equal-Weight Dry-Run

With all chunks at `WCET/N`, every split boundary divides execution time exactly
equally — there is no "heavy chunk" to target. This means:

- **OPT** searches many equivalent candidate masks and may fail to improve on tol-fb
  because any K-chunk split gives K×(WCET/K) = WCET/K max block, regardless of
  boundary placement.
- **tol-fb** applies its tolerance criterion against max_G_block. With uniform chunks
  a small K increase consistently reduces max_G_block, so tol-fb tends to succeed.
- The relative ordering of algorithms will differ from live-profiling results where
  vgg19's real 46-chunk non-uniform distribution provides meaningful optimisation targets.

For paper-accurate Fig.4 results, run with live profiling (Jetson AGX Orin, FP32):
```bash
python scripts/20_preflight_design.py --models alexnet resnet18 vgg19 --precision fp32
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --run-name fig4_live
```

---

## 6. Files Changed

| File | Change |
|------|--------|
| `src/optimization/candidate_space.py` | Added Priority 3 fallback (dry-run equal-weight distribution) |
| `scripts/15_compare_k1_timing_semantics.py` | New diagnostic script |
| `results/fig4_k1_semantics_diagnosis_report.md` | This report |
