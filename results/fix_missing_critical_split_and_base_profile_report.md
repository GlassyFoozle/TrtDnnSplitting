# Diagnosis and Fix: Missing `critical_split` and Base Profiling Workflow

**Date:** 2026-05-09  
**Commits:** follows `fbfa1a6` (K=1 accounting / G=0 fix)

---

## Symptom

Overnight live Fig.4 run
(`results/dnn_experiments/fig4_full8_yaml1_majorblocks_live_n50_uncapped/`)
completed in seconds rather than hours. The summary showed:

```
global_profile_budget_used=0
avg_real_profiles=0
split_triggered_pct=0
analysis_error_count=1554
```

All 1554 errors were:
```
ModuleNotFoundError: No module named 'src.splitting.critical_split'
```

---

## Root Cause

### 1. `src/splitting/critical_split.py` was never ported

`src/splitting/dag_aligned_split.py` imports `critical_split` at module level:
```python
from src.splitting.critical_split import make_critical_full_chunks  # line 19
```

The file `src/splitting/critical_split.py` was present in the prototype repo
(`TensorRTServer/trt_split_runtime_baseline/`) but was never copied into
`TrtDnnSplitting/`.

**Effect:** Any algorithm invocation that triggered a split evaluation (K>1) caused
an immediate `ModuleNotFoundError` when `dag_aligned_split` was first imported.
Only tasksets that were schedulable at K=1 (no split needed) produced a result.
This explains why ~32% showed SCHED at U=0.70 (the K=1 no-split subset) while
all others errored.

### 2. Equal-WCET fallback was silently active (previous session)

Prior fix `fbfa1a6` added an unconditional equal-weight WCET/N fallback as Priority 3
in `load_candidate_space()`. This caused dry-run tests to pass but masked the fact that
no real per-chunk profiling data existed. The fallback was appropriate for CI but should
not have been the default for production runs.

---

## Fixes Applied

### Fix 1: Port `src/splitting/critical_split.py`

Created `src/splitting/critical_split.py` (317 lines) from the prototype. Defines:

| Model | Strategy | Chunks |
|-------|----------|--------|
| AlexNet | MaxPool boundaries (pure chain → group at natural stages) | 5 |
| VGG19 | Pool stage boundaries | 7 |
| ResNet18 | One chunk per critical DAG node (module-only analysis) | 14 |

Key API:
```python
ChunkSpec = Tuple[nn.Module, Tuple[int, ...], Tuple[int, ...], str, str]

def make_critical_full_chunks(model_name: str, model: nn.Module) -> List[ChunkSpec]:
    ...

def available_critical_full_models() -> List[str]:
    ...
```

**Tests:** `tests/test_critical_split.py` (7 tests) — all pass.

### Fix 2: Write `scripts/21_profile_base_chunks.py` (new mandatory script)

Workflow:
1. Verify `cpp_runtime/build/table4_runner` exists (print build instructions and exit if not)
2. Verify `artifacts/split_configs/<model>/dag_aligned_full.json` exists
3. Export ONNX for each base chunk (skipped if already present)
4. Build per-chunk TRT engines via trtexec (skipped if already present)
5. Run `cpp_runtime/build/table4_runner` to measure GPU timing
6. Write `results/table4/<model>_cpp_dag_aligned_full_<precision>.json`
7. Import results into `results/optimization/.profiling_cache.json`

Options: `--models`, `--precision`, `--warmup`, `--iters`, `--force`, `--skip-cpp`

### Fix 3: Make equal-WCET fallback opt-in in `candidate_space.py`

`load_candidate_space()` now raises `RuntimeError` by default when no profiling data
is found, with actionable instructions:

```
RuntimeError: Missing dag_aligned_full profiling data for 'alexnet' (fp32).
Expected: results/table4/alexnet_cpp_dag_aligned_full_fp32.json

Run base chunk profiling first:
  conda run -n trt python scripts/21_profile_base_chunks.py --models alexnet --precision fp32

Or pass --allow-equal-wcet-fallback to scripts 30/40 for development use (produces approximate results).
```

Added `allow_equal_wcet_fallback: bool = False` parameter to:
- `load_candidate_space()` in `candidate_space.py`
- `load_dnn_taskset()` in `dnn_taskset_loader.py`
- `generate_dnn_taskset()` in `dnn_taskset_generator.py`
- `run_dnn_rta_algorithm()` in `dnn_algorithm_runner.py`
- `--allow-equal-wcet-fallback` CLI flag in scripts 30 and 40

---

## Verification

### Smoke test (Task 5)

```
conda run -n trt python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --algorithm-set full8 \
    --num-tasksets-override 2 \
    --dry-run \
    --allow-equal-wcet-fallback \
    --run-name fig4_full8_smoke_post_fix
```

Result:
```
Elapsed algorithm time: 3.01s
Result rows with errors: 0
Policy violations: 0
```

**`analysis_error_count=0`** — no `ModuleNotFoundError`, all algorithms complete.

### pytest

```
conda run -n trt python -m pytest tests/ -q
```

53 tests pass (46 prior + 7 new `test_critical_split.py` tests).

---

## Workflow Changes

| Script | Before | After |
|--------|--------|-------|
| `20_preflight_design.py` | "Run before any experiment" | Design preflight only (validates split-point engine builds; does NOT profile base chunks) |
| `21_profile_base_chunks.py` | Did not exist | **Mandatory first step** on any fresh clone — profiles base chunks and writes `results/table4/` |
| `30_run_yaml_fig4_experiment.py` | Silently used equal WCET if no profiling data | Fails with clear error unless profiling data exists or `--allow-equal-wcet-fallback` is passed |
| `40_run_fig5_design_time.py` | Same | Same |

---

## Files Changed

| File | Change |
|------|--------|
| `src/splitting/critical_split.py` | Created (ported from prototype) |
| `tests/test_critical_split.py` | Created (7 tests) |
| `scripts/21_profile_base_chunks.py` | Created (base chunk profiler) |
| `src/optimization/candidate_space.py` | Fallback now opt-in; default raises RuntimeError |
| `src/integration/dnn_taskset_loader.py` | Added `allow_equal_wcet_fallback` parameter |
| `src/integration/dnn_taskset_generator.py` | Added `allow_equal_wcet_fallback` parameter |
| `src/integration/dnn_algorithm_runner.py` | Added `allow_equal_wcet_fallback` parameter |
| `scripts/30_run_yaml_fig4_experiment.py` | Added `--allow-equal-wcet-fallback` CLI flag |
| `scripts/40_run_fig5_design_time.py` | Added `--allow-equal-wcet-fallback` CLI flag |
| `docs/EXPERIMENTS.md` | Updated preflight section; added script 21 workflow |
| `docs/TROUBLESHOOTING.md` | Added entries for critical_split and missing profiling data |
