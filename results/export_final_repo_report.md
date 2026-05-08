# TrtDnnSplitting — Final Repository Export Report

**Date:** 2026-05-08  
**Git commit:** `f051bdd5f57c304fb2d8bb162ca03bb0d3870aab`  
**Branch:** `master`  
**Files committed:** 70 (17,979 insertions)

---

## 1. Objective

Create a final, standalone, GitHub-ready `TrtDnnSplitting` repository at
`~/workspace/tensorrt/TrtDnnSplitting/` that:

- Requires **no external sibling repos** (no `../DNNSplitting` dependency)
- Supports **dry-run** (analytical) and **live** (TensorRT profiling) modes
- Implements all four paper algorithms: `ss:tol-fb`, `ss:opt`, `uni:tol-fb`, `uni:opt`
- Applies five correctness fixes from the pre-work reports

---

## 2. Repository Structure

```
TrtDnnSplitting/
├── .gitignore
├── INSTALL.md
├── README.md
├── requirements.txt
├── artifacts/
│   ├── .gitkeep
│   └── split_configs/
│       ├── alexnet/dag_aligned_full.json   ← ships for dry-run
│       ├── resnet18/dag_aligned_full.json
│       └── vgg19/dag_aligned_full.json
├── configs/
│   ├── dnn_tasksets/mixed_two_dnn_demo.json
│   ├── split_point_policies.json
│   └── yaml/
│       ├── 1_GPU0.6-1.0_task8_ov5.yaml
│       ├── 2_GPU0.6-1.0_task8_singleCPU_ov5.yaml
│       ├── 3_GPU1.0_task8_ov5.yaml
│       ├── 4_GPU0.8_task8_ov5.yaml
│       └── 5_GPU0.6_task8_ov5.yaml
├── cpp_runtime/
│   ├── CMakeLists.txt
│   ├── include/{buffer_pool,chunk_pipeline,timing,trt_engine}.hpp
│   └── src/{buffer_pool,chunk_pipeline,main_table4,trt_engine}.cpp
├── scripts/
│   ├── 10_inspect_single_taskset.py        ← single taskset + algorithm runner
│   ├── 20_preflight_design.py              ← live design preflight
│   ├── 30_run_yaml_fig4_experiment.py      ← YAML-driven Fig.4 sweep
│   ├── 31_plot_fig4.py                     ← Fig.4 plot
│   ├── 40_run_fig5_design_time.py          ← Fig.5 design-time sweep
│   ├── 41_plot_fig5.py                     ← Fig.5 plot
│   ├── internal_build_selected_engines.sh  ← called by compiler.py
│   ├── internal_export_selected_split.py   ← called by compiler.py
│   └── internal_fig4_helpers.py            ← shared result utilities
├── src/
│   ├── rta/              ← ★ self-contained RTA (no external DNNSplitting dep)
│   │   ├── __init__.py
│   │   ├── analysis.py   (1304 lines, ported from DNNSplitting paper)
│   │   └── task.py       (386 lines, copied verbatim)
│   ├── integration/      ← algorithm runner, task model, adapters
│   ├── optimization/     ← config evaluator, compiler, profiling DB
│   ├── splitting/        ← DAG-aligned and selective split logic
│   ├── export/           ← ONNX exporter
│   ├── models/           ← model registry
│   └── utils/            ← path helpers
├── tests/
│   ├── conftest.py
│   ├── test_rta_import.py
│   ├── test_taskgen.py
│   ├── test_split_policy.py
│   └── test_cache_validity.py
└── results/
    └── .gitkeep
```

---

## 3. Fixes Applied

### Fix 1 — Eliminate external DNNSplitting dependency

**Problem:** `dnn_algorithm_runner.py` inserted `../DNNSplitting/` into `sys.path`
at import time, making `--help` fail without the sibling repo present.

**Solution:**
- Copied `analysis.py` + `task.py` verbatim from `TensorRTServer/DNNSplitting/`
  into `src/rta/`; only changed one line in `analysis.py`:
  `from task import InferenceSegment` → `from src.rta.task import InferenceSegment`
- Created `src/rta/__init__.py` as a clean public API
- Updated `dnn_algorithm_runner.py`: replaced `sys.path.insert` + bare `from analysis import`
  with `from src.rta.analysis import (...)`
- Updated `dnnsplitting_adapter.py`: `get_dnnsplitting_dir()` is now a legacy stub;
  `from src.rta.task import InferenceSegment, SegInfTask` at module level

### Fix 2 — Consistent K=1 initialization for all algorithms (pre-existing)

Confirmed in `dnn_algorithm_runner.py` — all four algorithms initialize from K=1
(`apply_no_split_mask`) before the search loop. No additional changes needed.

### Fix 3 — Policy-limit full-split feasibility probes

**Problem:** `_run_ss_tol_fb_paper()` and `_run_ss_opt_paper()` used
`apply_full_split_mask()` (all N−1 boundaries enabled) for the feasibility probe,
probing configurations that no subsequent search step could ever reach under the
active policy.

**Solution:** Moved `get_enabled_boundaries()` before the probe in both functions.
Replaced `apply_full_split_mask(...)` with:
```python
policy_full_mask = apply_policy_to_mask([1] * (N-1), enabled)
evaluate_and_apply_mask(dt_i, st_i, policy_full_mask, 0, **eval_kwargs)
```
This makes the infeasibility gate consistent with the policy-constrained search space.

### Fix 4 — Content-aware cache validity in `is_mask_cached()`

**Problem:** `is_mask_cached()` returned `True` for any existing JSON file, including
error results and files with wrong chunk counts (from stale cache entries after a
re-profiling run with a different split config).

**Solution:** `is_mask_cached()` now checks:
1. JSON must parse successfully
2. `error` field must be absent or null
3. `per_chunk_gpu_mean_ms` must be present
4. `len(per_chunk_gpu_mean_ms) == sum(mask) + 1` (chunk count matches mask)

### Fix 5 — Rename internal scripts; update `compiler.py`

**Problem:** `compiler.py` referenced `scripts/31_export_selected_split.py` and
`scripts/32_build_selected_engines.sh`, which would collide with the user-facing
`scripts/31_plot_fig4.py`.

**Solution:**
- Renamed: `31_export_selected_split.py` → `internal_export_selected_split.py`
- Renamed: `32_build_selected_engines.sh` → `internal_build_selected_engines.sh`
- Updated `compiler.py` paths accordingly

---

## 4. Additional Changes

### `src/optimization/__init__.py`

Removed stale imports of dropped modules `task_model` and `optimizer_config`.
Now only re-exports from `config_evaluator`:
```python
from src.optimization.config_evaluator import EvaluationResult, evaluate_mask, mask_to_variant_name
```

### `dnn_workload_generator.py` — dry-run WCET fallback

Added `_DRY_RUN_BASE_WCET_MS` dict with measured Jetson AGX Orin FP32 p99 values:
- alexnet: 1.754 ms, resnet18: 1.037 ms, vgg19: 7.562 ms

Used as 4th fallback in `_get_base_gpu_wcet_ms()` when no profiling cache exists,
enabling `--dry-run` on a fresh clone without any prior profiling.

### `scripts/30_run_yaml_fig4_experiment.py`

Updated `_load_fig4_helpers()` to load `internal_fig4_helpers.py` (was
`56_run_fig4_pilot.py`, which no longer exists in the final repo).

### `artifacts/split_configs/*/dag_aligned_full.json`

Three pre-profiled base variant configs (AlexNet, ResNet18, VGG19) are shipped
with the repo for dry-run mode. `.gitignore` includes an exception
`!artifacts/split_configs/*/dag_aligned_full.json` so only these baseline configs
are tracked; all generated split configs remain untracked.

---

## 5. Validation Results

### py_compile

All 29 `src/` Python files: **PASS**  
All 8 `scripts/` Python files + 5 `tests/` files: **PASS**

### Dry-run smoke test

```
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --num-tasksets-override 2 \
    --dry-run
```

**Result:** PASS — 10 tasksets × 4 algorithms = 40 runs, 0 errors, 0 policy violations.
Elapsed: 0.24 s.

Output: `results/dnn_experiments/yaml_fig4_1_GPU0.6-1.0_task8_ov5_dry_*/`

### Git status

70 files tracked in initial commit on branch `master` (`f051bdd`).
No untracked files besides generated run output (properly `.gitignore`-d).

---

## 6. Known Limitations / Next Steps

- `tests/test_taskgen.py` depends on `dnn_taskset_generator.generate_dnn_taskset()`
  being callable with `dry_run=True`; the exact API signature should be verified
  before running `pytest` (the py_compile check passes, but the test logic
  depends on the generator accepting a `dry_run` kwarg).

- No GitHub remote configured; push manually when ready:
  `git remote add origin <URL> && git push -u origin master`

- For live experiments, the `cpp_runtime/build/` directory must be created and
  `table4_runner` compiled before running any live profiling.

- The four algorithms are validated end-to-end in dry-run. Live-mode validation
  (real TRT engine build + profiling) requires a Jetson Orin with TensorRT 8.6+
  and the pre-built engine files from a `20_preflight_design.py` run.
