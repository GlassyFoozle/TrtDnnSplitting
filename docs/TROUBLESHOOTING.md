# TrtDnnSplitting — Troubleshooting

## ImportError / ModuleNotFoundError

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Fix:** Always run from the repo root, not from inside `src/` or `scripts/`:
```bash
cd ~/workspace/tensorrt/TrtDnnSplitting
conda run -n trt python scripts/30_run_yaml_fig4_experiment.py ...
```

---

## FileNotFoundError: dag_aligned_full config not found

**Symptom:**
```
FileNotFoundError: dag_aligned_full config not found for 'resnet50'.
Run: conda run -n trt python scripts/26_generate_dag_aligned_configs.py --models resnet50
```

**Cause:** The model is not in `artifacts/split_configs/`. Only alexnet, resnet18, vgg19 ship
with the repo.

**Fix for dry-run:** Use only `--models alexnet resnet18 vgg19`.

**Fix for a new model:** Run `20_preflight_design.py` to generate and profile the configs:
```bash
python scripts/20_preflight_design.py --models resnet50 --precision fp32
```

---

## All algorithms report SCHED with 0 masks evaluated

**Symptom:** `SS_ours SCHED no-split masks=0 dry=0 cache=0 real=0`

**Cause:** The `_detect_rta_overload` check detected that even a fully loaded taskset is
trivially schedulable (very low utilization). This is correct behavior.

**Normal at low utilization.** Increase `--utilization` or lower the taskset period range.

---

## Result rows with errors: N > 0

**Symptom:** `Result rows with errors: 1` at end of Fig.4 run.

**Check:** `cat results/dnn_experiments/<run>/all_results.json | python -m json.tool | grep -A5 '"error"'`

**Common causes:**

| Error | Cause | Fix |
|-------|-------|-----|
| `list index out of range` (UNI) | G=0 tasks in G→SS back-conversion | Fixed in `src/rta/task.py` (present in repo) |
| `NumeratorExplosionError` | RTA numerics overflow | Expected; result marked UNSCHEDULABLE |
| `engine build failed` | Live mode, TRT not available | Use `--dry-run` |

---

## IndexError in convert_UNI_to_SS

**Symptom:** `IndexError: list index out of range` at `task.py:229` in `convert_UNI_to_SS`

**Cause (pre-fix):** When G=0 (no live profiling), `convert_SS_to_UNI` creates C sources
with indices 0 and 1 (for `cpu_pre_ms` and `cpu_post_ms`), but `c_list` was sized using
`original_segment_count+1 = 1`, missing index 1.

**Status:** Fixed in `src/rta/task.py` — `c_list` is now sized as
`max(original_segment_count+1, max_c_idx+1)`.

**Scope:** This fix only affects dry-run / fresh-clone mode where no live profiling has
been run. In live mode, G blocks are non-zero and the G=0 path is never triggered.

---

## pytest failures

**Symptom:** `ImportError` or `AttributeError` in tests.

**Check that tests run from repo root:**
```bash
cd ~/workspace/tensorrt/TrtDnnSplitting
conda run -n trt python -m pytest tests/ -q
```

**If `_get_base_gpu_wcet_ms` returns None:**
`_DRY_RUN_BASE_WCET_MS` only has entries for alexnet, resnet18, vgg19. For other models,
add an entry or run live profiling first.

---

## Live mode: TRT engine build fails

**Symptom:** `engine build failed` or `trtexec not found`

**Fix:**
1. Confirm you are on Jetson AGX Orin with TensorRT 8.6+
2. Run `20_preflight_design.py` first to build the base engines
3. Check `cpp_runtime/build/table4_runner` is compiled: see INSTALL.md §4

---

## Schedulability ratio CSV is empty / has NaN

**Symptom:** `schedulability_ratio.csv` has `NaN` for some algorithms.

**Cause:** All tasksets for that utilization returned errors. Check `all_results.json`.

---

## Plotting: "No data to plot"

**Symptom:** `31_plot_fig4.py` exits with "No data to plot" or produces an empty figure.

**Fix:** Ensure `schedulability_ratio.csv` is non-empty:
```bash
head -5 results/dnn_experiments/<run>/schedulability_ratio.csv
```
If empty, the experiment run failed early. Check `summary.md` for error counts.

---

## Policy violations > 0

**Symptom:** `Policy violations: N` at end of run.

**Cause:** An algorithm applied a split at a boundary disabled by the active policy.

**This should never happen.** File an issue with the run name and config.

---

## Interval cache ONNX is stale / incorrect

**Symptom:** After changing a model or base chunk definitions, cached chunk ONNXes produce
wrong timings.

**Cause:** `artifacts/chunk_cache/{model}/int_{start}_{end}/chunk.onnx` was built from an
older version of the model weights or chunk boundaries.

**Fix:** Delete the affected interval cache directories:
```bash
rm -rf artifacts/chunk_cache/<model>/int_<start>_<end>/
```
Or wipe the entire model's interval cache:
```bash
rm -rf artifacts/chunk_cache/<model>/
```
The interval cache is additive — removing it does not break correctness, only cache warm-up time.

---

## estimated_cold_total_s is 0.0 in Fig.5 CSV

**Symptom:** `mean_total_estimated_cold_s = 0.0` in `fig5_design_time_summary.csv`.

**Cause:** No `timing.json` files exist under `artifacts/chunk_cache/{model}/` because:
- This is the first live run (interval timing data is populated during live runs, not dry-runs).
- Or the interval cache was cleared before the run.

**Normal in dry-run mode.** In live mode, timing files are written on the first new export/build
per interval. On subsequent runs, `estimated_cold_total_s` is populated.

---

## Memory / OOM during live profiling

**Symptom:** `RuntimeError: CUDA out of memory` during TRT engine build.

**Fix:** Run models one at a time: `--models alexnet` then `--models resnet18` separately.
Avoid running multiple Fig.4 experiments in parallel on the same device.
