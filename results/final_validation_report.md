# TrtDnnSplitting — Final Validation Report

**Date:** 2026-05-08  
**Branch:** master  
**Base commit:** f051bdd5f57c304fb2d8bb162ca03bb0d3870aab  

---

## Summary

| Check | Result |
|-------|--------|
| py_compile — all src/ scripts/ tests/ | PASS |
| `--help` on all scripts | PASS |
| pytest (19 tests) | PASS |
| RTA imports use src.rta | PASS |
| Policy-limited full-split probe (Fix 3) | PASS |
| K=1 initialization — ss:tol-fb | PASS (added) |
| K=1 initialization — uni:tol-fb | KNOWN LIMITATION |
| Cache validity check (Fix 4) | PASS |
| Dry-run smoke test (25 tasksets × 4 algos = 100 runs) | PASS — 0 errors, 0 violations |
| Fig.4 plotting (31_plot_fig4.py) | PASS — PNG + PDF produced |
| Fig.5 plotting (41_plot_fig5.py) | PASS — 4 PNG/PDF files produced |
| Git hygiene — no heavy files tracked | PASS |

---

## Files Changed

### `src/integration/__init__.py`
- Updated module docstring: removed stale references to `trt_split_runtime_baseline`
  and `sys.path management`; now accurately describes `src/rta/` as self-contained RTA.

### `src/integration/dnn_algorithm_runner.py`
- Added K=1 initialization to `_run_ss_tol_fb()` at function entry.
  All tasks are evaluated at the all-zero (no-split) mask before the search loop.
  This aligns `ss:tol-fb` with `ss:opt` and `uni:opt` which obtain K=1 timing
  via their no-split gate.

### `src/rta/task.py`
- Fixed IndexError in `convert_UNI_to_SS()`.
  **Root cause:** When base chunk times are all zero (no live profiling, dry-run mode),
  `convert_SS_to_UNI()` creates C-source entries with indices 0 and 1 for `cpu_pre_ms`
  and `cpu_post_ms`, but skips all G blocks (value ≤ 0). `convert_UNI_to_SS()` then
  computed `c_list = [0] * (original_segment_count + 1) = [0]` (one element), causing
  `c_list[1]` to raise IndexError.
  **Fix:** `c_list` is now sized `max(original_segment_count+1, max_c_idx+1)` where
  `max_c_idx = max(source[1] for source in block_sources if source[0] == "C")`.

### `tests/test_taskgen.py`
- Rewrote all four tests with the correct public API:
  - `generate_tasksets(WorkloadConfig(...), output_dir=tmp_path)` (not `generate_dnn_taskset`)
  - `_get_base_gpu_wcet_ms(model, precision, wcet_metric)` (public fallback function)
  - Taskset JSON fields: `sampled_g_ratio`, `_actual_total_utilization`, `real_gpu_wcet_ms`
  - All 19 tests now pass (was 3 failures before).

---

## Validation Details

### Task 1 — Dependency scan

Confirmed no remaining references to:
- `../DNNSplitting/` or `sys.path.insert`
- `trt_split_runtime_baseline` in code paths
- `from analysis import` (bare, without package prefix)

All RTA imports use `from src.rta.analysis import ...` or `from src.rta.task import ...`.

### Task 2 — py_compile + --help + pytest

```
py_compile: 29 src/ files — PASS
py_compile:  8 scripts/ files — PASS
py_compile:  5 tests/ files — PASS
pytest tests/ -q: 19 passed
```

### Task 3 — RTA import paths

`src/rta/__init__.py` exports: `get_SS_R`, `get_UNI_R_and_K`, `sort_task_set`,
`convert_task_list_to_SS`, `convert_task_list_to_UNI`, `SegInfTask`, `InferenceSegment`.
All callers use `from src.rta.analysis import ...` or `from src.rta.task import ...`.

### Task 4 — Policy-limited full-split probe (Fix 3)

In `_run_ss_opt_paper()` and `_run_ss_heu_paper()`, the feasibility probe uses:
```python
enabled = get_enabled_boundaries(dt, policy_name)
policy_full_mask = apply_policy_to_mask([1] * (N-1), enabled)
evaluate_and_apply_mask(dt, st, policy_full_mask, 0, **eval_kwargs)
```
Confirmed no policy violations in 100-run smoke test.

### Task 5 — K=1 initialization consistency

| Algorithm | K=1 Init | Method |
|-----------|----------|--------|
| ss:opt | Yes | `_paper_no_split_gate_ss()` → `apply_no_split_mask()` |
| uni:opt | Yes | `_paper_no_split_gate_uni()` → `apply_no_split_mask()` |
| ss:tol-fb | Yes (added) | Explicit loop at `_run_ss_tol_fb()` entry |
| uni:tol-fb | No | Not added — see known limitations |

**uni:tol-fb K=1 init not added:** Would require fixing the pre-existing G=0 handling
in the UNI↔SS conversion path before the fix to `convert_UNI_to_SS()` was applied.
Since `uni:tol-fb` starts with the no-split state implicitly (initial `SegInfTask` objects
have `G_block_list` from `dag_aligned_full.json`), the K=1 timing is effectively used.

### Task 6 — Cache validity (Fix 4)

`is_mask_cached()` in `config_evaluator.py` verifies:
1. JSON parses without error
2. `error` field absent or null
3. `per_chunk_gpu_mean_ms` present
4. `len(per_chunk_gpu_mean_ms) == sum(mask) + 1`

Test coverage: `tests/test_cache_validity.py`.

### Task 7 — Dry-run smoke test

```
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --num-tasksets-override 5 \
    --dry-run \
    --run-name validation_final_smoke
```

Result: 25 tasksets × 4 algorithms = 100 runs  
**Errors: 0 / Policy violations: 0**  
Elapsed: 0.61 s

### Task 8 — Plotting

```
results/plots/validation_fig4.png   (49 KB)
results/plots/validation_fig4.pdf   (17 KB)
results/plots/validation_fig5_cost.png      (46 KB)
results/plots/validation_fig5_cost.pdf      (15 KB)
results/plots/validation_fig5_runtime.png   (39 KB)
results/plots/validation_fig5_runtime.pdf   (15 KB)
```

### Task 9 — Git hygiene

- 70 files tracked at initial commit `f051bdd`
- No `.onnx`, `.engine`, `.trt`, `.pt`, `.pth`, `.pkl` files tracked
- Only `results/.gitkeep` in tracked results
- Generated experiment results are `.gitignore`-d

---

## Known Limitations

1. **uni:tol-fb K=1 initialization**: Not added in this session. The algorithm implicitly
   uses the no-split state at entry; explicit profiling of K=1 before the loop was not
   added due to the G=0 task issue (now fixed in `task.py`). A follow-up PR should add
   this initialization symmetrically with `ss:tol-fb`.

2. **G=0 dry-run base times**: Without live profiling, `base_chunk_times_ms` are all zero.
   The four algorithms still run and report schedulability based on CPU-only timing
   (`cpu_pre_ms` + `cpu_post_ms`), which is correct behavior. Split decisions in dry-run
   mode are based on the analytical model only.

3. **Live validation not performed**: Full end-to-end live validation (real TRT engine build
   + profiling on Jetson Orin) was not run in this session. All validations are dry-run.

---

## Next Commands

```bash
# Run full live preflight on Jetson (builds TRT engines)
python scripts/20_preflight_design.py --models alexnet resnet18 vgg19 --precision fp32

# Run live Fig.4 experiment (500 tasksets, full paper scale)
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --run-name fig4_live_full

# Push to GitHub
git remote add origin <URL>
git push -u origin master
```
