# Full Fig.4 Algorithm Matrix — Report

**Date:** 2026-05-08  

---

## Algorithm Status

All eight SS + UNI algorithms are fully implemented in
`src/integration/dnn_algorithm_runner.py`. No porting was needed.

| Label | Model | Algorithm key | Implementation function | Status |
|-------|-------|--------------|------------------------|--------|
| SS-opt | SS | `opt` | `_run_ss_opt_paper()` | **FULL** |
| SS-heu | SS | `heu` | `_run_ss_heu_paper()` | **FULL** |
| SS-tol | SS | `tol` | `_run_ss_tol()` | **FULL** |
| SS-tol-fb | SS | `tol-fb` | `_run_ss_tol_fb()` | **FULL** |
| UNI-opt | UNI | `opt` | `_run_uni_opt_paper()` | **FULL** |
| UNI-heu | UNI | `heu` | `_run_uni_heu_paper()` | **FULL** |
| UNI-tol | UNI | `tol` | `_run_uni_tol()` | **FULL** |
| UNI-tol-fb | UNI | `tol-fb` | `_run_uni_tol_fb()` | **FULL** |

---

## Files Changed

### `scripts/30_run_yaml_fig4_experiment.py`

- Added `_ALGORITHM_SETS` dict with four predefined sets: `main4`, `full8`, `ss_only`, `uni_only`
- Renamed `_DEFAULT_ALGORITHMS` labels from `SS_ours/UNI_ours/SS_Buttazzo/UNI_Buttazzo` to
  `SS-tol-fb/UNI-tol-fb/SS-opt/UNI-opt` for consistency with the full8 naming scheme
- `_DEFAULT_ALGORITHMS` kept as backward-compatible alias pointing to `_ALGORITHM_SETS["main4"]`
- Added `--algorithm-set {main4,full8,ss_only,uni_only}` CLI option (default: `main4`)
- Added `--algorithms MODEL:ALGO[:LABEL]...` explicit override
- Added `_build_algorithm_list(args)` resolver
- Updated `write_summary()`, `write_run_config()`, and `main()` to use dynamic algorithm list
- Added "Algorithm set:" line to run output and summary

### `scripts/31_plot_fig4.py`

- Rewrote with extended `_ORDER_FULL8`, `_STYLE`, and `_PLOT_MODE_FILTERS` / `_PLOT_MODE_ORDER` dicts
- Added `--plot-mode {all,main4,ss_only,uni_only}` option (default: `all`)
- `load_series()` now normalizes legacy labels via `_LEGACY_ALIASES` dict
  (`SS_ours` → `SS-tol-fb`, etc.) so old CSVs still plot correctly
- `filter_series()` and `render_order()` helpers apply the mode filter and legend ordering
- Both matplotlib and Pillow backends updated for 8-algorithm support
- Output file gets `_<mode>` suffix when `--plot-mode` is not `all`

### `docs/EXPERIMENTS.md`

- Added "Full 2×4 Algorithm Matrix (full8)" section with:
  - Algorithm set table
  - Dry-run full8 sweep command
  - Plot commands for all/ss_only/uni_only modes
  - Explicit `--algorithms` override example
  - Live full8 command with `--global-max-real-profiles` cap

---

## Validation Results

### py_compile

44 Python files (src/ + scripts/ + tests/) — **PASS**

### pytest

19/19 tests — **PASS**

### Full8 dry-run smoke test

```
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --algorithm-set full8 \
    --num-tasksets-override 2 \
    --dry-run \
    --run-name fig4_full8_smoke
```

- 5 utilizations × 2 tasksets × 8 algorithms = **80 runs**
- Errors: **0**
- Policy violations: **0**
- Elapsed: 0.62 s

### Plots generated

| File | Description |
|------|-------------|
| `results/plots/fig4_full8_smoke.png/pdf` | All 8 algorithms |
| `results/plots/fig4_full8_smoke_ss_only.png/pdf` | SS-only 4 algorithms |
| `results/plots/fig4_full8_smoke_uni_only.png/pdf` | UNI-only 4 algorithms |

### Backward compatibility

Legacy CSV with `SS_ours/UNI_ours/SS_Buttazzo/UNI_Buttazzo` labels plots correctly
with `--plot-mode main4` via the `_LEGACY_ALIASES` normalization in `load_series()`.

---

## Example Commands

```bash
# Full8 dry-run
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --algorithm-set full8 \
    --num-tasksets-override 500 \
    --dry-run \
    --run-name fig4_full8

# Plot all 8
python scripts/31_plot_fig4.py \
    --run-dir results/dnn_experiments/fig4_full8 \
    --plot-mode all \
    --output fig4_full8 \
    --output-dir results/plots

# Live full8 with profile budget cap
python scripts/30_run_yaml_fig4_experiment.py \
    --config configs/yaml/1_GPU0.6-1.0_task8_ov5.yaml \
    --models alexnet resnet18 vgg19 \
    --split-policy major_blocks \
    --algorithm-set full8 \
    --global-max-real-profiles 200 \
    --run-name fig4_full8_live
```
