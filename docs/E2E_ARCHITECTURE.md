# TrtDnnSplitting — End-to-End Architecture

## High-Level Data Flow

```
Taskset JSON
    │
    ▼
generate_dnn_taskset()              ← dnn_taskset_loader.py
    │  Loads per-chunk base times from artifacts/split_configs/*/dag_aligned_full.json
    │  Falls back to zeros if no profiling available (dry-run safe)
    ▼
DNNBackedTask list
    │
    ▼
build_task_set_dict()               ← dnnsplitting_adapter.py
    │  Creates SegInfTask (src.rta.task) for each DNN task
    │  base_block_list = base_chunk_times_ms from dag_aligned_full
    ▼
run_dnn_rta_algorithm()             ← dnn_algorithm_runner.py
    │
    ├─ _dispatch_ss() ──────────────────────────────────────────────────────┐
    │      ├─ ss:tol-fb  → _run_ss_tol_fb()                                │
    │      └─ ss:opt     → _paper_no_split_gate_ss() → _run_ss_opt_paper() │
    │                                                                       │
    └─ _dispatch_uni() ─────────────────────────────────────────────────────┘
           ├─ uni:tol-fb → _run_uni_tol_fb()
           └─ uni:opt    → _paper_no_split_gate_uni() → _run_uni_opt_paper()
                │
                ▼
        evaluate_and_apply_mask()   ← mask_applicator.py
                │
                ├─ dry_run=True  → _apply_mask_to_chunk_times() + _patch_seg_task()
                └─ dry_run=False → evaluate_mask() → TRT engine build + profile
```

## Module Map

### src/rta/

Self-contained RTA ported from the DNNSplitting paper. No external dependencies.

| File | Description |
|------|-------------|
| `task.py` | `SegInfTask`, `InferenceSegment` — task model; SS↔UNI conversion |
| `analysis.py` | `get_SS_R()`, `get_UNI_R_and_K()`, `convert_task_list_to_SS/UNI()`, RTA core |

### src/integration/

Bridge layer: loads DNN tasks, runs algorithms, collects results.

| File | Role |
|------|------|
| `dnn_task.py` | `DNNBackedTask` dataclass |
| `dnn_taskset_loader.py` | Parse taskset JSON → `DNNBackedTask` list |
| `dnn_taskset_generator.py` | Generate single taskset JSON from params |
| `dnn_workload_generator.py` | `WorkloadConfig` + `generate_tasksets()` for batch generation |
| `dnnsplitting_adapter.py` | `dnn_task_to_seginftask()`, `build_task_set_dict()` |
| `mask_applicator.py` | `evaluate_and_apply_mask()`, dry/live dispatch |
| `dnn_algorithm_runner.py` | All four algorithm implementations + dispatcher |
| `split_point_policy.py` | `get_enabled_boundaries()`, `apply_policy_to_mask()` |
| `paper_style_search.py` | `search_optimal_ss_mask()`, `search_heuristic_ss_mask()`, etc. |

### src/optimization/

TRT engine builds and profiling cache management.

| File | Role |
|------|------|
| `config_evaluator.py` | `evaluate_mask()` — builds TRT engine and runs timing; manages interval cache |
| `candidate_space.py` | `load_candidate_space()` — loads dag_aligned_full configs |
| `compiler.py` | Orchestrates ONNX export + TRT engine build (subprocess wrappers) |
| `profiling_db.py` | `ProfilingDB` — flat JSON cache for per-variant profiling results |

### src/splitting/

Split-point generation and mask computation.

| File | Role |
|------|------|
| `dag_aligned_splitter.py` | Enumerate boundaries aligned to DNN layer graph |
| `selective_splitter.py` | Apply selective split patterns |

## Key Invariants

**base_chunk_times_ms**: Per-chunk GPU times from `dag_aligned_full.json`. Loaded once per
model; shared across all masks evaluated for that task. Zero in fresh clone (no live profiling).

**SegInfTask construction**: Uses `dummy_G = max(N, 1)` to pass integer-unit validation in
`InferenceSegment.__init__`, then overrides `base_block_list` with real float ms values.

**K=1 baseline timing**: The all-zero mask (K=1, no split) always uses `sum(base_chunk_times_ms)`
from the pre-profiled `dag_aligned_full` baseline — never triggers engine export/build/profile.
In live mode, `evaluate_and_apply_mask` short-circuits any all-zero mask to this baseline path,
returning `cache_hit=True`. If `base_chunk_times_ms` are all zero in live mode, an error is
returned directing the user to run `scripts/20_preflight_design.py`.

**K=1 initialization**: All four algorithms start from the all-zero mask (no-split) state:
- `ss:opt` and `uni:opt`: via `_paper_no_split_gate_ss/uni()`
- `ss:tol-fb`: explicit `apply_no_split_mask()` at function entry
- `uni:tol-fb`: implicit (no-split is the initial task state before the first tolerance check)

**Policy-limited feasibility probe**: `_run_ss_opt_paper()` and `_run_ss_heu_paper()` use
`apply_policy_to_mask([1]*(N-1), enabled)` as the full-split probe — not the raw all-ones mask.
This ensures the infeasibility gate is consistent with the policy-constrained search space.

**Cache validity (is_mask_cached)**: Returns `True` only if:
1. JSON parses successfully
2. No `error` field
3. `per_chunk_gpu_mean_ms` present
4. `len(per_chunk_gpu_mean_ms) == sum(mask) + 1`

**Interval-level cache**: `artifacts/chunk_cache/{model}/int_{start}_{end}/` stores the ONNX
and TRT engine for each merged base-chunk interval independently of the mask variant name.
When two different masks share a chunk with the same `source_chunk_ids = [start..end]`, the
second mask reuses the cached ONNX/engine rather than rebuilding it.
- ONNX export is done per-chunk inline (calls `export_module()` directly, not subprocess).
- Engine build falls back to whole-variant `build_engines()` if the all-or-nothing interval
  cache check fails for any engine, then populates the interval cache afterwards.
- `timing.json` inside each interval directory records `export_wall_s` and
  `build_{precision}_wall_s` for cold-cache design-time estimation.
- Interval cache is additive: deleting `artifacts/chunk_cache/` does not break correctness.

**Cold-cache design-time estimate**: `EvaluationResult.estimated_cold_total_s` sums the
per-interval export and build times for all chunks in the mask plus `profile_wall_s`. This
lets Fig.5 report the design-time cost that would have been paid on a cold interval cache,
even when the actual run benefited from caching.

## Task Model Lifecycle

```
DNNBackedTask
  ├─ base_chunk_times_ms  [N floats]   ← from dag_aligned_full.json
  ├─ initial_mask         [N-1 ints]   ← 0 = no split (initial state)
  └─ candidate_count      N

dnn_task_to_seginftask(dt, splitting_config)
  ↓
SegInfTask
  ├─ C_list       [cpu_pre_ms, cpu_post_ms]
  ├─ inference_segment_list[0].base_block_list  [N floats]
  └─ G = sum(base_chunk_times_ms)   (0 if no live profiling)

evaluate_and_apply_mask(dt, st, mask, chunk_idx, ...)
  └─ _patch_seg_task(task, chunk_times)
       ├─ seg.G_block_list = list(chunk_times)
       └─ task.G_segment_list[0] = list(chunk_times)
```

## SS ↔ UNI Conversion

`convert_SS_to_UNI()` merges all C and G blocks into a single UNI segment, tracking
`_UNI_block_sources` metadata for back-conversion. Blocks with value ≤ 0 are skipped.

`convert_UNI_to_SS()` uses `_UNI_block_sources` to reconstruct original C_list and
G_segment_list. `c_list` size is `max(original_segment_count+1, max_c_idx+1)` to handle
the G=0 case (fresh clone / no live profiling) where G blocks are 0 and are skipped by
`append_block`, leaving only C sources in `_UNI_block_sources`. Without this fix, the
second C source (index 1) would IndexError into a single-element `c_list`.
