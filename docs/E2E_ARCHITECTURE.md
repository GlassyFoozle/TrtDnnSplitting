# End-to-End Architecture

## Live data flow

```text
run_hayeonp_no_inflation.sh
  -> scripts/33_run_yaml_fig4_with_split_plots.py
     -> scripts/30_run_yaml_fig4_experiment.py
        -> generate YAML task sets
        -> run_dnn_rta_algorithm()
           -> SS/UNI heuristic or tolerance-feedback search
           -> evaluate_and_apply_mask()
              -> evaluate_mask()
                 -> generate selected split config
                 -> export missing interval ONNX
                 -> build missing interval TensorRT engine
                 -> profile with C++ table4_runner
                 -> update interval and exact-mask caches
     -> scripts/32_plot_early_stop_counters.py
     -> scripts/31_plot_fig4.py
```

## Versioned inputs

- `configs/yaml/gpu_util_configs/*.yaml`: workload distributions
- `configs/split_point_policies.json`: enabled boundaries per model and policy
- `artifacts/split_configs/<model>/dag_aligned_full.json`: baseline graph,
  chunk, tensor-shape, and ONNX/engine path metadata
- `src/`: task generation, RTA, split search, export, build, and cache logic
- `cpp_runtime/`: synchronized TensorRT profiling runtime

The baseline JSON files are metadata inputs. Device timing and TensorRT engines
are generated locally.

## Generated state

### Interval cache

`artifacts/chunk_cache/<model>/int_<start>_<end>/` is the canonical reusable
store for one contiguous group of base chunks. It contains ONNX, precision-
specific engines, and `timing.json`. Different masks reuse a shared interval
when their `source_chunk_ids` match.

### Exact-mask evaluation cache

`results/evaluations_no_inflation/<model>/` stores the assembled timing result
for a complete boundary mask. Cache validity checks reject malformed results,
results with errors, and timing arrays whose size does not match the mask.

### Measured best-K and profiling caches

The no-inflation runner uses separate files under `results/optimization/` so
raw-timing results cannot be confused with inflation-enabled experiments.
Monotonic cache adjustment is disabled, but ordinary measured best-K reuse is
enabled.

## K=1 semantics

K=1 is the all-zero boundary mask. In live mode, task generation requests a
device-measured single-interval result. When missing, the normal evaluator
exports the whole-model interval, builds its engine, profiles it, and caches
the result. K=1 is not estimated by summing independently measured split
chunks.

## Timing semantics

`table4_runner` reports CUDA-event GPU timing and CPU wall timing around
synchronized execution. The canonical experiment uses the maximum CPU wall
time for task WCET. `TRT_RAW_INTERVAL_TIMING=1` and
`TRT_DISABLE_MONOTONIC_INFLATION=1` preserve raw interval measurements.

## Algorithm path

`src/integration/dnn_algorithm_runner.py` constructs the SS or UNI task view,
checks the current configuration, invokes the selected split search, applies
measured masks, and performs the final RTA/tolerance validation. The local
`src/rta/` implementation is self-contained.

`src/integration/taskset_timing.py` records taskset-local search, profiling,
and optimization timing when the corresponding optional timing mode is
enabled. Aggregate timing columns are always emitted by the YAML runner.

## Portability boundary

Source, configs, and baseline metadata are portable. Generated TensorRT
engines and measured WCET are device/software-stack artifacts and must be
regenerated when compatibility or experimental comparability is uncertain.
