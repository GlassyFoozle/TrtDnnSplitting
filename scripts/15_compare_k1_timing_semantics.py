#!/usr/bin/env python3
"""
Inspect or profile K=1 (non-split) TensorRT timing.

K=1 means the all-zero boundary mask over dag_aligned_full base chunks. With
the interval-cache artifact layout, that mask is a single interval:

    artifacts/chunk_cache/<model>/int_0_<N-1>/

This script reports the measured K=1 timing used by task generation and, when
requested, profiles the all-zero mask through the current interval-cache path.
It also reports dag_aligned_full chunk sums as split-candidate metadata only;
those sums are not the non-split runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_PRECISION = "fp32"
DEFAULT_WCET_METRIC = "max"
POLICIES_PATH = REPO / "configs" / "split_point_policies.json"
EXCLUDED_POLICY_MODELS = {"vit_l_16"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Inspect/profile non-split K=1 TensorRT timing."
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Models to inspect. Default: all models in split_point_policies.json "
            "with dag_aligned_full artifacts, excluding vit_l_16."
        ),
    )
    ap.add_argument("--precision", default=DEFAULT_PRECISION, choices=["fp32", "fp16"])
    ap.add_argument("--wcet-metric", default=DEFAULT_WCET_METRIC, choices=["mean", "max"])
    ap.add_argument(
        "--profile-k1",
        action="store_true",
        help=(
            "Measure K=1 by reusing the existing whole-model interval engine. "
            "Does not export, rebuild, or update eval/profiling cache files."
        ),
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Accepted for old commands; --profile-k1 still reuses existing engines only.",
    )
    ap.add_argument(
        "--builder-optimization-level",
        type=int,
        default=None,
        choices=range(0, 6),
        metavar="{0..5}",
        help="Pass --builderOptimizationLevel=N to trtexec when rebuilding K=1 engines.",
    )
    ap.add_argument(
        "--rebuild-engine",
        action="store_true",
        help="Rebuild the K=1 interval engine before measuring.",
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--allow-missing-timing",
        action="store_true",
        help="Keep reporting other models when dag_aligned_full split timing is missing.",
    )
    return ap.parse_args()


def _policy_models() -> List[str]:
    data = json.loads(POLICIES_PATH.read_text())
    models = [
        name
        for name, policies in data.items()
        if isinstance(policies, dict) and not name.startswith("_")
    ]
    return sorted(m for m in models if m not in EXCLUDED_POLICY_MODELS)


def _default_models() -> List[str]:
    models = []
    for model in _policy_models():
        if _base_config_path(model).exists():
            models.append(model)
    return models


def _base_config_path(model: str) -> Path:
    return REPO / "artifacts" / "split_configs" / model / "dag_aligned_full.json"


def _load_base_config(model: str) -> Dict[str, Any]:
    return json.loads(_base_config_path(model).read_text())


def _base_chunk_count(model: str) -> int:
    return len(_load_base_config(model).get("chunks", []))


def _zero_mask(model: str) -> List[int]:
    return [0] * (_base_chunk_count(model) - 1)


def _k1_variant(model: str) -> str:
    from src.optimization.config_evaluator import mask_to_variant_name

    return mask_to_variant_name(model, _zero_mask(model))


def _eval_path(model: str, precision: str) -> Path:
    return REPO / "results" / "evaluations" / model / f"{_k1_variant(model)}_{precision}.json"


def _interval_timing_path(model: str) -> Path:
    n = _base_chunk_count(model)
    return REPO / "artifacts" / "chunk_cache" / model / f"int_0_{n - 1}" / "timing.json"


def _interval_engine_path(model: str, precision: str) -> Path:
    n = _base_chunk_count(model)
    return (
        REPO
        / "artifacts"
        / "chunk_cache"
        / model
        / f"int_0_{n - 1}"
        / f"chunk_{precision}.engine"
    )


def _interval_onnx_path(model: str) -> Path:
    n = _base_chunk_count(model)
    return REPO / "artifacts" / "chunk_cache" / model / f"int_0_{n - 1}" / "chunk.onnx"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _metric_from_eval(data: Dict[str, Any], metric: str) -> Optional[float]:
    if data.get("error"):
        return None
    key = "per_chunk_gpu_mean_ms" if metric == "mean" else "per_chunk_gpu_max_ms"
    values = data.get(key)
    if not values and metric == "max":
        values = data.get("per_chunk_gpu_p99_ms")
    if not values or len(values) != 1:
        return None
    return float(sum(values))


def _metric_from_interval(data: Dict[str, Any], precision: str, metric: str) -> Optional[float]:
    key = f"gpu_{metric}_ms_{precision}"
    value = data.get(key)
    if value is None and metric == "max":
        value = data.get(f"gpu_p99_ms_{precision}")
    if value is None:
        return None
    return float(value)


def _measured_k1(model: str, precision: str, metric: str) -> Tuple[Optional[float], str]:
    eval_data = _read_json(_eval_path(model, precision))
    if eval_data is not None and eval_data.get("mask") == _zero_mask(model):
        value = _metric_from_eval(eval_data, metric)
        if value is not None:
            return value, f"eval:{_eval_path(model, precision).relative_to(REPO)}"

    int_data = _read_json(_interval_timing_path(model))
    if int_data is not None:
        expected_ids = list(range(_base_chunk_count(model)))
        if int_data.get("source_chunk_ids") in (None, expected_ids):
            value = _metric_from_interval(int_data, precision, metric)
            if value is not None:
                return value, f"interval:{_interval_timing_path(model).relative_to(REPO)}"

    return None, "missing"


def _k1_runner_config(model: str, precision: str) -> Dict[str, Any]:
    """Build a temporary K=1 config that points at existing interval-cache artifacts."""
    base = _load_base_config(model)
    chunks = base.get("chunks", [])
    if not chunks:
        raise RuntimeError(f"{model}: dag_aligned_full config has no chunks")

    n = len(chunks)
    interval_rel = f"artifacts/chunk_cache/{model}/int_0_{n - 1}"
    variant = _k1_variant(model)
    source_ids = list(range(n))

    return {
        "model": model,
        "variant": variant,
        "base_variant": "dag_aligned_full",
        "n_chunks": 1,
        "input_shape": base.get("input_shape", chunks[0].get("input_shape")),
        "candidate_count": n,
        "boundary_count": n - 1,
        "active_boundary_count": 0,
        "mask": [0] * (n - 1),
        "merged_groups": [source_ids],
        "chunks": [
            {
                "id": 0,
                "name": "chunk0",
                "chunk_name": "chunk0",
                "description": f"{chunks[0].get('chunk_name', 'chunk0')}.."
                f"{chunks[-1].get('chunk_name', f'chunk{n - 1}')} ({n} merged)",
                "source_chunk_ids": source_ids,
                "input_shape": chunks[0]["input_shape"],
                "output_shape": chunks[-1]["output_shape"],
                "onnx": f"{interval_rel}/chunk.onnx",
                "engine_fp32": f"{interval_rel}/chunk_fp32.engine",
                "engine_fp16": f"{interval_rel}/chunk_fp16.engine",
            }
        ],
        "full_model": {
            "onnx": f"artifacts/onnx/{model}/{variant}/full.onnx",
            "engine_fp32": f"artifacts/engines/{model}/{variant}/full_fp32.engine",
            "engine_fp16": f"artifacts/engines/{model}/{variant}/full_fp16.engine",
        },
    }


def _table4_output_path(model: str, precision: str) -> Path:
    return REPO / "results" / "table4" / f"{model}_cpp_{_k1_variant(model)}_{precision}.json"


def _measure_k1_direct(model: str, precision: str, warmup: int, iters: int) -> Dict[str, Any]:
    """
    Measure K=1 with the existing interval-cache engine, then restore/delete the
    table4 JSON that the C++ runner writes so no cache/result state is updated.
    """
    runner = next(
        (p for p in (
            REPO / "cpp_runtime" / "build_sync_wall" / "table4_runner",
            REPO / "cpp_runtime" / "build" / "table4_runner",
        ) if p.exists()),
        REPO / "cpp_runtime" / "build_sync_wall" / "table4_runner",
    )
    if not runner.exists():
        raise RuntimeError(f"{model}: missing C++ runner: {runner}")
    engine = _interval_engine_path(model, precision)
    if not engine.exists():
        raise RuntimeError(f"{model}: missing existing K=1 engine: {engine}")

    cfg = _k1_runner_config(model, precision)
    out_path = _table4_output_path(model, precision)
    previous = out_path.read_bytes() if out_path.exists() else None

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) as fh:
        json.dump(cfg, fh, indent=2)
        fh.flush()
        cmd = [
            str(runner),
            "--config",
            fh.name,
            "--repo",
            str(REPO),
            "--precision",
            precision,
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--sync-wall",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(f"{model}: table4_runner failed\n{detail}")
            data = _read_json(out_path)
            if data is None:
                raise RuntimeError(f"{model}: table4_runner did not write {out_path}")
            return data
        finally:
            if previous is None:
                out_path.unlink(missing_ok=True)
            else:
                out_path.write_bytes(previous)


def _metric_from_table4(data: Dict[str, Any], metric: str) -> Optional[float]:
    if metric == "mean":
        value = data.get("total_chunked_cpu_wall_mean_ms", data.get("total_chunked_gpu_mean_ms"))
    else:
        value = (
            data.get("total_chunked_cpu_wall_max_ms")
            or data.get("total_chunked_cpu_wall_p99_ms")
            or data.get("total_chunked_gpu_max_ms")
            or data.get("total_chunked_gpu_p99_ms")
        )
    return None if value is None else float(value)


def _ensure_or_rebuild_k1_engine(
    model: str,
    precision: str,
    rebuild: bool,
) -> None:
    engine = _interval_engine_path(model, precision)
    if engine.exists() and not rebuild:
        return
    onnx = _interval_onnx_path(model)
    if not onnx.exists():
        raise RuntimeError(
            f"{model}: K=1 ONNX missing: {onnx}. "
            "Generate the K=1 interval artifact before rebuilding."
        )

    from src.optimization.compiler import build_single_engine

    action = "rebuild" if engine.exists() else "build"
    print(
        f"  [{model}] {action} K=1 engine "
        f"({onnx.relative_to(REPO)} -> {engine.relative_to(REPO)})",
        flush=True,
    )
    ok, _wall = build_single_engine(onnx, engine, precision=precision)
    if not ok:
        raise RuntimeError(f"{model}: failed to {action} K=1 engine")


def _candidate_space(model: str, precision: str, allow_missing: bool):
    from src.optimization.candidate_space import load_candidate_space

    return load_candidate_space(
        model,
        precision,
        allow_missing_timing_for_live=allow_missing,
    )


def _gen_k1_value(model: str, precision: str, metric: str) -> Optional[float]:
    from src.integration.dnn_workload_generator import _get_base_gpu_wcet_ms

    return _get_base_gpu_wcet_ms(model, precision, metric)


def _format_ms(value: Optional[float]) -> str:
    return "missing" if value is None else f"{value:.4f}"


def _print_rows(rows: Iterable[Dict[str, Any]]) -> None:
    header = (
        f"{'Model':<20} {'N':>4} {'K1 measured':>12} {'TaskGen G':>12} "
        f"{'DagFull sum':>12} {'Delta':>10} {'K1 source'}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        delta = None
        if row["k1"] is not None and row["dag_sum"] is not None:
            delta = row["dag_sum"] - row["k1"]
        print(
            f"{row['model']:<20} {row['n']:>4} "
            f"{_format_ms(row['k1']):>12} {_format_ms(row['gen_g']):>12} "
            f"{_format_ms(row['dag_sum']):>12} {_format_ms(delta):>10} "
            f"{row['k1_source']}"
        )


def main() -> int:
    args = parse_args()
    if args.builder_optimization_level is not None:
        os.environ["TRT_BUILDER_OPT_LEVEL"] = str(args.builder_optimization_level)
        print(f"TensorRT builderOptimizationLevel={args.builder_optimization_level}", flush=True)
    models = args.models or _default_models()
    missing_cfg = [m for m in models if not _base_config_path(m).exists()]
    if missing_cfg:
        print(f"[error] Missing dag_aligned_full artifacts for: {missing_cfg}", file=sys.stderr)
        print(
            "Run: conda run -n trt python scripts/26_generate_dag_aligned_configs.py "
            f"--models {' '.join(missing_cfg)}",
            file=sys.stderr,
        )
        return 2

    live_profiles: Dict[str, Dict[str, Any]] = {}
    if args.profile_k1:
        print(
            f"Measuring K=1 with existing interval engines only "
            f"({args.precision}, warmup={args.warmup}, iters={args.iters}; "
            "no cache/result update)"
        )
        for model_idx, model in enumerate(models, start=1):
            print(f"\n=== K=1 model {model_idx}/{len(models)}: {model} ===", flush=True)
            _ensure_or_rebuild_k1_engine(model, args.precision, args.rebuild_engine)
            data = _measure_k1_direct(model, args.precision, args.warmup, args.iters)
            live_profiles[model] = data
            value = _metric_from_table4(data, args.wcet_metric)
            per = data.get("chunks", [{}])[0].get(f"gpu_{args.wcet_metric}_ms")
            if per is None and args.wcet_metric == "max":
                per = data.get("chunks", [{}])[0].get("gpu_p99_ms")
            print(
                f"  {model:<20} total_{args.wcet_metric}={_format_ms(value)} ms  "
                f"chunk0_{args.wcet_metric}={_format_ms(float(per) if per is not None else None)} ms"
            )
        print()

    print("=" * 96)
    print("K=1 timing semantics: current interval-cache layout")
    print("=" * 96)
    print(f"Repo: {REPO}")
    print(f"Models: {', '.join(models)}")
    print(f"Metric: {args.wcet_metric}  Precision: {args.precision}")
    print()

    rows: List[Dict[str, Any]] = []
    ok = True
    for model in models:
        n = _base_chunk_count(model)
        if model in live_profiles:
            k1_value = _metric_from_table4(live_profiles[model], args.wcet_metric)
            k1_source = "live_profile_existing_engine:no_cache_update"
        else:
            k1_value, k1_source = _measured_k1(model, args.precision, args.wcet_metric)
        gen_g = _gen_k1_value(model, args.precision, args.wcet_metric)
        try:
            cs = _candidate_space(model, args.precision, args.allow_missing_timing)
            dag_values = cs.chunk_gpu_mean_ms if args.wcet_metric == "mean" else cs.chunk_gpu_max_ms
            dag_sum = sum(dag_values)
            dag_source = cs.timing_source
        except Exception as exc:
            dag_sum = None
            dag_source = f"missing ({exc})"
        if k1_value is None:
            ok = False
        rows.append(
            {
                "model": model,
                "n": n,
                "k1": k1_value,
                "gen_g": gen_g,
                "dag_sum": dag_sum,
                "dag_source": dag_source,
                "k1_source": k1_source,
            }
        )

    _print_rows(rows)
    print()
    print("Notes:")
    print("  - K1 measured is the non-split all-zero mask, one interval int_0_(N-1).")
    if args.profile_k1:
        print("  - --profile-k1 reused existing engines and restored/deleted table4 output after reading.")
    print("  - TaskGen G should match K1 measured; dag_aligned_full sum is only split metadata.")
    print("  - vit_l_16 is intentionally excluded because its dag_aligned_full artifact is absent.")
    print()
    print("dag_aligned_full timing sources:")
    for row in rows:
        print(f"  {row['model']}: {row['dag_source']}")

    if not ok:
        print()
        print("RESULT: Missing real K=1 timing for at least one model.")
        print("To profile K=1 with the current interval-cache path:")
        print(
            "  conda run -n trt python scripts/15_compare_k1_timing_semantics.py "
            f"--profile-k1 --models {' '.join(models)} --precision {args.precision} "
            f"--warmup {args.warmup} --iters {args.iters}"
        )
        return 1

    print()
    print("RESULT: K=1 timing is available for every requested model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
