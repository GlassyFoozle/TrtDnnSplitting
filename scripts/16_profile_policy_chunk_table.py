#!/usr/bin/env python3
"""
Measure per-chunk TensorRT runtime for a split-point policy.

This is an inspection-only script. It reuses existing interval-cache engines,
runs cpp_runtime/build/table4_runner with a temporary config, prints a table,
and restores/deletes the table4 JSON that the runner writes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
POLICIES_PATH = REPO / "configs" / "split_point_policies.json"
RUNNER = next(
    (p for p in (
        REPO / "cpp_runtime" / "build_sync_wall" / "table4_runner",
        REPO / "cpp_runtime" / "build" / "table4_runner",
    ) if p.exists()),
    REPO / "cpp_runtime" / "build_sync_wall" / "table4_runner",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Print per-chunk TRT timing table for a split policy."
    )
    ap.add_argument("--models", nargs="+", default=["alexnet", "resnet18", "vgg19"])
    ap.add_argument("--policy", default="trt_fusion_safe")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--builder-optimization-level",
        type=int,
        default=None,
        choices=range(0, 6),
        metavar="{0..5}",
        help="Pass --builderOptimizationLevel=N to trtexec when building missing/rebuilt engines.",
    )
    ap.add_argument(
        "--rebuild-engines",
        action="store_true",
        help="Rebuild policy interval engines before measuring, even when they already exist.",
    )
    ap.add_argument(
        "--no-build-missing",
        action="store_true",
        help="Fail if policy interval engines are missing instead of creating them.",
    )
    return ap.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _base_config_path(model: str) -> Path:
    return REPO / "artifacts" / "split_configs" / model / "dag_aligned_full.json"


def _enabled_boundaries(model: str, policy: str, boundary_count: int) -> List[int]:
    policies = _read_json(POLICIES_PATH)
    model_policies = policies.get(model, {})
    enabled = model_policies.get(policy)
    if enabled is None:
        raise KeyError(f"{model}: policy {policy!r} not found in {POLICIES_PATH}")
    return sorted(i for i in enabled if isinstance(i, int) and 0 <= i < boundary_count)


def _mask_to_groups(mask: List[int]) -> List[List[int]]:
    groups: List[List[int]] = []
    current = [0]
    for boundary_idx, bit in enumerate(mask):
        if bit:
            groups.append(current)
            current = [boundary_idx + 1]
        else:
            current.append(boundary_idx + 1)
    groups.append(current)
    return groups


def _policy_groups(model: str, policy: str) -> tuple[Dict[str, Any], List[int], List[List[int]]]:
    base = _read_json(_base_config_path(model))
    n = len(base["chunks"])
    enabled = set(_enabled_boundaries(model, policy, n - 1))
    mask = [1 if i in enabled else 0 for i in range(n - 1)]
    return base, mask, _mask_to_groups(mask)


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _variant_name(model: str, policy: str) -> str:
    return f"{model}_{policy}_inspect"


def _table4_output_path(model: str, variant: str, precision: str) -> Path:
    return REPO / "results" / "table4" / f"{model}_cpp_{variant}_{precision}.json"


def _build_temp_config(
    model: str,
    policy: str,
    precision: str,
    require_engines: bool = True,
) -> tuple[Dict[str, Any], List[List[int]]]:
    base, mask, groups = _policy_groups(model, policy)
    base_chunks = base["chunks"]
    variant = _variant_name(model, policy)
    chunk_cfgs = []

    for chunk_id, group in enumerate(groups):
        first = base_chunks[group[0]]
        last = base_chunks[group[-1]]
        int_dir = REPO / "artifacts" / "chunk_cache" / model / f"int_{group[0]}_{group[-1]}"
        engine = int_dir / f"chunk_{precision}.engine"
        onnx = int_dir / "chunk.onnx"
        if require_engines and not engine.exists():
            raise FileNotFoundError(
                f"{model}: missing engine for policy chunk {chunk_id}, "
                f"interval int_{group[0]}_{group[-1]}: {engine}"
            )

        src_names = [base_chunks[i].get("chunk_name", f"chunk{i}") for i in group]
        chunk_cfgs.append(
            {
                "id": chunk_id,
                "name": f"chunk{chunk_id}",
                "chunk_name": f"chunk{chunk_id}",
                "description": (
                    first.get("description", first.get("chunk_name", f"chunk{group[0]}"))
                    if len(group) == 1
                    else f"{src_names[0]}..{src_names[-1]} ({len(group)} merged)"
                ),
                "source_chunk_ids": group,
                "source_chunk_names": src_names,
                "input_shape": first["input_shape"],
                "output_shape": last["output_shape"],
                "onnx": _rel(onnx),
                "engine_fp32": _rel(int_dir / "chunk_fp32.engine"),
                "engine_fp16": _rel(int_dir / "chunk_fp16.engine"),
            }
        )

    cfg = {
        "model": model,
        "variant": variant,
        "base_variant": "dag_aligned_full",
        "policy": policy,
        "n_chunks": len(groups),
        "input_shape": base.get("input_shape", base_chunks[0]["input_shape"]),
        "candidate_count": len(base_chunks),
        "boundary_count": len(base_chunks) - 1,
        "active_boundary_count": sum(mask),
        "mask": mask,
        "merged_groups": groups,
        "chunks": chunk_cfgs,
        "full_model": {
            "onnx": f"artifacts/onnx/{model}/{variant}/full.onnx",
            "engine_fp32": f"artifacts/engines/{model}/{variant}/full_fp32.engine",
            "engine_fp16": f"artifacts/engines/{model}/{variant}/full_fp16.engine",
        },
    }
    return cfg, groups


def _missing_policy_artifacts(
    cfg: Dict[str, Any],
    precision: str,
) -> tuple[List[str], List[str]]:
    missing_onnx = []
    missing_engines = []
    for chunk in cfg["chunks"]:
        ids = chunk["source_chunk_ids"]
        label = f"int_{ids[0]}_{ids[-1]}"
        if not (REPO / chunk["onnx"]).exists():
            missing_onnx.append(label)
        if not (REPO / chunk[f"engine_{precision}"]).exists():
            missing_engines.append(label)
    return missing_onnx, missing_engines


def _ensure_policy_artifacts(
    model: str,
    cfg: Dict[str, Any],
    groups: List[List[int]],
    precision: str,
    rebuild_engines: bool = False,
) -> None:
    missing_onnx, missing_engines = _missing_policy_artifacts(cfg, precision)
    if not missing_onnx and not missing_engines and not rebuild_engines:
        return

    action = "rebuilding" if rebuild_engines else "creating missing"
    print(
        f"[{model}] {action} interval artifacts: "
        f"onnx={len(missing_onnx)} engine={len(missing_engines)}",
        flush=True,
    )
    if missing_onnx:
        shown = ", ".join(missing_onnx[:12])
        suffix = "..." if len(missing_onnx) > 12 else ""
        print(f"  missing ONNX: {shown}{suffix}", flush=True)
    if missing_engines:
        shown = ", ".join(missing_engines[:12])
        suffix = "..." if len(missing_engines) > 12 else ""
        print(f"  missing engines: {shown}{suffix}", flush=True)

    from src.optimization.config_evaluator import (
        _build_engines_with_interval_cache,
        _export_chunks_with_interval_cache,
    )

    if missing_onnx:
        _export_chunks_with_interval_cache(model, cfg, groups, force=False)
    else:
        print(f"  [interval_cache] all {len(groups)} ONNX(es) from interval cache", flush=True)
    _, engine_misses, _ = _build_engines_with_interval_cache(
        model,
        cfg,
        groups,
        precision,
        force=rebuild_engines,
    )
    missing_onnx, missing_engines = _missing_policy_artifacts(cfg, precision)
    if missing_onnx or missing_engines:
        raise RuntimeError(
            f"{model}: failed to create all policy artifacts; "
            f"missing_onnx={missing_onnx[:8]} missing_engines={missing_engines[:8]}"
        )
    if engine_misses:
        verb = "rebuilt" if rebuild_engines else "built"
        print(f"[{model}] {verb} {engine_misses} interval engine(s)", flush=True)


def _run_table4_no_update(
    cfg: Dict[str, Any],
    precision: str,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    if not RUNNER.exists():
        raise FileNotFoundError(f"table4_runner not found: {RUNNER}")

    model = cfg["model"]
    variant = cfg["variant"]
    out_path = _table4_output_path(model, variant, precision)
    previous = out_path.read_bytes() if out_path.exists() else None

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=True) as fh:
        json.dump(cfg, fh, indent=2)
        fh.flush()
        cmd = [
            str(RUNNER),
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
            if not out_path.exists():
                raise RuntimeError(f"{model}: table4_runner did not write {out_path}")
            return _read_json(out_path)
        finally:
            if previous is None:
                out_path.unlink(missing_ok=True)
            else:
                out_path.write_bytes(previous)


def _shape_str(shape: Any) -> str:
    if not isinstance(shape, list):
        return str(shape)
    return "x".join(str(x) for x in shape)


def _print_model_table(cfg: Dict[str, Any], result: Dict[str, Any]) -> None:
    chunks = result.get("chunks", [])
    cfg_chunks = cfg["chunks"]
    total_mean = result.get("total_chunked_cpu_wall_mean_ms", 0.0)
    total_p99 = result.get("total_chunked_cpu_wall_p99_ms", 0.0)
    total_max = result.get("total_chunked_cpu_wall_max_ms", 0.0)

    print()
    print(f"## {cfg['model']} / {cfg['policy']}  K={cfg['n_chunks']}")
    print(
        f"total_mean={total_mean:.4f} ms  "
        f"total_p99={total_p99:.4f} ms  total_max={total_max:.4f} ms"
    )
    header = (
        f"{'id':>2}  {'interval':>9}  {'base_ids':<18}  {'input':<14}  "
        f"{'output':<14}  {'mean_ms':>9}  {'p99_ms':>9}  {'max_ms':>9}  description"
    )
    print(header)
    print("-" * len(header))
    for cfg_chunk, measured in zip(cfg_chunks, chunks):
        ids = cfg_chunk["source_chunk_ids"]
        interval = f"{ids[0]}-{ids[-1]}"
        base_ids = ",".join(str(i) for i in ids)
        print(
            f"{cfg_chunk['id']:>2}  {interval:>9}  {base_ids:<18}  "
            f"{_shape_str(cfg_chunk['input_shape']):<14}  "
            f"{_shape_str(cfg_chunk['output_shape']):<14}  "
            f"{float(measured['cpu_mean_ms']):>9.4f}  "
            f"{float(measured['cpu_p99_ms']):>9.4f}  "
            f"{float(measured['cpu_max_ms']):>9.4f}  "
            f"{cfg_chunk['description']}"
        )


def main() -> int:
    args = parse_args()
    if args.builder_optimization_level is not None:
        os.environ["TRT_BUILDER_OPT_LEVEL"] = str(args.builder_optimization_level)
        print(f"TensorRT builderOptimizationLevel={args.builder_optimization_level}", flush=True)
    for model_idx, model in enumerate(args.models, start=1):
        print(
            f"\n=== model {model_idx}/{len(args.models)}: {model} / {args.policy} ===",
            flush=True,
        )
        cfg, groups = _build_temp_config(
            model,
            args.policy,
            args.precision,
            require_engines=args.no_build_missing,
        )
        if not args.no_build_missing:
            _ensure_policy_artifacts(
                model,
                cfg,
                groups,
                args.precision,
                rebuild_engines=args.rebuild_engines,
            )
        result = _run_table4_no_update(
            cfg,
            args.precision,
            args.warmup,
            args.iters,
        )
        _print_model_table(cfg, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
