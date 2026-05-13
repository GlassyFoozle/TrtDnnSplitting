#!/usr/bin/env python3
"""
27_reprofile_cached_intervals.py — Re-profile existing interval-cache engines.

This script does not export ONNX or build TensorRT engines. It scans
artifacts/chunk_cache/<model>/int_<start>_<end>/chunk_<precision>.engine,
runs table4_runner against each existing engine, and writes fresh GPU
mean/p99/max plus interval-specific profiling wall time back to timing.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TABLE4_RUNNER = REPO / "cpp_runtime" / "build" / "table4_runner"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Re-profile existing chunk_cache interval engines.")
    ap.add_argument("--models", nargs="+", default=None, help="Models to scan (default: all in chunk_cache)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="Re-profile even when max timing already exists")
    ap.add_argument("--dry-run", action="store_true", help="List intervals without running table4_runner")
    ap.add_argument("--limit", type=int, default=None, help="Profile at most N intervals")
    return ap.parse_args()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _discover_models(requested: Optional[list[str]]) -> list[str]:
    if requested:
        return sorted({m.lower() for m in requested})
    cache_dir = REPO / "artifacts" / "chunk_cache"
    if not cache_dir.exists():
        return []
    return sorted(p.name for p in cache_dir.iterdir() if p.is_dir())


def _parse_interval_name(path: Path) -> Optional[tuple[int, int]]:
    parts = path.name.split("_")
    if len(parts) != 3 or parts[0] != "int":
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _load_base_config(model: str) -> dict:
    path = REPO / "artifacts" / "split_configs" / model / "dag_aligned_full.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing base config for {model}: {path}")
    return json.loads(path.read_text())


def _interval_dirs(model: str, precision: str) -> Iterable[Path]:
    model_dir = REPO / "artifacts" / "chunk_cache" / model
    if not model_dir.exists():
        return []
    engine_name = f"chunk_{precision}.engine"
    return (
        p for p in sorted(model_dir.iterdir())
        if p.is_dir() and (p / engine_name).exists() and _parse_interval_name(p) is not None
    )


def _make_profile_config(model: str, interval_dir: Path, precision: str, base_cfg: dict) -> dict:
    parsed = _parse_interval_name(interval_dir)
    if parsed is None:
        raise ValueError(f"Invalid interval directory: {interval_dir}")
    start, end = parsed
    chunks = base_cfg["chunks"]
    first = chunks[start]
    last = chunks[end]
    engine = interval_dir / f"chunk_{precision}.engine"
    rel_engine = str(engine.relative_to(REPO))
    variant = f"reprofile_{interval_dir.name}"
    return {
        "model": model,
        "variant": variant,
        "n_chunks": 1,
        "input_shape": base_cfg["input_shape"],
        "chunks": [
            {
                "id": 0,
                "name": "chunk0",
                "chunk_name": interval_dir.name,
                "description": f"cached interval {interval_dir.name}",
                "input_shape": first["input_shape"],
                "output_shape": last["output_shape"],
                "engine_fp32": rel_engine if precision == "fp32" else "",
                "engine_fp16": rel_engine if precision == "fp16" else "",
            }
        ],
        "full_model": {
            "engine_fp32": str((interval_dir / "missing_full_fp32.engine").relative_to(REPO)),
            "engine_fp16": str((interval_dir / "missing_full_fp16.engine").relative_to(REPO)),
        },
    }


def _run_table4(config_path: Path, precision: str, warmup: int, iters: int) -> tuple[Optional[Path], float]:
    cmd = [
        str(TABLE4_RUNNER),
        "--config", str(config_path),
        "--repo", str(REPO),
        "--precision", precision,
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            print(f"    {line}")
    if proc.returncode != 0:
        print(f"    FAILED: table4_runner exit {proc.returncode}", file=sys.stderr)
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        return None, wall
    out = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    out_path = Path(out) if out else None
    if out_path and out_path.exists():
        return out_path, wall
    return None, wall


def _update_timing(
    timing_path: Path,
    model: str,
    interval_dir: Path,
    precision: str,
    result_json: Path,
    profile_wall_s: float,
    warmup: int,
    iters: int,
) -> None:
    parsed = _parse_interval_name(interval_dir)
    if parsed is None:
        raise ValueError(f"Invalid interval directory: {interval_dir}")
    start, end = parsed
    result = json.loads(result_json.read_text())
    chunks = result.get("chunks") or []
    if len(chunks) != 1:
        raise RuntimeError(f"Expected one chunk in {result_json}, got {len(chunks)}")
    ch = chunks[0]
    timing = _load_json(timing_path)
    timing.update({
        "model": model,
        "source_chunk_ids": list(range(start, end + 1)),
        "start_idx": start,
        "end_idx": end,
        "precision": precision,
        f"gpu_mean_ms_{precision}": ch.get("gpu_mean_ms"),
        f"gpu_p99_ms_{precision}": ch.get("gpu_p99_ms"),
        f"gpu_max_ms_{precision}": ch.get("gpu_max_ms"),
        f"profile_wall_s_{precision}": profile_wall_s,
        f"profile_warmup_{precision}": int(warmup),
        f"profile_iters_{precision}": int(iters),
        f"profile_result_json_{precision}": str(result_json.relative_to(REPO)),
        f"profile_timestamp_{precision}": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _write_json(timing_path, timing)


def main() -> int:
    args = parse_args()
    if not TABLE4_RUNNER.exists() and not args.dry_run:
        print(f"ERROR: table4_runner not found: {TABLE4_RUNNER}", file=sys.stderr)
        return 1

    models = _discover_models(args.models)
    if not models:
        print("No chunk_cache models found.")
        return 0

    profiled = 0
    skipped = 0
    failed = 0
    considered = 0

    for model in models:
        try:
            base_cfg = _load_base_config(model)
        except Exception as exc:
            print(f"[{model}] skip: {exc}")
            continue

        for interval_dir in _interval_dirs(model, args.precision):
            if args.limit is not None and profiled >= args.limit:
                print(f"Limit reached ({args.limit}).")
                print(f"Summary: considered={considered} profiled={profiled} skipped={skipped} failed={failed}")
                return 0
            considered += 1
            timing_path = interval_dir / "timing.json"
            timing = _load_json(timing_path)
            max_key = f"gpu_max_ms_{args.precision}"
            warm_key = f"profile_warmup_{args.precision}"
            iter_key = f"profile_iters_{args.precision}"
            if (
                not args.force
                and timing.get(max_key) is not None
                and timing.get(warm_key) == args.warmup
                and timing.get(iter_key) == args.iters
            ):
                skipped += 1
                continue

            print(f"[{model}] {interval_dir.name} ({args.precision})")
            if args.dry_run:
                profiled += 1
                continue

            with tempfile.TemporaryDirectory(prefix="trtdnn_reprofile_") as td:
                cfg_path = Path(td) / f"{model}_{interval_dir.name}.json"
                cfg_path.write_text(json.dumps(
                    _make_profile_config(model, interval_dir, args.precision, base_cfg),
                    indent=2,
                ))
                out_path, wall = _run_table4(cfg_path, args.precision, args.warmup, args.iters)
                if out_path is None:
                    failed += 1
                    continue
                _update_timing(
                    timing_path,
                    model,
                    interval_dir,
                    args.precision,
                    out_path,
                    wall,
                    args.warmup,
                    args.iters,
                )
                profiled += 1

    print(f"Summary: considered={considered} profiled={profiled} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
