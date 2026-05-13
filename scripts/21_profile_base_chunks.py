#!/usr/bin/env python3
"""
21_profile_base_chunks.py — Profile dag_aligned_full base chunks and write
results/table4/<model>_cpp_dag_aligned_full_<precision>.json.

This script MUST be run before any paper dry-run or live Fig.4/Fig.5
experiment.  Without real per-chunk timing data the schedulability analysis
uses approximated equal-weight allocation and produces results that differ
from the paper.

Workflow
--------
1. Verify dag_aligned_full.json split config exists for each model.
2. Export ONNX for each chunk (skipped if already present).
3. Build per-chunk TRT engines via trtexec (skipped if already present).
4. Run cpp_runtime/build/table4_runner for each model to measure GPU timing.
5. Write results/table4/<model>_cpp_dag_aligned_full_<precision>.json.
6. Import results into results/optimization/.profiling_cache.json.

Prerequisites
-------------
A. Build the C++ profiler (one-time; requires cmake ≥ 3.18):
     cd cpp_runtime
     mkdir -p build && cd build
     cmake .. -DCMAKE_BUILD_TYPE=Release
     make -j$(nproc)
   This produces cpp_runtime/build/table4_runner.

B. Ensure TRTEXEC env var is set if trtexec is not at the default path:
     export TRTEXEC=/usr/src/tensorrt/bin/trtexec

Usage
-----
  conda run -n trt python scripts/21_profile_base_chunks.py \\
      --models alexnet resnet18 vgg19 \\
      --precision fp32 \\
      --warmup 20 \\
      --iters 200

  # Force re-export / re-build even if artifacts already exist:
  conda run -n trt python scripts/21_profile_base_chunks.py \\
      --models alexnet --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TRTEXEC_DEFAULT = "/usr/src/tensorrt/bin/trtexec"
TABLE4_RUNNER   = REPO / "cpp_runtime" / "build" / "table4_runner"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Profile dag_aligned_full base chunks.")
    ap.add_argument("--models", nargs="+", default=["alexnet", "resnet18", "vgg19"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--force", action="store_true",
                    help="Re-export ONNX and re-build engines even if present")
    ap.add_argument("--skip-cpp", action="store_true",
                    help="Skip table4_runner; just export ONNX and build engines")
    return ap.parse_args()


# ── Prerequisites ─────────────────────────────────────────────────────────────

def _check_table4_runner() -> Path:
    """Verify table4_runner exists; print build instructions and exit if not."""
    if TABLE4_RUNNER.exists():
        return TABLE4_RUNNER
    print(
        "\n[error] cpp_runtime/build/table4_runner not found.\n"
        "\n"
        "Build it with:\n"
        "    cd cpp_runtime\n"
        "    mkdir -p build && cd build\n"
        "    cmake .. -DCMAKE_BUILD_TYPE=Release\n"
        "    make -j$(nproc)\n"
        "\n"
        "cmake ≥ 3.18 and CUDA/TensorRT dev headers are required.\n"
        "On Jetson (JetPack 6 / L4T R36):  sudo apt install cmake\n"
        "Then rerun this script.\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _trtexec() -> Path:
    path = Path(os.environ.get("TRTEXEC", TRTEXEC_DEFAULT))
    if not path.exists():
        print(
            f"\n[error] trtexec not found at {path}.\n"
            "Set TRTEXEC=/path/to/trtexec or install TensorRT.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return path


def _check_split_config(model: str) -> Path:
    p = REPO / "artifacts" / "split_configs" / model / "dag_aligned_full.json"
    if not p.exists():
        print(
            f"\n[error] dag_aligned_full.json not found for {model!r}.\n"
            f"Run: conda run -n trt python scripts/26_generate_dag_aligned_configs.py"
            f" --models {model}\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return p


# ── ONNX export ───────────────────────────────────────────────────────────────

def _export_onnx(cfg_path: Path, precision: str, force: bool) -> None:
    from src.models.registry import build_model
    from src.export.onnx_exporter import export_module
    from src.splitting.dag_aligned_split import make_dag_aligned_chunks

    cfg = json.loads(cfg_path.read_text())
    model_name = cfg["model"]
    model = build_model(model_name).eval()
    chunks = make_dag_aligned_chunks(model_name, model)

    print(f"  Exporting {len(chunks)} ONNX chunks for {model_name}…")
    for spec, chunk_cfg in zip(chunks, cfg["chunks"]):
        onnx_rel = chunk_cfg["onnx"]
        onnx_path = REPO / onnx_rel
        if onnx_path.exists() and not force:
            print(f"    [{chunk_cfg['id']:2d}] skip (exists): {onnx_path.name}")
            continue
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_module(spec.module.cpu(), tuple(spec.input_shape), onnx_path)
        print(f"    [{chunk_cfg['id']:2d}] exported → {onnx_path.name}")


# ── Engine build ──────────────────────────────────────────────────────────────

def _build_engines(cfg_path: Path, precision: str, force: bool, trtexec_bin: Path) -> None:
    cfg = json.loads(cfg_path.read_text())
    model_name = cfg["model"]
    engine_key = f"engine_{precision}"

    print(f"  Building {precision.upper()} engines for {model_name}…")
    for chunk_cfg in cfg["chunks"]:
        onnx_path   = REPO / chunk_cfg["onnx"]
        engine_path = REPO / chunk_cfg[engine_key]

        if not onnx_path.exists():
            print(f"    [{chunk_cfg['id']:2d}] ERROR: ONNX missing: {onnx_path}")
            sys.exit(1)

        if engine_path.exists() and not force:
            print(f"    [{chunk_cfg['id']:2d}] skip (exists): {engine_path.name}")
            continue

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        log_dir = REPO / "artifacts" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{model_name}_21_chunk{chunk_cfg['id']}_{precision}.log"

        cmd = [
            str(trtexec_bin),
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--noDataTransfers",
            "--iterations=100",
        ]
        if precision == "fp16":
            cmd.append("--fp16")

        t0 = time.time()
        with log_path.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"    [{chunk_cfg['id']:2d}] BUILD FAILED ({elapsed:.1f}s). "
                  f"See {log_path.name}")
            sys.exit(1)
        print(f"    [{chunk_cfg['id']:2d}] built ({elapsed:.1f}s) → {engine_path.name}")


# ── table4_runner ─────────────────────────────────────────────────────────────

def _run_table4(
    cfg_path: Path,
    precision: str,
    warmup: int,
    iters: int,
    runner: Path,
) -> Path:
    cfg = json.loads(cfg_path.read_text())
    model_name = cfg["model"]
    out_path = (
        REPO / "results" / "table4"
        / f"{model_name}_cpp_dag_aligned_full_{precision}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(runner),
        "--config", str(cfg_path),
        "--repo",   str(REPO),
        "--precision", precision,
        "--warmup", str(warmup),
        "--iters",  str(iters),
    ]
    print(f"  Running table4_runner for {model_name} ({precision})…")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  [error] table4_runner failed for {model_name} ({elapsed:.1f}s).",
              file=sys.stderr)
        sys.exit(1)
    print(f"  Done in {elapsed:.1f}s.")

    if out_path.exists():
        return out_path
    # runner may write to a different path; search for it
    candidates = list((REPO / "results" / "table4").glob(
        f"{model_name}_cpp_*_{precision}.json"
    ))
    if candidates:
        return candidates[0]
    print(f"  [warn] table4_runner completed but output not found at {out_path}")
    return out_path


# ── Profiling cache import ────────────────────────────────────────────────────

def _import_to_cache(table4_paths: List[Path], precision: str) -> None:
    """Import table4 results into the profiling cache."""
    try:
        from src.optimization.profiling_db import ProfilingDB
        cache_path = REPO / "results" / "optimization" / ".profiling_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        db = ProfilingDB(cache_path)
        imported = 0
        for p in table4_paths:
            if db.import_from_cpp_result(p):
                imported += 1
        print(f"\n[cache] Imported {imported} model(s) into {cache_path.relative_to(REPO)}")
    except Exception as exc:
        print(f"[warn] Cache import failed (non-fatal): {exc}")


# ── Summary print ─────────────────────────────────────────────────────────────

def _print_summary(table4_paths: List[Path], precision: str) -> None:
    print("\n" + "=" * 60)
    print(f"Base chunk profiling summary ({precision.upper()})")
    print("=" * 60)
    print(f"{'Model':<12}  {'N':>4}  {'sum_mean (ms)':>14}  {'sum_p99 (ms)':>13}  {'sum_max (ms)':>13}")
    print("-" * 68)
    for p in table4_paths:
        if not p.exists():
            print(f"  {p.stem}: MISSING")
            continue
        data = json.loads(p.read_text())
        model = data.get("model", "?")
        chunks = data.get("chunks", [])
        mean_sum = sum(c["gpu_mean_ms"] for c in chunks)
        p99_sum  = sum(c["gpu_p99_ms"]  for c in chunks)
        max_sum  = sum(c.get("gpu_max_ms", 0.0) for c in chunks)
        print(
            f"  {model:<12}  {len(chunks):>4}  "
            f"{mean_sum:>14.4f}  {p99_sum:>13.4f}  {max_sum:>13.4f}"
        )
    print()
    print("These values will now be used by scripts 30, 40, and 15.")
    print("Run scripts/15_compare_k1_timing_semantics.py to verify consistency.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    runner: Optional[Path] = None
    if not args.skip_cpp:
        runner = _check_table4_runner()
    trtexec_bin = _trtexec()

    print(f"Models: {args.models}")
    print(f"Precision: {args.precision}  warmup={args.warmup}  iters={args.iters}")
    print()

    table4_paths: List[Path] = []

    for model in args.models:
        print(f"── {model} ─────────────────────────────────────────")
        cfg_path = _check_split_config(model)

        _export_onnx(cfg_path, args.precision, args.force)
        _build_engines(cfg_path, args.precision, args.force, trtexec_bin)

        if not args.skip_cpp and runner is not None:
            out = _run_table4(cfg_path, args.precision, args.warmup, args.iters, runner)
            table4_paths.append(out)

        print()

    if table4_paths:
        _import_to_cache(table4_paths, args.precision)
        _print_summary(table4_paths, args.precision)
        print("\nBase profiling complete.  You can now run:")
        print(f"  python scripts/30_run_yaml_fig4_experiment.py --models {' '.join(args.models)} ...")
    elif args.skip_cpp:
        print("ONNX export and engine build complete (--skip-cpp: table4_runner not run).")
        print("To finish profiling, build cpp_runtime and rerun without --skip-cpp.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
