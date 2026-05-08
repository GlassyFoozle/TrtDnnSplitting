"""
compiler.py — Thin wrapper around the existing export/build infrastructure.

Given a selected split config (from selective_split.py), this module:
  1. Exports merged-chunk ONNX files (calls script 31 logic)
  2. Builds TRT engines via trtexec (calls script 32 logic)

Supports dry_run (print commands without executing) and caching
(skip steps if artifacts already exist and force=False).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent


def _onnx_exists(cfg: dict) -> bool:
    return all((REPO / c["onnx"]).exists() for c in cfg["chunks"])


def _engines_exist(cfg: dict, precision: str) -> bool:
    key = f"engine_{precision}"
    return all((REPO / c[key]).exists() for c in cfg["chunks"])


def export_onnx(
    model_name: str,
    variant_name: str,
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Export ONNX files for a selected split config.
    Returns True if export was run (or would be run in dry_run mode).
    """
    cfg_path = REPO / "artifacts" / "split_configs" / model_name / f"{variant_name}.json"
    if not cfg_path.exists():
        print(f"  [compiler] Config not found: {cfg_path.relative_to(REPO)}", file=sys.stderr)
        return False

    cfg = json.loads(cfg_path.read_text())
    if not force and _onnx_exists(cfg):
        print(f"  [compiler] ONNX already exists for {model_name}/{variant_name} — skipping (use force=True to override)")
        return False

    cmd = [
        "conda", "run", "-n", "trt",
        "python", str(REPO / "scripts" / "internal_export_selected_split.py"),
        "--model", model_name,
        "--name", variant_name,
        "--skip-full",
    ]

    if dry_run:
        print(f"  [DRY-RUN] export ONNX: {' '.join(cmd)}")
        return True

    print(f"  [compiler] Exporting ONNX for {model_name}/{variant_name}...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [compiler] ONNX export FAILED (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def build_engines(
    model_name: str,
    variant_name: str,
    precision: str = "fp32",
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Build TRT engines for a selected split config.
    Returns True if build was run (or would be run in dry_run mode).
    """
    cfg_path = REPO / "artifacts" / "split_configs" / model_name / f"{variant_name}.json"
    if not cfg_path.exists():
        print(f"  [compiler] Config not found: {cfg_path.relative_to(REPO)}", file=sys.stderr)
        return False

    cfg = json.loads(cfg_path.read_text())
    if not force and _engines_exist(cfg, precision):
        print(f"  [compiler] Engines already exist for {model_name}/{variant_name} ({precision}) — skipping")
        return False

    fp32_flag = "--fp32-only" if precision == "fp32" else "--fp16-only"
    cmd = [
        "bash",
        str(REPO / "scripts" / "internal_build_selected_engines.sh"),
        "--model", model_name,
        "--name", variant_name,
        fp32_flag,
    ]

    if dry_run:
        print(f"  [DRY-RUN] build engines: {' '.join(cmd)}")
        return True

    print(f"  [compiler] Building TRT engines for {model_name}/{variant_name} ({precision})...")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [compiler] Engine build FAILED (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def build_single_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp32",
    dry_run: bool = False,
) -> Tuple[bool, float]:
    """
    Build a single TRT engine from an ONNX file via trtexec.

    Returns (success, wall_seconds). On dry_run returns (True, 0.0).
    Falls back gracefully when trtexec is not present (non-Jetson).
    """
    from pathlib import Path as _Path
    trtexec = _Path(os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec"))
    if not trtexec.exists():
        print(
            f"  [compiler] trtexec not found at {trtexec} — cannot build {onnx_path.name}",
            file=sys.stderr,
        )
        return False, 0.0

    if not onnx_path.exists():
        print(f"  [compiler] ONNX not found: {onnx_path}", file=sys.stderr)
        return False, 0.0

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--noDataTransfers",
        "--iterations=100",
    ]
    if precision == "fp16":
        cmd.append("--fp16")

    if dry_run:
        print(f"  [DRY-RUN] build single engine: {' '.join(cmd)}")
        return True, 0.0

    log_dir = REPO / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{onnx_path.stem}_{precision}.log"

    print(f"  [build] {onnx_path.name} -> {engine_path.name}")
    t0 = time.perf_counter()
    with log_path.open("w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - t0

    if proc.returncode != 0:
        print(
            f"  [compiler] Engine build FAILED for {onnx_path.name}. See {log_path}",
            file=sys.stderr,
        )
        return False, wall
    return True, wall


def compile_config(
    model_name: str,
    variant_name: str,
    precision: str = "fp32",
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Full compile pipeline: export ONNX then build engines.
    Returns a status dict.
    """
    onnx_ok = export_onnx(model_name, variant_name, force=force, dry_run=dry_run)
    engine_ok = build_engines(model_name, variant_name, precision=precision, force=force, dry_run=dry_run)
    return {
        "model": model_name,
        "variant": variant_name,
        "precision": precision,
        "onnx_exported": onnx_ok,
        "engines_built": engine_ok,
        "status": "ok" if (onnx_ok or not force) and engine_ok else "partial",
    }
