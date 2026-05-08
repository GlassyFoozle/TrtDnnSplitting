"""
config_evaluator.py — Profile-in-the-loop mask evaluator.

Takes a boundary mask, compiles it to a selected split config, exports ONNXs,
builds TRT engines, and runs the C++ profiler.  Results are cached in
ProfilingDB so repeated calls with the same mask return immediately.

This is the primary reusable backend for the future splitting algorithm.

Flow
----
  mask
  → [selective_split]   generate / save selected config
  → [compiler]          export merged-chunk ONNXs
  → [compiler]          build TRT engines
  → [C++ table4_runner] measure GPU latency
  → [ProfilingDB]       cache result
  → EvaluationResult

Naming
------
If no variant_name is given, a deterministic name is derived:
  <model>_mask_<8-char SHA-256 of mask string>_k<n_chunks>
e.g.  alexnet_mask_a3f7c291_k4

Output locations
----------------
  configs:    artifacts/split_configs/<model>/<variant>.json
  eval JSON:  results/evaluations/<model>/<variant>_<precision>.json
  raw C++:    results/evaluations/<model>/<variant>_<precision>_cpp_raw.json
  cache:      results/optimization/.profiling_cache.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

_CPP_RUNNER = REPO / "cpp_runtime" / "build" / "table4_runner"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    # Identity
    model_name: str
    variant_name: str
    base_variant: str
    precision: str
    mask: List[int]
    groups: List[List[int]]
    n_chunks: int

    # Artifact paths (relative to REPO for portability)
    config_path: str = ""
    result_json_path: str = ""

    # Pipeline status
    exported: bool = False
    built: bool = False
    profiled: bool = False
    cache_hit: bool = False

    # Timing (None = not measured)
    full_gpu_mean_ms: Optional[float] = None
    full_gpu_p99_ms: Optional[float] = None
    chunked_gpu_mean_ms: Optional[float] = None
    chunked_gpu_p99_ms: Optional[float] = None
    overhead_ms: Optional[float] = None
    overhead_pct: Optional[float] = None

    # Per-chunk timing lists (indexed by merged-chunk id)
    per_chunk_gpu_mean_ms: Optional[List[float]] = None
    per_chunk_gpu_p99_ms: Optional[List[float]] = None
    per_chunk_cpu_wall_mean_ms: Optional[List[float]] = None
    per_chunk_cpu_wall_p99_ms: Optional[List[float]] = None

    # Diagnostics
    notes: str = ""
    error: Optional[str] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def mask_str(self) -> str:
        return "".join(str(b) for b in self.mask)

    def ok(self) -> bool:
        return self.error is None and self.chunked_gpu_mean_ms is not None

    def summary(self) -> str:
        lines = [
            f"EvaluationResult: {self.model_name}/{self.variant_name}  ({self.precision})",
            f"  mask    : {self.mask_str()}  K={self.n_chunks}",
            f"  status  : exported={self.exported} built={self.built} "
            f"profiled={self.profiled} cache_hit={self.cache_hit}",
        ]
        if self.chunked_gpu_mean_ms is not None:
            lines.append(
                f"  chunked : {self.chunked_gpu_mean_ms:.4f} ms (p99={self.chunked_gpu_p99_ms:.4f} ms)"
            )
        if self.full_gpu_mean_ms is not None:
            lines.append(
                f"  full    : {self.full_gpu_mean_ms:.4f} ms"
            )
        if self.overhead_pct is not None:
            lines.append(f"  overhead: {self.overhead_pct:+.2f}%")
        if self.per_chunk_gpu_mean_ms:
            for ci, (m, p) in enumerate(zip(
                self.per_chunk_gpu_mean_ms,
                self.per_chunk_gpu_p99_ms or [None] * len(self.per_chunk_gpu_mean_ms),
            )):
                p_str = f"  p99={p:.4f}" if p else ""
                lines.append(f"  chunk{ci} : {m:.4f} ms{p_str}")
        if self.notes:
            lines.append(f"  notes   : {self.notes}")
        if self.error:
            lines.append(f"  ERROR   : {self.error}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Variant naming ─────────────────────────────────────────────────────────────

def mask_to_variant_name(model_name: str, mask: List[int]) -> str:
    """
    Derive a deterministic variant name from a mask.
    e.g. alexnet_mask_a3f7c291_k4
    """
    mask_str = "".join(str(b) for b in mask)
    h = hashlib.sha256(mask_str.encode()).hexdigest()[:8]
    k = sum(mask) + 1
    return f"{model_name}_mask_{h}_k{k}"


# ── Path helpers ───────────────────────────────────────────────────────────────

def _eval_dir(model_name: str) -> Path:
    return REPO / "results" / "evaluations" / model_name


def _eval_json_path(model_name: str, variant_name: str, precision: str) -> Path:
    return _eval_dir(model_name) / f"{variant_name}_{precision}.json"


def _cpp_raw_path(model_name: str, variant_name: str, precision: str) -> Path:
    return _eval_dir(model_name) / f"{variant_name}_{precision}_cpp_raw.json"


def _cpp_table4_output_path(model_name: str, variant_name: str, precision: str) -> Path:
    """Where table4_runner writes its output by default."""
    return REPO / "results" / "table4" / f"{model_name}_cpp_{variant_name}_{precision}.json"


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_db() -> "ProfilingDB":
    from src.optimization.profiling_db import ProfilingDB
    cache_path = REPO / "results" / "optimization" / ".profiling_cache.json"
    db = ProfilingDB(cache_path)
    db.import_all_cpp_results(REPO)
    return db


def is_mask_cached(model_name: str, mask: List[int], precision: str) -> bool:
    """Return True if a valid, completed evaluation result exists for this mask.

    Checks beyond mere file existence (Fix 4):
    - JSON must be parseable
    - must not have an error field set
    - per_chunk_gpu_mean_ms must be present and have length == sum(mask)+1
    """
    variant_name = mask_to_variant_name(model_name, mask)
    p = _eval_json_path(model_name, variant_name, precision)
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
    except Exception:
        return False
    if d.get("error"):
        return False
    chunk_times = d.get("per_chunk_gpu_mean_ms")
    if not chunk_times:
        return False
    expected_chunks = sum(mask) + 1
    return len(chunk_times) == expected_chunks


def _load_cached_result(model_name: str, variant_name: str, precision: str) -> Optional[EvaluationResult]:
    """Try loading a previously saved EvaluationResult JSON."""
    p = _eval_json_path(model_name, variant_name, precision)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        r = EvaluationResult.from_dict(d)
        r.cache_hit = True
        return r
    except Exception:
        return None


def _save_result(result: EvaluationResult) -> Path:
    """Persist EvaluationResult to results/evaluations/<model>/."""
    out = _eval_json_path(result.model_name, result.variant_name, result.precision)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    return out


# ── C++ profiler ───────────────────────────────────────────────────────────────

def _run_cpp_profiler(
    model_name: str,
    variant_name: str,
    precision: str,
    warmup: int,
    iters: int,
) -> Optional[Path]:
    """
    Run table4_runner on the selected split config.
    Returns path to the output JSON (results/table4/...) on success, None on failure.
    """
    if not _CPP_RUNNER.exists():
        print(f"  [evaluator] C++ runner not found: {_CPP_RUNNER}", file=sys.stderr)
        return None

    cfg_path = REPO / "artifacts" / "split_configs" / model_name / f"{variant_name}.json"
    if not cfg_path.exists():
        print(f"  [evaluator] Config not found: {cfg_path}", file=sys.stderr)
        return None

    cmd = [
        str(_CPP_RUNNER),
        "--config", str(cfg_path),
        "--repo", str(REPO),
        "--precision", precision,
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]

    print(f"  [evaluator] Running C++ profiler: {model_name}/{variant_name} ({precision})")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print stderr (progress output) to console
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"    {line}")

    if result.returncode != 0:
        print(f"  [evaluator] C++ runner FAILED (exit {result.returncode})", file=sys.stderr)
        return None

    # The runner prints the output path to stdout
    out_path_str = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    expected = _cpp_table4_output_path(model_name, variant_name, precision)
    if expected.exists():
        return expected
    if out_path_str and Path(out_path_str).exists():
        return Path(out_path_str)
    return None


def _parse_cpp_result(json_path: Path) -> dict:
    """Parse C++ table4_runner output JSON into a timing dict."""
    d = json.loads(json_path.read_text())
    chunks = d.get("chunks", [])
    # C++ runner returns 0.0 for the full engine when it isn't present → treat as None
    full_mean = d.get("full_engine_gpu_mean_ms") or None
    full_p99  = d.get("full_engine_gpu_p99_ms")  or None
    if full_mean is not None and full_mean <= 0.0:
        full_mean = None
        full_p99  = None
    return {
        "full_gpu_mean_ms": full_mean,
        "full_gpu_p99_ms": full_p99,
        "chunked_gpu_mean_ms": d.get("total_chunked_gpu_mean_ms"),
        "chunked_gpu_p99_ms": d.get("total_chunked_gpu_p99_ms"),
        "per_chunk_gpu_mean_ms": [c["gpu_mean_ms"] for c in chunks],
        "per_chunk_gpu_p99_ms": [c["gpu_p99_ms"] for c in chunks],
        "per_chunk_cpu_wall_mean_ms": [c.get("cpu_mean_ms") for c in chunks],
        "per_chunk_cpu_wall_p99_ms": [c.get("cpu_p99_ms") for c in chunks],
    }


# ── Python TRT fallback profiler ───────────────────────────────────────────────

def _run_python_profiler(
    model_name: str,
    variant_name: str,
    precision: str,
    warmup: int,
    iters: int,
) -> Optional[dict]:
    """
    Fall back to script 33 (Python TRT) when C++ runner is unavailable.
    Returns parsed timing dict or None.
    """
    cmd = [
        "conda", "run", "-n", "trt",
        "python", str(REPO / "scripts" / "33_profile_selected_split.py"),
        "--model", model_name,
        "--name", variant_name,
        "--precision", precision,
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    print(f"  [evaluator] Running Python TRT profiler (C++ fallback): {model_name}/{variant_name}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        return None

    profile_json = (
        REPO / "results" / "selected_splits" / model_name
        / f"{variant_name}_{precision}_profile.json"
    )
    if not profile_json.exists():
        return None

    pd = json.loads(profile_json.read_text())
    trt = pd.get("trt_profiling") or {}
    if not trt:
        return None

    chunks = trt.get("chunks", [])
    full_info = trt.get("full_engine") or {}
    full_mean = (full_info.get("gpu_ms") or {}).get("mean")
    chunked_ms = (trt.get("total_gpu_ms") or {}).get("mean")

    return {
        "full_gpu_mean_ms": full_mean,
        "full_gpu_p99_ms": (full_info.get("gpu_ms") or {}).get("p99"),
        "chunked_gpu_mean_ms": chunked_ms,
        "chunked_gpu_p99_ms": (trt.get("total_gpu_ms") or {}).get("p99"),
        "per_chunk_gpu_mean_ms": [c["gpu_ms"]["mean"] for c in chunks],
        "per_chunk_gpu_p99_ms": [c["gpu_ms"]["p99"] for c in chunks],
        "per_chunk_cpu_wall_mean_ms": None,
        "per_chunk_cpu_wall_p99_ms": None,
    }


# ── Main evaluator ─────────────────────────────────────────────────────────────

def evaluate_mask(
    model_name: str,
    mask: "List[int] | str",
    variant_name: Optional[str] = None,
    base_variant: str = "dag_aligned_full",
    precision: str = "fp32",
    warmup: int = 20,
    iters: int = 200,
    use_cpp: bool = True,
    export: bool = True,
    build: bool = True,
    profile: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> EvaluationResult:
    """
    Evaluate a boundary mask end-to-end: generate config → export → build → profile.

    Parameters
    ----------
    model_name    : e.g. "alexnet"
    mask          : binary list, binary string ("010110…"), "all", or "none"
    variant_name  : override auto-generated name
    base_variant  : base split config to derive from (default: dag_aligned_full)
    precision     : "fp32" or "fp16"
    warmup / iters: profiling parameters
    use_cpp       : use C++ table4_runner (True) or Python TRT fallback (False)
    export        : export ONNX chunks if not already present
    build         : build TRT engines if not already present
    profile       : run GPU profiler
    force         : re-export/rebuild/reprofile even if artifacts exist
    dry_run       : print plan without executing anything

    Returns
    -------
    EvaluationResult with timing data populated (or error set on failure).
    """
    from src.splitting.selective_split import (
        load_base_config, parse_boundary_mask,
        make_selected_split_config, save_selected_config,
    )
    from src.optimization.compiler import export_onnx, build_engines, _engines_exist, _onnx_exists

    # ── 1. Parse mask ─────────────────────────────────────────────────────────
    base_cfg = load_base_config(model_name, base_variant)
    n_base = len(base_cfg["chunks"])

    try:
        mask_list = parse_boundary_mask(mask, n_base) if isinstance(mask, str) else list(mask)
    except ValueError as e:
        return EvaluationResult(
            model_name=model_name, variant_name=variant_name or "unknown",
            base_variant=base_variant, precision=precision,
            mask=[], groups=[], n_chunks=0,
            error=f"Mask parse error: {e}",
        )

    from src.splitting.selective_split import compute_merge_groups
    groups = compute_merge_groups(mask_list)
    n_chunks = len(groups)

    # ── 2. Determine variant name ──────────────────────────────────────────────
    if variant_name is None:
        variant_name = mask_to_variant_name(model_name, mask_list)

    print(f"\n[evaluator] {model_name}/{variant_name}  K={n_chunks}  ({precision})")
    print(f"  mask: {''.join(str(b) for b in mask_list)}")

    # ── 3. Cache check ────────────────────────────────────────────────────────
    if not force and not dry_run:
        cached = _load_cached_result(model_name, variant_name, precision)
        if cached is not None and cached.ok():
            print(f"  → cache hit (result JSON exists)")
            return cached

    # ── 4. Generate and save config ───────────────────────────────────────────
    if dry_run:
        print(f"  [DRY-RUN] would generate config: "
              f"artifacts/split_configs/{model_name}/{variant_name}.json")
        print(f"  [DRY-RUN] groups: {groups}")
        if export:
            print(f"  [DRY-RUN] would export ONNX: artifacts/onnx/{model_name}/{variant_name}/")
        if build:
            print(f"  [DRY-RUN] would build engines: artifacts/engines/{model_name}/{variant_name}/")
        if profile:
            runner_str = "C++ table4_runner" if use_cpp else "Python TRT"
            print(f"  [DRY-RUN] would profile with {runner_str} (warmup={warmup}, iters={iters})")
        return EvaluationResult(
            model_name=model_name, variant_name=variant_name,
            base_variant=base_variant, precision=precision,
            mask=mask_list, groups=groups, n_chunks=n_chunks,
            notes="dry_run",
        )

    cfg = make_selected_split_config(
        model_name=model_name,
        base_variant=base_variant,
        mask=mask_list,
        selected_name=variant_name,
        base_cfg=base_cfg,
    )
    cfg_path = save_selected_config(cfg, model_name, variant_name)
    print(f"  config → {cfg_path.relative_to(REPO)}")

    result = EvaluationResult(
        model_name=model_name,
        variant_name=variant_name,
        base_variant=base_variant,
        precision=precision,
        mask=mask_list,
        groups=groups,
        n_chunks=n_chunks,
        config_path=str(cfg_path.relative_to(REPO)),
    )

    # ── 5. Export ONNX ────────────────────────────────────────────────────────
    if export:
        already = _onnx_exists(cfg)
        if already and not force:
            print(f"  ONNX already present — skipping export")
            result.exported = False
        else:
            ok = export_onnx(model_name, variant_name, force=force)
            result.exported = ok
            if not ok:
                result.error = "ONNX export failed"
                _save_result(result)
                return result

    # ── 6. Build engines ──────────────────────────────────────────────────────
    if build:
        already = _engines_exist(cfg, precision)
        if already and not force:
            print(f"  Engines already present — skipping build")
            result.built = False
        else:
            ok = build_engines(model_name, variant_name, precision=precision, force=force)
            result.built = ok
            if not ok:
                result.error = "Engine build failed"
                _save_result(result)
                return result

    # ── 7. Profile ────────────────────────────────────────────────────────────
    if not profile:
        result.notes = "profile=False — no timing measured"
        _save_result(result)
        return result

    # C++ preferred
    timing: Optional[dict] = None
    cpp_raw_json: Optional[Path] = None

    if use_cpp and _CPP_RUNNER.exists():
        # Check if engines are present before running
        if not _engines_exist(cfg, precision):
            result.error = "Engines not built — cannot profile"
            _save_result(result)
            return result

        cpp_out = _run_cpp_profiler(model_name, variant_name, precision, warmup, iters)
        if cpp_out is not None and cpp_out.exists():
            result.profiled = True
            timing = _parse_cpp_result(cpp_out)

            # Copy raw C++ JSON to evaluations dir
            cpp_raw_dst = _cpp_raw_path(model_name, variant_name, precision)
            cpp_raw_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cpp_out, cpp_raw_dst)
            cpp_raw_json = cpp_raw_dst
            result.notes = f"cpp_runner; raw_json={cpp_raw_dst.relative_to(REPO)}"
        else:
            print(f"  [evaluator] C++ profiler failed — trying Python fallback", file=sys.stderr)
    elif use_cpp and not _CPP_RUNNER.exists():
        print(f"  [evaluator] C++ runner not built — falling back to Python TRT", file=sys.stderr)

    # Python fallback
    if timing is None:
        timing = _run_python_profiler(model_name, variant_name, precision, warmup, iters)
        if timing is not None:
            result.profiled = True
            result.notes = (result.notes + "; python_trt_fallback").lstrip("; ")
        else:
            result.error = "All profilers failed"
            _save_result(result)
            return result

    # ── 8. Populate timing fields ─────────────────────────────────────────────
    result.full_gpu_mean_ms = timing.get("full_gpu_mean_ms")
    result.full_gpu_p99_ms = timing.get("full_gpu_p99_ms")
    result.chunked_gpu_mean_ms = timing.get("chunked_gpu_mean_ms")
    result.chunked_gpu_p99_ms = timing.get("chunked_gpu_p99_ms")
    result.per_chunk_gpu_mean_ms = timing.get("per_chunk_gpu_mean_ms")
    result.per_chunk_gpu_p99_ms = timing.get("per_chunk_gpu_p99_ms")
    result.per_chunk_cpu_wall_mean_ms = timing.get("per_chunk_cpu_wall_mean_ms")
    result.per_chunk_cpu_wall_p99_ms = timing.get("per_chunk_cpu_wall_p99_ms")

    if result.full_gpu_mean_ms and result.chunked_gpu_mean_ms:
        result.overhead_ms = result.chunked_gpu_mean_ms - result.full_gpu_mean_ms
        result.overhead_pct = (result.overhead_ms / result.full_gpu_mean_ms) * 100.0

    # ── 9. Update ProfilingDB ──────────────────────────────────────────────────
    try:
        db = _load_db()
        db.put(
            model_name, variant_name, precision,
            full_gpu_mean_ms=result.full_gpu_mean_ms,
            per_chunk_gpu_mean_ms=result.per_chunk_gpu_mean_ms,
            per_chunk_gpu_p99_ms=result.per_chunk_gpu_p99_ms,
            total_chunked_gpu_mean_ms=result.chunked_gpu_mean_ms,
            source_json=str(cpp_raw_json) if cpp_raw_json else "",
        )
    except Exception as e:
        print(f"  [evaluator] WARNING: ProfilingDB update failed: {e}", file=sys.stderr)

    # ── 10. Save and return ───────────────────────────────────────────────────
    result_path = _save_result(result)
    result.result_json_path = str(result_path.relative_to(REPO))
    result_path.write_text(json.dumps(result.to_dict(), indent=2))  # update with path field
    return result
