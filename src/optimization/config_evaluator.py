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
  → [interval cache]    reuse existing ONNX/engines for known chunk intervals
  → [export]            export any uncached merged-chunk ONNXs (per-chunk, inline Python)
  → [build]             build TRT engines (whole-variant; skipped if all from cache)
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
  intervals:  artifacts/chunk_cache/<model>/int_{start}_{end}/ (ONNX + engines + timing)
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
from typing import List, Optional, Tuple

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
    full_gpu_max_ms: Optional[float] = None
    chunked_gpu_mean_ms: Optional[float] = None
    chunked_gpu_p99_ms: Optional[float] = None
    chunked_gpu_max_ms: Optional[float] = None
    overhead_ms: Optional[float] = None
    overhead_pct: Optional[float] = None

    # Per-chunk timing lists (indexed by merged-chunk id)
    per_chunk_gpu_mean_ms: Optional[List[float]] = None
    per_chunk_gpu_p99_ms: Optional[List[float]] = None
    per_chunk_gpu_max_ms: Optional[List[float]] = None
    per_chunk_cpu_wall_mean_ms: Optional[List[float]] = None
    per_chunk_cpu_wall_p99_ms: Optional[List[float]] = None
    per_chunk_cpu_wall_max_ms: Optional[List[float]] = None

    # Interval cache accounting (populated by evaluate_mask when not a mask-level cache hit)
    interval_cache_hits: int = 0       # combined ONNX + engine hits
    interval_cache_misses: int = 0     # combined ONNX + engine misses
    interval_onnx_cache_hits: int = 0
    interval_onnx_cache_misses: int = 0
    interval_engine_cache_hits: int = 0
    interval_engine_cache_misses: int = 0

    # Wall-clock time for each pipeline phase (0.0 if phase was skipped via cache)
    export_wall_s: Optional[float] = None
    build_wall_s: Optional[float] = None
    profile_wall_s: Optional[float] = None
    profile_warmup: Optional[int] = None
    profile_iters: Optional[int] = None
    interval_engine_build_wall_s: float = 0.0

    # Cold-cache design-time estimates: what this mask would have cost without caching
    estimated_cold_export_s: Optional[float] = None
    estimated_cold_build_s: Optional[float] = None
    estimated_cold_total_s: Optional[float] = None

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
            p99_str = (
                f"{self.chunked_gpu_p99_ms:.4f}"
                if self.chunked_gpu_p99_ms is not None else "n/a"
            )
            max_str = (
                f"{self.chunked_gpu_max_ms:.4f}"
                if self.chunked_gpu_max_ms is not None else "n/a"
            )
            lines.append(
                f"  chunked : {self.chunked_gpu_mean_ms:.4f} ms "
                f"(p99={p99_str} ms, max={max_str} ms)"
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


# ── Boundary-mask helpers (torch-free; mirrors selective_split.compute_merge_groups) ──

def _compute_merge_groups(mask: List[int]) -> List[List[int]]:
    """Given a boundary mask of length N-1, return groups of consecutive base-chunk indices."""
    n_chunks = len(mask) + 1
    groups: List[List[int]] = []
    current: List[int] = [0]
    for boundary_idx, bit in enumerate(mask):
        if bit == 1:
            groups.append(current)
            current = [boundary_idx + 1]
        else:
            current.append(boundary_idx + 1)
    groups.append(current)
    return groups


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


# ── Interval cache helpers ─────────────────────────────────────────────────────

def _interval_dir(model_name: str, source_chunk_ids: List[int]) -> Path:
    start, end = source_chunk_ids[0], source_chunk_ids[-1]
    return REPO / "artifacts" / "chunk_cache" / model_name / f"int_{start}_{end}"


def _interval_onnx_path(model_name: str, source_chunk_ids: List[int]) -> Path:
    return _interval_dir(model_name, source_chunk_ids) / "chunk.onnx"


def _interval_engine_path(model_name: str, source_chunk_ids: List[int], precision: str) -> Path:
    return _interval_dir(model_name, source_chunk_ids) / f"chunk_{precision}.engine"


def _interval_timing_path(model_name: str, source_chunk_ids: List[int]) -> Path:
    return _interval_dir(model_name, source_chunk_ids) / "timing.json"


def _load_interval_timing(model_name: str, source_chunk_ids: List[int]) -> dict:
    p = _interval_timing_path(model_name, source_chunk_ids)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_interval_timing(model_name: str, source_chunk_ids: List[int], timing: dict) -> None:
    p = _interval_timing_path(model_name, source_chunk_ids)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(timing, indent=2))


def _export_chunks_with_interval_cache(
    model_name: str,
    cfg: dict,
    groups: List[List[int]],
    force: bool = False,
    device: str = "cuda",
) -> Tuple[int, int, float]:
    """
    Export ONNX for each chunk, using the interval cache as the canonical store.

    Since config chunk paths now point directly to the interval cache directory,
    a cache hit means the file already exists at the config path — no copy needed.
    For cache misses, exports directly to the interval cache path.

    Returns (interval_onnx_hits, interval_onnx_misses, total_export_wall_s).
    """
    from src.models.registry import build_model
    from src.splitting.dag_aligned_split import make_dag_aligned_chunks
    from src.splitting.selective_split import build_merged_module
    from src.export.onnx_exporter import export_module

    chunk_configs = cfg["chunks"]
    needs_export: List[int] = []

    for i, grp in enumerate(groups):
        # Config path IS the interval cache path — no copy needed on hit.
        onnx_path = REPO / chunk_configs[i]["onnx"]
        if not force and onnx_path.exists():
            pass  # cache hit
        else:
            needs_export.append(i)

    cache_hits = len(groups) - len(needs_export)
    cache_misses = len(needs_export)
    total_wall = 0.0

    if not needs_export:
        print(f"  [interval_cache] all {len(groups)} ONNX(es) from interval cache")
        return cache_hits, cache_misses, total_wall

    if cache_hits > 0:
        print(f"  [interval_cache] {cache_hits}/{len(groups)} ONNX(es) from interval cache")

    # Load base model once for all missing chunk exports
    model = build_model(model_name)
    base_specs = make_dag_aligned_chunks(model_name, model)

    for i in needs_export:
        grp = groups[i]
        chunk_cfg = chunk_configs[i]
        onnx_path = REPO / chunk_cfg["onnx"]  # interval cache path
        in_shape = tuple(chunk_cfg["input_shape"])

        base_modules = [base_specs[j].module for j in grp]
        merged_mod, _ = build_merged_module(base_modules, in_shape)

        print(f"  [export] chunk{i} base_chunks={grp}")
        t0 = time.perf_counter()
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        export_module(merged_mod, in_shape, onnx_path, device=device)
        wall = time.perf_counter() - t0
        total_wall += wall

        timing = _load_interval_timing(model_name, grp)
        timing.update({"model": model_name, "source_chunk_ids": grp, "export_wall_s": wall})
        _save_interval_timing(model_name, grp, timing)

    return cache_hits, cache_misses, total_wall


def _build_engines_with_interval_cache(
    model_name: str,
    cfg: dict,
    groups: List[List[int]],
    precision: str,
    force: bool = False,
) -> Tuple[int, int, float]:
    """
    Build TRT engines per-chunk, using the interval cache as the canonical store.

    Since config chunk paths now point directly to the interval cache directory,
    a cache hit means the engine file already exists — no copy needed.
    For cache misses, builds directly to the interval cache path.

    Returns (engine_cache_hits, engine_cache_misses, total_build_wall_s).
    Fails fast (returns immediately) if any per-chunk build fails.
    """
    from src.optimization.compiler import build_single_engine

    chunk_configs = cfg["chunks"]
    hits = 0
    misses = 0
    total_wall = 0.0

    for i, grp in enumerate(groups):
        # Config path IS the interval cache path — no copy needed.
        eng_path  = REPO / chunk_configs[i][f"engine_{precision}"]
        onnx_path = REPO / chunk_configs[i]["onnx"]

        if not force and eng_path.exists():
            hits += 1
            print(f"  [interval_cache] chunk{i} engine from cache (base_chunks={grp})")
        else:
            # Cache miss: build directly into the interval cache directory.
            eng_path.parent.mkdir(parents=True, exist_ok=True)
            ok, wall = build_single_engine(onnx_path, eng_path, precision=precision)
            total_wall += wall
            if not ok:
                return hits, misses + 1, total_wall
            misses += 1
            timing = _load_interval_timing(model_name, grp)
            timing.update({f"build_{precision}_wall_s": wall})
            _save_interval_timing(model_name, grp, timing)

    if misses > 0:
        print(f"  [interval_cache] {hits}/{len(groups)} engine(s) from cache, {misses} built")
    elif hits > 0:
        print(f"  [interval_cache] all {hits}/{len(groups)} engine(s) from interval cache")
    return hits, misses, total_wall


def _estimate_cold_cost(
    model_name: str,
    groups: List[List[int]],
    precision: str,
    actual_profile_wall_s: float,
) -> dict:
    """
    Sum per-interval recorded export+build times to estimate what this mask would
    have cost on a completely cold interval cache (no reuse).
    """
    total_export = 0.0
    total_build = 0.0
    for grp in groups:
        t = _load_interval_timing(model_name, grp)
        total_export += float(t.get("export_wall_s") or 0.0)
        total_build += float(t.get(f"build_{precision}_wall_s") or 0.0)
    has_data = total_export > 0 or total_build > 0
    return {
        "estimated_cold_export_s": total_export if has_data else None,
        "estimated_cold_build_s": total_build if has_data else None,
        "estimated_cold_total_s": (
            total_export + total_build + actual_profile_wall_s if has_data else None
        ),
    }


# ── Interval GPU timing helpers ────────────────────────────────────────────────

def _backfill_interval_gpu_timing(
    model_name: str,
    groups: List[List[int]],
    precision: str,
    result: "EvaluationResult",
) -> None:
    """After profiling, write per-chunk GPU timing into each interval's timing.json.

    This enables cache-only assembly (Task D): once all required intervals for a
    mask have gpu_mean_ms + gpu_p99_ms + gpu_max_ms, the mask can be served
    without re-profiling for max-WCET analysis.
    """
    means = result.per_chunk_gpu_mean_ms or []
    p99s  = result.per_chunk_gpu_p99_ms  or []
    maxs  = result.per_chunk_gpu_max_ms  or []
    if not means or len(means) != len(groups):
        return
    for i, grp in enumerate(groups):
        t = _load_interval_timing(model_name, grp)
        update = {
            "model":           model_name,
            "source_chunk_ids": grp,
            "start_idx":       grp[0],
            "end_idx":         grp[-1],
            "precision":       precision,
            f"gpu_mean_ms_{precision}": means[i],
            f"gpu_p99_ms_{precision}":  p99s[i] if i < len(p99s) else means[i],
        }
        if i < len(maxs) and maxs[i] is not None:
            update[f"gpu_max_ms_{precision}"] = maxs[i]
        if result.profile_wall_s is not None:
            update[f"source_eval_profile_wall_s_{precision}"] = result.profile_wall_s
        if result.profile_warmup is not None:
            update[f"profile_warmup_{precision}"] = result.profile_warmup
        if result.profile_iters is not None:
            update[f"profile_iters_{precision}"] = result.profile_iters
        t.update(update)
        _save_interval_timing(model_name, grp, t)


def can_assemble_from_intervals(
    model_name: str,
    mask: List[int],
    precision: str,
) -> bool:
    """Return True if all required interval timing.json files have GPU timing data.

    Used by cache-only mode to serve masks without re-profiling.
    """
    groups = _compute_merge_groups(mask)
    for grp in groups:
        t = _load_interval_timing(model_name, grp)
        if not (
            t.get(f"gpu_mean_ms_{precision}")
            and t.get(f"gpu_p99_ms_{precision}")
            and t.get(f"gpu_max_ms_{precision}")
        ):
            return False
    return True


def assemble_from_intervals(
    model_name: str,
    mask: List[int],
    precision: str,
    base_variant: str = "dag_aligned_full",
) -> "EvaluationResult":
    """
    Assemble an EvaluationResult from per-interval GPU timing data.

    Called when the mask-level eval JSON is missing but all intervals have timing.
    Writes the assembled result to results/evaluations/ so future calls hit cache.
    """
    variant_name = mask_to_variant_name(model_name, mask)
    groups = _compute_merge_groups(mask)
    n_chunks = len(groups)

    means = []
    p99s  = []
    maxs  = []
    for grp in groups:
        t = _load_interval_timing(model_name, grp)
        means.append(float(t.get(f"gpu_mean_ms_{precision}", 0.0)))
        p99s.append(float(t.get(f"gpu_p99_ms_{precision}", 0.0)))
        maxs.append(float(t.get(f"gpu_max_ms_{precision}", 0.0)))

    chunked_mean = sum(means) if means else None
    chunked_p99  = sum(p99s)  if p99s  else None
    chunked_max  = sum(maxs) if maxs else None

    result = EvaluationResult(
        model_name=model_name,
        variant_name=variant_name,
        base_variant=base_variant,
        precision=precision,
        mask=list(mask),
        groups=groups,
        n_chunks=n_chunks,
        exported=False,
        built=False,
        profiled=False,
        cache_hit=True,
        chunked_gpu_mean_ms=chunked_mean,
        chunked_gpu_p99_ms=chunked_p99,
        chunked_gpu_max_ms=chunked_max,
        per_chunk_gpu_mean_ms=means,
        per_chunk_gpu_p99_ms=p99s,
        per_chunk_gpu_max_ms=maxs,
        notes="assembled_from_interval_timing",
    )
    out = _save_result(result)
    result.result_json_path = str(out.relative_to(REPO))
    out.write_text(json.dumps(result.to_dict(), indent=2))
    return result


def backfill_interval_gpu_timing_from_evals(
    model_name: str,
    precision: str = "fp32",
) -> int:
    """
    One-time utility: populate interval timing.json GPU fields from existing
    results/evaluations/ JSONs.  Returns number of intervals updated.
    """
    from src.splitting.selective_split import compute_merge_groups
    eval_dir = REPO / "results" / "evaluations" / model_name
    if not eval_dir.exists():
        return 0
    updated = 0
    for p in sorted(eval_dir.glob(f"*_{precision}.json")):
        if p.stem.endswith("_cpp_raw"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("error"):
            continue
        mask  = d.get("mask", [])
        means = d.get("per_chunk_gpu_mean_ms") or []
        p99s  = d.get("per_chunk_gpu_p99_ms")  or []
        maxs  = d.get("per_chunk_gpu_max_ms")  or []
        if not mask or not means:
            continue
        groups = compute_merge_groups(mask)
        if len(means) != len(groups):
            continue
        for i, grp in enumerate(groups):
            t = _load_interval_timing(model_name, grp)
            key_mean = f"gpu_mean_ms_{precision}"
            key_p99  = f"gpu_p99_ms_{precision}"
            key_max  = f"gpu_max_ms_{precision}"
            if t.get(key_mean) is None or (maxs and t.get(key_max) is None):
                update = {
                    "model":            model_name,
                    "source_chunk_ids": grp,
                    "start_idx":        grp[0],
                    "end_idx":          grp[-1],
                    "precision":        precision,
                    key_mean:           means[i],
                    key_p99:            p99s[i] if i < len(p99s) else means[i],
                }
                if i < len(maxs) and maxs[i] is not None:
                    update[key_max] = maxs[i]
                t.update(update)
                _save_interval_timing(model_name, grp, t)
                updated += 1
    return updated


# ── Mask-level cache helpers ───────────────────────────────────────────────────

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
    - per_chunk_gpu_max_ms must be present and have length == sum(mask)+1
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
    chunk_times = d.get("per_chunk_gpu_max_ms")
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
    full_max  = d.get("full_engine_gpu_max_ms")  or None
    if full_mean is not None and full_mean <= 0.0:
        full_mean = None
        full_p99  = None
        full_max  = None
    return {
        "full_gpu_mean_ms": full_mean,
        "full_gpu_p99_ms": full_p99,
        "full_gpu_max_ms": full_max,
        "chunked_gpu_mean_ms": d.get("total_chunked_gpu_mean_ms"),
        "chunked_gpu_p99_ms": d.get("total_chunked_gpu_p99_ms"),
        "chunked_gpu_max_ms": d.get("total_chunked_gpu_max_ms"),
        "per_chunk_gpu_mean_ms": [c["gpu_mean_ms"] for c in chunks],
        "per_chunk_gpu_p99_ms": [c["gpu_p99_ms"] for c in chunks],
        "per_chunk_gpu_max_ms": [c.get("gpu_max_ms") for c in chunks],
        "per_chunk_cpu_wall_mean_ms": [c.get("cpu_mean_ms") for c in chunks],
        "per_chunk_cpu_wall_p99_ms": [c.get("cpu_p99_ms") for c in chunks],
        "per_chunk_cpu_wall_max_ms": [c.get("cpu_max_ms") for c in chunks],
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
    full_gpu_ms = full_info.get("gpu_ms") or {}
    total_gpu_ms = trt.get("total_gpu_ms") or {}

    return {
        "full_gpu_mean_ms": full_mean,
        "full_gpu_p99_ms": full_gpu_ms.get("p99"),
        "full_gpu_max_ms": full_gpu_ms.get("max"),
        "chunked_gpu_mean_ms": chunked_ms,
        "chunked_gpu_p99_ms": total_gpu_ms.get("p99"),
        "chunked_gpu_max_ms": total_gpu_ms.get("max"),
        "per_chunk_gpu_mean_ms": [c["gpu_ms"]["mean"] for c in chunks],
        "per_chunk_gpu_p99_ms": [c["gpu_ms"]["p99"] for c in chunks],
        "per_chunk_gpu_max_ms": [c["gpu_ms"].get("max") for c in chunks],
        "per_chunk_cpu_wall_mean_ms": None,
        "per_chunk_cpu_wall_p99_ms": None,
        "per_chunk_cpu_wall_max_ms": None,
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
    # ── 0. Early cache check (torch-free) ────────────────────────────────────
    # For list masks we can compute the variant name and check the cache before
    # importing selective_split (which imports torch at module level).
    if not force and not dry_run and isinstance(mask, list) and variant_name is None:
        _early_variant = mask_to_variant_name(model_name, list(mask))
        _early_cached = _load_cached_result(model_name, _early_variant, precision)
        if _early_cached is not None and _early_cached.ok():
            print(f"\n[evaluator] {model_name}/{_early_variant}  ({precision})  → cache hit (early)")
            return _early_cached
    elif not force and not dry_run and isinstance(mask, list) and variant_name is not None:
        _early_cached = _load_cached_result(model_name, variant_name, precision)
        if _early_cached is not None and _early_cached.ok():
            print(f"\n[evaluator] {model_name}/{variant_name}  ({precision})  → cache hit (early)")
            return _early_cached

    from src.splitting.selective_split import (
        load_base_config, parse_boundary_mask,
        make_selected_split_config, save_selected_config,
    )
    from src.optimization.compiler import export_onnx, _engines_exist, _onnx_exists

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

    # ── 5. Export ONNX (per-chunk with interval cache) ───────────────────────
    if export:
        onnx_hits, onnx_misses, onnx_wall = _export_chunks_with_interval_cache(
            model_name, cfg, groups, force=force
        )
        result.export_wall_s = onnx_wall
        result.interval_onnx_cache_hits = onnx_hits
        result.interval_onnx_cache_misses = onnx_misses
        result.interval_cache_hits += onnx_hits
        result.interval_cache_misses += onnx_misses
        result.exported = onnx_misses > 0

        if not _onnx_exists(cfg):
            result.error = "ONNX export failed"
            _save_result(result)
            return result

    # ── 6. Build engines (per-chunk with interval cache) ─────────────────────
    if build:
        eng_hits, eng_misses, eng_build_wall = _build_engines_with_interval_cache(
            model_name, cfg, groups, precision, force
        )
        result.interval_engine_cache_hits = eng_hits
        result.interval_engine_cache_misses = eng_misses
        result.interval_engine_build_wall_s = eng_build_wall
        result.interval_cache_hits += eng_hits
        result.interval_cache_misses += eng_misses
        result.build_wall_s = eng_build_wall
        result.built = eng_misses > 0

        if not _engines_exist(cfg, precision):
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
    t0_profile = time.perf_counter()

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

    result.profile_wall_s = time.perf_counter() - t0_profile
    result.profile_warmup = int(warmup)
    result.profile_iters = int(iters)

    # ── 8. Populate timing fields ─────────────────────────────────────────────
    result.full_gpu_mean_ms = timing.get("full_gpu_mean_ms")
    result.full_gpu_p99_ms = timing.get("full_gpu_p99_ms")
    result.full_gpu_max_ms = timing.get("full_gpu_max_ms")
    result.chunked_gpu_mean_ms = timing.get("chunked_gpu_mean_ms")
    result.chunked_gpu_p99_ms = timing.get("chunked_gpu_p99_ms")
    result.chunked_gpu_max_ms = timing.get("chunked_gpu_max_ms")
    result.per_chunk_gpu_mean_ms = timing.get("per_chunk_gpu_mean_ms")
    result.per_chunk_gpu_p99_ms = timing.get("per_chunk_gpu_p99_ms")
    result.per_chunk_gpu_max_ms = timing.get("per_chunk_gpu_max_ms")
    result.per_chunk_cpu_wall_mean_ms = timing.get("per_chunk_cpu_wall_mean_ms")
    result.per_chunk_cpu_wall_p99_ms = timing.get("per_chunk_cpu_wall_p99_ms")
    result.per_chunk_cpu_wall_max_ms = timing.get("per_chunk_cpu_wall_max_ms")

    # Backfill per-interval GPU timing so cache-only mode can assemble masks.
    _backfill_interval_gpu_timing(model_name, groups, precision, result)

    if result.full_gpu_mean_ms and result.chunked_gpu_mean_ms:
        result.overhead_ms = result.chunked_gpu_mean_ms - result.full_gpu_mean_ms
        result.overhead_pct = (result.overhead_ms / result.full_gpu_mean_ms) * 100.0

    # ── 9. Update ProfilingDB ──────────────────────────────────────────────────
    try:
        db = _load_db()
        db.put(
            model_name, variant_name, precision,
            full_gpu_mean_ms=result.full_gpu_mean_ms,
            full_gpu_max_ms=result.full_gpu_max_ms,
            per_chunk_gpu_mean_ms=result.per_chunk_gpu_mean_ms,
            per_chunk_gpu_p99_ms=result.per_chunk_gpu_p99_ms,
            per_chunk_gpu_max_ms=result.per_chunk_gpu_max_ms,
            total_chunked_gpu_mean_ms=result.chunked_gpu_mean_ms,
            total_chunked_gpu_max_ms=result.chunked_gpu_max_ms,
            source_json=str(cpp_raw_json) if cpp_raw_json else "",
        )
    except Exception as e:
        print(f"  [evaluator] WARNING: ProfilingDB update failed: {e}", file=sys.stderr)

    # ── 10. Cold-cache design-time estimate ───────────────────────────────────
    cold = _estimate_cold_cost(model_name, groups, precision, result.profile_wall_s or 0.0)
    result.estimated_cold_export_s = cold["estimated_cold_export_s"]
    result.estimated_cold_build_s = cold["estimated_cold_build_s"]
    result.estimated_cold_total_s = cold["estimated_cold_total_s"]

    # ── 11. Save and return ───────────────────────────────────────────────────
    result_path = _save_result(result)
    result.result_json_path = str(result_path.relative_to(REPO))
    result_path.write_text(json.dumps(result.to_dict(), indent=2))  # update with path field
    return result
