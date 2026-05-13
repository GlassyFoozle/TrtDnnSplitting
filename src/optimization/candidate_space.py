"""
candidate_space.py — Load and describe the candidate split-point universe.

The candidate universe for a model is its `dag_aligned_full` config:
  N base chunks → N-1 candidate boundaries.

This module provides:
  - CandidateSpace: wrapper around a dag_aligned_full config with timing data
  - load_candidate_space(model_name, precision): build a CandidateSpace

Timing data priority:
  1. ProfilingDB object (if provided)
  2. results/table4/<model>_cpp_dag_aligned_full_<precision>.json
  3. singleton interval cache timing files, artifacts/chunk_cache/<model>/int_i_i/timing.json
  4. [live mode only] positive placeholder timing via allow_missing_timing_for_live=True
  5. [opt-in only] equal-weight fallback via allow_equal_wcet_fallback=True

By default, if no profiling data exists the function raises RuntimeError
directing the user to run scripts/21_profile_base_chunks.py. Using
allow_equal_wcet_fallback=True restores the equal-distribution approximation
(WCET/N per chunk) for development or CI environments without hardware.
Using allow_missing_timing_for_live=True keeps only the metadata and inserts a
placeholder that must be replaced by measured mask timing before RTA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
_BASE_VARIANT = "dag_aligned_full"


@dataclass
class CandidateSpace:
    model_name: str
    base_variant: str
    candidate_count: int           # N base chunks
    boundary_count: int            # N-1 boundaries
    chunk_names: List[str]
    chunk_descriptions: List[str]
    input_shapes: List[list]
    output_shapes: List[list]
    # Per-chunk timing (may be 0.0 if not profiled)
    chunk_gpu_mean_ms: List[float] = field(default_factory=list)
    chunk_gpu_p99_ms: List[float] = field(default_factory=list)
    chunk_gpu_max_ms: List[float] = field(default_factory=list)
    timing_is_placeholder: bool = False
    timing_source: str = ""
    config_path: str = ""

    @property
    def has_timing(self) -> bool:
        return (not self.timing_is_placeholder) and any(t > 0.0 for t in self.chunk_gpu_mean_ms)

    @property
    def total_estimated_ms(self) -> float:
        return sum(self.chunk_gpu_mean_ms)

    def chunk_summary(self) -> str:
        lines = [
            f"CandidateSpace: {self.model_name} / {self.base_variant}",
            f"  N={self.candidate_count} chunks  boundaries={self.boundary_count}  "
            f"has_timing={self.has_timing}",
        ]
        for i in range(self.candidate_count):
            t = self.chunk_gpu_max_ms[i] if self.chunk_gpu_max_ms else 0.0
            lines.append(
                f"  [{i:2d}] {self.chunk_names[i]:20s}  "
                f"{str(self.input_shapes[i]):22s} → {str(self.output_shapes[i]):22s}  "
                f"max={t:.4f}ms"
            )
        return "\n".join(lines)


def _read_singleton_interval_timings(
    model_name: str,
    precision: str,
    n_chunks: int,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    means: List[float] = []
    p99s: List[float] = []
    maxs: List[float] = []
    for i in range(n_chunks):
        p = REPO / "artifacts" / "chunk_cache" / model_name / f"int_{i}_{i}" / "timing.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
            mean = d.get(f"gpu_mean_ms_{precision}")
            p99 = d.get(f"gpu_p99_ms_{precision}")
            max_v = d.get(f"gpu_max_ms_{precision}")
            if mean is None or p99 is None or max_v is None:
                return None
            means.append(float(mean))
            p99s.append(float(p99))
            maxs.append(float(max_v))
        except Exception:
            return None
    return means, p99s, maxs


def load_candidate_space(
    model_name: str,
    precision: str = "fp32",
    profiling_db: Optional[object] = None,  # ProfilingDB or None
    allow_equal_wcet_fallback: bool = False,
    allow_missing_timing_for_live: bool = False,
) -> CandidateSpace:
    """
    Load the CandidateSpace for model_name from its dag_aligned_full config.

    Timing data priority:
      1. profiling_db (if provided and has entry for this model/variant/precision)
      2. results/table4/<model>_cpp_dag_aligned_full_<precision>.json
      3. singleton interval cache timing files
      4. [live mode] placeholder timing if allow_missing_timing_for_live=True
      5. [opt-in] equal-weight fallback if allow_equal_wcet_fallback=True

    Raises RuntimeError if no timing data is found and
    neither allow_missing_timing_for_live nor allow_equal_wcet_fallback is set.
    Run scripts/21_profile_base_chunks.py to create canonical profiling data.
    """
    cfg_path = REPO / "artifacts" / "split_configs" / model_name / f"{_BASE_VARIANT}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"dag_aligned_full config not found for {model_name!r}.\n"
            f"Run: conda run -n trt python scripts/26_generate_dag_aligned_configs.py --models {model_name}"
        )

    cfg = json.loads(cfg_path.read_text())
    chunks = cfg["chunks"]
    n = len(chunks)

    chunk_names = [c["chunk_name"] for c in chunks]
    chunk_descs = [c.get("description", "") for c in chunks]
    input_shapes = [c["input_shape"] for c in chunks]
    output_shapes = [c["output_shape"] for c in chunks]

    # Try to get timing data
    per_chunk_means: List[float] = [0.0] * n
    per_chunk_p99: List[float] = [0.0] * n
    per_chunk_max: List[float] = [0.0] * n
    timing_source = ""

    # Priority 1: profiling DB
    if profiling_db is not None:
        means = profiling_db.get_per_chunk_means(model_name, _BASE_VARIANT, precision)
        if means and len(means) == n:
            per_chunk_means = list(means)
            entry = profiling_db.get(model_name, _BASE_VARIANT, precision) or {}
            per_chunk_p99 = list(entry.get("per_chunk_gpu_p99_ms", [0.0] * n))
            per_chunk_max = list(entry.get("per_chunk_gpu_max_ms", [0.0] * n))
            if len(per_chunk_p99) != n:
                per_chunk_p99 = [0.0] * n
            if len(per_chunk_max) != n:
                per_chunk_max = [0.0] * n
            if any(t > 0.0 for t in per_chunk_means):
                timing_source = "profiling_db"

    # Priority 2: scan C++ result JSON
    if all(t == 0.0 for t in per_chunk_means):
        cpp_path = (
            REPO / "results" / "table4"
            / f"{model_name}_cpp_{_BASE_VARIANT}_{precision}.json"
        )
        if cpp_path.exists():
            try:
                d = json.loads(cpp_path.read_text())
                cpp_chunks = d.get("chunks", [])
                if len(cpp_chunks) == n:
                    per_chunk_means = [c["gpu_mean_ms"] for c in cpp_chunks]
                    per_chunk_p99 = [c["gpu_p99_ms"] for c in cpp_chunks]
                    per_chunk_max = [c.get("gpu_max_ms", 0.0) for c in cpp_chunks]
                    timing_source = str(cpp_path.relative_to(REPO))
            except Exception:
                pass

    # Priority 3: singleton interval-cache timing produced by script 27.
    interval_timings = _read_singleton_interval_timings(model_name, precision, n)
    if interval_timings is not None:
        int_means, int_p99, int_max = interval_timings
        if all(t == 0.0 for t in per_chunk_means) or all(t == 0.0 for t in per_chunk_max):
            per_chunk_means = int_means
            per_chunk_p99 = int_p99
            per_chunk_max = int_max
            timing_source = "singleton_interval_timing"

    if any(t > 0.0 for t in per_chunk_means) and all(t == 0.0 for t in per_chunk_max):
        if allow_missing_timing_for_live:
            per_chunk_means = [1.0] * n
            per_chunk_p99 = [1.0] * n
            per_chunk_max = [1.0] * n
            timing_source = "live_placeholder_missing_max"
        elif not allow_equal_wcet_fallback:
            table4_path = (
                REPO / "results" / "table4"
                / f"{model_name}_cpp_{_BASE_VARIANT}_{precision}.json"
            )
            raise RuntimeError(
                f"Missing max timing data for {model_name!r} ({precision}).\n"
                f"Existing profiling data has mean/p99 but no gpu_max_ms fields.\n"
                f"Re-profile cached/base chunks first, for example:\n"
                f"  conda run -n trt python scripts/21_profile_base_chunks.py "
                f"--models {model_name} --precision {precision}\n"
                f"or use scripts/27_reprofile_cached_intervals.py for interval cache.\n"
                f"Expected updated table4 JSON: {table4_path}"
            )

    timing_is_placeholder = timing_source.startswith("live_placeholder")

    # Priority 4: live-mode placeholder when metadata exists but canonical base
    # chunk timing does not.  The positive value preserves N GPU blocks through
    # SS->UNI conversion; it must never be consumed by RTA.
    if all(t == 0.0 for t in per_chunk_means) and allow_missing_timing_for_live:
        per_chunk_means = [1.0] * n
        per_chunk_p99 = [1.0] * n
        per_chunk_max = [1.0] * n
        timing_source = "live_placeholder_missing_base_timing"
        timing_is_placeholder = True

    # Priority 5: opt-in equal-weight fallback for development/CI without hardware.
    # These reference values match measured K=1 max totals on Jetson AGX Orin (FP32).
    # They must be kept in sync with _DRY_RUN_BASE_WCET_MS in dnn_workload_generator.py.
    if all(t == 0.0 for t in per_chunk_means):
        if allow_equal_wcet_fallback:
            _DRY_RUN_WCET_MS = {
                "alexnet": 1.754,
                "resnet18": 1.037,
                "vit_l_16": 25.43,
                "vgg19": 7.562,
            }
            wcet = _DRY_RUN_WCET_MS.get(model_name.lower())
            if wcet is not None:
                per_chunk_means = [wcet / n] * n
                per_chunk_p99 = [wcet / n] * n
                per_chunk_max = [wcet / n] * n
                timing_source = "equal_wcet_fallback"
        else:
            table4_path = (
                REPO / "results" / "table4"
                / f"{model_name}_cpp_{_BASE_VARIANT}_{precision}.json"
            )
            raise RuntimeError(
                f"Missing dag_aligned_full profiling data for {model_name!r} "
                f"({precision}).\n"
                f"Expected: {table4_path}\n\n"
                f"Run base chunk profiling first:\n"
                f"  conda run -n trt python scripts/21_profile_base_chunks.py "
                f"--models {model_name} --precision {precision}\n\n"
                f"Or pass --allow-equal-wcet-fallback to scripts 30/40 for "
                f"development use (produces approximate results)."
            )

    return CandidateSpace(
        model_name=model_name,
        base_variant=_BASE_VARIANT,
        candidate_count=n,
        boundary_count=n - 1,
        chunk_names=chunk_names,
        chunk_descriptions=chunk_descs,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        chunk_gpu_mean_ms=per_chunk_means,
        chunk_gpu_p99_ms=per_chunk_p99,
        chunk_gpu_max_ms=per_chunk_max,
        timing_is_placeholder=timing_is_placeholder,
        timing_source=timing_source,
        config_path=str(cfg_path),
    )
