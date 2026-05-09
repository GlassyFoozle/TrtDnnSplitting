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
  3. [opt-in only] equal-weight fallback via allow_equal_wcet_fallback=True

By default, if no profiling data exists the function raises RuntimeError
directing the user to run scripts/21_profile_base_chunks.py. Using
allow_equal_wcet_fallback=True restores the equal-distribution approximation
(WCET/N per chunk) for development or CI environments without hardware.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
    config_path: str = ""

    @property
    def has_timing(self) -> bool:
        return any(t > 0.0 for t in self.chunk_gpu_mean_ms)

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
            t = self.chunk_gpu_mean_ms[i] if self.chunk_gpu_mean_ms else 0.0
            lines.append(
                f"  [{i:2d}] {self.chunk_names[i]:20s}  "
                f"{str(self.input_shapes[i]):22s} → {str(self.output_shapes[i]):22s}  "
                f"{t:.4f}ms"
            )
        return "\n".join(lines)


def load_candidate_space(
    model_name: str,
    precision: str = "fp32",
    profiling_db: Optional[object] = None,  # ProfilingDB or None
    allow_equal_wcet_fallback: bool = False,
) -> CandidateSpace:
    """
    Load the CandidateSpace for model_name from its dag_aligned_full config.

    Timing data priority:
      1. profiling_db (if provided and has entry for this model/variant/precision)
      2. results/table4/<model>_cpp_dag_aligned_full_<precision>.json
      3. [opt-in] equal-weight fallback if allow_equal_wcet_fallback=True

    Raises RuntimeError if no timing data is found and
    allow_equal_wcet_fallback is False. Run scripts/21_profile_base_chunks.py
    to create the required profiling data.
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

    # Priority 1: profiling DB
    if profiling_db is not None:
        means = profiling_db.get_per_chunk_means(model_name, _BASE_VARIANT, precision)
        if means and len(means) == n:
            per_chunk_means = list(means)
            p99 = profiling_db.get(model_name, _BASE_VARIANT, precision)
            per_chunk_p99 = list((p99 or {}).get("per_chunk_gpu_p99_ms", [0.0] * n))

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
            except Exception:
                pass

    # Priority 3: opt-in equal-weight fallback for development/CI without hardware.
    # These reference values match measured p99 totals on Jetson AGX Orin (FP32).
    # They must be kept in sync with _DRY_RUN_BASE_WCET_MS in dnn_workload_generator.py.
    if all(t == 0.0 for t in per_chunk_means):
        if allow_equal_wcet_fallback:
            _DRY_RUN_WCET_MS = {"alexnet": 1.754, "resnet18": 1.037, "vgg19": 7.562}
            wcet = _DRY_RUN_WCET_MS.get(model_name.lower())
            if wcet is not None:
                per_chunk_means = [wcet / n] * n
                per_chunk_p99 = [wcet / n] * n
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
        config_path=str(cfg_path),
    )
