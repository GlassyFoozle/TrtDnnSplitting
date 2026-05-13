"""
dnn_task.py — DNNBackedTask: metadata model for a real-DNN inference task.

A DNNBackedTask wraps a single DNN model instance configured for schedulability
analysis. It holds the TRT profiling results and the mapping to DNNSplitting
SegInfTask parameters.

Key design choices:
  - Timing in milliseconds (float) throughout.
  - per_splitting_overhead = 0.0: base_chunk_times_ms are measured per-chunk
    GPU times; overhead is already captured in measurements.
  - wcet_metric selects which timing column drives analysis:
      "max"  → gpu_max_ms  (WCET/default)
      "p99"  → deprecated alias for max in analysis paths
      "mean" → gpu_mean_ms (development-only optimistic path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DNNBackedTask:
    # --- Identity ---
    task_name: str               # logical task name, e.g. "tau1_alexnet"
    model_name: str              # TRT model key, e.g. "alexnet"
    precision: str               # "fp32" / "fp16"

    # --- Real-time parameters ---
    period_ms: float             # task period T (ms)
    deadline_ms: float           # relative deadline D (ms); usually == period_ms
    priority: float              # scheduling priority (lower = higher priority typical)
    cpu_id: int                  # CPU core assignment (0-based)

    # --- CPU execution times ---
    cpu_pre_ms: float            # pre-GPU CPU segment C₀ (ms)
    cpu_post_ms: float           # post-GPU CPU segment C_last (ms)

    # --- TRT candidate space ---
    base_variant: str            # "dag_aligned_full" (universe of split points)
    candidate_count: int         # N base chunks in dag_aligned_full
    boundary_count: int          # N-1 candidate boundaries

    # --- Current configuration ---
    initial_mask: List[int]       # boundary mask at task creation (all-1 = max split)
    selected_variant_name: str   # e.g. "alexnet_balanced4" or SHA-based name
    selected_config_path: str    # path to the selected split config JSON
    profile_result_path: str     # path to EvaluationResult JSON

    # --- Timing data (from TRT profiling) ---
    wcet_metric: str             # "max" or "mean"; "p99" aliases max
    base_chunk_times_ms: List[float]   # per-chunk GPU times for dag_aligned_full
    current_chunk_times_ms: List[float]  # per-chunk GPU times for selected config
    base_timing_placeholder: bool = False
    current_timing_measured: bool = False

    # --- Optional notes ---
    notes: str = ""

    # --- Derived (populated after construction) ---
    # These are computed from the segment list and chunk times.
    _segment_count: int = field(default=0, repr=False)

    def __post_init__(self):
        if len(self.base_chunk_times_ms) != self.candidate_count:
            raise ValueError(
                f"base_chunk_times_ms length {len(self.base_chunk_times_ms)} "
                f"!= candidate_count {self.candidate_count}"
            )
        if len(self.initial_mask) != self.boundary_count:
            raise ValueError(
                f"initial_mask length {len(self.initial_mask)} "
                f"!= boundary_count {self.boundary_count}"
            )
        self._segment_count = sum(self.initial_mask) + 1

    @property
    def total_gpu_ms(self) -> float:
        """Total GPU time for the selected config (sum of current_chunk_times_ms)."""
        return sum(self.current_chunk_times_ms)

    @property
    def total_cpu_ms(self) -> float:
        return self.cpu_pre_ms + self.cpu_post_ms

    @property
    def n_active_chunks(self) -> int:
        """Number of GPU chunks in the selected split config."""
        return len(self.current_chunk_times_ms)

    @property
    def max_chunk_ms(self) -> float:
        """Largest single GPU chunk time."""
        return max(self.current_chunk_times_ms) if self.current_chunk_times_ms else 0.0

    def summary(self) -> str:
        lines = [
            f"DNNBackedTask: {self.task_name}  ({self.model_name}/{self.precision})",
            f"  T={self.period_ms}ms  D={self.deadline_ms}ms  cpu={self.cpu_id}",
            f"  C_pre={self.cpu_pre_ms:.4f}ms  C_post={self.cpu_post_ms:.4f}ms",
            f"  GPU total={self.total_gpu_ms:.4f}ms  chunks={self.n_active_chunks}  "
            f"max_chunk={self.max_chunk_ms:.4f}ms",
            f"  wcet_metric={self.wcet_metric}  variant={self.selected_variant_name}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "precision": self.precision,
            "period_ms": self.period_ms,
            "deadline_ms": self.deadline_ms,
            "priority": self.priority,
            "cpu_id": self.cpu_id,
            "cpu_pre_ms": self.cpu_pre_ms,
            "cpu_post_ms": self.cpu_post_ms,
            "base_variant": self.base_variant,
            "candidate_count": self.candidate_count,
            "boundary_count": self.boundary_count,
            "initial_mask": self.initial_mask,
            "selected_variant_name": self.selected_variant_name,
            "selected_config_path": self.selected_config_path,
            "profile_result_path": self.profile_result_path,
            "wcet_metric": self.wcet_metric,
            "base_chunk_times_ms": self.base_chunk_times_ms,
            "current_chunk_times_ms": self.current_chunk_times_ms,
            "base_timing_placeholder": self.base_timing_placeholder,
            "current_timing_measured": self.current_timing_measured,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DNNBackedTask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
