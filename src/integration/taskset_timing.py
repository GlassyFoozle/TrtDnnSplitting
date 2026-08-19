"""Taskset-local design/search timing for profile-in-the-loop algorithms."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO = Path(__file__).resolve().parent.parent.parent


def mask_interval_groups(mask: List[int]) -> List[List[int]]:
    """Return consecutive base-chunk groups represented by a boundary mask."""
    if not mask:
        return [[0]]
    groups: List[List[int]] = []
    current = [0]
    for boundary_idx, bit in enumerate(mask):
        if int(bit) == 1:
            groups.append(current)
            current = [boundary_idx + 1]
        else:
            current.append(boundary_idx + 1)
    groups.append(current)
    return groups


@dataclass
class TasksetTimingRecorder:
    """
    Measure one taskset/algorithm run with a fresh logical profiling cache.

    A configuration is unique by (model, precision, mask). Profiling is charged
    once per unique configuration. TensorRT optimization is estimated once per
    unique interval engine, allowing different masks to reuse shared chunks.
    """

    enabled: bool = False
    virtual_profile: bool = False
    cache_aware_profile: bool = False
    search_wall_s: float = 0.0
    search_compute_wall_s: float = 0.0
    time_to_valid_config_s: Optional[float] = None
    valid_config_found: bool = False

    profiling_wall_s: float = 0.0
    estimated_optimize_wall_s: float = 0.0
    actual_export_wall_s: float = 0.0
    actual_build_wall_s: float = 0.0
    actual_pipeline_wall_s: float = 0.0

    unique_configurations_encountered: int = 0
    unique_configurations_profiled: int = 0
    unique_chunks_optimized: int = 0
    missing_build_timing_count: int = 0
    profile_failure_count: int = 0
    missing_profile_timing_count: int = 0
    virtual_profile_iterations: int = 0

    _seen_configurations: set = field(default_factory=set, repr=False)
    _seen_chunks: set = field(default_factory=set, repr=False)
    _pending_build_timings: set = field(default_factory=set, repr=False)
    events: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def begin_configuration(
        self,
        model_name: str,
        precision: str,
        mask: List[int],
        warmup: int = 0,
        iters: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Register a configuration and return its event only on first sight."""
        if not self.enabled:
            return None
        normalized_mask = tuple(int(v) for v in mask)
        config_key = (str(model_name), str(precision), normalized_mask)
        if config_key in self._seen_configurations:
            return None
        self._seen_configurations.add(config_key)
        self.unique_configurations_encountered += 1

        new_chunks = []
        for group in mask_interval_groups(list(normalized_mask)):
            chunk_key = (
                str(model_name),
                str(precision),
                int(group[0]),
                int(group[-1]),
            )
            if chunk_key in self._seen_chunks:
                continue
            self._seen_chunks.add(chunk_key)
            self.unique_chunks_optimized += 1
            new_chunks.append([int(group[0]), int(group[-1])])
            build_s = self._load_build_wall_s(chunk_key)
            if build_s is None:
                self._pending_build_timings.add(chunk_key)
            else:
                self.estimated_optimize_wall_s += build_s

        event: Dict[str, Any] = {
            "sequence": len(self.events) + 1,
            "model": str(model_name),
            "precision": str(precision),
            "mask": list(normalized_mask),
            "new_chunks": new_chunks,
            "profile_wall_s": 0.0,
            "virtual_profile_wall_s": 0.0,
            "wall_mean_sum_ms": 0.0,
            "profile_time_source": (
                "virtual_wall_mean"
                if self.virtual_profile or self.cache_aware_profile
                else "actual"
            ),
            "actual_export_wall_s": 0.0,
            "actual_build_wall_s": 0.0,
            "did_profile": False,
            "success": False,
            "error": "",
        }
        if self.virtual_profile or self.cache_aware_profile:
            groups = mask_interval_groups(list(normalized_mask))
            wall_means = self._load_configuration_wall_means(
                str(model_name), str(precision), list(normalized_mask), groups
            )
            missing = sum(value is None for value in wall_means)
            wall_mean_sum_ms = sum(float(value) for value in wall_means if value is not None)
            iterations = max(0, int(warmup)) + max(0, int(iters))
            virtual_s = wall_mean_sum_ms * iterations / 1000.0
            event.update({
                "profile_wall_s": virtual_s,
                "virtual_profile_wall_s": virtual_s,
                "wall_mean_sum_ms": wall_mean_sum_ms,
                "profile_iterations": iterations,
                "missing_wall_mean_chunks": missing,
                "did_profile": False,
                "success": missing == 0,
            })
            self.virtual_profile_iterations = iterations
            self.profiling_wall_s += virtual_s
            self.missing_profile_timing_count += missing
            if missing == 0:
                self.unique_configurations_profiled += 1
            else:
                self.profile_failure_count += 1
        self.events.append(event)
        return event

    def finish_configuration(self, event: Optional[Dict[str, Any]], result: Any) -> None:
        """Complete a first-seen event using a MaskApplication/Evaluation result."""
        if event is None:
            return
        profile_s = float(getattr(result, "profile_wall_s", 0.0) or 0.0)
        export_s = float(getattr(result, "export_wall_s", 0.0) or 0.0)
        build_s = float(getattr(result, "build_wall_s", 0.0) or 0.0)
        cache_hit = bool(getattr(result, "cache_hit", False))
        if cache_hit:
            export_s = 0.0
            build_s = 0.0
        did_profile = bool(getattr(result, "did_profile", False)) or bool(
            getattr(result, "profiled", False)
        )
        success = bool(getattr(result, "success", False))
        if not hasattr(result, "success"):
            ok = getattr(result, "ok", None)
            success = bool(ok()) if callable(ok) else not bool(getattr(result, "error", None))

        cached_profile_s = float(event.get("virtual_profile_wall_s", 0.0) or 0.0)
        hybrid_actual_profile = bool(
            self.cache_aware_profile
            and not cache_hit
            and did_profile
            and success
        )
        if hybrid_actual_profile:
            # begin_configuration provisionally charged cached wall-mean cost.
            # Replace it with the measured profile duration on a real cache miss.
            self.profiling_wall_s += profile_s - cached_profile_s
            event["virtual_profile_wall_s"] = 0.0
            event["profile_time_source"] = "actual_cache_miss"
            reported_profile_s = profile_s
            missing = int(event.get("missing_wall_mean_chunks", 0) or 0)
            if missing > 0:
                self.missing_profile_timing_count = max(
                    0, self.missing_profile_timing_count - missing
                )
                self.profile_failure_count = max(0, self.profile_failure_count - 1)
                self.unique_configurations_profiled += 1
        elif self.cache_aware_profile and not cache_hit and not success:
            # A real cache miss was attempted but did not produce usable timing.
            # Do not leave a provisional cached estimate charged as successful.
            self.profiling_wall_s = max(0.0, self.profiling_wall_s - cached_profile_s)
            event["virtual_profile_wall_s"] = 0.0
            event["profile_time_source"] = "actual_cache_miss_failed"
            reported_profile_s = 0.0
            if int(event.get("missing_wall_mean_chunks", 0) or 0) == 0:
                self.unique_configurations_profiled = max(
                    0, self.unique_configurations_profiled - 1
                )
                self.profile_failure_count += 1
        elif self.virtual_profile or self.cache_aware_profile:
            reported_profile_s = cached_profile_s
        else:
            reported_profile_s = profile_s
        event.update({
            "profile_wall_s": reported_profile_s,
            "actual_export_wall_s": export_s,
            "actual_build_wall_s": build_s,
            "did_profile": did_profile if not self.virtual_profile else False,
            "success": success,
            "error": str(getattr(result, "error", None) or ""),
        })
        if not self.virtual_profile and not self.cache_aware_profile:
            self.profiling_wall_s += profile_s
        self.actual_export_wall_s += export_s
        self.actual_build_wall_s += build_s
        actual_profile_s = profile_s if hybrid_actual_profile or not (
            self.virtual_profile or self.cache_aware_profile
        ) else 0.0
        self.actual_pipeline_wall_s += actual_profile_s + export_s + build_s
        if not self.virtual_profile and not self.cache_aware_profile:
            if did_profile and profile_s > 0.0:
                self.unique_configurations_profiled += 1
            else:
                self.profile_failure_count += 1
        self._backfill_pending_build_timings()

    def finalize_search(self, search_wall_s: float, valid_config_found: bool) -> None:
        self.search_wall_s = float(search_wall_s)
        self.valid_config_found = bool(valid_config_found)
        self.time_to_valid_config_s = self.search_wall_s if self.valid_config_found else None
        self.search_compute_wall_s = max(
            0.0, self.search_wall_s - self.actual_pipeline_wall_s
        )
        self._backfill_pending_build_timings()
        self.missing_build_timing_count = len(self._pending_build_timings)

    @property
    def optimize_plus_profiling_wall_s(self) -> float:
        return self.estimated_optimize_wall_s + self.profiling_wall_s

    @property
    def estimated_search_wall_s(self) -> float:
        return self.search_compute_wall_s + self.profiling_wall_s

    @property
    def requires_actual_profile(self) -> bool:
        return self.enabled and not self.virtual_profile and not self.cache_aware_profile

    @property
    def virtual_cache_only(self) -> bool:
        return self.enabled and self.virtual_profile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profiling_wall_s": self.profiling_wall_s,
            "virtual_profiling_wall_s": (
                self.profiling_wall_s if self.virtual_profile else 0.0
            ),
            "profiling_time_source": (
                "virtual_wall_mean"
                if self.virtual_profile
                else "cached_estimate_or_actual_miss"
                if self.cache_aware_profile
                else "actual"
            ),
            "virtual_profile_iterations": self.virtual_profile_iterations,
            "estimated_optimize_wall_s": self.estimated_optimize_wall_s,
            "optimize_plus_profiling_wall_s": self.optimize_plus_profiling_wall_s,
            "search_wall_s": self.search_wall_s,
            "search_compute_wall_s": self.search_compute_wall_s,
            "estimated_search_wall_s": self.estimated_search_wall_s,
            "time_to_valid_config_s": self.time_to_valid_config_s,
            "valid_config_found": self.valid_config_found,
            "unique_configurations_encountered": self.unique_configurations_encountered,
            "unique_configurations_profiled": self.unique_configurations_profiled,
            "unique_chunks_optimized": self.unique_chunks_optimized,
            "missing_build_timing_count": self.missing_build_timing_count,
            "profile_failure_count": self.profile_failure_count,
            "missing_profile_timing_count": self.missing_profile_timing_count,
            "actual_export_wall_s": self.actual_export_wall_s,
            "actual_build_wall_s": self.actual_build_wall_s,
            "actual_pipeline_wall_s": self.actual_pipeline_wall_s,
        }

    @staticmethod
    def _timing_path(chunk_key: Tuple[str, str, int, int]) -> Path:
        model_name, _precision, start, end = chunk_key
        return (
            REPO
            / "artifacts"
            / "chunk_cache"
            / model_name
            / f"int_{start}_{end}"
            / "timing.json"
        )

    def _load_build_wall_s(
        self, chunk_key: Tuple[str, str, int, int]
    ) -> Optional[float]:
        _model_name, precision, _start, _end = chunk_key
        path = self._timing_path(chunk_key)
        if not path.exists():
            return None
        try:
            timing = json.loads(path.read_text())
            value = timing.get(f"build_{precision}_wall_s")
            if value is None:
                return None
            return float(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _backfill_pending_build_timings(self) -> None:
        resolved = []
        for chunk_key in self._pending_build_timings:
            build_s = self._load_build_wall_s(chunk_key)
            if build_s is None:
                continue
            self.estimated_optimize_wall_s += build_s
            resolved.append(chunk_key)
        for chunk_key in resolved:
            self._pending_build_timings.remove(chunk_key)

    def _load_configuration_wall_means(
        self,
        model_name: str,
        precision: str,
        mask: List[int],
        groups: List[List[int]],
    ) -> List[Optional[float]]:
        values = [self._load_interval_wall_mean(model_name, precision, group) for group in groups]
        if all(value is not None for value in values):
            return values

        eval_values = self._load_evaluation_wall_means(model_name, precision, mask)
        if len(eval_values) == len(values):
            values = [
                value if value is not None else eval_values[idx]
                for idx, value in enumerate(values)
            ]
        return values

    @staticmethod
    def _load_interval_wall_mean(
        model_name: str, precision: str, group: List[int]
    ) -> Optional[float]:
        path = (
            REPO
            / "artifacts"
            / "chunk_cache"
            / model_name
            / f"int_{group[0]}_{group[-1]}"
            / "timing.json"
        )
        try:
            timing = json.loads(path.read_text())
            value = timing.get(
                f"wall_mean_ms_{precision}", timing.get(f"gpu_mean_ms_{precision}")
            )
            return float(value) if value is not None else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _load_evaluation_wall_means(
        model_name: str, precision: str, mask: List[int]
    ) -> List[Optional[float]]:
        mask_text = "".join(str(int(bit)) for bit in mask)
        digest = hashlib.sha256(mask_text.encode()).hexdigest()[:8]
        variant = f"{model_name}_mask_{digest}_k{sum(mask) + 1}"
        root = Path(
            os.environ.get(
                "EVALUATION_CACHE_DIR", str(REPO / "results" / "evaluations")
            )
        )
        if not root.is_absolute():
            root = REPO / root
        path = root / model_name / f"{variant}_{precision}.json"
        try:
            data = json.loads(path.read_text())
            raw = data.get("per_chunk_cpu_wall_mean_ms") or data.get(
                "per_chunk_gpu_mean_ms"
            ) or []
            return [float(value) if value is not None else None for value in raw]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []
