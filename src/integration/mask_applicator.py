"""
mask_applicator.py — Apply a TensorRT boundary mask to a DNNBackedTask + SegInfTask pair.

The core operation of the DNN-aware splitting pipeline:
  mask (N-1 boundaries) → evaluate_mask() → measured K-chunk times → patch SegInfTask

Key design:
  - base_block_list stays as the N individual dag_aligned_full chunk times (immutable).
  - After evaluation, G_block_list is OVERRIDDEN with the K measured times.
    RTA never uses sum-of-base-blocks estimates as a timing substitute.
  - splitting_config is updated to reflect the new mask.
  - dry_run=True cannot produce schedulability timing; it returns success=False.

Per-chunk timing column:
  wcet_metric="max"  → per_chunk_gpu_max_ms  (default WCET)
  wcet_metric="p99"  → deprecated alias for max in analysis paths
  wcet_metric="mean" → per_chunk_gpu_mean_ms (development-only optimistic path)
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
_K_SPLIT_CACHE_PATH = REPO / "results" / "optimization" / "measured_k_split_cache.json"
_K_SPLIT_CACHE_VERSION = 2

if TYPE_CHECKING:
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.live_budget import LiveProfileBudget


def _select_measured_chunk_times(eval_result, wcet_metric: str) -> List[float]:
    """Return measured per-chunk GPU timing from an EvaluationResult."""
    metric = (wcet_metric or "max").lower()
    if metric in ("max", "p99"):
        chunk_times = eval_result.per_chunk_gpu_max_ms or []
        if chunk_times:
            return list(chunk_times)
        if metric == "p99":
            legacy = eval_result.per_chunk_gpu_p99_ms or []
            if legacy:
                return list(legacy)
    return list(eval_result.per_chunk_gpu_mean_ms or [])


def _print_live_eval_event(message: str) -> None:
    """Print live build/profile events even when experiment progress captures stdout."""
    stream = getattr(sys, "__stdout__", None) or sys.stdout
    print(message, file=stream, flush=True)


@dataclass
class MaskApplicationResult:
    """Result of evaluate_and_apply_mask()."""
    success: bool
    mask: List[int]
    k_chunks: int

    # Profiling provenance
    cache_hit: bool = False
    did_export: bool = False
    did_build: bool = False
    did_profile: bool = False
    dry_run: bool = False
    is_k1_baseline: bool = False  # True when K=1 no-split shortcut was taken

    # Interval-level cache accounting (0 when dry_run or mask-level cache hit)
    interval_cache_hits: int = 0       # combined ONNX + engine hits
    interval_cache_misses: int = 0     # combined ONNX + engine misses
    interval_onnx_cache_hits: int = 0
    interval_onnx_cache_misses: int = 0
    interval_engine_cache_hits: int = 0
    interval_engine_cache_misses: int = 0
    interval_timing_cache_hit: bool = False  # served from interval GPU timing (no re-profile)

    # Wall-clock timing per pipeline phase (0.0 when not measured)
    export_wall_s: float = 0.0
    build_wall_s: float = 0.0
    profile_wall_s: float = 0.0
    interval_engine_build_wall_s: float = 0.0

    # Cold-cache design-time estimate (None when interval timing data is unavailable)
    estimated_cold_total_s: Optional[float] = None

    # Timing
    selected_chunk_times: List[float] = field(default_factory=list)
    max_block: float = 0.0
    total_gpu: float = 0.0

    # Paths
    profile_result_path: str = ""
    variant_name: str = ""
    model_name: str = ""

    error: Optional[str] = None


def evaluate_and_apply_mask(
    dnn_task: "DNNBackedTask",
    seg_task,                   # SegInfTask (DNNSplitting)
    mask: List[int],
    segment_idx: int = 0,
    *,
    precision: str = "fp32",
    wcet_metric: str = "max",   # "max" or "mean"; "p99" is a deprecated alias
    use_cpp: bool = True,
    force: bool = False,
    export: bool = True,
    build: bool = True,
    dry_run: bool = False,
    warmup: int = 20,
    iters: int = 200,
    live_budget: "Optional[LiveProfileBudget]" = None,
    verbose_evaluator: bool = False,
    enable_monotonic_k_split_cache: bool = False,
) -> MaskApplicationResult:
    """
    Evaluate a boundary mask via TRT profiling (or cache) and apply measured
    per-chunk times to the SegInfTask's InferenceSegment.

    Parameters
    ----------
    dnn_task    : DNNBackedTask supplying model_name, precision, base metadata.
    seg_task    : SegInfTask whose inference_segment_list[segment_idx] will be updated.
    mask        : binary list of length N-1 (dag_aligned_full boundaries).
    segment_idx : which InferenceSegment to update (always 0 for single-segment tasks).
    wcet_metric : "max" (WCET/default) or "mean" (optimistic); "p99" aliases max.
    dry_run     : ask evaluator for a plan only; no timing is applied.

    Returns
    -------
    MaskApplicationResult — always set, success=False on error.
    """
    from src.optimization.config_evaluator import evaluate_mask as _eval_mask

    seg = seg_task.inference_segment_list[segment_idx]
    base_times = seg.base_block_list   # N individual base chunk times
    N = len(base_times)

    if len(mask) != N - 1:
        return MaskApplicationResult(
            success=False, mask=mask, k_chunks=0,
            error=f"mask length {len(mask)} != N-1 = {N - 1}",
        )

    k = sum(mask) + 1

    # dry_run never supplies measured timing. Keep it non-mutating so RTA cannot
    # accidentally consume base-sum estimates.
    if dry_run:
        return MaskApplicationResult(
            success=False, mask=list(mask), k_chunks=k, dry_run=True,
            error="dry_run does not provide measured per-chunk timing",
        )

    # Cache order for live/evaluation mode:
    #   1. exact mask EvaluationResult JSON (handled by evaluate_mask below),
    #   2. interval timing cache assembly,
    #   3. live export/build/profile.
    if not force:
        from src.optimization.config_evaluator import (
            is_mask_cached, can_assemble_from_intervals, assemble_from_intervals,
        )
        if (
            not is_mask_cached(dnn_task.model_name, mask, precision)
            and can_assemble_from_intervals(dnn_task.model_name, mask, precision)
        ):
            assembled = assemble_from_intervals(dnn_task.model_name, mask, precision)
            if assembled.ok():
                chunk_times = _select_measured_chunk_times(assembled, wcet_metric)
                if chunk_times and len(chunk_times) == k:
                    _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)
                    dnn_task.current_chunk_times_ms = list(chunk_times)
                    dnn_task.current_timing_measured = True
                    dnn_task.selected_variant_name = assembled.variant_name
                    dnn_task.profile_result_path = assembled.result_json_path
                    return MaskApplicationResult(
                        success=True,
                        mask=list(mask),
                        k_chunks=k,
                        cache_hit=True,
                        interval_timing_cache_hit=True,
                        selected_chunk_times=list(chunk_times),
                        max_block=max(chunk_times),
                        total_gpu=sum(chunk_times),
                        variant_name=assembled.variant_name,
                        profile_result_path=assembled.result_json_path,
                    )

    # ── live budget pre-check (real eval only) ────────────────────────────────
    live_cache_miss_variant = ""
    if live_budget is not None:
        from src.optimization.config_evaluator import (
            is_mask_cached, mask_to_variant_name,
            can_assemble_from_intervals, assemble_from_intervals,
        )
        if not is_mask_cached(dnn_task.model_name, mask, precision):
            variant_name = mask_to_variant_name(dnn_task.model_name, mask)
            live_cache_miss_variant = variant_name
            reason = live_budget.check_before_real_eval(dnn_task.model_name, variant_name)
            if reason is not None:
                # Try assembling from interval timing before accepting skip.
                if can_assemble_from_intervals(dnn_task.model_name, mask, precision):
                    assembled = assemble_from_intervals(
                        dnn_task.model_name, mask, precision
                    )
                    if assembled.ok():
                        chunk_times = _select_measured_chunk_times(assembled, wcet_metric)
                        if chunk_times and len(chunk_times) == k:
                            _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)
                            dnn_task.current_chunk_times_ms = list(chunk_times)
                            dnn_task.current_timing_measured = True
                            dnn_task.selected_variant_name = assembled.variant_name
                            # Count as interval timing cache hit (not skip)
                            return MaskApplicationResult(
                                success=True, mask=list(mask), k_chunks=k,
                                cache_hit=True,
                                interval_timing_cache_hit=True,
                                selected_chunk_times=list(chunk_times),
                                max_block=max(chunk_times),
                                total_gpu=sum(chunk_times),
                                variant_name=assembled.variant_name,
                                profile_result_path=assembled.result_json_path,
                            )
                live_budget.record_skip()
                return MaskApplicationResult(
                    success=False, mask=list(mask), k_chunks=k,
                    model_name=dnn_task.model_name,
                    error=reason,
                )
            if verbose_evaluator:
                _print_live_eval_event(
                    f"[live] real profile/build start: {dnn_task.model_name}/{variant_name} "
                    f"K={k}"
                )

    # ── real evaluation ───────────────────────────────────────────────────────
    eval_result = _eval_mask(
        model_name=dnn_task.model_name,
        mask=mask,
        precision=precision,
        warmup=warmup,
        iters=iters,
        use_cpp=use_cpp,
        force=force,
        export=export,
        build=build,
    )

    if verbose_evaluator and not eval_result.cache_hit and (
        eval_result.exported or eval_result.built or eval_result.profiled
    ):
        phases = []
        if eval_result.exported:
            phases.append(f"export={float(eval_result.export_wall_s or 0.0):.1f}s")
        if eval_result.built:
            phases.append(f"build={float(eval_result.build_wall_s or 0.0):.1f}s")
        if eval_result.profiled:
            phases.append(f"profile={float(eval_result.profile_wall_s or 0.0):.1f}s")
        _print_live_eval_event(
            "[live] real eval done: "
            f"{eval_result.model_name}/{eval_result.variant_name} "
            f"K={eval_result.n_chunks} " + " ".join(phases)
        )

    if eval_result.error or not eval_result.ok():
        error_msg = eval_result.error or "EvaluationResult not ok (no timing)"
        return MaskApplicationResult(
            success=False, mask=list(mask), k_chunks=k,
            cache_hit=eval_result.cache_hit,
            did_export=eval_result.exported,
            did_build=eval_result.built,
            did_profile=eval_result.profiled,
            interval_cache_hits=eval_result.interval_cache_hits,
            interval_cache_misses=eval_result.interval_cache_misses,
            interval_onnx_cache_hits=eval_result.interval_onnx_cache_hits,
            interval_onnx_cache_misses=eval_result.interval_onnx_cache_misses,
            interval_engine_cache_hits=eval_result.interval_engine_cache_hits,
            interval_engine_cache_misses=eval_result.interval_engine_cache_misses,
            interval_engine_build_wall_s=float(eval_result.interval_engine_build_wall_s),
            export_wall_s=float(eval_result.export_wall_s or 0.0),
            build_wall_s=float(eval_result.build_wall_s or 0.0),
            profile_wall_s=float(eval_result.profile_wall_s or 0.0),
            estimated_cold_total_s=eval_result.estimated_cold_total_s,
            variant_name=eval_result.variant_name,
            profile_result_path=eval_result.result_json_path,
            error=error_msg,
        )

    # Select timing column
    chunk_times = _select_measured_chunk_times(eval_result, wcet_metric)
    if not chunk_times:
        return MaskApplicationResult(
            success=False, mask=list(mask), k_chunks=k,
            cache_hit=eval_result.cache_hit,
            did_export=eval_result.exported,
            did_build=eval_result.built,
            did_profile=eval_result.profiled,
            variant_name=eval_result.variant_name,
            profile_result_path=eval_result.result_json_path,
            error="Measured per-chunk GPU timing unavailable",
        )

    # Verify chunk count matches expected K
    if len(chunk_times) != k:
        return MaskApplicationResult(
            success=False, mask=list(mask), k_chunks=k,
            cache_hit=eval_result.cache_hit,
            did_export=eval_result.exported,
            did_build=eval_result.built,
            did_profile=eval_result.profiled,
            selected_chunk_times=list(chunk_times),
            max_block=max(chunk_times) if chunk_times else 0.0,
            total_gpu=sum(chunk_times),
            variant_name=eval_result.variant_name,
            profile_result_path=eval_result.result_json_path,
            error=f"Measured chunk count {len(chunk_times)} != expected K={k}",
        )

    _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)

    # Update DNNBackedTask metadata
    dnn_task.current_chunk_times_ms = list(chunk_times)
    dnn_task.current_timing_measured = True
    dnn_task.selected_variant_name = eval_result.variant_name
    dnn_task.selected_config_path = eval_result.config_path
    dnn_task.profile_result_path = eval_result.result_json_path

    # Charge real profile against global budget
    if live_budget is not None and not eval_result.cache_hit:
        live_budget.record_real_profile()

    return MaskApplicationResult(
        success=True,
        mask=list(mask),
        k_chunks=k,
        cache_hit=eval_result.cache_hit,
        did_export=eval_result.exported,
        did_build=eval_result.built,
        did_profile=eval_result.profiled,
        interval_cache_hits=eval_result.interval_cache_hits,
        interval_cache_misses=eval_result.interval_cache_misses,
        interval_onnx_cache_hits=eval_result.interval_onnx_cache_hits,
        interval_onnx_cache_misses=eval_result.interval_onnx_cache_misses,
        interval_engine_cache_hits=eval_result.interval_engine_cache_hits,
        interval_engine_cache_misses=eval_result.interval_engine_cache_misses,
        interval_engine_build_wall_s=float(eval_result.interval_engine_build_wall_s),
        export_wall_s=float(eval_result.export_wall_s or 0.0),
        build_wall_s=float(eval_result.build_wall_s or 0.0),
        profile_wall_s=float(eval_result.profile_wall_s or 0.0),
        estimated_cold_total_s=eval_result.estimated_cold_total_s,
        selected_chunk_times=list(chunk_times),
        max_block=max(chunk_times) if chunk_times else 0.0,
        total_gpu=sum(chunk_times),
        variant_name=eval_result.variant_name,
        profile_result_path=eval_result.result_json_path,
    )


def apply_no_split_mask(
    dnn_task: "DNNBackedTask",
    seg_task,
    segment_idx: int = 0,
    *,
    dry_run: bool = False,
    **kwargs,
) -> MaskApplicationResult:
    """Apply K=1 (all boundaries off) mask.

    K=1 uses the same measured evaluator/cache path as every other mask.
    """
    seg = seg_task.inference_segment_list[segment_idx]
    N = len(seg.base_block_list)
    mask = [0] * (N - 1)
    return evaluate_and_apply_mask(
        dnn_task, seg_task, mask, segment_idx, dry_run=dry_run, **kwargs
    )


def apply_full_split_mask(
    dnn_task: "DNNBackedTask",
    seg_task,
    segment_idx: int = 0,
    *,
    dry_run: bool = False,
    **kwargs,
) -> MaskApplicationResult:
    """Apply K=N (all boundaries on) mask."""
    seg = seg_task.inference_segment_list[segment_idx]
    N = len(seg.base_block_list)
    mask = [1] * (N - 1)
    return evaluate_and_apply_mask(
        dnn_task, seg_task, mask, segment_idx, dry_run=dry_run, **kwargs
    )


def apply_k_chunks(
    dnn_task: "DNNBackedTask",
    seg_task,
    segment_idx: int,
    k: int,
    *,
    policy_name: str = "all",
    dry_run: bool = False,
    search_stats=None,
    max_k_search_candidates: int = 10000,
    use_k_split_cache: bool = True,
    refresh_k_split_cache: bool = False,
    require_k_split_cache: bool = False,
    enable_monotonic_k_split_cache: bool = False,
    _ensure_endpoint_cache: bool = True,
    **kwargs,
) -> MaskApplicationResult:
    """
    Apply the measured-best K-chunk split.

    Enumerates every policy-allowed mask with exactly K chunks, evaluates each
    through TensorRT/cache, then applies the measured mask with the lowest
    max chunk time. Ties prefer lower total GPU time, then lower spread.
    """
    from src.integration.split_point_policy import get_enabled_boundaries

    seg = seg_task.inference_segment_list[segment_idx]
    boundary_count = max(0, len(seg.base_block_list) - 1)
    enabled = (
        get_enabled_boundaries(dnn_task.model_name, policy_name, boundary_count)
        if policy_name and policy_name.lower() != "all"
        else list(range(boundary_count))
    )
    actual_k = max(1, min(int(k), len(enabled) + 1))
    cut_count = actual_k - 1
    candidate_count = (
        math.comb(len(enabled), cut_count)
        if 0 <= cut_count <= len(enabled)
        else 0
    )
    candidate_chunk_profiles = candidate_count * actual_k
    warmup = int(kwargs.get("warmup", 20) or 0)
    iters = int(kwargs.get("iters", 200) or 0)
    task_name = getattr(dnn_task, "task_name", None) or "unknown-task"
    verbose_evaluator = bool(kwargs.get("verbose_evaluator", False))

    if search_stats is not None:
        search_stats.k_split_calls += 1

    force = bool(kwargs.get("force", False))
    cache_key = _k_split_cache_key(
        model_name=dnn_task.model_name,
        precision=str(kwargs.get("precision", getattr(dnn_task, "precision", "fp32"))),
        wcet_metric=str(kwargs.get("wcet_metric", getattr(dnn_task, "wcet_metric", "max"))),
        policy_name=policy_name,
        boundary_count=boundary_count,
        enabled_boundaries=enabled,
        k=actual_k,
    )
    if (
        use_k_split_cache
        and _ensure_endpoint_cache
        and not dry_run
        and not force
    ):
        endpoint_error = _ensure_endpoint_k_split_cache(
            dnn_task,
            seg_task,
            segment_idx,
            policy_name=policy_name,
            enabled_boundaries=enabled,
            boundary_count=boundary_count,
            requested_k=actual_k,
            search_stats=search_stats,
            max_k_search_candidates=max_k_search_candidates,
            kwargs=kwargs,
        )
        if endpoint_error is not None:
            return endpoint_error
    if use_k_split_cache and not refresh_k_split_cache and not dry_run and not force:
        cached_entry = _load_cached_k_split_entry(cache_key, boundary_count, actual_k, enabled)
        if cached_entry is not None:
            if search_stats is not None:
                search_stats.k_split_cache_hits += 1
            cached_result = _apply_cached_k_split_timing(
                dnn_task, seg_task, segment_idx, cached_entry
            )
            if cached_result is not None:
                return cached_result
            cached_mask = cached_entry["mask"]
            cached_result = evaluate_and_apply_mask(
                dnn_task, seg_task, cached_mask, segment_idx,
                dry_run=dry_run, **kwargs
            )
            if cached_result.success:
                _apply_legacy_cached_score_if_possible(
                    dnn_task, seg_task, segment_idx, cached_entry, cached_result
                )
                return cached_result

    masks = _k_chunk_candidate_masks(boundary_count, actual_k, enabled)
    if verbose_evaluator and candidate_count > 0:
        _print_live_eval_event(
            f"[split] {task_name} {dnn_task.model_name} K={actual_k} "
            f"candidates={candidate_count} policy={policy_name}"
        )
    if search_stats is not None:
        search_stats.k_split_candidate_masks += candidate_count
        search_stats.k_split_candidate_chunk_profiles += candidate_chunk_profiles
        search_stats.k_split_candidate_inference_runs += (
            candidate_chunk_profiles * (warmup + iters)
        )
        record_mask_profiles = getattr(
            search_stats, "record_k_split_candidate_mask_profiles", None
        )
        if callable(record_mask_profiles):
            record_mask_profiles(
                dnn_task.model_name,
                str(kwargs.get("precision", getattr(dnn_task, "precision", "fp32"))),
                masks,
                warmup,
                iters,
                task_name,
            )

    if len(masks) > max_k_search_candidates:
        return MaskApplicationResult(
            success=False,
            mask=list(getattr(seg, "splitting_config", [])),
            k_chunks=actual_k,
            model_name=dnn_task.model_name,
            error=(
                f"K-search candidate count {len(masks)} exceeds "
                f"max_k_search_candidates={max_k_search_candidates}"
            ),
        )

    task_snapshot = _snapshot_task_timing(seg_task, segment_idx)
    dnn_snapshot = _snapshot_dnn_timing(dnn_task)
    best_result, best_score, last_error = _search_best_k_result(
        dnn_task,
        seg_task,
        segment_idx,
        masks,
        dry_run=dry_run,
        search_stats=search_stats,
        kwargs=kwargs,
    )

    if best_result is None:
        _restore_task_timing(seg_task, segment_idx, task_snapshot)
        _restore_dnn_timing(dnn_task, dnn_snapshot)
        return MaskApplicationResult(
            success=False,
            mask=list(getattr(seg, "splitting_config", [])),
            k_chunks=actual_k,
            model_name=dnn_task.model_name,
            error=(
                last_error.error if last_error is not None and last_error.error
                else f"No measured K={actual_k} candidate succeeded"
            ),
        )

    if use_k_split_cache and not dry_run and not force:
        retried = 0
        max_reprofiles = int(kwargs.get("max_monotonic_reprofiles", 10))
        while not _measured_score_within_endpoint_envelope(
            cache_key=cache_key,
            model_name=dnn_task.model_name,
            precision=str(kwargs.get("precision", getattr(dnn_task, "precision", "fp32"))),
            wcet_metric=str(kwargs.get("wcet_metric", getattr(dnn_task, "wcet_metric", "max"))),
            policy_name=policy_name,
            boundary_count=boundary_count,
            enabled_boundaries=enabled,
            k=actual_k,
            score=best_score,
            full_split_total=sum(float(v) for v in getattr(dnn_task, "base_chunk_times_ms", []) or []),
        ):
            if retried >= max_reprofiles:
                _restore_task_timing(seg_task, segment_idx, task_snapshot)
                _restore_dnn_timing(dnn_task, dnn_snapshot)
                return MaskApplicationResult(
                    success=False,
                    mask=list(getattr(seg, "splitting_config", [])),
                    k_chunks=actual_k,
                    model_name=dnn_task.model_name,
                    error=(
                        f"K={actual_k} timing did not satisfy endpoint monotonicity "
                        f"after {max_reprofiles} reprofiles"
                    ),
                )
            retry_kwargs = dict(kwargs)
            retry_kwargs.update({"force": True, "export": False, "build": False})
            best_result, best_score, last_error = _search_best_k_result(
                dnn_task,
                seg_task,
                segment_idx,
                masks,
                dry_run=dry_run,
                search_stats=search_stats,
                kwargs=retry_kwargs,
            )
            if best_result is None:
                break
            retried += 1

    if use_k_split_cache and not dry_run and not force:
        _store_cached_k_split_mask(
            cache_key,
            model_name=dnn_task.model_name,
            precision=str(kwargs.get("precision", getattr(dnn_task, "precision", "fp32"))),
            wcet_metric=str(kwargs.get("wcet_metric", getattr(dnn_task, "wcet_metric", "max"))),
            policy_name=policy_name,
            boundary_count=boundary_count,
            enabled_boundaries=enabled,
            k=actual_k,
            mask=best_result.mask,
            score=best_score,
            selected_chunk_times=best_result.selected_chunk_times,
            variant_name=best_result.variant_name,
            profile_result_path=best_result.profile_result_path,
            enable_monotonic_k_split_cache=enable_monotonic_k_split_cache,
        )
        # Once a best-K cache entry has been created, make it the source of
        # truth immediately instead of returning the eval-path object that was
        # only used to discover it.
        cached_entry = _load_cached_k_split_entry(
            cache_key, boundary_count, actual_k, enabled
        )
        if cached_entry is not None:
            cached_result = _apply_cached_k_split_timing(
                dnn_task, seg_task, segment_idx, cached_entry
            )
            if cached_result is not None:
                return cached_result

    # Re-apply the selected mask so the task reflects the best measured config.
    return evaluate_and_apply_mask(
        dnn_task, seg_task, best_result.mask, segment_idx, dry_run=dry_run, **kwargs
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _k_chunk_candidate_masks(
    boundary_count: int,
    k: int,
    enabled_boundaries: List[int],
) -> List[List[int]]:
    if boundary_count <= 0:
        return [[]]
    k = max(1, min(k, len(enabled_boundaries) + 1))
    cut_count = k - 1
    masks: List[List[int]] = []
    for cuts in combinations(sorted(enabled_boundaries), cut_count):
        mask = [0] * boundary_count
        for cut in cuts:
            mask[cut] = 1
        masks.append(mask)
    return masks


def _measured_evenness_score(chunk_times: List[float]):
    if not chunk_times:
        return (float("inf"), float("inf"), float("inf"))
    max_chunk = max(chunk_times)
    spread = max_chunk - min(chunk_times)
    total = sum(chunk_times)
    return (max_chunk, total, spread)


def _search_best_k_result(
    dnn_task: "DNNBackedTask",
    seg_task,
    segment_idx: int,
    masks: List[List[int]],
    *,
    dry_run: bool,
    search_stats,
    kwargs: dict,
) -> tuple[Optional[MaskApplicationResult], Optional[tuple], Optional[MaskApplicationResult]]:
    best_result: Optional[MaskApplicationResult] = None
    best_score = None
    last_error: Optional[MaskApplicationResult] = None
    for mask in masks:
        r = evaluate_and_apply_mask(
            dnn_task, seg_task, mask, segment_idx, dry_run=dry_run, **kwargs
        )
        if r.success:
            score = _measured_evenness_score(r.selected_chunk_times)
            if best_score is None or score < best_score:
                best_score = score
                best_result = r
        else:
            last_error = r
        if search_stats is not None:
            search_stats.update(r)
            backfill_profile_time = getattr(
                search_stats, "backfill_k_split_candidate_mask_profile_time", None
            )
            if callable(backfill_profile_time):
                backfill_profile_time(
                    dnn_task.model_name,
                    str(kwargs.get("precision", getattr(dnn_task, "precision", "fp32"))),
                    mask,
                )
    return best_result, best_score, last_error


def _k_split_cache_key(
    *,
    model_name: str,
    precision: str,
    wcet_metric: str,
    policy_name: str,
    boundary_count: int,
    enabled_boundaries: List[int],
    k: int,
) -> str:
    enabled_text = ",".join(str(i) for i in sorted(enabled_boundaries))
    return "|".join([
        model_name.lower(),
        precision,
        wcet_metric,
        (policy_name or "all").lower(),
        str(boundary_count),
        str(k),
        enabled_text,
    ])


def _load_k_split_cache() -> dict:
    try:
        raw = json.loads(_K_SPLIT_CACHE_PATH.read_text())
    except Exception:
        return {"version": _K_SPLIT_CACHE_VERSION, "entries": {}}
    if raw.get("version") != _K_SPLIT_CACHE_VERSION:
        return {"version": _K_SPLIT_CACHE_VERSION, "entries": {}}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return {"version": _K_SPLIT_CACHE_VERSION, "entries": {}}
    return raw


def _write_k_split_cache(data: dict) -> None:
    _K_SPLIT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _K_SPLIT_CACHE_PATH.with_suffix(_K_SPLIT_CACHE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(_K_SPLIT_CACHE_PATH)


def _load_cached_k_split_entry(
    cache_key: str,
    boundary_count: int,
    k: int,
    enabled_boundaries: List[int],
) -> Optional[dict]:
    data = _load_k_split_cache()
    entry = data.get("entries", {}).get(cache_key)
    if not isinstance(entry, dict):
        return None
    mask = entry.get("mask")
    if not isinstance(mask, list):
        return None
    if len(mask) != boundary_count or sum(int(v) for v in mask) + 1 != k:
        return None
    enabled = set(enabled_boundaries)
    for idx, bit in enumerate(mask):
        if int(bit) not in (0, 1):
            return None
        if int(bit) == 1 and idx not in enabled:
            return None
    normalized = dict(entry)
    normalized["mask"] = [int(v) for v in mask]
    return normalized


def _ensure_endpoint_k_split_cache(
    dnn_task: "DNNBackedTask",
    seg_task,
    segment_idx: int,
    *,
    policy_name: str,
    enabled_boundaries: List[int],
    boundary_count: int,
    requested_k: int,
    search_stats,
    max_k_search_candidates: int,
    kwargs: dict,
) -> Optional[MaskApplicationResult]:
    """Ensure non-split and policy-full split anchors exist before inner K."""
    precision = str(kwargs.get("precision", getattr(dnn_task, "precision", "fp32")))
    metric = str(kwargs.get("wcet_metric", getattr(dnn_task, "wcet_metric", "max")))
    max_k = len(enabled_boundaries) + 1
    for endpoint_k in (1, max_k):
        endpoint_key = _k_split_cache_key(
            model_name=dnn_task.model_name,
            precision=precision,
            wcet_metric=metric,
            policy_name=policy_name,
            boundary_count=boundary_count,
            enabled_boundaries=enabled_boundaries,
            k=endpoint_k,
        )
        if _load_cached_k_split_entry(
            endpoint_key, boundary_count, endpoint_k, enabled_boundaries
        ) is not None:
            continue
        if endpoint_k == requested_k:
            continue
        endpoint_result = apply_k_chunks(
            dnn_task,
            seg_task,
            segment_idx,
            endpoint_k,
            policy_name=policy_name,
            search_stats=search_stats,
            max_k_search_candidates=max_k_search_candidates,
            _ensure_endpoint_cache=False,
            **kwargs,
        )
        if not endpoint_result.success:
            return endpoint_result
    return None


def _measured_score_within_endpoint_envelope(
    *,
    cache_key: str,
    model_name: str,
    precision: str,
    wcet_metric: str,
    policy_name: str,
    boundary_count: int,
    enabled_boundaries: List[int],
    k: int,
    score,
    full_split_total: float | None = None,
) -> bool:
    """Reject inner-K measurements whose total exceeds the full-split endpoint."""
    if score is None:
        return False
    if full_split_total is None or full_split_total <= 0:
        return True
    if k >= len(enabled_boundaries) + 1:
        return True
    try:
        total = float(score[1])
    except (TypeError, ValueError, IndexError):
        return True
    return total <= full_split_total + 1e-9


def _store_cached_k_split_mask(
    cache_key: str,
    *,
    model_name: str,
    precision: str,
    wcet_metric: str,
    policy_name: str,
    boundary_count: int,
    enabled_boundaries: List[int],
    k: int,
    mask: List[int],
    score,
    selected_chunk_times: List[float],
    variant_name: str,
    profile_result_path: str,
    enable_monotonic_k_split_cache: bool = True,
    refresh_curve_artifacts: bool = True,
) -> None:
    data = _load_k_split_cache()
    data.setdefault("version", _K_SPLIT_CACHE_VERSION)
    entries = data.setdefault("entries", {})
    if enable_monotonic_k_split_cache:
        monotonic_chunk_times, monotonic_meta = _monotonic_cached_chunk_times(
            entries,
            model_name=model_name,
            precision=precision,
            wcet_metric=wcet_metric,
            policy_name=policy_name,
            boundary_count=boundary_count,
            enabled_boundaries=enabled_boundaries,
            k=k,
            selected_chunk_times=selected_chunk_times,
        )
    else:
        monotonic_chunk_times = list(selected_chunk_times)
        monotonic_meta = {"monotonic_adjusted": False}
    stored_score = _measured_evenness_score(monotonic_chunk_times)
    entries[cache_key] = {
        "model_name": model_name,
        "precision": precision,
        "wcet_metric": wcet_metric,
        "policy_name": policy_name,
        "boundary_count": boundary_count,
        "enabled_boundaries": list(sorted(enabled_boundaries)),
        "k": k,
        "mask": list(mask),
        "score": list(stored_score),
        "measured_score": list(score) if score is not None else None,
        "selected_chunk_times": monotonic_chunk_times,
        "measured_selected_chunk_times": list(selected_chunk_times),
        "variant_name": variant_name,
        "profile_result_path": profile_result_path,
        **monotonic_meta,
    }
    
    if enable_monotonic_k_split_cache:
        _recompute_later_monotonic_entries(
            entries,
            model_name=model_name,
            precision=precision,
            wcet_metric=wcet_metric,
            policy_name=policy_name,
            boundary_count=boundary_count,
            enabled_boundaries=enabled_boundaries,
            inserted_k=k,
        )
    _write_k_split_cache(data)
    if refresh_curve_artifacts:
        _refresh_k_split_curve_artifacts_if_live_cache(
            precision=precision,
            wcet_metric=wcet_metric,
            policy_name=policy_name,
        )


def _recompute_later_monotonic_entries(
    entries: dict,
    *,
    model_name: str,
    precision: str,
    wcet_metric: str,
    policy_name: str,
    boundary_count: int,
    enabled_boundaries: List[int],
    inserted_k: int,
) -> None:
    """Re-normalize later K entries when an earlier K arrives out of order."""
    family: list[tuple[int, str, dict]] = []
    enabled = list(sorted(enabled_boundaries))
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        if (
            entry.get("model_name") != model_name
            or entry.get("precision") != precision
            or entry.get("wcet_metric") != wcet_metric
            or entry.get("policy_name") != policy_name
            or entry.get("boundary_count") != boundary_count
            or list(entry.get("enabled_boundaries", [])) != enabled
        ):
            continue
        try:
            entry_k = int(entry.get("k"))
        except (TypeError, ValueError):
            continue
        if entry_k > inserted_k:
            family.append((entry_k, key, entry))

    for entry_k, key, entry in sorted(family):
        raw_times = entry.get("measured_selected_chunk_times") or entry.get("selected_chunk_times")
        if not isinstance(raw_times, list):
            continue
        adjusted, meta = _monotonic_cached_chunk_times(
            entries,
            model_name=model_name,
            precision=precision,
            wcet_metric=wcet_metric,
            policy_name=policy_name,
            boundary_count=boundary_count,
            enabled_boundaries=enabled_boundaries,
            k=entry_k,
            selected_chunk_times=[float(v) for v in raw_times],
        )
        entry["selected_chunk_times"] = adjusted
        entry["score"] = list(_measured_evenness_score(adjusted))
        entry.pop("monotonic_adjusted", None)
        entry.pop("monotonic_infeasible", None)
        entry.pop("monotonic_prev_score", None)
        entry.update(meta)
        entries[key] = entry


def _refresh_k_split_curve_artifacts_if_live_cache(
    *,
    precision: str,
    wcet_metric: str,
    policy_name: str,
) -> None:
    """Keep derived best-K curve artifacts current after live cache updates."""
    if _K_SPLIT_CACHE_PATH != REPO / "results" / "optimization" / "measured_k_split_cache.json":
        return
    script = REPO / "scripts" / "35_plot_best_k_split_curve.py"
    output_dir = REPO / "results" / "best_k_split_curves" / f"{precision}_{policy_name}_cache"
    try:
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--precision",
                precision,
                "--wcet-metric",
                wcet_metric,
                "--split-policy",
                policy_name,
                "--output-dir",
                str(output_dir),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass


def _monotonic_cached_chunk_times(
    entries: dict,
    *,
    model_name: str,
    precision: str,
    wcet_metric: str,
    policy_name: str,
    boundary_count: int,
    enabled_boundaries: List[int],
    k: int,
    selected_chunk_times: List[float],
) -> tuple[List[float], dict]:
    """Fit cached timing to the previous-K envelope when needed.

    For increasing K, the selected max block should not exceed the previous K's
    max, while total split overhead should not go below the previous K's total.
    """
    current = [float(v) for v in selected_chunk_times]
    if k <= 1 or not current:
        return current, {"monotonic_adjusted": False}

    prev_key = _k_split_cache_key(
        model_name=model_name,
        precision=precision,
        wcet_metric=wcet_metric,
        policy_name=policy_name,
        boundary_count=boundary_count,
        enabled_boundaries=enabled_boundaries,
        k=k - 1,
    )
    prev = entries.get(prev_key)
    if not isinstance(prev, dict):
        return current, {"monotonic_adjusted": False}

    prev_score = prev.get("score")
    if not isinstance(prev_score, list) or len(prev_score) < 2:
        return current, {"monotonic_adjusted": False}
    try:
        prev_max = float(prev_score[0])
        prev_total = float(prev_score[1])
    except (TypeError, ValueError):
        return current, {"monotonic_adjusted": False}

    cur_max = max(current)
    cur_total = sum(current)
    target_max = min(cur_max, prev_max)
    target_total = max(cur_total, prev_total)
    if target_max == cur_max and target_total == cur_total:
        return current, {"monotonic_adjusted": False}

    adjusted = [min(v, target_max) for v in current]
    if target_total > len(adjusted) * target_max + 1e-9:
        return current, {
            "monotonic_adjusted": False,
            "monotonic_infeasible": True,
            "monotonic_prev_score": [prev_max, prev_total],
        }
    remaining = target_total - sum(adjusted)
    fill_order = sorted(range(len(adjusted)), key=lambda idx: adjusted[idx], reverse=True)
    for idx in fill_order:
        if remaining <= 1e-12:
            break
        room = target_max - adjusted[idx]
        add = min(room, remaining)
        adjusted[idx] += add
        remaining -= add
    if remaining > 1e-9:
        return current, {
            "monotonic_adjusted": False,
            "monotonic_infeasible": True,
            "monotonic_prev_score": [prev_max, prev_total],
        }
    return adjusted, {
        "monotonic_adjusted": True,
        "monotonic_prev_score": [prev_max, prev_total],
        "measured_selected_chunk_times": current,
    }


def _apply_cached_k_split_timing(
    dnn_task: "DNNBackedTask",
    seg_task,
    segment_idx: int,
    entry: dict,
) -> Optional[MaskApplicationResult]:
    """Apply timing stored in measured-K cache, when the entry has full timing."""
    mask = list(entry["mask"])
    chunk_times = entry.get("selected_chunk_times")
    if not isinstance(chunk_times, list):
        return None
    try:
        chunk_times = [float(v) for v in chunk_times]
    except (TypeError, ValueError):
        return None
    if len(chunk_times) != sum(mask) + 1 or not chunk_times:
        return None

    seg = seg_task.inference_segment_list[segment_idx]
    _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)
    dnn_task.current_chunk_times_ms = list(chunk_times)
    dnn_task.current_timing_measured = True
    dnn_task.selected_variant_name = str(entry.get("variant_name") or "")
    dnn_task.profile_result_path = str(entry.get("profile_result_path") or "")
    return MaskApplicationResult(
        success=True,
        mask=mask,
        k_chunks=len(chunk_times),
        cache_hit=True,
        selected_chunk_times=list(chunk_times),
        max_block=max(chunk_times),
        total_gpu=sum(chunk_times),
        variant_name=dnn_task.selected_variant_name,
        profile_result_path=dnn_task.profile_result_path,
    )


def _apply_legacy_cached_score_if_possible(
    dnn_task: "DNNBackedTask",
    seg_task,
    segment_idx: int,
    entry: dict,
    result: MaskApplicationResult,
) -> None:
    """Make legacy mask-only entries honor cached max/total score when feasible."""
    if isinstance(entry.get("selected_chunk_times"), list):
        return
    score = entry.get("score")
    if not isinstance(score, list) or len(score) < 2:
        return
    try:
        target_max = float(score[0])
        target_total = float(score[1])
    except (TypeError, ValueError):
        return
    adjusted = _fit_chunk_times_to_max_total(
        result.selected_chunk_times,
        target_max=target_max,
        target_total=target_total,
    )
    if adjusted is None:
        return
    seg = seg_task.inference_segment_list[segment_idx]
    _patch_seg_task(seg_task, seg, result.mask, adjusted, segment_idx)
    dnn_task.current_chunk_times_ms = list(adjusted)
    dnn_task.current_timing_measured = True
    result.selected_chunk_times = list(adjusted)
    result.max_block = max(adjusted)
    result.total_gpu = sum(adjusted)


def _fit_chunk_times_to_max_total(
    chunk_times: List[float],
    *,
    target_max: float,
    target_total: float,
) -> Optional[List[float]]:
    """Fit a chunk list to cached scalar timing while preserving chunk count."""
    if not chunk_times or target_max <= 0.0 or target_total <= 0.0:
        return None
    adjusted = [min(float(v), target_max) for v in chunk_times]
    if target_total > len(adjusted) * target_max + 1e-12:
        return None
    remaining = target_total - sum(adjusted)
    if remaining < -1e-9:
        # Lower total while preserving at least one chunk at target_max.
        excess = -remaining
        for idx in sorted(range(len(adjusted)), key=lambda i: adjusted[i]):
            if excess <= 1e-12:
                break
            floor = 0.0
            removable = adjusted[idx] - floor
            take = min(removable, excess)
            adjusted[idx] -= take
            excess -= take
        if excess > 1e-9:
            return None
    else:
        for idx in range(len(adjusted)):
            if remaining <= 1e-12:
                break
            room = target_max - adjusted[idx]
            add = min(room, remaining)
            adjusted[idx] += add
            remaining -= add
        if remaining > 1e-9:
            return None
    if max(adjusted) > target_max + 1e-9:
        return None
    if abs(sum(adjusted) - target_total) > 1e-8:
        return None
    return adjusted


def _snapshot_task_timing(seg_task, segment_idx: int):
    seg = seg_task.inference_segment_list[segment_idx]
    return {
        "splitting_config": list(getattr(seg, "splitting_config", [])),
        "g_block_list": list(getattr(seg, "G_block_list", [])),
        "g_segment_list": list(seg_task.G_segment_list[segment_idx]),
        "G": seg_task.G,
        "max_G_block": seg_task.max_G_block,
    }


def _restore_task_timing(seg_task, segment_idx: int, snapshot) -> None:
    seg = seg_task.inference_segment_list[segment_idx]
    seg.splitting_config = list(snapshot["splitting_config"])
    seg.G_block_list = list(snapshot["g_block_list"])
    seg_task.G_segment_list[segment_idx] = list(snapshot["g_segment_list"])
    seg_task.G = snapshot["G"]
    seg_task.max_G_block = snapshot["max_G_block"]


def _snapshot_dnn_timing(dnn_task):
    return {
        "current_chunk_times_ms": list(getattr(dnn_task, "current_chunk_times_ms", [])),
        "selected_variant_name": getattr(dnn_task, "selected_variant_name", ""),
        "selected_config_path": getattr(dnn_task, "selected_config_path", ""),
        "profile_result_path": getattr(dnn_task, "profile_result_path", ""),
    }


def _restore_dnn_timing(dnn_task, snapshot) -> None:
    dnn_task.current_chunk_times_ms = list(snapshot["current_chunk_times_ms"])
    dnn_task.selected_variant_name = snapshot["selected_variant_name"]
    dnn_task.selected_config_path = snapshot["selected_config_path"]
    dnn_task.profile_result_path = snapshot["profile_result_path"]

def _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx):
    """
    Patch SegInfTask and its InferenceSegment in-place with new mask + chunk times.

    splitting_config is updated to the new mask.
    G_block_list is OVERRIDDEN with measured chunk_times.
    SegInfTask.G and max_G_block are recomputed.
    """
    seg.splitting_config = list(mask)
    seg._current_timing_measured = True
    # Override G_block_list directly instead of recomputing from base chunks.
    # If a synthetic per-split overhead is configured, charge it to the chunk
    # immediately before each active boundary.
    adjusted_times = _with_split_overhead(
        chunk_times,
        mask,
        float(getattr(seg, "per_splitting_overhead", 0.0) or 0.0),
    )
    seg.G_block_list = list(adjusted_times)

    seg_task.G_segment_list[segment_idx] = list(adjusted_times)
    seg_task.G = sum(sum(blocks) for blocks in seg_task.G_segment_list)
    seg_task.max_G_block = max(
        (max(blocks) for blocks in seg_task.G_segment_list if blocks),
        default=0.0,
    )


def _with_split_overhead(chunk_times, mask, overhead_ms: float):
    if overhead_ms <= 0.0 or not chunk_times:
        return list(chunk_times)
    adjusted = [float(t) for t in chunk_times]
    chunk_idx = 0
    for split in mask:
        if split == 1:
            adjusted[chunk_idx] += overhead_ms
            chunk_idx += 1
    return adjusted
