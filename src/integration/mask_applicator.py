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
from itertools import combinations
import sys
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
    dry_run: bool = False,
    warmup: int = 20,
    iters: int = 200,
    live_budget: "Optional[LiveProfileBudget]" = None,
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
            print(
                f"[live] real profile/build start: {dnn_task.model_name}/{variant_name} "
                f"K={k}",
                flush=True,
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

    if search_stats is not None:
        search_stats.k_split_calls += 1
        search_stats.k_split_candidate_masks += candidate_count
        search_stats.k_split_candidate_chunk_profiles += candidate_chunk_profiles
        search_stats.k_split_candidate_inference_runs += (
            candidate_chunk_profiles * (warmup + iters)
        )

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
    if use_k_split_cache and not dry_run and not force:
        cached_mask = _load_cached_k_split_mask(cache_key, boundary_count, actual_k, enabled)
        if cached_mask is not None:
            if search_stats is not None:
                search_stats.k_split_cache_hits += 1
            cached_result = evaluate_and_apply_mask(
                dnn_task, seg_task, cached_mask, segment_idx,
                dry_run=dry_run, **kwargs
            )
            if cached_result.success:
                return cached_result

    masks = _k_chunk_candidate_masks(boundary_count, actual_k, enabled)

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
        )

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


def _load_cached_k_split_mask(
    cache_key: str,
    boundary_count: int,
    k: int,
    enabled_boundaries: List[int],
) -> Optional[List[int]]:
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
    return [int(v) for v in mask]


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
) -> None:
    data = _load_k_split_cache()
    data.setdefault("version", _K_SPLIT_CACHE_VERSION)
    entries = data.setdefault("entries", {})
    entries[cache_key] = {
        "model_name": model_name,
        "precision": precision,
        "wcet_metric": wcet_metric,
        "policy_name": policy_name,
        "boundary_count": boundary_count,
        "enabled_boundaries": list(sorted(enabled_boundaries)),
        "k": k,
        "mask": list(mask),
        "score": list(score) if score is not None else None,
    }
    _write_k_split_cache(data)


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
    # Override G_block_list directly — do NOT call _compute_block_list().
    # This preserves real measured timing.
    seg.G_block_list = list(chunk_times)

    seg_task.G_segment_list[segment_idx] = list(chunk_times)
    seg_task.G = sum(sum(blocks) for blocks in seg_task.G_segment_list)
    seg_task.max_G_block = max(
        (max(blocks) for blocks in seg_task.G_segment_list if blocks),
        default=0.0,
    )
