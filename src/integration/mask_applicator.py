"""
mask_applicator.py — Apply a TensorRT boundary mask to a DNNBackedTask + SegInfTask pair.

The core operation of the DNN-aware splitting pipeline:
  mask (N-1 boundaries) → evaluate_mask() → measured K-chunk times → patch SegInfTask

Key design:
  - base_block_list stays as the N individual dag_aligned_full chunk times (immutable).
  - After evaluation, G_block_list is OVERRIDDEN with the K measured (or estimated) times.
    This means RTA uses real profiled merged-chunk latency, not the sum-of-base-blocks estimate.
  - splitting_config is updated to reflect the new mask.
  - dry_run=True skips TRT and uses sum-of-base-blocks estimates.

Per-chunk timing column:
  wcet_metric="p99"  → per_chunk_gpu_p99_ms  (default, conservative)
  wcet_metric="mean" → per_chunk_gpu_mean_ms  (optimistic)
"""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

if TYPE_CHECKING:
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.live_budget import LiveProfileBudget


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

    # Interval-level cache accounting (0 when dry_run or mask-level cache hit)
    interval_cache_hits: int = 0
    interval_cache_misses: int = 0

    # Wall-clock timing per pipeline phase (0.0 when not measured)
    export_wall_s: float = 0.0
    build_wall_s: float = 0.0
    profile_wall_s: float = 0.0

    # Cold-cache design-time estimate (None when interval timing data is unavailable)
    estimated_cold_total_s: Optional[float] = None

    # Timing
    selected_chunk_times: List[float] = field(default_factory=list)
    max_block: float = 0.0
    total_gpu: float = 0.0

    # Paths
    profile_result_path: str = ""
    variant_name: str = ""

    error: Optional[str] = None


def evaluate_and_apply_mask(
    dnn_task: "DNNBackedTask",
    seg_task,                   # SegInfTask (DNNSplitting)
    mask: List[int],
    segment_idx: int = 0,
    *,
    precision: str = "fp32",
    wcet_metric: str = "p99",   # "p99" or "mean"
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
    wcet_metric : "p99" (conservative) or "mean" (optimistic).
    dry_run     : skip TRT; use sum-of-base-blocks estimate instead.

    Returns
    -------
    MaskApplicationResult — always set, success=False on error.
    """
    from src.optimization.config_evaluator import evaluate_mask as _eval_mask
    from src.integration.dnn_taskset_loader import _apply_mask_to_chunk_times

    seg = seg_task.inference_segment_list[segment_idx]
    base_times = seg.base_block_list   # N individual base chunk times
    N = len(base_times)

    if len(mask) != N - 1:
        return MaskApplicationResult(
            success=False, mask=mask, k_chunks=0,
            error=f"mask length {len(mask)} != N-1 = {N - 1}",
        )

    k = sum(mask) + 1

    # ── K=1 baseline: all-zero mask always uses pre-profiled dag_aligned_full ─
    # Never triggers engine export/build/profile — only masks with active
    # split boundaries (k > 1) should enter the live profiling pipeline.
    if k == 1:
        if not dry_run and all(t == 0.0 for t in base_times):
            return MaskApplicationResult(
                success=False, mask=list(mask), k_chunks=1,
                error=(
                    f"K=1 baseline timing unavailable for {dnn_task.model_name!r}: "
                    "base_chunk_times_ms are all zero. "
                    "Run scripts/20_preflight_design.py to profile the base variant first."
                ),
            )
        chunk_times = _apply_mask_to_chunk_times(base_times, mask)
        _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)
        dnn_task.current_chunk_times_ms = list(chunk_times)
        return MaskApplicationResult(
            success=True, mask=list(mask), k_chunks=1,
            dry_run=dry_run,
            cache_hit=not dry_run,  # live mode: baseline read counts as cache hit
            selected_chunk_times=chunk_times,
            max_block=chunk_times[0] if chunk_times else 0.0,
            total_gpu=sum(chunk_times),
        )

    # ── dry_run: estimate from base block sums ────────────────────────────────
    if dry_run:
        chunk_times = _apply_mask_to_chunk_times(base_times, mask)
        _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)
        dnn_task.current_chunk_times_ms = list(chunk_times)
        return MaskApplicationResult(
            success=True, mask=list(mask), k_chunks=k,
            dry_run=True,
            selected_chunk_times=chunk_times,
            max_block=max(chunk_times) if chunk_times else 0.0,
            total_gpu=sum(chunk_times),
        )

    # ── live budget pre-check (real eval only) ────────────────────────────────
    live_cache_miss_variant = ""
    if live_budget is not None:
        from src.optimization.config_evaluator import is_mask_cached, mask_to_variant_name
        if not is_mask_cached(dnn_task.model_name, mask, precision):
            variant_name = mask_to_variant_name(dnn_task.model_name, mask)
            live_cache_miss_variant = variant_name
            reason = live_budget.check_before_real_eval(dnn_task.model_name, variant_name)
            if reason is not None:
                live_budget.record_skip()
                chunk_times = _apply_mask_to_chunk_times(
                    seg.base_block_list, mask
                ) if seg.base_block_list else []
                _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)
                dnn_task.current_chunk_times_ms = list(chunk_times)
                return MaskApplicationResult(
                    success=False, mask=list(mask), k_chunks=k,
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
        # Fallback: use estimated times from base blocks
        chunk_times = _apply_mask_to_chunk_times(base_times, mask)
        error_msg = eval_result.error or "EvaluationResult not ok (no timing)"
        _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)
        dnn_task.current_chunk_times_ms = list(chunk_times)
        return MaskApplicationResult(
            success=False, mask=list(mask), k_chunks=k,
            cache_hit=eval_result.cache_hit,
            did_export=eval_result.exported,
            did_build=eval_result.built,
            did_profile=eval_result.profiled,
            interval_cache_hits=eval_result.interval_cache_hits,
            interval_cache_misses=eval_result.interval_cache_misses,
            export_wall_s=float(eval_result.export_wall_s or 0.0),
            build_wall_s=float(eval_result.build_wall_s or 0.0),
            profile_wall_s=float(eval_result.profile_wall_s or 0.0),
            estimated_cold_total_s=eval_result.estimated_cold_total_s,
            selected_chunk_times=chunk_times,
            max_block=max(chunk_times) if chunk_times else 0.0,
            total_gpu=sum(chunk_times),
            variant_name=eval_result.variant_name,
            profile_result_path=eval_result.result_json_path,
            error=error_msg,
        )

    # Select timing column
    if wcet_metric == "p99":
        chunk_times = eval_result.per_chunk_gpu_p99_ms or []
    else:
        chunk_times = eval_result.per_chunk_gpu_mean_ms or []

    # Fallback to mean if p99 unavailable
    if not chunk_times:
        chunk_times = eval_result.per_chunk_gpu_mean_ms or []

    # Final fallback: estimated from base sums
    if not chunk_times:
        chunk_times = _apply_mask_to_chunk_times(base_times, mask)

    # Verify chunk count matches expected K
    if len(chunk_times) != k:
        # Use estimated if count mismatch
        chunk_times = _apply_mask_to_chunk_times(base_times, mask)

    _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx)

    # Update DNNBackedTask metadata
    dnn_task.current_chunk_times_ms = list(chunk_times)
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

    Always uses the pre-profiled dag_aligned_full baseline timing; never
    triggers engine export/build/profile regardless of dry_run.
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
    **kwargs,
) -> MaskApplicationResult:
    """
    Apply a balanced K-chunk split using DP partitioning on base_block_list.

    Uses balanced_splitter to find active boundaries that minimize the maximum
    group time. If policy_name is not "all", disabled policy boundaries are
    forced to 0 and K is clamped to the maximum chunks the policy can express.
    """
    from src.optimization.balanced_splitter import (
        balanced_split, policy_aware_balanced_split,
    )
    from src.integration.split_point_policy import get_enabled_boundaries

    seg = seg_task.inference_segment_list[segment_idx]
    base_times = seg.base_block_list
    boundary_count = max(0, len(base_times) - 1)

    if policy_name and policy_name.lower() != "all":
        enabled = get_enabled_boundaries(
            dnn_task.model_name, policy_name, boundary_count
        )
        plan = policy_aware_balanced_split(
            base_times, k, enabled, model_name=dnn_task.model_name
        )
    else:
        plan = balanced_split(base_times, k, model_name=dnn_task.model_name)
    return evaluate_and_apply_mask(
        dnn_task, seg_task, plan.mask, segment_idx, dry_run=dry_run, **kwargs
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _patch_seg_task(seg_task, seg, mask, chunk_times, segment_idx):
    """
    Patch SegInfTask and its InferenceSegment in-place with new mask + chunk times.

    splitting_config is updated to the new mask.
    G_block_list is OVERRIDDEN with the (measured or estimated) chunk_times.
    SegInfTask.G and max_G_block are recomputed.
    """
    seg.splitting_config = list(mask)
    # Override G_block_list directly — do NOT call _compute_block_list().
    # This preserves real measured timing rather than sum-of-base-blocks estimate.
    seg.G_block_list = list(chunk_times)

    seg_task.G_segment_list[segment_idx] = list(chunk_times)
    seg_task.G = sum(sum(blocks) for blocks in seg_task.G_segment_list)
    seg_task.max_G_block = max(
        (max(blocks) for blocks in seg_task.G_segment_list if blocks),
        default=0.0,
    )
