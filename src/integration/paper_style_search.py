"""
paper_style_search.py — Paper-style OPT and HEU candidate mask search.

Implements the BFS-OPT and greedy-HEU algorithms from DNNSplitting analysis.py
with a DNN-aware profiling backend.

Original paper algorithm pattern
---------------------------------
OPT (BFS):
  Start from no-split config.
  BFS over configs by adding one boundary at a time (constrained to enabled set).
  For each config that fails tolerance → expand it (add more boundaries).
  For each config that meets tolerance → record as feasible candidate.
  Apply the feasible candidate with minimum WCET (C + G).

HEU (greedy):
  Start from no-split config.
  Each round: generate all one-boundary extensions of the current best config.
  Profile all extensions; pick the one with minimum max_G_block.
  Move to that config. Repeat until tolerance met or no enabled boundaries remain.

DNN-aware difference:
  Original: `apply_SS_splitting_config(task, config)` uses synthetic integer timing.
  DNN-aware: `evaluate_and_apply_mask(...)` calls TRT profiling (cache-first).

Profiling budget:
  `max_profiles` limits total evaluate_and_apply_mask calls per search.
  `max_candidates` limits total configs enqueued (BFS-OPT only).
  When budget is exhausted the best-so-far config is applied and returned.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class CandidateSearchResult:
    found: bool               # tolerance was met
    best_mask: List[int]      # best mask found (may not meet tolerance if found=False)
    best_max_block: float
    profiles_used: int        # number of evaluate_and_apply_mask calls
    candidates_tried: int     # total configs evaluated


# ── Internal helpers ──────────────────────────────────────────────────────────

def _add_enabled_split_points(
    cur_mask: List[int],
    enabled_boundaries: List[int],
    seen: set,
    queue: deque,
) -> None:
    """
    Generate all one-bit additions of cur_mask restricted to enabled_boundaries.
    New configs (not in seen) are appended to queue.
    """
    for b in enabled_boundaries:
        if cur_mask[b] != 0:
            continue
        next_mask = list(cur_mask)
        next_mask[b] = 1
        key = tuple(next_mask)
        if key not in seen:
            seen.add(key)
            queue.append(next_mask)


# ── Generic search primitives ─────────────────────────────────────────────────

def _bfs_search(
    initial_mask: List[int],
    enabled_boundaries: List[int],
    apply_fn: Callable[[List[int]], Tuple[bool, float, float]],
    tolerance: float,
    max_profiles: int = 500,
    max_candidates: int = 10000,
) -> CandidateSearchResult:
    """
    BFS over enabled-boundary subsets, expanding non-feasible nodes.

    apply_fn(mask) -> (success, max_block, total_gpu)
      Each call is one profiling_count increment (cache-first in practice).

    Returns best feasible mask (minimum total_gpu = minimum C+G per paper),
    or best-so-far if budget exhausted.
    """
    N = len(initial_mask)
    base_mask = [0] * N
    queue: deque = deque([base_mask])
    seen = {tuple(base_mask)}

    profiles_used = 0
    candidates_tried = 0
    best_mask: Optional[List[int]] = None
    best_max_block = math.inf
    best_total_gpu = math.inf  # paper objective: minimum total WCET among feasible
    found = False

    while queue and profiles_used < max_profiles and candidates_tried < max_candidates:
        cur_mask = queue.popleft()
        success, max_block, total_gpu = apply_fn(cur_mask)
        profiles_used += 1
        candidates_tried += 1

        if not success:
            # Profiling failed; treat as infeasible, expand
            _add_enabled_split_points(cur_mask, enabled_boundaries, seen, queue)
            continue

        if max_block > tolerance:
            # Infeasible: add one-boundary extensions to BFS queue
            _add_enabled_split_points(cur_mask, enabled_boundaries, seen, queue)
        else:
            # Feasible: track best by minimum total_gpu (paper: min C+G)
            found = True
            if total_gpu < best_total_gpu:
                best_total_gpu = total_gpu
                best_max_block = max_block
                best_mask = list(cur_mask)
            # Do NOT expand feasible configs

        # Track overall best for infeasible fallback reporting
        if best_mask is None or (not found and max_block < best_max_block):
            if not found:
                best_max_block = max_block
                best_mask = list(cur_mask)

    if best_mask is None:
        best_mask = list(initial_mask)

    return CandidateSearchResult(
        found=found,
        best_mask=best_mask,
        best_max_block=best_max_block if best_mask is not None else math.inf,
        profiles_used=profiles_used,
        candidates_tried=candidates_tried,
    )


def _greedy_search(
    initial_mask: List[int],
    enabled_boundaries: List[int],
    apply_fn: Callable[[List[int]], Tuple[bool, float]],
    tolerance: float,
    max_profiles: int = 500,
) -> CandidateSearchResult:
    """
    Greedy BFS-HEU: each round, probe all one-boundary extensions of current best,
    pick the one with minimum max_G_block.

    Follows RTA_SS_heu / RTA_UNI_heu from analysis.py.
    apply_fn(mask) -> (success, max_block)
    """
    current_mask = list(initial_mask)
    active_set = {b for b in enabled_boundaries if current_mask[b] == 1}

    profiles_used = 0
    candidates_tried = 0
    best_max_block = math.inf
    found = False

    # Initial probe
    success, max_block = apply_fn(current_mask)
    profiles_used += 1
    candidates_tried += 1
    best_max_block = max_block

    if success and max_block <= tolerance:
        return CandidateSearchResult(
            found=True, best_mask=list(current_mask),
            best_max_block=max_block,
            profiles_used=profiles_used, candidates_tried=candidates_tried,
        )

    seen = {tuple(current_mask)}

    while profiles_used < max_profiles:
        # Generate one-boundary extensions of current
        extensions: List[List[int]] = []
        for b in enabled_boundaries:
            if current_mask[b] != 0:
                continue
            ext = list(current_mask)
            ext[b] = 1
            key = tuple(ext)
            if key not in seen:
                seen.add(key)
                extensions.append(ext)

        if not extensions:
            break  # No enabled boundaries remain

        # Profile all extensions, pick the one with minimum max_G_block
        best_ext_mask: Optional[List[int]] = None
        best_ext_max_block = math.inf

        for ext_mask in extensions:
            if profiles_used >= max_profiles:
                break
            success, max_block = apply_fn(ext_mask)
            profiles_used += 1
            candidates_tried += 1

            if success and max_block < best_ext_max_block:
                best_ext_max_block = max_block
                best_ext_mask = list(ext_mask)

        if best_ext_mask is None:
            break  # All extensions failed to evaluate

        # Move to best extension
        current_mask = best_ext_mask
        best_max_block = best_ext_max_block
        active_set.update(b for b in enabled_boundaries if current_mask[b] == 1)

        if best_ext_max_block <= tolerance:
            found = True
            break

    return CandidateSearchResult(
        found=found,
        best_mask=list(current_mask),
        best_max_block=best_max_block,
        profiles_used=profiles_used,
        candidates_tried=candidates_tried,
    )


# ── SS-aware search functions ─────────────────────────────────────────────────

def search_optimal_ss_mask(
    dnn_task,
    seg_task,
    seg_idx: int,
    tolerance: float,
    enabled_boundaries: List[int],
    eval_kwargs: dict,
    stats,
    *,
    max_profiles: int = 500,
    max_candidates: int = 10000,
) -> CandidateSearchResult:
    """
    Paper-style BFS-OPT for a single SS task segment.

    Enumerates enabled-boundary subsets via BFS, expanding configs that fail
    tolerance. Returns the feasible config with minimum max_G_block (proxy for
    minimum WCET). Each evaluate_and_apply_mask call = one profiling_count step.

    After search, re-applies the best mask (cache hit) so seg_task reflects it.
    """
    from src.integration.mask_applicator import evaluate_and_apply_mask

    N = len(seg_task.inference_segment_list[seg_idx].base_block_list)

    def apply_fn(mask: List[int]) -> Tuple[bool, float, float]:
        r = evaluate_and_apply_mask(dnn_task, seg_task, mask, seg_idx, **eval_kwargs)
        stats.update(r)
        return r.success, r.max_block, r.total_gpu

    result = _bfs_search(
        initial_mask=[0] * (N - 1),
        enabled_boundaries=enabled_boundaries,
        apply_fn=apply_fn,
        tolerance=tolerance,
        max_profiles=max_profiles,
        max_candidates=max_candidates,
    )

    # Re-apply best (cache hit — already profiled above)
    if result.best_mask:
        r = evaluate_and_apply_mask(
            dnn_task, seg_task, result.best_mask, seg_idx, **eval_kwargs
        )
        stats.update(r)

    return result


def search_heuristic_ss_mask(
    dnn_task,
    seg_task,
    seg_idx: int,
    tolerance: float,
    enabled_boundaries: List[int],
    eval_kwargs: dict,
    stats,
    *,
    max_profiles: int = 500,
) -> CandidateSearchResult:
    """
    Paper-style greedy HEU for a single SS task segment.

    Each round: probe all one-boundary extensions of current best config,
    pick the one with minimum max_G_block. Repeat until tolerance met.

    After search, re-applies the best mask (cache hit) so seg_task reflects it.
    """
    from src.integration.mask_applicator import evaluate_and_apply_mask

    seg = seg_task.inference_segment_list[seg_idx]
    N = len(seg.base_block_list)
    # Always start HEU from no-split (per paper: RTA_SS_heu starts from non_splitting_config)
    initial_mask = [0] * (N - 1)

    def apply_fn(mask: List[int]) -> Tuple[bool, float]:
        r = evaluate_and_apply_mask(dnn_task, seg_task, mask, seg_idx, **eval_kwargs)
        stats.update(r)
        return r.success, r.max_block

    result = _greedy_search(
        initial_mask=initial_mask,
        enabled_boundaries=enabled_boundaries,
        apply_fn=apply_fn,
        tolerance=tolerance,
        max_profiles=max_profiles,
    )

    # Re-apply best (cache hit)
    if result.best_mask:
        r = evaluate_and_apply_mask(
            dnn_task, seg_task, result.best_mask, seg_idx, **eval_kwargs
        )
        stats.update(r)

    return result


# ── UNI-aware search functions ────────────────────────────────────────────────

def search_optimal_uni_mask(
    dnn_task,
    st_orig,
    ut_target,
    seg_idx: int,
    tolerance: float,
    enabled_boundaries: List[int],
    eval_kwargs: dict,
    stats,
    *,
    max_profiles: int = 500,
    max_candidates: int = 10000,
) -> CandidateSearchResult:
    """
    Paper-style BFS-OPT for a UNI task segment.

    enabled_boundaries are in TRT (GPU) space (N_gpu-1 boundaries).
    Each apply_fn call evaluates a TRT mask and reconstructs UNI G_block_list.
    """
    from src.integration.dnn_algorithm_runner import _uni_apply_raw_mask

    seg = st_orig.inference_segment_list[seg_idx]
    N_gpu = len(seg.base_block_list)

    def apply_fn(trt_mask: List[int]) -> Tuple[bool, float, float]:
        r = _uni_apply_raw_mask(dnn_task, st_orig, ut_target, seg_idx, trt_mask, eval_kwargs)
        stats.update(r)
        # ut_target.G is the total UNI WCET (sum of all segments) — paper's C+G objective
        return r.success, ut_target.max_G_block, ut_target.G

    result = _bfs_search(
        initial_mask=[0] * (N_gpu - 1),
        enabled_boundaries=enabled_boundaries,
        apply_fn=apply_fn,
        tolerance=tolerance,
        max_profiles=max_profiles,
        max_candidates=max_candidates,
    )

    if result.best_mask:
        _uni_apply_raw_mask(dnn_task, st_orig, ut_target, seg_idx, result.best_mask, eval_kwargs)

    return result


def search_heuristic_uni_mask(
    dnn_task,
    st_orig,
    ut_target,
    seg_idx: int,
    tolerance: float,
    enabled_boundaries: List[int],
    eval_kwargs: dict,
    stats,
    *,
    max_profiles: int = 500,
) -> CandidateSearchResult:
    """
    Paper-style greedy HEU for a UNI task segment.
    """
    from src.integration.dnn_algorithm_runner import _uni_apply_raw_mask

    seg = st_orig.inference_segment_list[seg_idx]
    N_gpu = len(seg.base_block_list)

    def apply_fn(trt_mask: List[int]) -> Tuple[bool, float]:
        r = _uni_apply_raw_mask(dnn_task, st_orig, ut_target, seg_idx, trt_mask, eval_kwargs)
        stats.update(r)
        return r.success, ut_target.max_G_block

    result = _greedy_search(
        initial_mask=[0] * (N_gpu - 1),
        enabled_boundaries=enabled_boundaries,
        apply_fn=apply_fn,
        tolerance=tolerance,
        max_profiles=max_profiles,
    )

    if result.best_mask:
        _uni_apply_raw_mask(dnn_task, st_orig, ut_target, seg_idx, result.best_mask, eval_kwargs)

    return result
