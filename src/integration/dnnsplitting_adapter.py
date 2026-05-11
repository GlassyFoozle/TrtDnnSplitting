"""
dnnsplitting_adapter.py — Convert DNNBackedTask → SegInfTask for RTA.

Key issue: InferenceSegment.__init__ enforces G_segment >= max_block_count as an
integer-unit invariant. Real DNN tasks in milliseconds violate this (e.g., 1.757ms
total with 22 chunks). Workaround: pass dummy_G = max(N, 1) to satisfy the check,
then immediately override base_block_list, splitting_config, and G_block_list with
real measured values.

per_splitting_overhead is set to 0.0 because base_chunk_times_ms are already
measured per-chunk GPU times that include all real overhead.

RTA logic is in src/rta/ (adapted from DNNSplitting; formulas unchanged).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

REPO = Path(__file__).resolve().parent.parent.parent

from src.rta.task import InferenceSegment, SegInfTask  # noqa: E402


def get_dnnsplitting_dir() -> Path:
    """Legacy stub — RTA is now internal in src/rta/. Returns REPO for compatibility."""
    return REPO


def dnn_task_to_seginftask(
    dnn_task,
    splitting_config: Optional[List[int]] = None,
) -> object:
    """
    Convert a DNNBackedTask into a DNNSplitting SegInfTask.

    Parameters
    ----------
    dnn_task : DNNBackedTask
        The task to convert.
    splitting_config : list of int, optional
        A boundary mask of length (N-1) where 1=split, 0=merge.
        If None, uses dnn_task.initial_mask (all-1 = max split).

    Returns
    -------
    SegInfTask
        Ready for use with DNNSplitting RTA functions.

    Notes
    -----
    The returned SegInfTask has:
      - C_list = [cpu_pre_ms, cpu_post_ms]
      - One InferenceSegment backed by current_chunk_times_ms
      - per_splitting_overhead = 0.0
      - splitting_config applied as given (or initial_mask)
      - period, deadline, priority from dnn_task

    The InferenceSegment is constructed with a dummy G_segment (= max(N, 1))
    to pass the integer-unit validation in __init__, then its internals are
    overridden with real float millisecond values.
    """
    # base_chunk_times_ms: N individual base chunks (dag_aligned_full granularity)
    # initial_mask: N-1 boundary mask over the base chunks
    # current_chunk_times_ms: already-merged groups — NOT used as base_block_list
    base_times = list(dnn_task.base_chunk_times_ms)
    N = len(base_times)  # number of base chunks

    if splitting_config is None:
        splitting_config = list(dnn_task.initial_mask)

    if len(splitting_config) != N - 1:
        raise ValueError(
            f"splitting_config length {len(splitting_config)} != "
            f"N-1 = {N - 1} (N = candidate_count = {N})"
        )

    # --- Build InferenceSegment with the dummy-G workaround ---
    # InferenceSegment.__init__ checks G_segment >= max_block_count (integer-unit invariant).
    # Real DNN timings in ms violate this (1.757ms < 22 blocks).
    # Pass dummy_G = max(N, 1) to pass validation, then override internals.
    dummy_G = max(N, 1)
    seg = InferenceSegment(
        G_segment=dummy_G,
        max_block_count=N,
        per_splitting_overhead=0.0,
    )
    # Override with real measured millisecond values
    seg.G_segment = float(sum(base_times))
    seg.base_block_list = list(base_times)
    seg.max_block_count = N
    seg.splitting_config = list(splitting_config)
    current_times = list(getattr(dnn_task, "current_chunk_times_ms", []) or [])
    expected_k = sum(splitting_config) + 1
    if len(current_times) == expected_k:
        seg.G_block_list = [float(t) for t in current_times]
    else:
        seg.G_block_list = seg._compute_block_list()

    # --- Build segment_list for SegInfTask ---
    # SegInfTask.__init__ also calls InferenceSegment(G_segment, max_block_count, ...).
    # Pass dummy_G here too so it passes validation.
    segment_list = [
        {
            "C": float(dnn_task.cpu_pre_ms),
            "G_segment": float(dummy_G),   # dummy; overridden below
            "max_block_count": N,
            "per_splitting_overhead": 0.0,
        },
        {
            "C": float(dnn_task.cpu_post_ms),
            "G_segment": 0,
            "max_block_count": 1,
            "per_splitting_overhead": 0.0,
        },
    ]

    # DNN tasksets use the common real-time convention "1 = highest priority".
    # DNNSplitting.sort_task_set orders larger numeric priority first, so pass
    # the negated value to preserve deadline-monotonic order without changing
    # source taskset schemas or DNNSplitting core code.
    dnnsplitting_priority = -float(dnn_task.priority)

    task = SegInfTask(
        id=dnn_task.task_name,
        segment_list=segment_list,
        period=float(dnn_task.period_ms),
        deadline=float(dnn_task.deadline_ms),
        priority=dnnsplitting_priority,
        cpu=dnn_task.cpu_id,
    )

    if not task.is_valid():
        raise RuntimeError(
            f"SegInfTask construction failed for task {dnn_task.task_name!r}. "
            "Check that base_chunk_times_ms is non-empty."
        )

    # Replace the InferenceSegment built by SegInfTask.__init__ with our
    # properly-timed one (real ms base_block_list + correct splitting_config).
    task.inference_segment_list[0] = seg
    task.G_segment_list[0] = seg.G_block_list
    task.G = sum(sum(blocks) for blocks in task.G_segment_list)
    task.max_G_block = max(
        (max(blocks) for blocks in task.G_segment_list if blocks),
        default=0.0,
    )

    return task


def build_task_set_dict(
    dnn_tasks,
    splitting_configs: Optional[dict] = None,
) -> dict:
    """
    Build a DNNSplitting task_set dict from a list of DNNBackedTask.

    Parameters
    ----------
    dnn_tasks : list of DNNBackedTask
    splitting_configs : dict {task_name: List[int]}, optional
        Per-task splitting config overrides. If None, each task's initial_mask is used.

    Returns
    -------
    dict  {"cpus": {cpu_id: [SegInfTask, ...]}}
        Suitable for passing directly to DNNSplitting analysis functions.
    """
    if splitting_configs is None:
        splitting_configs = {}

    cpus: dict = {}
    for dt in dnn_tasks:
        cfg = splitting_configs.get(dt.task_name, None)
        st = dnn_task_to_seginftask(dt, splitting_config=cfg)
        cpu_key = dt.cpu_id
        cpus.setdefault(cpu_key, []).append(st)

    return {"cpus": cpus}
