"""
dnn_taskset_generator.py — Generate a DNNBackedTask list from a taskset JSON spec,
with optional EvaluationResult overlay.

The generator is the main entry point for script 50. It:
  1. Loads the taskset JSON (via dnn_taskset_loader.load_dnn_taskset)
  2. Optionally overlays per-task EvaluationResult data (evaluated masks)
  3. Returns the list of DNNBackedTask objects

EvaluationResult overlay logic:
  If an EvaluationResult JSON exists for the task's selected_variant_name,
  replace current_chunk_times_ms with the per-chunk timing from that result.
  This makes the task reflect the actual evaluated configuration rather than
  the mask-derived estimate from dag_aligned_full timings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.integration.dnn_task import DNNBackedTask
from src.integration.dnn_taskset_loader import load_dnn_taskset, _apply_mask_to_chunk_times

REPO = Path(__file__).resolve().parent.parent.parent
EVALS_DIR = REPO / "results" / "evaluations"


def generate_dnn_taskset(
    taskset_path: str | Path,
    profiling_db=None,
    overlay_evaluations: bool = True,
    allow_equal_wcet_fallback: bool = False,
) -> List[DNNBackedTask]:
    """
    Generate a list of DNNBackedTask from a taskset JSON spec.

    Parameters
    ----------
    taskset_path : str or Path
        Path to the DNN taskset JSON (absolute or relative to configs/dnn_tasksets/).
    profiling_db : ProfilingDB, optional
        Passed to load_candidate_space for timing data resolution.
    overlay_evaluations : bool
        If True, overlay per-chunk timing from EvaluationResult JSONs when available.
    allow_equal_wcet_fallback : bool
        Passed to load_candidate_space; enables equal-weight WCET/N fallback for
        development/CI use when profiling data is missing.

    Returns
    -------
    list of DNNBackedTask
    """
    tasks = load_dnn_taskset(taskset_path, profiling_db=profiling_db,
                             allow_equal_wcet_fallback=allow_equal_wcet_fallback)

    if overlay_evaluations:
        for task in tasks:
            _try_overlay_evaluation(task)

    return tasks


def _try_overlay_evaluation(task: DNNBackedTask) -> None:
    """
    If an EvaluationResult JSON exists for task.selected_variant_name, overlay
    its per-chunk timing into task.current_chunk_times_ms.
    """
    if not task.selected_variant_name:
        return

    precision = task.precision
    eval_path = (
        EVALS_DIR / task.model_name
        / f"{task.selected_variant_name}_{precision}.json"
    )
    if not eval_path.exists():
        return

    try:
        data = json.loads(eval_path.read_text())
        metric = task.wcet_metric

        if metric == "p99":
            per_chunk = data.get("per_chunk_gpu_p99_ms")
        else:
            per_chunk = data.get("per_chunk_gpu_mean_ms")

        if per_chunk and len(per_chunk) > 0:
            task.current_chunk_times_ms = list(per_chunk)
            # Update profile_result_path to this file
            task.profile_result_path = str(eval_path)
    except Exception:
        pass  # Silently skip malformed evaluation results


def update_task_with_evaluation(
    task: DNNBackedTask,
    eval_result_path: str | Path,
) -> bool:
    """
    Update a DNNBackedTask's current_chunk_times_ms from an EvaluationResult JSON.

    Returns True if the update succeeded, False otherwise.
    """
    eval_path = Path(eval_result_path)
    if not eval_path.exists():
        return False

    try:
        data = json.loads(eval_path.read_text())
        metric = task.wcet_metric

        if metric == "p99":
            per_chunk = data.get("per_chunk_gpu_p99_ms")
        else:
            per_chunk = data.get("per_chunk_gpu_mean_ms")

        if not per_chunk:
            return False

        mask = data.get("mask", [])
        new_variant = data.get("variant_name", task.selected_variant_name)
        config_path = data.get("config_path", task.selected_config_path)

        task.current_chunk_times_ms = list(per_chunk)
        task.profile_result_path = str(eval_path)
        if mask:
            task.initial_mask = list(mask)
        if new_variant:
            task.selected_variant_name = new_variant
        if config_path:
            task.selected_config_path = config_path
        return True
    except Exception:
        return False
