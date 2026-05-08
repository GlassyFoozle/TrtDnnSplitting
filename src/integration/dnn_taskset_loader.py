"""
dnn_taskset_loader.py — Load and validate DNN taskset JSON configs.

A DNN taskset JSON file describes a set of real-DNN inference tasks backed by
TRT profiling data. The loader resolves per-chunk timing from the profiling
artifacts and constructs DNNBackedTask objects.

JSON format (configs/dnn_tasksets/*.json):
{
  "name": "...",
  "description": "...",            # optional
  "precision": "fp32",
  "wcet_metric": "p99",            # "p99" or "mean"; default "p99"
  "tasks": [
    {
      "task_name": "tau1_alexnet",
      "model_name": "alexnet",
      "period_ms": 50.0,
      "deadline_ms": 50.0,
      "priority": 1,
      "cpu_id": 0,
      "cpu_pre_ms": 0.5,           # pre-GPU CPU time (ms)
      "cpu_post_ms": 0.2,          # post-GPU CPU time (ms)
      "target_chunks": 4,          # requested K chunks (used to build initial_mask)
      "variant_name": "...",       # optional: override selected variant name
      "notes": ""                  # optional
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from src.integration.dnn_task import DNNBackedTask
from src.optimization.candidate_space import load_candidate_space

REPO = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = REPO / "configs" / "dnn_tasksets"

REQUIRED_TASK_FIELDS = {
    "task_name", "model_name", "period_ms", "deadline_ms",
    "priority", "cpu_id", "cpu_pre_ms", "cpu_post_ms",
}


def validate_dnn_taskset_json(data: dict) -> List[str]:
    """
    Validate a DNN taskset JSON dict.

    Returns a list of error strings (empty if valid).
    """
    errors = []
    if "tasks" not in data or not isinstance(data["tasks"], list):
        errors.append("Missing or non-list 'tasks' field")
        return errors
    if not data["tasks"]:
        errors.append("'tasks' list is empty")

    for i, t in enumerate(data["tasks"]):
        prefix = f"tasks[{i}]"
        missing = REQUIRED_TASK_FIELDS - set(t.keys())
        if missing:
            errors.append(f"{prefix}: missing fields: {sorted(missing)}")
        if "target_chunks" not in t and "initial_mask" not in t:
            errors.append(f"{prefix}: must have 'target_chunks' or 'initial_mask'")
        for ms_field in ("period_ms", "deadline_ms", "cpu_pre_ms", "cpu_post_ms"):
            val = t.get(ms_field)
            if val is not None and (not isinstance(val, (int, float)) or val < 0):
                errors.append(f"{prefix}: {ms_field} must be a non-negative number")
    return errors


def _build_initial_mask_for_k(boundary_count: int, k: int) -> List[int]:
    """
    Build an initial boundary mask that produces exactly k chunks from N boundaries.

    Places k-1 active boundaries evenly distributed across the N positions.
    """
    if k <= 1:
        return [0] * boundary_count
    if k >= boundary_count + 1:
        return [1] * boundary_count

    # Place k-1 split points evenly: use step = N/(k-1)
    mask = [0] * boundary_count
    n_splits = k - 1
    for i in range(n_splits):
        pos = int(round((i + 1) * boundary_count / k)) - 1
        pos = max(0, min(boundary_count - 1, pos))
        mask[pos] = 1
    return mask


def load_dnn_taskset(
    path: str | Path,
    profiling_db=None,
) -> List[DNNBackedTask]:
    """
    Load a DNN taskset JSON and return a list of DNNBackedTask objects.

    Timing data is resolved from dag_aligned_full C++ profiling results.
    Falls back to zeros if no profiling data is available (suitable for
    structure-only tests).

    Parameters
    ----------
    path : str or Path
        Path to the taskset JSON file. If relative, resolved from configs/dnn_tasksets/.
    profiling_db : ProfilingDB, optional
        Used to load per-chunk timing. If None, loads directly from result JSONs.

    Returns
    -------
    list of DNNBackedTask
    """
    path = Path(path)
    if not path.is_absolute():
        # Try as-is from CWD first, then under CONFIGS_DIR
        if not path.exists():
            path = CONFIGS_DIR / path.name if (CONFIGS_DIR / path.name).exists() else CONFIGS_DIR / path

    raw = json.loads(path.read_text())
    errors = validate_dnn_taskset_json(raw)
    if errors:
        raise ValueError(
            f"Invalid DNN taskset JSON at {path}:\n" + "\n".join(f"  {e}" for e in errors)
        )

    precision = raw.get("precision", "fp32")
    wcet_metric = raw.get("wcet_metric", "p99")

    tasks: List[DNNBackedTask] = []
    for spec in raw["tasks"]:
        model_name = spec["model_name"]
        task_name = spec["task_name"]

        # Load candidate space for timing data
        try:
            cs = load_candidate_space(model_name, precision, profiling_db)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"[{task_name}] {exc}\n"
                "Run script 26 and 29 first, or set timing to zeros via task-level override."
            ) from exc

        N = cs.candidate_count
        boundary_count = cs.boundary_count

        # Select timing column
        if wcet_metric == "p99":
            base_chunk_times = cs.chunk_gpu_p99_ms
        else:
            base_chunk_times = cs.chunk_gpu_mean_ms

        # Build initial_mask
        if "initial_mask" in spec:
            initial_mask = list(spec["initial_mask"])
            if len(initial_mask) != boundary_count:
                raise ValueError(
                    f"[{task_name}] initial_mask length {len(initial_mask)} "
                    f"!= boundary_count {boundary_count}"
                )
        else:
            target_k = int(spec.get("target_chunks", 1))
            initial_mask = _build_initial_mask_for_k(boundary_count, target_k)

        # Compute current_chunk_times_ms from initial_mask
        # Merge consecutive base chunks according to the mask (0=merge into left)
        current_chunk_times = _apply_mask_to_chunk_times(base_chunk_times, initial_mask)

        # Variant / config references (may be empty strings if not yet evaluated)
        variant_name = spec.get("variant_name", "")
        selected_config_path = spec.get("selected_config_path", "")
        profile_result_path = spec.get("profile_result_path", "")

        dt = DNNBackedTask(
            task_name=task_name,
            model_name=model_name,
            precision=precision,
            period_ms=float(spec["period_ms"]),
            deadline_ms=float(spec["deadline_ms"]),
            priority=float(spec["priority"]),
            cpu_id=int(spec["cpu_id"]),
            cpu_pre_ms=float(spec["cpu_pre_ms"]),
            cpu_post_ms=float(spec["cpu_post_ms"]),
            base_variant=cs.base_variant,
            candidate_count=N,
            boundary_count=boundary_count,
            initial_mask=initial_mask,
            selected_variant_name=variant_name,
            selected_config_path=selected_config_path,
            profile_result_path=profile_result_path,
            wcet_metric=wcet_metric,
            base_chunk_times_ms=list(base_chunk_times),
            current_chunk_times_ms=list(current_chunk_times),
            notes=spec.get("notes", ""),
        )
        tasks.append(dt)

    return tasks


def _apply_mask_to_chunk_times(
    base_times: List[float],
    mask: List[int],
) -> List[float]:
    """
    Merge base chunk times according to a boundary mask.

    mask[i] = 1 → split between chunk i and chunk i+1 (start a new group)
    mask[i] = 0 → merge chunk i+1 into the current group

    Returns a list of merged group times (sum of constituent chunks).
    """
    if not base_times:
        return []
    N = len(base_times)
    assert len(mask) == N - 1, f"mask length {len(mask)} != N-1 = {N - 1}"

    groups: List[float] = []
    current = base_times[0]
    for i, m in enumerate(mask):
        if m == 1:
            groups.append(current)
            current = base_times[i + 1]
        else:
            current += base_times[i + 1]
    groups.append(current)
    return groups
