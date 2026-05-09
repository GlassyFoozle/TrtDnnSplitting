"""
test_k1_accounting.py — Tests for K=1 baseline accounting correctness.

Verifies:
  - K=1 evaluations set is_k1_baseline=True in MaskApplicationResult
  - ProfilingStats.update() routes K=1 to baseline_k1_hits, NOT real_profiles
  - K=1 does not invoke evaluate_mask() (short-circuits before TRT pipeline)
  - SS and UNI no-split accounting are consistent (both count baseline_k1_hits)
  - ProfilingStats.to_dict() includes baseline_k1_hits
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_taskset_json(base_times: List[float], all_zeros: bool = False) -> Path:
    """Write a one-task taskset JSON and return its path."""
    spec = {
        "name": "test_ts",
        "precision": "fp32",
        "wcet_metric": "p99",
        "tasks": [{
            "task_name": "t1",
            "model_name": "alexnet",
            "precision": "fp32",
            "period_ms": 10.0,
            "deadline_ms": 10.0,
            "priority": 1,
            "cpu_id": 0,
            "cpu_pre_ms": 0.1,
            "cpu_post_ms": 0.1,
            "target_chunks": 1,
            "wcet_metric": "p99",
        }],
    }
    fd = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
    json.dump(spec, fd)
    fd.close()
    return Path(fd.name)


def _load_first_task(tmp_path: Path):
    """Return (DNNBackedTask, SegInfTask) for a one-task alexnet taskset."""
    from src.integration.dnn_taskset_loader import load_dnn_taskset
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask

    p = _make_taskset_json([])
    tasks = load_dnn_taskset(p, allow_equal_wcet_fallback=True)
    dt = tasks[0]
    seg_task = dnn_task_to_seginftask(dt)
    return dt, seg_task


def _load_full_taskset(n_tasks: int = 4):
    """Return (sorted_list, task_map) from a generated n-task taskset."""
    import json, tempfile
    from src.integration.dnn_taskset_generator import generate_dnn_taskset
    from src.integration.dnnsplitting_adapter import build_task_set_dict
    from src.rta.analysis import sort_task_set

    spec = {
        "name": "test",
        "precision": "fp32",
        "wcet_metric": "p99",
        "tasks": [
            {
                "task_name": f"t{i}",
                "model_name": "alexnet",
                "precision": "fp32",
                "period_ms": 10.0 + i * 5.0,
                "deadline_ms": 10.0 + i * 5.0,
                "priority": i + 1,
                "cpu_id": 0,
                "cpu_pre_ms": 0.1,
                "cpu_post_ms": 0.1,
                "target_chunks": 1,
                "wcet_metric": "p99",
            }
            for i in range(n_tasks)
        ],
    }
    fd = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
    json.dump(spec, fd)
    fd.close()
    dnn_tasks = generate_dnn_taskset(Path(fd.name), overlay_evaluations=False,
                                     allow_equal_wcet_fallback=True)

    task_set = build_task_set_dict(dnn_tasks)
    sorted_list = sort_task_set(task_set)
    task_map = {}
    for dt in dnn_tasks:
        for st in sorted_list:
            if str(st.id) == str(dt.task_name):
                task_map[str(dt.task_name)] = (dt, st)
                break
    return sorted_list, task_map, dnn_tasks


# ── MaskApplicationResult.is_k1_baseline ─────────────────────────────────────

def test_k1_baseline_flag_set_on_dry_run_success(tmp_path):
    """evaluate_and_apply_mask with all-zeros mask in dry-run sets is_k1_baseline=True."""
    from src.integration.mask_applicator import evaluate_and_apply_mask

    dt, seg_task = _load_first_task(tmp_path)
    N = dt.candidate_count
    mask = [0] * (N - 1)

    result = evaluate_and_apply_mask(dt, seg_task, mask, dry_run=True)

    assert result.is_k1_baseline is True
    assert result.k_chunks == 1
    assert result.success is True


def test_k1_baseline_flag_set_on_live_error_path(tmp_path):
    """K=1 error path (all base times zero, live mode) sets is_k1_baseline=True."""
    from src.integration.mask_applicator import evaluate_and_apply_mask

    dt, seg_task = _load_first_task(tmp_path)
    # Ensure base times are all-zero (fresh clone / no preflight)
    seg = seg_task.inference_segment_list[0]
    seg.base_block_list = [0.0] * dt.candidate_count
    dt.base_chunk_times_ms = [0.0] * dt.candidate_count

    N = dt.candidate_count
    mask = [0] * (N - 1)

    result = evaluate_and_apply_mask(dt, seg_task, mask, dry_run=False)

    assert result.is_k1_baseline is True
    assert result.success is False
    assert "all zero" in (result.error or "")


# ── ProfilingStats.update() routing ──────────────────────────────────────────

def _make_k1_result(**kwargs):
    from src.integration.mask_applicator import MaskApplicationResult
    return MaskApplicationResult(
        success=True, mask=[0, 0], k_chunks=1,
        is_k1_baseline=True,
        dry_run=kwargs.get("dry_run", True),
        cache_hit=kwargs.get("cache_hit", False),
        **{k: v for k, v in kwargs.items() if k not in ("dry_run", "cache_hit")},
    )


def test_k1_does_not_increment_real_profiles():
    from src.integration.dnn_algorithm_runner import ProfilingStats

    stats = ProfilingStats()
    stats.update(_make_k1_result())

    assert stats.baseline_k1_hits == 1
    assert stats.real_profiles == 0
    assert stats.dry_run_evaluations == 0
    assert stats.cache_hits == 0
    assert stats.masks_evaluated == 1


def test_k1_live_does_not_increment_cache_hits():
    """K=1 in live mode (cache_hit=True) still goes to baseline_k1_hits."""
    from src.integration.dnn_algorithm_runner import ProfilingStats

    stats = ProfilingStats()
    stats.update(_make_k1_result(dry_run=False, cache_hit=True))

    assert stats.baseline_k1_hits == 1
    assert stats.cache_hits == 0
    assert stats.real_profiles == 0


def test_multiple_k1_updates_accumulate():
    from src.integration.dnn_algorithm_runner import ProfilingStats

    stats = ProfilingStats()
    for _ in range(8):
        stats.update(_make_k1_result())

    assert stats.baseline_k1_hits == 8
    assert stats.masks_evaluated == 8
    assert stats.real_profiles == 0


def test_non_k1_still_routes_to_real_profiles():
    """Non-K=1 mask with no special flags still goes to real_profiles."""
    from src.integration.mask_applicator import MaskApplicationResult
    from src.integration.dnn_algorithm_runner import ProfilingStats

    stats = ProfilingStats()
    r = MaskApplicationResult(
        success=True, mask=[1, 0], k_chunks=2,
        is_k1_baseline=False, dry_run=False, cache_hit=False,
    )
    stats.update(r)

    assert stats.real_profiles == 1
    assert stats.baseline_k1_hits == 0


def test_profiling_stats_to_dict_includes_k1():
    from src.integration.dnn_algorithm_runner import ProfilingStats

    stats = ProfilingStats()
    stats.update(_make_k1_result())
    d = stats.to_dict()

    assert "baseline_k1_hits" in d
    assert d["baseline_k1_hits"] == 1
    assert d["real_profiles"] == 0


# ── K=1 does not invoke evaluate_mask() ──────────────────────────────────────

def test_k1_mask_does_not_call_evaluate_mask(monkeypatch, tmp_path):
    """Verify K=1 short-circuits before evaluate_mask() is ever called."""
    from src.integration.mask_applicator import evaluate_and_apply_mask

    called = []

    def mock_evaluate_mask(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("evaluate_mask should never be called for K=1 mask")

    # K=1 shortcut is before the `from src.optimization.config_evaluator import evaluate_mask`
    # line in evaluate_and_apply_mask. Patching the source module is enough.
    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        mock_evaluate_mask,
    )

    dt, seg_task = _load_first_task(tmp_path)
    N = dt.candidate_count
    mask = [0] * (N - 1)

    result = evaluate_and_apply_mask(dt, seg_task, mask, dry_run=True)
    assert result.is_k1_baseline is True
    assert called == [], "evaluate_mask was unexpectedly called"


# ── SS accounting with K=1 init ───────────────────────────────────────────────

def test_ss_single_counts_baseline_k1_hits(monkeypatch):
    """_run_ss_single calls apply_no_split_mask per task → baseline_k1_hits == N."""
    from src.integration.dnn_algorithm_runner import (
        _run_ss_single, DNNAlgorithmResult,
    )
    from src.integration.mask_applicator import MaskApplicationResult

    n = 4
    sorted_list, task_map, _ = _load_full_taskset(n_tasks=n)

    applied = []

    def mock_apply_no_split(dt, st, seg_idx=0, **kwargs):
        r = MaskApplicationResult(
            success=True, mask=[0] * (st.inference_segment_list[0].base_block_list.__len__() - 1),
            k_chunks=1, is_k1_baseline=True, dry_run=True,
            selected_chunk_times=[1.5], max_block=1.5, total_gpu=1.5,
        )
        applied.append(r)
        # Apply the K=1 patch so RTA sees sensible G
        seg = st.inference_segment_list[0]
        seg.G_block_list = [1.5]
        st.G_segment_list[0] = [1.5]
        st.G = 1.5
        st.max_G_block = 1.5
        return r

    monkeypatch.setattr(
        "src.integration.dnn_algorithm_runner.apply_no_split_mask",
        mock_apply_no_split,
    )

    result = DNNAlgorithmResult(
        rta_model="SS", algorithm="single", taskset_path="", precision="fp32",
        wcet_metric="p99", dry_run=True,
    )

    _run_ss_single(sorted_list, task_map, result, {"dry_run": True})

    assert result.stats.baseline_k1_hits == n, (
        f"Expected {n} baseline_k1_hits, got {result.stats.baseline_k1_hits}"
    )
    assert result.stats.real_profiles == 0
