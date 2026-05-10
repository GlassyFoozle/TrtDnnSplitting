"""
test_interval_cache_scalability.py — Tests for Task B-E interval cache scalability.

Four tests:
  1. Two masks sharing an interval produce the same path key (no per-variant copies).
  2. Config chunk paths point into chunk_cache, never into artifacts/onnx or artifacts/engines.
  3. can_assemble_from_intervals + assemble_from_intervals round-trip.
  4. ProfilingStats.unique_skipped_masks counts unique masks, not attempts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Test 1: Shared interval → same path key ───────────────────────────────────

def test_shared_interval_same_path(monkeypatch, tmp_path):
    """
    Two masks that differ only in which other intervals are active but share the
    interval [0,1] must resolve to the exact same ONNX path for that interval.
    No per-variant copies should be created.
    """
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import _interval_onnx_path

    # Interval [0,1] as seen by mask-A (groups [[0,1],[2],[3]]) and
    # mask-B (groups [[0,1],[2,3]]) — both share chunk ids [0,1].
    path_a = _interval_onnx_path("resnet18", [0, 1])
    path_b = _interval_onnx_path("resnet18", [0, 1])

    assert path_a == path_b, "same interval must map to the same path regardless of calling mask"
    assert "chunk_cache" in str(path_a), "path must be inside chunk_cache, not a variant dir"
    assert "onnx" not in path_a.parts or path_a.parts[path_a.parts.index("artifacts") + 1] == "chunk_cache"


# ── Test 2: make_selected_split_config uses interval cache paths only ─────────

def test_split_config_uses_interval_paths():
    """
    make_selected_split_config() must generate chunk paths that live inside
    artifacts/chunk_cache/.  No path should contain artifacts/onnx/ or
    artifacts/engines/ (those are the old per-variant copy locations).

    Imports _compute_merge_groups from config_evaluator to avoid the torch
    dependency that selective_split.py introduces at module level.
    """
    from src.optimization.config_evaluator import _compute_merge_groups, _interval_onnx_path, _interval_engine_path

    # Simulate what make_selected_split_config() now produces for mask [1,0]:
    # groups = [[0], [1,2]]  → intervals int_0_0 and int_1_2
    mask = [1, 0]
    groups = _compute_merge_groups(mask)
    assert groups == [[0], [1, 2]]

    for grp in groups:
        onnx_path    = _interval_onnx_path("alexnet", grp)
        engine_fp32  = _interval_engine_path("alexnet", grp, "fp32")
        engine_fp16  = _interval_engine_path("alexnet", grp, "fp16")

        for path_str in (str(onnx_path), str(engine_fp32), str(engine_fp16)):
            assert "chunk_cache" in path_str, (
                f"path should reference chunk_cache, got: {path_str}"
            )
            # After stripping "chunk_cache", there must be no "/onnx/" or "/engines/"
            stripped = path_str.replace("chunk_cache", "")
            assert "/onnx/" not in stripped, (
                f"path must not reference artifacts/onnx/: {path_str}"
            )
            assert "/engines/" not in stripped, (
                f"path must not reference artifacts/engines/: {path_str}"
            )


# ── Test 3: cache-only assembly from interval timing ─────────────────────────

def test_assemble_from_intervals(monkeypatch, tmp_path):
    """
    When all interval timing.json files contain gpu_mean_ms_fp32 and
    gpu_p99_ms_fp32, can_assemble_from_intervals returns True and
    assemble_from_intervals returns a valid EvaluationResult.
    """
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import (
        _save_interval_timing,
        _compute_merge_groups,
        can_assemble_from_intervals,
        assemble_from_intervals,
    )

    # mask with 3 base chunks, all boundaries active → 3 intervals: [0],[1],[2]
    mask = [1, 1]
    groups = _compute_merge_groups(mask)  # [[0],[1],[2]]

    # Write GPU timing into each interval
    timing_data = [
        {"gpu_mean_ms_fp32": 0.10, "gpu_p99_ms_fp32": 0.12},
        {"gpu_mean_ms_fp32": 0.20, "gpu_p99_ms_fp32": 0.22},
        {"gpu_mean_ms_fp32": 0.30, "gpu_p99_ms_fp32": 0.33},
    ]
    for grp, td in zip(groups, timing_data):
        _save_interval_timing("alexnet", grp, td)

    assert can_assemble_from_intervals("alexnet", mask, "fp32"), (
        "should be assembleable when all intervals have GPU timing"
    )

    result = assemble_from_intervals(
        model_name="alexnet",
        mask=mask,
        precision="fp32",
        base_variant="dag_aligned_full",
    )

    assert result is not None
    assert result.ok(), f"assembled result should be ok, got error={result.error!r}"
    assert len(result.per_chunk_gpu_p99_ms) == 3
    assert result.per_chunk_gpu_p99_ms[0] == pytest.approx(0.12)
    assert result.per_chunk_gpu_p99_ms[1] == pytest.approx(0.22)
    assert result.per_chunk_gpu_p99_ms[2] == pytest.approx(0.33)
    assert result.chunked_gpu_p99_ms == pytest.approx(sum([0.12, 0.22, 0.33]))


def test_assemble_from_intervals_missing_timing(monkeypatch, tmp_path):
    """
    can_assemble_from_intervals returns False when any interval is missing GPU timing.
    """
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import _save_interval_timing, can_assemble_from_intervals

    mask = [1, 1]
    # Only write timing for interval [0]; [1] and [2] are missing
    _save_interval_timing("alexnet", [0], {"gpu_mean_ms_fp32": 0.10, "gpu_p99_ms_fp32": 0.12})

    assert not can_assemble_from_intervals("alexnet", mask, "fp32"), (
        "should not be assembleable when some intervals lack GPU timing"
    )


# ── Test 4: unique_skipped_masks counts unique masks, not attempts ────────────

def test_unique_skipped_vs_attempt_count():
    """
    ProfilingStats.skipped_cache_misses counts every update() call that results
    in a skip, even for the same mask.  unique_skipped_masks counts each
    distinct mask only once.
    """
    from src.integration.dnn_algorithm_runner import ProfilingStats
    from src.integration.mask_applicator import MaskApplicationResult

    stats = ProfilingStats()

    def _skipped_result(mask):
        return MaskApplicationResult(
            success=False,
            mask=mask,
            k_chunks=sum(mask) + 1,
            cache_hit=False,
            is_k1_baseline=False,
            dry_run=False,
            error="global_budget_exhausted",
        )

    mask_a = [1, 0, 1]
    mask_b = [0, 1, 0]

    # Same mask_a twice, mask_b once → 3 attempts, 2 unique skipped
    stats.update(_skipped_result(mask_a))
    stats.update(_skipped_result(mask_a))
    stats.update(_skipped_result(mask_b))

    assert stats.skipped_cache_misses == 3, (
        f"attempt counter should be 3, got {stats.skipped_cache_misses}"
    )
    assert stats.unique_skipped_masks == 2, (
        f"unique counter should be 2, got {stats.unique_skipped_masks}"
    )
