"""
test_interval_cache.py — Unit tests for the interval-level ONNX/engine cache.

Tests cover:
  - Interval key / path helpers return consistent, non-colliding paths.
  - Cache hit: ONNX copied from interval cache to variant path.
  - Cache miss: export proceeds; interval cache is populated afterward.
  - Engine cache check/populate round-trip.
  - Cold-cache estimation when interval timing data exists.
  - Cold-cache estimation returns None when no timing data is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """Create a minimal directory tree that mirrors the real repo layout."""
    (tmp_path / "artifacts" / "chunk_cache").mkdir(parents=True)
    (tmp_path / "artifacts" / "onnx").mkdir(parents=True)
    (tmp_path / "artifacts" / "engines").mkdir(parents=True)
    return tmp_path


# ── Interval path helpers ─────────────────────────────────────────────────────

def test_interval_paths_are_unique(tmp_path):
    from src.optimization.config_evaluator import _interval_onnx_path, _interval_engine_path

    p1 = _interval_onnx_path("alexnet", [0])
    p2 = _interval_onnx_path("alexnet", [1, 2, 3])
    p3 = _interval_onnx_path("resnet18", [0])

    assert p1 != p2, "different chunk ranges must produce different paths"
    assert p1 != p3, "different models must produce different paths"
    assert p1.name == "chunk.onnx"
    assert p1.parent.name == "int_0_0"
    assert p2.parent.name == "int_1_3"

    e32 = _interval_engine_path("alexnet", [0], "fp32")
    e16 = _interval_engine_path("alexnet", [0], "fp16")
    assert e32 != e16
    assert e32.name == "chunk_fp32.engine"
    assert e16.name == "chunk_fp16.engine"


# ── Interval timing DB ────────────────────────────────────────────────────────

def test_interval_timing_roundtrip(monkeypatch, tmp_path):
    """_load_interval_timing returns empty dict when no file; round-trips after save."""
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import _load_interval_timing, _save_interval_timing

    timing = _load_interval_timing("alexnet", [0])
    assert timing == {}

    _save_interval_timing("alexnet", [0], {"export_wall_s": 1.23, "build_fp32_wall_s": 5.0})
    timing = _load_interval_timing("alexnet", [0])
    assert timing["export_wall_s"] == pytest.approx(1.23)
    assert timing["build_fp32_wall_s"] == pytest.approx(5.0)


def test_parse_cpp_result_preserves_max_fields(tmp_path):
    """C++ table4 JSON parsing keeps max timing fields for recording/caching."""
    from src.optimization.config_evaluator import _parse_cpp_result

    p = tmp_path / "table4.json"
    p.write_text(json.dumps({
        "full_engine_gpu_mean_ms": 1.0,
        "full_engine_gpu_p99_ms": 1.2,
        "full_engine_gpu_max_ms": 1.4,
        "total_chunked_gpu_mean_ms": 2.0,
        "total_chunked_gpu_p99_ms": 2.2,
        "total_chunked_gpu_max_ms": 2.5,
        "chunks": [
            {
                "gpu_mean_ms": 0.7,
                "gpu_p99_ms": 0.8,
                "gpu_max_ms": 0.9,
                "cpu_mean_ms": 0.71,
                "cpu_p99_ms": 0.81,
                "cpu_max_ms": 0.91,
            },
            {
                "gpu_mean_ms": 1.3,
                "gpu_p99_ms": 1.4,
                "gpu_max_ms": 1.6,
                "cpu_mean_ms": 1.31,
                "cpu_p99_ms": 1.41,
                "cpu_max_ms": 1.61,
            },
        ],
    }))

    parsed = _parse_cpp_result(p)
    assert parsed["full_gpu_max_ms"] == pytest.approx(1.4)
    assert parsed["chunked_gpu_max_ms"] == pytest.approx(2.5)
    assert parsed["per_chunk_gpu_max_ms"] == pytest.approx([0.9, 1.6])
    assert parsed["per_chunk_cpu_wall_max_ms"] == pytest.approx([0.91, 1.61])


# ── Cold-cache estimation ─────────────────────────────────────────────────────

def test_estimate_cold_cost_no_data(monkeypatch, tmp_path):
    """Returns None fields when no interval timing data exists."""
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import _estimate_cold_cost

    result = _estimate_cold_cost("alexnet", [[0], [1, 2]], "fp32", actual_profile_wall_s=2.0)
    assert result["estimated_cold_export_s"] is None
    assert result["estimated_cold_build_s"] is None
    assert result["estimated_cold_total_s"] is None


def test_estimate_cold_cost_with_data(monkeypatch, tmp_path):
    """Sums per-interval export + build times; adds actual_profile_wall_s."""
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import (
        _estimate_cold_cost, _save_interval_timing,
    )

    _save_interval_timing("alexnet", [0], {"export_wall_s": 1.0, "build_fp32_wall_s": 3.0})
    _save_interval_timing("alexnet", [1, 2], {"export_wall_s": 2.0, "build_fp32_wall_s": 4.0})

    result = _estimate_cold_cost("alexnet", [[0], [1, 2]], "fp32", actual_profile_wall_s=5.0)
    assert result["estimated_cold_export_s"] == pytest.approx(3.0)
    assert result["estimated_cold_build_s"] == pytest.approx(7.0)
    assert result["estimated_cold_total_s"] == pytest.approx(15.0)  # 3+7+5


# ── Engine interval cache (path-consolidation architecture) ───────────────────

def test_interval_engine_path_is_canonical(monkeypatch, tmp_path):
    """
    _interval_engine_path returns a path inside artifacts/chunk_cache, not
    inside artifacts/engines/.  The path is stable — two calls with the same
    arguments return the same result (no per-variant copies).
    """
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import _interval_engine_path

    p1 = _interval_engine_path("alexnet", [0], "fp32")
    p2 = _interval_engine_path("alexnet", [0], "fp32")
    assert p1 == p2, "same interval must always map to same path"
    assert "chunk_cache" in str(p1), "engine must be inside chunk_cache"
    assert "engines" not in str(p1).replace("chunk_cache", ""), \
        "engine must not be in a separate /engines/ tree"


def test_interval_engine_exists_after_write(monkeypatch, tmp_path):
    """
    After writing a fake engine to the interval cache path, the path exists
    and can be read back without any copy step.
    """
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import _interval_engine_path

    groups = [[0], [1, 2, 3]]
    for grp in groups:
        eng = _interval_engine_path("alexnet", grp, "fp32")
        eng.parent.mkdir(parents=True, exist_ok=True)
        eng.write_bytes(b"fake_engine")

    # Both intervals are present — no copy needed
    for grp in groups:
        eng = _interval_engine_path("alexnet", grp, "fp32")
        assert eng.exists()
        assert eng.read_bytes() == b"fake_engine"
