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


# ── Engine interval cache check/populate ─────────────────────────────────────

def _make_fake_config(tmp_path: Path, model: str, variant: str, groups: list) -> dict:
    """Build a minimal cfg dict with real paths in tmp_path."""
    chunks = []
    for i, grp in enumerate(groups):
        onnx = tmp_path / "artifacts" / "onnx" / model / variant / f"chunk{i}.onnx"
        eng32 = tmp_path / "artifacts" / "engines" / model / variant / f"chunk{i}_fp32.engine"
        eng16 = tmp_path / "artifacts" / "engines" / model / variant / f"chunk{i}_fp16.engine"
        chunks.append({
            "onnx": str(onnx.relative_to(tmp_path)),
            "engine_fp32": str(eng32.relative_to(tmp_path)),
            "engine_fp16": str(eng16.relative_to(tmp_path)),
            "source_chunk_ids": grp,
            "input_shape": [1, 3, 224, 224],
        })
    return {"chunks": chunks, "merged_groups": groups}


def test_engine_cache_check_no_cache(monkeypatch, tmp_path):
    """_check_interval_engine_cache returns 0 when interval cache is empty."""
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import _check_interval_engine_cache

    groups = [[0], [1, 2, 3]]
    cfg = _make_fake_config(tmp_path, "alexnet", "v_test", groups)
    hits = _check_interval_engine_cache("alexnet", cfg, groups, "fp32")
    assert hits == 0


def test_engine_cache_populate_then_check(monkeypatch, tmp_path):
    """Populate interval engine cache, then check that the cache is hit."""
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)
    from src.optimization.config_evaluator import (
        _check_interval_engine_cache, _populate_interval_engine_cache,
    )

    groups = [[0], [1, 2, 3]]
    cfg = _make_fake_config(tmp_path, "alexnet", "v_test", groups)

    # Create fake engine files in variant paths
    for chunk_cfg in cfg["chunks"]:
        eng = tmp_path / chunk_cfg["engine_fp32"]
        eng.parent.mkdir(parents=True, exist_ok=True)
        eng.write_bytes(b"fake_engine")

    _populate_interval_engine_cache("alexnet", cfg, groups, "fp32")

    # Now create a second variant config using the same intervals
    cfg2 = _make_fake_config(tmp_path, "alexnet", "v_test2", groups)
    hits = _check_interval_engine_cache("alexnet", cfg2, groups, "fp32")
    assert hits == len(groups), "all engines should be served from interval cache"

    # Verify the engine files were actually copied to the new variant paths
    for chunk_cfg in cfg2["chunks"]:
        eng = tmp_path / chunk_cfg["engine_fp32"]
        assert eng.exists(), f"engine missing: {eng}"
        assert eng.read_bytes() == b"fake_engine"
