"""
test_fig5_cli.py — Tests for script 40 algorithm CLI: label normalization,
algorithm sets, unknown algorithm validation, and per-interval engine cache.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_script():
    """Load 40_run_fig5_design_time.py as a module without executing main()."""
    spec_path = REPO / "scripts" / "40_run_fig5_design_time.py"
    spec = importlib.util.spec_from_file_location("fig5_script", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Label alias normalization ─────────────────────────────────────────────────

def test_parse_algorithms_canonical_form():
    """Canonical 'model:algo' strings are accepted unchanged."""
    mod = _load_script()
    specs = mod.parse_algorithms(["ss:opt", "uni:tol-fb", "ss:heu"])
    assert specs == [("ss", "opt"), ("uni", "tol-fb"), ("ss", "heu")]


def test_parse_algorithms_paper_label_aliases():
    """Paper-style labels (SS-tol-fb, UNI-opt, etc.) are normalized."""
    mod = _load_script()
    specs = mod.parse_algorithms(["SS-tol-fb", "UNI-opt", "SS-heu", "UNI-tol"])
    assert specs == [
        ("ss", "tol-fb"),
        ("uni", "opt"),
        ("ss", "heu"),
        ("uni", "tol"),
    ]


def test_parse_algorithms_dash_case_aliases():
    """Lowercase dash forms (ss-opt, uni-heu) are normalized."""
    mod = _load_script()
    specs = mod.parse_algorithms(["ss-opt", "uni-heu"])
    assert specs == [("ss", "opt"), ("uni", "heu")]


def test_parse_algorithms_unknown_model_raises():
    mod = _load_script()
    with pytest.raises(ValueError, match="Unknown RTA model"):
        mod.parse_algorithms(["bogus:opt"])


def test_parse_algorithms_unknown_algorithm_raises():
    mod = _load_script()
    with pytest.raises(ValueError, match="Unknown"):
        mod.parse_algorithms(["ss:nonexistent"])


def test_algorithm_set_main4():
    mod = _load_script()
    expected = [
        ("ss", "tol-fb"), ("uni", "tol-fb"), ("ss", "opt"), ("uni", "opt"),
    ]
    specs = mod.parse_algorithms(mod._ALGORITHM_SETS["main4"])
    assert specs == expected


def test_algorithm_set_full8():
    mod = _load_script()
    specs = mod.parse_algorithms(mod._ALGORITHM_SETS["full8"])
    assert ("ss", "opt") in specs
    assert ("uni", "opt") in specs
    assert ("ss", "heu") in specs
    assert ("uni", "heu") in specs
    assert len(specs) == 8


def test_algorithm_set_ss_only():
    mod = _load_script()
    specs = mod.parse_algorithms(mod._ALGORITHM_SETS["ss_only"])
    assert all(m == "ss" for m, _ in specs)
    assert len(specs) == 4


def test_algorithm_set_uni_only():
    mod = _load_script()
    specs = mod.parse_algorithms(mod._ALGORITHM_SETS["uni_only"])
    assert all(m == "uni" for m, _ in specs)
    assert len(specs) == 4


# ── Per-interval engine cache ─────────────────────────────────────────────────

def _make_fake_cfg(tmp_path: Path, model: str, variant: str, groups: list) -> dict:
    """Build a minimal config dict using interval cache paths (the new canonical layout)."""
    from src.optimization.config_evaluator import _interval_onnx_path, _interval_engine_path
    chunks = []
    for i, grp in enumerate(groups):
        onnx_abs  = _interval_onnx_path(model, grp)
        eng32_abs = _interval_engine_path(model, grp, "fp32")
        eng16_abs = _interval_engine_path(model, grp, "fp16")
        # Config stores paths relative to REPO (= tmp_path in tests)
        chunks.append({
            "onnx": str(onnx_abs.relative_to(tmp_path)),
            "engine_fp32": str(eng32_abs.relative_to(tmp_path)),
            "engine_fp16": str(eng16_abs.relative_to(tmp_path)),
            "source_chunk_ids": grp,
            "input_shape": [1, 3, 224, 224],
        })
    return {"chunks": chunks, "merged_groups": groups}


def test_build_engines_with_interval_cache_all_miss(monkeypatch, tmp_path):
    """When interval cache is empty, build_single_engine is called per chunk."""
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)

    calls = []

    def mock_build(onnx_path, engine_path, precision="fp32", dry_run=False):
        calls.append(str(onnx_path))
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(b"fake_engine")
        return True, 0.5

    # Patch the source so the inner import picks it up
    import src.optimization.compiler as compiler_mod
    monkeypatch.setattr(compiler_mod, "build_single_engine", mock_build)

    groups = [[0], [1, 2]]
    cfg = _make_fake_cfg(tmp_path, "alexnet", "v1", groups)

    # Create ONNX files (needed by build path)
    for chunk_cfg in cfg["chunks"]:
        onnx = tmp_path / chunk_cfg["onnx"]
        onnx.parent.mkdir(parents=True, exist_ok=True)
        onnx.write_bytes(b"fake_onnx")

    from src.optimization.config_evaluator import _build_engines_with_interval_cache
    hits, misses, wall = _build_engines_with_interval_cache(
        "alexnet", cfg, groups, "fp32"
    )

    assert misses == 2
    assert hits == 0
    assert wall == pytest.approx(1.0)
    assert len(calls) == 2


def test_build_engines_with_interval_cache_all_hit(monkeypatch, tmp_path):
    """When all engines are in interval cache (= config paths), build_single_engine is never called."""
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)

    import src.optimization.compiler as compiler_mod

    def must_not_build(*a, **kw):
        raise AssertionError("build_single_engine should not be called on cache hit")

    monkeypatch.setattr(compiler_mod, "build_single_engine", must_not_build)

    groups = [[0], [1, 2]]
    cfg = _make_fake_cfg(tmp_path, "alexnet", "v1", groups)

    # Pre-populate the interval engine cache — same paths the config now references.
    for chunk_cfg in cfg["chunks"]:
        eng = tmp_path / chunk_cfg["engine_fp32"]
        eng.parent.mkdir(parents=True, exist_ok=True)
        eng.write_bytes(b"cached_engine")

    from src.optimization.config_evaluator import _build_engines_with_interval_cache
    hits, misses, wall = _build_engines_with_interval_cache(
        "alexnet", cfg, groups, "fp32"
    )

    assert hits == 2
    assert misses == 0
    assert wall == 0.0

    # Engine files remain in place (no copy step needed in new architecture).
    for chunk_cfg in cfg["chunks"]:
        eng = tmp_path / chunk_cfg["engine_fp32"]
        assert eng.exists()
        assert eng.read_bytes() == b"cached_engine"


def test_build_engines_with_interval_cache_timing_recorded(monkeypatch, tmp_path):
    """Per-chunk build timing is written to timing.json in interval cache dir."""
    monkeypatch.setattr("src.optimization.config_evaluator.REPO", tmp_path)

    import src.optimization.compiler as compiler_mod

    def mock_build(onnx_path, engine_path, precision="fp32", dry_run=False):
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(b"built_engine")
        return True, 2.5

    monkeypatch.setattr(compiler_mod, "build_single_engine", mock_build)

    groups = [[0]]
    cfg = _make_fake_cfg(tmp_path, "alexnet", "v1", groups)
    onnx = tmp_path / cfg["chunks"][0]["onnx"]
    onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.write_bytes(b"onnx")

    from src.optimization.config_evaluator import (
        _build_engines_with_interval_cache, _load_interval_timing,
    )
    _build_engines_with_interval_cache("alexnet", cfg, groups, "fp32")

    timing = _load_interval_timing("alexnet", [0])
    assert timing.get("build_fp32_wall_s") == pytest.approx(2.5)
