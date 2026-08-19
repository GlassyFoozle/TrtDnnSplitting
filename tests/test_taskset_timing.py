from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.integration.taskset_timing import TasksetTimingRecorder


def _write_build_timing(root, model, start, end, seconds, wall_mean_ms=None):
    path = root / "artifacts" / "chunk_cache" / model / f"int_{start}_{end}"
    path.mkdir(parents=True, exist_ok=True)
    data = {"build_fp32_wall_s": seconds}
    if wall_mean_ms is not None:
        data["wall_mean_ms_fp32"] = wall_mean_ms
    (path / "timing.json").write_text(json.dumps(data))


def test_recorder_profiles_each_configuration_once_and_reuses_chunks(tmp_path, monkeypatch):
    import src.integration.taskset_timing as timing_module

    monkeypatch.setattr(timing_module, "REPO", tmp_path)
    _write_build_timing(tmp_path, "alexnet", 0, 0, 1.0)
    _write_build_timing(tmp_path, "alexnet", 1, 2, 2.0)
    _write_build_timing(tmp_path, "alexnet", 1, 1, 3.0)
    _write_build_timing(tmp_path, "alexnet", 2, 2, 4.0)

    recorder = TasksetTimingRecorder(enabled=True)
    first = recorder.begin_configuration("alexnet", "fp32", [1, 0])
    assert first is not None
    assert recorder.begin_configuration("alexnet", "fp32", [1, 0]) is None
    recorder.finish_configuration(
        first,
        SimpleNamespace(
            success=True,
            did_profile=True,
            profile_wall_s=0.5,
            export_wall_s=0.0,
            build_wall_s=0.0,
            error=None,
        ),
    )

    second = recorder.begin_configuration("alexnet", "fp32", [1, 1])
    assert second is not None
    recorder.finish_configuration(
        second,
        SimpleNamespace(
            success=True,
            did_profile=True,
            profile_wall_s=0.7,
            export_wall_s=0.0,
            build_wall_s=0.0,
            error=None,
        ),
    )
    recorder.finalize_search(2.0, valid_config_found=True)

    assert recorder.unique_configurations_encountered == 2
    assert recorder.unique_configurations_profiled == 2
    assert recorder.unique_chunks_optimized == 4
    assert recorder.profiling_wall_s == 1.2
    assert recorder.estimated_optimize_wall_s == 10.0
    assert recorder.optimize_plus_profiling_wall_s == 11.2
    assert recorder.search_compute_wall_s == 0.8
    assert recorder.time_to_valid_config_s == 2.0
    assert recorder.missing_build_timing_count == 0


def test_recorder_reports_missing_build_metadata(tmp_path, monkeypatch):
    import src.integration.taskset_timing as timing_module

    monkeypatch.setattr(timing_module, "REPO", tmp_path)
    recorder = TasksetTimingRecorder(enabled=True)
    event = recorder.begin_configuration("resnet18", "fp32", [0])
    recorder.finish_configuration(
        event,
        SimpleNamespace(
            success=False,
            did_profile=False,
            profile_wall_s=0.0,
            export_wall_s=0.0,
            build_wall_s=0.0,
            error="missing engine",
        ),
    )
    recorder.finalize_search(0.1, valid_config_found=False)

    assert recorder.profile_failure_count == 1
    assert recorder.missing_build_timing_count == 1
    assert recorder.time_to_valid_config_s is None


def test_virtual_profile_uses_cached_wall_mean_and_warmup(tmp_path, monkeypatch):
    import src.integration.taskset_timing as timing_module

    monkeypatch.setattr(timing_module, "REPO", tmp_path)
    _write_build_timing(tmp_path, "alexnet", 0, 0, 1.0, wall_mean_ms=4.0)
    _write_build_timing(tmp_path, "alexnet", 1, 2, 2.0, wall_mean_ms=6.0)

    recorder = TasksetTimingRecorder(enabled=True, virtual_profile=True)
    event = recorder.begin_configuration(
        "alexnet", "fp32", [1, 0], warmup=20, iters=200
    )
    recorder.finalize_search(0.1, valid_config_found=True)

    assert event is not None
    assert event["wall_mean_sum_ms"] == 10.0
    assert event["virtual_profile_wall_s"] == 2.2
    assert recorder.profiling_wall_s == 2.2
    assert recorder.unique_configurations_profiled == 1
    assert recorder.virtual_profile_iterations == 220
    assert recorder.actual_pipeline_wall_s == 0.0
    assert recorder.search_compute_wall_s == 0.1
    assert recorder.estimated_search_wall_s == pytest.approx(2.3)
    assert recorder.missing_profile_timing_count == 0


def test_cache_aware_profile_estimates_hits_and_measures_misses(tmp_path, monkeypatch):
    import src.integration.taskset_timing as timing_module

    monkeypatch.setattr(timing_module, "REPO", tmp_path)
    _write_build_timing(tmp_path, "alexnet", 0, 0, 1.0, wall_mean_ms=4.0)

    recorder = TasksetTimingRecorder(enabled=True, cache_aware_profile=True)
    cached = recorder.begin_configuration("alexnet", "fp32", [], warmup=20, iters=200)
    assert cached is not None
    recorder.finish_configuration(
        cached,
        SimpleNamespace(
            success=True,
            cache_hit=True,
            profiled=True,
            profile_wall_s=99.0,
            export_wall_s=0.0,
            build_wall_s=0.0,
            error=None,
        ),
    )
    assert recorder.profiling_wall_s == pytest.approx(0.88)
    assert cached["profile_time_source"] == "virtual_wall_mean"

    missing = recorder.begin_configuration("resnet18", "fp32", [], warmup=20, iters=200)
    assert missing is not None
    recorder.finish_configuration(
        missing,
        SimpleNamespace(
            success=True,
            cache_hit=False,
            profiled=True,
            profile_wall_s=1.25,
            export_wall_s=0.5,
            build_wall_s=2.0,
            error=None,
        ),
    )
    assert recorder.profiling_wall_s == pytest.approx(2.13)
    assert recorder.actual_pipeline_wall_s == pytest.approx(3.75)
    assert missing["profile_time_source"] == "actual_cache_miss"
    assert recorder.requires_actual_profile is False
    assert recorder.virtual_cache_only is False
