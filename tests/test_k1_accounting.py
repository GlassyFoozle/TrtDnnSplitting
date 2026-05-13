"""Tests for K=1 measured-mask accounting correctness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _make_taskset_json() -> Path:
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


def _load_first_task():
    from src.integration.dnn_taskset_loader import load_dnn_taskset
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask

    tasks = load_dnn_taskset(_make_taskset_json(), allow_equal_wcet_fallback=True)
    dt = tasks[0]
    return dt, dnn_task_to_seginftask(dt)


def _load_full_taskset(n_tasks: int = 4):
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
    dnn_tasks = generate_dnn_taskset(
        Path(fd.name), overlay_evaluations=False, allow_equal_wcet_fallback=True
    )
    task_set = build_task_set_dict(dnn_tasks)
    sorted_list = sort_task_set(task_set)
    task_map = {}
    for dt in dnn_tasks:
        for st in sorted_list:
            if str(st.id) == str(dt.task_name):
                task_map[str(dt.task_name)] = (dt, st)
                break
    return sorted_list, task_map


def _fake_eval_result(*, chunk_times, cache_hit=False, profiled=True, error=None):
    return SimpleNamespace(
        error=error,
        ok=lambda: error is None,
        cache_hit=cache_hit,
        exported=False,
        built=False,
        profiled=profiled,
        interval_cache_hits=0,
        interval_cache_misses=0,
        interval_onnx_cache_hits=0,
        interval_onnx_cache_misses=0,
        interval_engine_cache_hits=0,
        interval_engine_cache_misses=0,
        interval_engine_build_wall_s=0.0,
        export_wall_s=0.0,
        build_wall_s=0.0,
        profile_wall_s=0.0,
        estimated_cold_total_s=None,
        per_chunk_gpu_max_ms=list(chunk_times),
        per_chunk_gpu_p99_ms=list(chunk_times),
        per_chunk_gpu_mean_ms=list(chunk_times),
        variant_name="fake",
        result_json_path="",
        config_path="",
    )


def test_k1_mask_calls_evaluate_mask_and_applies_measured_timing(monkeypatch):
    from src.integration.mask_applicator import evaluate_and_apply_mask

    dt, seg_task = _load_first_task()
    mask = [0] * (dt.candidate_count - 1)
    calls = []

    def mock_evaluate_mask(model_name, mask, **kwargs):
        calls.append((model_name, list(mask), kwargs))
        return _fake_eval_result(chunk_times=[3.5], cache_hit=False)

    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        mock_evaluate_mask,
    )

    result = evaluate_and_apply_mask(dt, seg_task, mask, dry_run=False)

    assert calls
    assert result.success is True
    assert result.is_k1_baseline is False
    assert result.k_chunks == 1
    assert result.selected_chunk_times == [3.5]
    assert seg_task.inference_segment_list[0].G_block_list == [3.5]


def test_k1_mask_fails_without_measured_timing(monkeypatch):
    from src.integration.mask_applicator import evaluate_and_apply_mask

    dt, seg_task = _load_first_task()
    mask = [0] * (dt.candidate_count - 1)

    def mock_evaluate_mask(model_name, mask, **kwargs):
        return _fake_eval_result(chunk_times=[])

    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        mock_evaluate_mask,
    )

    result = evaluate_and_apply_mask(dt, seg_task, mask, dry_run=False)

    assert result.success is False
    assert result.is_k1_baseline is False
    assert "Measured per-chunk GPU timing unavailable" in (result.error or "")


def test_dry_run_does_not_apply_base_sum_estimate():
    from src.integration.mask_applicator import evaluate_and_apply_mask

    dt, seg_task = _load_first_task()
    before = list(seg_task.inference_segment_list[0].G_block_list)
    mask = [0] * (dt.candidate_count - 1)

    result = evaluate_and_apply_mask(dt, seg_task, mask, dry_run=True)

    assert result.success is False
    assert result.dry_run is True
    assert seg_task.inference_segment_list[0].G_block_list == before


def test_k1_real_profile_counts_as_real_profile():
    from src.integration.dnn_algorithm_runner import ProfilingStats
    from src.integration.mask_applicator import MaskApplicationResult

    stats = ProfilingStats()
    stats.update(MaskApplicationResult(
        success=True, mask=[0, 0], k_chunks=1,
        is_k1_baseline=False, dry_run=False, cache_hit=False,
    ))

    assert stats.baseline_k1_hits == 0
    assert stats.real_profiles == 1
    assert stats.cache_hits == 0


def test_k1_cache_hit_counts_as_cache_hit():
    from src.integration.dnn_algorithm_runner import ProfilingStats
    from src.integration.mask_applicator import MaskApplicationResult

    stats = ProfilingStats()
    stats.update(MaskApplicationResult(
        success=True, mask=[0, 0], k_chunks=1,
        is_k1_baseline=False, dry_run=False, cache_hit=True,
    ))

    assert stats.baseline_k1_hits == 0
    assert stats.real_profiles == 0
    assert stats.cache_hits == 1


def test_ss_single_counts_measured_k1_profiles(monkeypatch):
    from src.integration.dnn_algorithm_runner import (
        DNNAlgorithmResult,
        _run_ss_single,
    )
    from src.integration.mask_applicator import MaskApplicationResult

    sorted_list, task_map = _load_full_taskset(n_tasks=4)

    def mock_apply_no_split(dt, st, seg_idx=0, **kwargs):
        r = MaskApplicationResult(
            success=True,
            mask=[0] * (len(st.inference_segment_list[0].base_block_list) - 1),
            k_chunks=1,
            is_k1_baseline=False,
            dry_run=False,
            cache_hit=False,
            selected_chunk_times=[1.5],
            max_block=1.5,
            total_gpu=1.5,
        )
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
        wcet_metric="p99", dry_run=False,
    )

    _run_ss_single(sorted_list, task_map, result, {"dry_run": False})

    assert result.stats.baseline_k1_hits == 0
    assert result.stats.real_profiles == 4
