from __future__ import annotations

from types import SimpleNamespace


def _fake_eval_result(mask, chunk_times):
    return SimpleNamespace(
        error=None,
        ok=lambda: True,
        cache_hit=False,
        exported=False,
        built=False,
        profiled=True,
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
        variant_name="mask_" + "".join(str(b) for b in mask),
        result_json_path="",
        config_path="",
    )


def test_apply_k_chunks_profiles_all_k_masks_and_uses_measured_best(monkeypatch):
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration.mask_applicator import apply_k_chunks

    dnn_task = DNNBackedTask(
        task_name="t1",
        model_name="unknown_model_for_all_policy",
        precision="fp32",
        period_ms=10.0,
        deadline_ms=10.0,
        priority=1,
        cpu_id=0,
        cpu_pre_ms=0.0,
        cpu_post_ms=0.0,
        base_variant="dag_aligned_full",
        candidate_count=4,
        boundary_count=3,
        initial_mask=[0, 0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="p99",
        base_chunk_times_ms=[10.0, 1.0, 1.0, 10.0],
        current_chunk_times_ms=[22.0],
    )
    seg_task = dnn_task_to_seginftask(dnn_task)

    measured = {
        (1, 0, 0): [12.0, 12.0],
        (0, 1, 0): [20.0, 3.0],
        (0, 0, 1): [7.0, 7.5],
    }
    calls = []

    def mock_evaluate_mask(model_name, mask, **kwargs):
        calls.append(tuple(mask))
        return _fake_eval_result(mask, measured[tuple(mask)])

    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        mock_evaluate_mask,
    )

    result = apply_k_chunks(
        dnn_task, seg_task, 0, 2, policy_name="all", use_k_split_cache=False
    )

    assert result.success is True
    assert set(calls[:3]) == set(measured)
    assert calls[-1] == (0, 0, 1)
    assert result.mask == [0, 0, 1]
    assert result.selected_chunk_times == [7.0, 7.5]
    assert seg_task.inference_segment_list[0].G_block_list == [7.0, 7.5]


def test_apply_k_chunks_respects_major_blocks_policy(monkeypatch):
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration.mask_applicator import apply_k_chunks

    dnn_task = DNNBackedTask(
        task_name="t1",
        model_name="alexnet",
        precision="fp32",
        period_ms=10.0,
        deadline_ms=10.0,
        priority=1,
        cpu_id=0,
        cpu_pre_ms=0.0,
        cpu_post_ms=0.0,
        base_variant="dag_aligned_full",
        candidate_count=22,
        boundary_count=21,
        initial_mask=[0] * 21,
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="p99",
        base_chunk_times_ms=[1.0] * 22,
        current_chunk_times_ms=[22.0],
    )
    seg_task = dnn_task_to_seginftask(dnn_task)
    calls = []

    def mock_evaluate_mask(model_name, mask, **kwargs):
        calls.append(list(mask))
        return _fake_eval_result(mask, [1.0, 1.0])

    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        mock_evaluate_mask,
    )
    monkeypatch.setattr(
        "src.optimization.config_evaluator.can_assemble_from_intervals",
        lambda *args, **kwargs: False,
    )

    result = apply_k_chunks(
        dnn_task, seg_task, 0, 2, policy_name="major_blocks", use_k_split_cache=False
    )

    assert result.success is True
    evaluated_boundaries = {mask.index(1) for mask in calls[:-1]}
    assert evaluated_boundaries == {2, 5, 7, 9, 12, 14, 17, 20}


def test_apply_k_chunks_reuses_persistent_best_mask_cache(monkeypatch, tmp_path):
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration import mask_applicator
    from src.integration.mask_applicator import apply_k_chunks

    monkeypatch.setattr(
        mask_applicator,
        "_K_SPLIT_CACHE_PATH",
        tmp_path / "measured_k_split_cache.json",
    )

    dnn_task = DNNBackedTask(
        task_name="t1",
        model_name="cache_test_model",
        precision="fp32",
        period_ms=10.0,
        deadline_ms=10.0,
        priority=1,
        cpu_id=0,
        cpu_pre_ms=0.0,
        cpu_post_ms=0.0,
        base_variant="dag_aligned_full",
        candidate_count=4,
        boundary_count=3,
        initial_mask=[0, 0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="p99",
        base_chunk_times_ms=[10.0, 1.0, 1.0, 10.0],
        current_chunk_times_ms=[22.0],
    )
    measured = {
        (1, 0, 0): [12.0, 12.0],
        (0, 1, 0): [20.0, 3.0],
        (0, 0, 1): [7.0, 7.5],
    }
    calls = []

    def mock_evaluate_mask(model_name, mask, **kwargs):
        calls.append(tuple(mask))
        return _fake_eval_result(mask, measured[tuple(mask)])

    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        mock_evaluate_mask,
    )

    seg_task = dnn_task_to_seginftask(dnn_task)
    first = apply_k_chunks(dnn_task, seg_task, 0, 2, policy_name="all")

    assert first.success is True
    assert set(calls[:3]) == set(measured)
    assert calls[-1] == (0, 0, 1)

    calls.clear()
    dnn_task_2 = DNNBackedTask.from_dict(dnn_task.to_dict())
    seg_task_2 = dnn_task_to_seginftask(dnn_task_2)
    second = apply_k_chunks(dnn_task_2, seg_task_2, 0, 2, policy_name="all")

    assert second.success is True
    assert second.mask == [0, 0, 1]
    assert calls == [(0, 0, 1)]
