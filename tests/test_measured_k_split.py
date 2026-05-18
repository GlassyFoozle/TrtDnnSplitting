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
        (0, 0, 0): [10.0],
        (1, 0, 0): [12.0, 12.0],
        (0, 1, 0): [20.0, 3.0],
        (0, 0, 1): [7.0, 7.5],
        (1, 1, 1): [5.0, 5.0, 5.0, 5.0],
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
    assert set(calls[:3]) == {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
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
        (0, 0, 0): [10.0],
        (1, 0, 0): [12.0, 12.0],
        (0, 1, 0): [20.0, 3.0],
        (0, 0, 1): [7.0, 7.5],
        (1, 1, 1): [5.0, 5.0, 5.0, 5.0],
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
    assert set(calls[:5]) == set(measured)
    assert calls[-1] == (0, 0, 1)

    calls.clear()
    dnn_task_2 = DNNBackedTask.from_dict(dnn_task.to_dict())
    seg_task_2 = dnn_task_to_seginftask(dnn_task_2)
    second = apply_k_chunks(dnn_task_2, seg_task_2, 0, 2, policy_name="all")

    assert second.success is True
    assert second.mask == [0, 0, 1]
    assert second.selected_chunk_times == [7.0, 7.5]
    assert calls == []


def test_apply_k_chunks_falls_back_to_eval_for_legacy_cache_entry(monkeypatch, tmp_path):
    import json

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
    seg_task = dnn_task_to_seginftask(dnn_task)
    key = "cache_test_model|fp32|p99|all|3|2|0,1,2"
    mask_applicator._K_SPLIT_CACHE_PATH.write_text(json.dumps({
        "version": 2,
        "entries": {
            key: {
                "model_name": "cache_test_model",
                "precision": "fp32",
                "wcet_metric": "p99",
                "policy_name": "all",
                "boundary_count": 3,
                "enabled_boundaries": [0, 1, 2],
                "k": 2,
                "mask": [0, 0, 1],
                "score": [7.5, 14.5, 0.5],
            }
        },
    }))
    calls = []

    def mock_evaluate_mask(model_name, mask, **kwargs):
        calls.append(tuple(mask))
        measured = {
            (0, 0, 0): [10.0],
            (0, 0, 1): [7.0, 7.5],
            (1, 1, 1): [5.0, 5.0, 5.0, 5.0],
        }
        return _fake_eval_result(mask, measured[tuple(mask)])

    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        mock_evaluate_mask,
    )

    result = apply_k_chunks(dnn_task, seg_task, 0, 2, policy_name="all")

    assert result.success is True
    assert max(result.selected_chunk_times) == 7.5
    assert sum(result.selected_chunk_times) == 14.5
    assert calls == [(0, 0, 0), (1, 1, 1), (0, 0, 1)]


def test_k_split_cache_monotonicizes_against_previous_k(monkeypatch, tmp_path):
    import json

    from src.integration import mask_applicator

    monkeypatch.setattr(
        mask_applicator,
        "_K_SPLIT_CACHE_PATH",
        tmp_path / "measured_k_split_cache.json",
    )
    prev_key = "toy|fp32|max|all|3|2|0,1,2"
    mask_applicator._K_SPLIT_CACHE_PATH.write_text(json.dumps({
        "version": 2,
        "entries": {
            prev_key: {
                "model_name": "toy",
                "precision": "fp32",
                "wcet_metric": "max",
                "policy_name": "all",
                "boundary_count": 3,
                "enabled_boundaries": [0, 1, 2],
                "k": 2,
                "mask": [1, 0, 0],
                "score": [5.0, 9.0, 1.0],
                "selected_chunk_times": [5.0, 4.0],
            }
        },
    }))

    mask_applicator._store_cached_k_split_mask(
        "toy|fp32|max|all|3|3|0,1,2",
        model_name="toy",
        precision="fp32",
        wcet_metric="max",
        policy_name="all",
        boundary_count=3,
        enabled_boundaries=[0, 1, 2],
        k=3,
        mask=[1, 1, 0],
        score=(6.0, 8.0, 5.0),
        selected_chunk_times=[6.0, 1.0, 1.0],
        variant_name="toy_k3",
        profile_result_path="",
    )

    data = json.loads(mask_applicator._K_SPLIT_CACHE_PATH.read_text())
    entry = data["entries"]["toy|fp32|max|all|3|3|0,1,2"]
    assert entry["selected_chunk_times"] == [5.0, 3.0, 1.0]
    assert entry["score"] == [5.0, 9.0, 4.0]
    assert entry["measured_score"] == [6.0, 8.0, 5.0]
    assert entry["monotonic_adjusted"] is True


def test_k_split_cache_re_monotonicizes_later_k_when_previous_arrives_late(monkeypatch, tmp_path):
    import json

    from src.integration import mask_applicator

    monkeypatch.setattr(
        mask_applicator,
        "_K_SPLIT_CACHE_PATH",
        tmp_path / "measured_k_split_cache.json",
    )

    mask_applicator._store_cached_k_split_mask(
        "toy|fp32|max|all|3|3|0,1,2",
        model_name="toy",
        precision="fp32",
        wcet_metric="max",
        policy_name="all",
        boundary_count=3,
        enabled_boundaries=[0, 1, 2],
        k=3,
        mask=[1, 1, 0],
        score=(4.0, 8.0, 2.0),
        selected_chunk_times=[4.0, 3.0, 1.0],
        variant_name="toy_k3",
        profile_result_path="",
    )
    mask_applicator._store_cached_k_split_mask(
        "toy|fp32|max|all|3|2|0,1,2",
        model_name="toy",
        precision="fp32",
        wcet_metric="max",
        policy_name="all",
        boundary_count=3,
        enabled_boundaries=[0, 1, 2],
        k=2,
        mask=[1, 0, 0],
        score=(5.0, 9.0, 1.0),
        selected_chunk_times=[5.0, 4.0],
        variant_name="toy_k2",
        profile_result_path="",
    )

    data = json.loads(mask_applicator._K_SPLIT_CACHE_PATH.read_text())
    later = data["entries"]["toy|fp32|max|all|3|3|0,1,2"]
    assert later["selected_chunk_times"] == [4.0, 4.0, 1.0]
    assert later["score"] == [4.0, 9.0, 3.0]
    assert later["measured_selected_chunk_times"] == [4.0, 3.0, 1.0]
    assert later["monotonic_adjusted"] is True


def test_k_split_cache_can_keep_raw_timings_when_monotonic_repair_disabled(monkeypatch, tmp_path):
    import json

    from src.integration import mask_applicator

    monkeypatch.setattr(
        mask_applicator,
        "_K_SPLIT_CACHE_PATH",
        tmp_path / "measured_k_split_cache.json",
    )
    prev_key = "toy|fp32|max|all|3|2|0,1,2"
    mask_applicator._K_SPLIT_CACHE_PATH.write_text(json.dumps({
        "version": 2,
        "entries": {
            prev_key: {
                "model_name": "toy",
                "precision": "fp32",
                "wcet_metric": "max",
                "policy_name": "all",
                "boundary_count": 3,
                "enabled_boundaries": [0, 1, 2],
                "k": 2,
                "mask": [1, 0, 0],
                "score": [5.0, 9.0, 1.0],
                "selected_chunk_times": [5.0, 4.0],
            }
        },
    }))

    mask_applicator._store_cached_k_split_mask(
        "toy|fp32|max|all|3|3|0,1,2",
        model_name="toy",
        precision="fp32",
        wcet_metric="max",
        policy_name="all",
        boundary_count=3,
        enabled_boundaries=[0, 1, 2],
        k=3,
        mask=[1, 1, 0],
        score=(6.0, 8.0, 5.0),
        selected_chunk_times=[6.0, 1.0, 1.0],
        variant_name="toy_k3",
        profile_result_path="",
        enable_monotonic_k_split_cache=False,
    )

    data = json.loads(mask_applicator._K_SPLIT_CACHE_PATH.read_text())
    entry = data["entries"]["toy|fp32|max|all|3|3|0,1,2"]
    assert entry["selected_chunk_times"] == [6.0, 1.0, 1.0]
    assert entry["score"] == [6.0, 8.0, 5.0]
    assert entry["measured_selected_chunk_times"] == [6.0, 1.0, 1.0]
    assert entry["monotonic_adjusted"] is False


def test_apply_k_chunks_reprofiles_inner_k_when_total_exceeds_full_split(monkeypatch, tmp_path):
    import json

    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration import mask_applicator
    from src.integration.mask_applicator import apply_k_chunks

    monkeypatch.setattr(
        mask_applicator,
        "_K_SPLIT_CACHE_PATH",
        tmp_path / "measured_k_split_cache.json",
    )
    task = DNNBackedTask(
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
        candidate_count=3,
        boundary_count=2,
        initial_mask=[0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="max",
        base_chunk_times_ms=[3.0, 3.0, 3.0],
        current_chunk_times_ms=[9.0],
    )
    calls = []

    def mock_evaluate_mask(model_name, mask, **kwargs):
        key = tuple(mask)
        calls.append((key, bool(kwargs.get("force"))))
        if key == (0, 0):
            return _fake_eval_result(mask, [6.0])
        if key == (1, 1):
            return _fake_eval_result(mask, [3.0, 3.0, 3.0])
        if kwargs.get("force"):
            return _fake_eval_result(mask, [4.0, 4.5])
        return _fake_eval_result(mask, [5.0, 5.0])

    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        mock_evaluate_mask,
    )

    result = apply_k_chunks(task, dnn_task_to_seginftask(task), 0, 2, policy_name="all")

    assert result.success is True
    assert result.selected_chunk_times == [4.0, 4.5]
    assert any(force for _, force in calls)
    data = json.loads(mask_applicator._K_SPLIT_CACHE_PATH.read_text())
    k2_entry = next(e for e in data["entries"].values() if e["k"] == 2)
    assert k2_entry["score"][1] == 8.5


def test_apply_k_chunks_skips_persistent_best_mask_cache_in_dry_run(monkeypatch):
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration import mask_applicator
    from src.integration.mask_applicator import apply_k_chunks

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
    seg_task = dnn_task_to_seginftask(dnn_task)

    monkeypatch.setattr(
        mask_applicator,
        "_load_cached_k_split_entry",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry_run must not read measured_k_split_cache")
        ),
    )

    result = apply_k_chunks(dnn_task, seg_task, 0, 2, policy_name="all", dry_run=True)

    assert result.success is False
    assert result.error == "dry_run does not provide measured per-chunk timing"


def test_apply_k_chunks_skips_persistent_best_mask_cache_when_forced(monkeypatch):
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration import mask_applicator
    from src.integration.mask_applicator import apply_k_chunks

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
    seg_task = dnn_task_to_seginftask(dnn_task)

    monkeypatch.setattr(
        mask_applicator,
        "_load_cached_k_split_entry",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("force=True must not read measured_k_split_cache")
        ),
    )
    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        lambda model_name, mask, **kwargs: _fake_eval_result(mask, [1.0, 1.0]),
    )
    monkeypatch.setattr(
        "src.optimization.config_evaluator.can_assemble_from_intervals",
        lambda *args, **kwargs: False,
    )

    result = apply_k_chunks(
        dnn_task, seg_task, 0, 2, policy_name="all", force=True
    )

    assert result.success is True


def test_measured_optimistic_ss_uses_apply_k_chunks_result(monkeypatch):
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration.dnn_algorithm_runner import _get_measured_optimistic_SS_R

    dnn_tasks = []
    sorted_tasks = []
    task_map = {}
    for idx, priority in enumerate((1, 2), start=1):
        dt = DNNBackedTask(
            task_name=f"t{idx}",
            model_name="cache_test_model",
            precision="fp32",
            period_ms=100.0,
            deadline_ms=100.0,
            priority=priority,
            cpu_id=0,
            cpu_pre_ms=0.0,
            cpu_post_ms=0.0,
            base_variant="dag_aligned_full",
            candidate_count=2,
            boundary_count=1,
            initial_mask=[0],
            selected_variant_name="test",
            selected_config_path="",
            profile_result_path="",
            wcet_metric="p99",
            base_chunk_times_ms=[9.0, 9.0],
            current_chunk_times_ms=[18.0],
        )
        st = dnn_task_to_seginftask(dt)
        dnn_tasks.append(dt)
        sorted_tasks.append(st)
        task_map[str(st.id)] = (dt, st)

    calls = []

    def mock_apply_k_chunks(dt, st, segment_idx, k, **kwargs):
        calls.append((dt.task_name, k, kwargs["policy_name"]))
        seg = st.inference_segment_list[segment_idx]
        seg.splitting_config = [1]
        seg.G_block_list = [2.0, 3.0]
        st.G_segment_list[segment_idx] = list(seg.G_block_list)
        st.G = sum(seg.G_block_list)
        st.max_G_block = 3.0
        return SimpleNamespace(success=True)

    monkeypatch.setattr(
        "src.integration.dnn_algorithm_runner.apply_k_chunks",
        mock_apply_k_chunks,
    )

    optimistic = _get_measured_optimistic_SS_R(
        sorted_tasks, task_map, {}, policy_name="all"
    )

    assert optimistic is not None
    assert calls == [("t1", 2, "all"), ("t2", 2, "all")]
    # R_0 = own G (18) + lower blocking. Measured max block gives 21;
    # legacy base chunks [9, 9] would have produced 27.
    assert optimistic[0] == 21.0


def test_apply_k_chunks_cache_miss_creates_cache_and_returns_cached_timing(monkeypatch, tmp_path):
    import json

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
        candidate_count=3,
        boundary_count=2,
        initial_mask=[0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="max",
        base_chunk_times_ms=[3.0, 3.0, 3.0],
        current_chunk_times_ms=[9.0],
    )
    measured = {
        (0, 0): [6.0],
        (1, 0): [5.0, 5.0],
        (0, 1): [4.0, 4.5],
        (1, 1): [3.0, 3.0, 3.0],
    }
    monkeypatch.setattr(
        "src.optimization.config_evaluator.evaluate_mask",
        lambda model_name, mask, **kwargs: _fake_eval_result(mask, measured[tuple(mask)]),
    )
    seg_task = dnn_task_to_seginftask(dnn_task)

    result = apply_k_chunks(dnn_task, seg_task, 0, 2, policy_name="all")

    assert result.success is True
    assert result.cache_hit is True
    assert result.selected_chunk_times == [4.0, 4.5]
    data = json.loads((tmp_path / "measured_k_split_cache.json").read_text())
    k2_entry = next(e for e in data["entries"].values() if e["k"] == 2)
    assert k2_entry["selected_chunk_times"] == [4.0, 4.5]
