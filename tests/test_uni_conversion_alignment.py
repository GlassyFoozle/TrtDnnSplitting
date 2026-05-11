from __future__ import annotations

from types import SimpleNamespace


def test_uni_measured_blocks_keep_cpu_pre_post_separate():
    from src.integration.dnn_algorithm_runner import _uni_g_blocks_from_measured_chunks

    dnn_task = SimpleNamespace(cpu_pre_ms=0.2, cpu_post_ms=0.3)

    assert _uni_g_blocks_from_measured_chunks(dnn_task, [1.1]) == [0.2, 1.1, 0.3]
    assert _uni_g_blocks_from_measured_chunks(dnn_task, [0.4, 0.5]) == [
        0.2,
        0.4,
        0.5,
        0.3,
    ]


def test_uni_measured_blocks_omit_zero_cpu_blocks_like_ss_to_uni():
    from src.integration.dnn_algorithm_runner import _uni_g_blocks_from_measured_chunks

    dnn_task = SimpleNamespace(cpu_pre_ms=0.0, cpu_post_ms=0.3)

    assert _uni_g_blocks_from_measured_chunks(dnn_task, [1.1, 0.0, 1.2]) == [
        1.1,
        0.0,
        1.2,
        0.3,
    ]


def test_uni_trt_mask_expansion_keeps_cpu_boundaries_fixed():
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration.dnn_algorithm_runner import (
        _patch_uni_measured_blocks,
        _uni_config_from_trt_mask,
    )
    from src.rta.analysis import convert_task_SS_to_UNI

    dnn_task = DNNBackedTask(
        task_name="t1",
        model_name="alexnet",
        precision="fp32",
        period_ms=10.0,
        deadline_ms=10.0,
        priority=1,
        cpu_id=0,
        cpu_pre_ms=0.2,
        cpu_post_ms=0.3,
        base_variant="dag_aligned_full",
        candidate_count=3,
        boundary_count=2,
        initial_mask=[0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="p99",
        base_chunk_times_ms=[1.0, 2.0, 3.0],
        current_chunk_times_ms=[6.0],
    )
    ss_task = dnn_task_to_seginftask(dnn_task)
    uni_task = convert_task_SS_to_UNI(ss_task)

    assert uni_task.inference_segment_list[0].G_block_list == [0.2, 6.0, 0.3]
    assert _uni_config_from_trt_mask(uni_task, [1, 0]) == [1, 1, 0, 1]

    _patch_uni_measured_blocks(dnn_task, uni_task, 0, [1, 0], [1.25, 4.75])

    seg = uni_task.inference_segment_list[0]
    assert seg.splitting_config == [1, 1, 0, 1]
    assert seg.G_block_list == [0.2, 1.25, 4.75, 0.3]
    assert uni_task.G_segment_list[0] == [0.2, 1.25, 4.75, 0.3]
    assert uni_task.G == 6.5
    assert uni_task.max_G_block == 4.75


def test_uni_to_ss_preserves_measured_gpu_chunks_and_split_capacity():
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration.dnn_algorithm_runner import _patch_uni_measured_blocks
    from src.rta.analysis import convert_task_SS_to_UNI, convert_task_UNI_to_SS

    dnn_task = DNNBackedTask(
        task_name="t1",
        model_name="alexnet",
        precision="fp32",
        period_ms=10.0,
        deadline_ms=10.0,
        priority=1,
        cpu_id=0,
        cpu_pre_ms=0.2,
        cpu_post_ms=0.3,
        base_variant="dag_aligned_full",
        candidate_count=3,
        boundary_count=2,
        initial_mask=[0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="p99",
        base_chunk_times_ms=[1.0, 2.0, 3.0],
        current_chunk_times_ms=[6.0],
    )
    uni_task = convert_task_SS_to_UNI(dnn_task_to_seginftask(dnn_task))
    _patch_uni_measured_blocks(dnn_task, uni_task, 0, [1, 0], [1.25, 4.75])

    ss_task = convert_task_UNI_to_SS(uni_task)
    seg = ss_task.inference_segment_list[0]

    assert ss_task.C_list == [0.2, 0.3]
    assert seg.base_block_list == [1.0, 2.0, 3.0]
    assert seg.splitting_config == [1, 0]
    assert seg.G_block_list == [1.25, 4.75]
    assert seg.size == 2
    assert seg.max_block_count == 3
    assert ss_task.G == 6.0
    assert ss_task.max_G_block == 4.75


def test_ss_to_uni_preserves_measured_k1_gpu_timing():
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.rta.analysis import convert_task_SS_to_UNI

    dnn_task = DNNBackedTask(
        task_name="t1",
        model_name="alexnet",
        precision="fp32",
        period_ms=10.0,
        deadline_ms=10.0,
        priority=1,
        cpu_id=0,
        cpu_pre_ms=0.2,
        cpu_post_ms=0.3,
        base_variant="dag_aligned_full",
        candidate_count=3,
        boundary_count=2,
        initial_mask=[0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="p99",
        base_chunk_times_ms=[0.4, 0.6, 0.8],
        current_chunk_times_ms=[1.1],
    )
    ss_task = dnn_task_to_seginftask(dnn_task)
    ss_task.inference_segment_list[0].G_block_list = [1.1]
    ss_task.G_segment_list[0] = [1.1]
    ss_task.G = 1.1
    ss_task.max_G_block = 1.1

    uni_task = convert_task_SS_to_UNI(ss_task)

    assert uni_task.inference_segment_list[0].G_block_list == [0.2, 1.1, 0.3]
    assert uni_task.G == 1.6
    assert uni_task.max_G_block == 1.1


def test_adapter_initializes_from_current_chunk_times_not_base_sum():
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask

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
        candidate_count=3,
        boundary_count=2,
        initial_mask=[0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="p99",
        base_chunk_times_ms=[1.0, 2.0, 3.0],
        current_chunk_times_ms=[4.5],
    )

    seg_task = dnn_task_to_seginftask(dnn_task)

    assert seg_task.inference_segment_list[0].G_block_list == [4.5]
    assert seg_task.G == 4.5


def test_measured_full_split_uni_probe_uses_measured_full_split(monkeypatch):
    from src.integration.dnn_task import DNNBackedTask
    from src.integration.dnnsplitting_adapter import dnn_task_to_seginftask
    from src.integration.dnn_algorithm_runner import _measured_full_split_uni_probe

    dnn_task = DNNBackedTask(
        task_name="t1",
        model_name="alexnet",
        precision="fp32",
        period_ms=10.0,
        deadline_ms=10.0,
        priority=1,
        cpu_id=0,
        cpu_pre_ms=0.2,
        cpu_post_ms=0.3,
        base_variant="dag_aligned_full",
        candidate_count=3,
        boundary_count=2,
        initial_mask=[0, 0],
        selected_variant_name="test",
        selected_config_path="",
        profile_result_path="",
        wcet_metric="p99",
        base_chunk_times_ms=[1.0, 2.0, 3.0],
        current_chunk_times_ms=[4.5],
    )
    ss_task = dnn_task_to_seginftask(dnn_task)

    def mock_apply_full_split_mask(dt, st, segment_idx, **kwargs):
        seg = st.inference_segment_list[segment_idx]
        seg.splitting_config = [1, 1]
        seg.G_block_list = [0.9, 1.1, 1.3]
        st.G_segment_list[segment_idx] = [0.9, 1.1, 1.3]
        st.G = 3.3
        st.max_G_block = 1.3

        class Result:
            success = True
            mask = [1, 1]
            selected_chunk_times = [0.9, 1.1, 1.3]

        return Result()

    monkeypatch.setattr(
        "src.integration.dnn_algorithm_runner.apply_full_split_mask",
        mock_apply_full_split_mask,
    )

    ok, uni_probe, _ = _measured_full_split_uni_probe(dnn_task, ss_task, {})

    assert ok is True
    assert uni_probe.inference_segment_list[0].G_block_list == [
        0.2, 0.9, 1.1, 1.3, 0.3
    ]
    assert ss_task.inference_segment_list[0].G_block_list == [4.5]
