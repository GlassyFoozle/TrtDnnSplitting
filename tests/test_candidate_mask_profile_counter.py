from src.integration.dnn_algorithm_runner import ProfilingStats


def test_k_split_candidate_mask_profiles_use_per_run_interval_cache():
    stats = ProfilingStats()

    # N=3 base chunks, K=2 has two candidate masks:
    # [1, 0] -> intervals [0], [1, 2]
    # [0, 1] -> intervals [0, 1], [2]
    stats.record_k_split_candidate_mask_profiles(
        "toy", "fp32", [[1, 0], [0, 1]], warmup=2, iters=3
    )
    assert stats.k_split_candidate_mask_profiles == 2
    assert stats.k_split_candidate_chunk_profiles_with_reuse == 4
    assert stats.k_split_unique_model_masks == 2
    assert stats.k_split_unique_model_chunks == 4
    assert stats.k_split_unique_task_masks == 2
    assert stats.k_split_unique_task_chunks == 4
    assert stats.k_split_candidate_mask_inference_runs == 10

    # Same masks need no new e2e profiles in the same taskset-algorithm run.
    stats.record_k_split_candidate_mask_profiles(
        "toy", "fp32", [[1, 0], [0, 1]], warmup=2, iters=3
    )
    assert stats.k_split_candidate_mask_profiles == 2
    assert stats.k_split_candidate_chunk_profiles_with_reuse == 4
    assert stats.k_split_unique_model_masks == 2
    assert stats.k_split_unique_model_chunks == 4
    assert stats.k_split_candidate_mask_inference_runs == 10

    # Full split introduces interval [1], so one new mask profile is needed.
    stats.record_k_split_candidate_mask_profiles(
        "toy", "fp32", [[1, 1]], warmup=2, iters=3
    )
    assert stats.k_split_candidate_mask_profiles == 3
    assert stats.k_split_candidate_chunk_profiles_with_reuse == 5
    assert stats.k_split_unique_model_masks == 3
    assert stats.k_split_unique_model_chunks == 5
    assert stats.k_split_candidate_mask_inference_runs == 15


def test_k_split_unique_candidate_types_are_counted_by_model_and_task():
    stats = ProfilingStats()

    stats.record_k_split_candidate_mask_profiles(
        "toy", "fp32", [[1, 0]], task_name="tau1_toy"
    )
    stats.record_k_split_candidate_mask_profiles(
        "toy", "fp32", [[1, 0]], task_name="tau2_toy"
    )
    assert stats.k_split_unique_model_masks == 1
    assert stats.k_split_unique_model_chunks == 2
    assert stats.k_split_unique_task_masks == 2
    assert stats.k_split_unique_task_chunks == 4
