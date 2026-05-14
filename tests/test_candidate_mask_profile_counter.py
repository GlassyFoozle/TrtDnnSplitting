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
    assert stats.k_split_candidate_mask_inference_runs == 10

    # Same masks need no new e2e profiles in the same taskset-algorithm run.
    stats.record_k_split_candidate_mask_profiles(
        "toy", "fp32", [[1, 0], [0, 1]], warmup=2, iters=3
    )
    assert stats.k_split_candidate_mask_profiles == 2
    assert stats.k_split_candidate_mask_inference_runs == 10

    # Full split introduces interval [1], so one new mask profile is needed.
    stats.record_k_split_candidate_mask_profiles(
        "toy", "fp32", [[1, 1]], warmup=2, iters=3
    )
    assert stats.k_split_candidate_mask_profiles == 3
    assert stats.k_split_candidate_mask_inference_runs == 15
