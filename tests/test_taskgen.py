"""
test_taskgen.py — Unit tests for dnn_workload_generator and dnn_taskset_generator.
"""
import pytest


def test_workload_generator_g_ratio():
    """G_ratio field from YAML maps to correct CPU + GPU times."""
    from src.integration.dnn_workload_generator import generate_workload_from_yaml_params

    params = {
        "g_ratio": 0.8,
        "utilization": 0.5,
        "n_tasks": 1,
        "n_cpus": 1,
        "model": "alexnet",
        "max_block_count": 4,
        "period_range": [20.0, 20.0],
    }
    tasks = generate_workload_from_yaml_params(params, seed=0)
    assert len(tasks) == 1
    t = tasks[0]
    # G / (C + G) should be approximately g_ratio
    G = t.gpu_ms
    C = t.cpu_ms
    assert G > 0 and C >= 0
    ratio = G / (C + G)
    assert abs(ratio - 0.8) < 0.05, f"G_ratio={ratio:.3f}, expected ~0.8"


def test_generate_dnn_taskset_dry_run():
    """generate_dnn_taskset with dry_run=True produces tasks with base chunk times."""
    from src.integration.dnn_taskset_generator import generate_dnn_taskset

    tasks = generate_dnn_taskset(
        model_names=["alexnet"],
        n_tasks=2,
        g_ratio=0.7,
        utilization_per_task=0.4,
        policy="major_blocks",
        dry_run=True,
        seed=42,
    )
    assert len(tasks) == 2
    for t in tasks:
        assert len(t.base_chunk_times_ms) > 0
        assert len(t.initial_mask) == len(t.base_chunk_times_ms) - 1
        assert all(b in (0, 1) for b in t.initial_mask)


def test_taskset_period_utilization():
    """Period should satisfy: U = G / T where G = sum(base_chunk_times_ms)."""
    from src.integration.dnn_taskset_generator import generate_dnn_taskset

    U_target = 0.3
    tasks = generate_dnn_taskset(
        model_names=["resnet18"],
        n_tasks=1,
        g_ratio=0.6,
        utilization_per_task=U_target,
        policy="major_blocks",
        dry_run=True,
        seed=7,
    )
    t = tasks[0]
    G = sum(t.base_chunk_times_ms)
    T = t.period_ms
    U_actual = G / T
    assert abs(U_actual - U_target) < 0.05, f"U={U_actual:.4f}, expected ~{U_target}"
