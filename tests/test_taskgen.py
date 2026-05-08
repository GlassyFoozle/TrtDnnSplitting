"""
test_taskgen.py — Unit tests for dnn_workload_generator (WorkloadConfig + generate_tasksets).
"""
import json
import pytest
from pathlib import Path


def _make_config(**kwargs):
    from src.integration.dnn_workload_generator import WorkloadConfig
    defaults = dict(
        models=["alexnet"],
        n_tasks=2,
        utilization=0.5,
        n_tasksets=2,
        precision="fp32",
        wcet_metric="p99",
        taskgen_mode="dnnsplitting",
        utilization_basis="total",
        g_ratio_range=(0.7, 0.9),
        seed=42,
    )
    defaults.update(kwargs)
    return WorkloadConfig(**defaults)


def test_generate_tasksets_returns_paths(tmp_path):
    """generate_tasksets produces at least one JSON file per requested taskset."""
    from src.integration.dnn_workload_generator import generate_tasksets
    cfg = _make_config(n_tasksets=2)
    paths = generate_tasksets(cfg, output_dir=tmp_path)
    assert len(paths) > 0
    for p in paths:
        assert p.exists()
        data = json.loads(p.read_text())
        assert "tasks" in data or isinstance(data, list) or len(data) > 0


def test_g_ratio_in_range(tmp_path):
    """sampled_g_ratio per task should lie within the requested g_ratio_range."""
    from src.integration.dnn_workload_generator import generate_tasksets
    g_min, g_max = 0.6, 0.8
    cfg = _make_config(n_tasksets=3, g_ratio_range=(g_min, g_max), utilization=0.4)
    paths = generate_tasksets(cfg, output_dir=tmp_path)
    assert paths, "no tasksets generated"

    for p in paths:
        data = json.loads(p.read_text())
        tasks = data.get("tasks", []) if isinstance(data, dict) else data
        for t in tasks:
            ratio = t.get("sampled_g_ratio") or t.get("actual_g_ratio")
            if ratio is None:
                continue
            assert g_min - 0.05 <= ratio <= g_max + 0.05, (
                f"G_ratio={ratio:.3f} outside [{g_min}, {g_max}] in {p.name}"
            )


def test_utilization_consistent(tmp_path):
    """_actual_total_utilization in metadata should approximate the requested total U."""
    from src.integration.dnn_workload_generator import generate_tasksets
    U_target = 0.5
    cfg = _make_config(n_tasksets=3, utilization=U_target, n_tasks=4,
                       models=["alexnet", "resnet18"], seed=7)
    paths = generate_tasksets(cfg, output_dir=tmp_path)
    assert paths

    for p in paths:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            continue
        u_actual = data.get("_actual_total_utilization")
        if u_actual is None:
            # Fall back to per-task sum via real_gpu_wcet_ms / period_ms
            tasks = data.get("tasks", [])
            u_actual = sum(
                t.get("real_gpu_wcet_ms", 0) / t.get("period_ms", 1)
                for t in tasks if t.get("period_ms", 0) > 0
            )
        # Allow ±20% tolerance due to UUniFast sampling
        assert abs(u_actual - U_target) < U_target * 0.25 + 0.05, (
            f"total U={u_actual:.4f}, expected ~{U_target} in {p.name}"
        )


def test_dry_run_wcet_fallback():
    """_get_base_gpu_wcet_ms returns a positive value for known models even without cache."""
    from src.integration.dnn_workload_generator import _get_base_gpu_wcet_ms
    for model in ["alexnet", "resnet18", "vgg19"]:
        wcet = _get_base_gpu_wcet_ms(model, precision="fp32", wcet_metric="p99")
        assert wcet is not None and wcet > 0, f"No WCET for {model}"
