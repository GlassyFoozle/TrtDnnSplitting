"""
test_cache_validity.py — Unit tests for the Fix-4 is_mask_cached() improvements.
"""
import json
import pytest


@pytest.fixture
def tmp_eval_dir(tmp_path, monkeypatch):
    """Redirect REPO so eval JSONs land under tmp_path."""
    import src.optimization.config_evaluator as ce
    monkeypatch.setattr(ce, "REPO", tmp_path)
    return tmp_path


def _write_eval_json(tmp_path, model, variant, precision, data):
    p = tmp_path / "results" / "evaluations" / model
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{variant}_{precision}.json").write_text(json.dumps(data))


def test_cache_miss_no_file(tmp_eval_dir):
    from src.optimization.config_evaluator import is_mask_cached
    assert not is_mask_cached("alexnet", [1, 0, 1], "fp32")


def test_cache_miss_error_field(tmp_eval_dir):
    from src.optimization.config_evaluator import is_mask_cached, mask_to_variant_name
    mask = [1, 0, 1]
    variant = mask_to_variant_name("alexnet", mask)
    _write_eval_json(tmp_eval_dir, "alexnet", variant, "fp32", {
        "error": "ONNX export failed",
        "per_chunk_gpu_mean_ms": None,
        "per_chunk_gpu_max_ms": None,
    })
    assert not is_mask_cached("alexnet", mask, "fp32")


def test_cache_miss_wrong_chunk_count(tmp_eval_dir):
    from src.optimization.config_evaluator import is_mask_cached, mask_to_variant_name
    mask = [1, 0, 1]  # 3 splits → 2 active boundaries → 3 chunks
    variant = mask_to_variant_name("alexnet", mask)
    _write_eval_json(tmp_eval_dir, "alexnet", variant, "fp32", {
        "error": None,
        "per_chunk_gpu_max_ms": [1.0, 2.0],  # wrong: should be 3 chunks
    })
    assert not is_mask_cached("alexnet", mask, "fp32")


def test_cache_hit_valid(tmp_eval_dir):
    from src.optimization.config_evaluator import is_mask_cached, mask_to_variant_name
    mask = [1, 0, 1]
    variant = mask_to_variant_name("alexnet", mask)
    expected_chunks = sum(mask) + 1  # = 3
    _write_eval_json(tmp_eval_dir, "alexnet", variant, "fp32", {
        "error": None,
        "per_chunk_gpu_mean_ms": [1.0, 2.0, 3.0],
        "per_chunk_gpu_p99_ms": [1.1, 2.1, 3.1],
        "per_chunk_gpu_max_ms": [1.2, 2.2, 3.2],
    })
    assert is_mask_cached("alexnet", mask, "fp32")


def test_cache_miss_corrupt_json(tmp_eval_dir):
    from src.optimization.config_evaluator import is_mask_cached, mask_to_variant_name
    mask = [1, 1]
    variant = mask_to_variant_name("alexnet", mask)
    p = tmp_eval_dir / "results" / "evaluations" / "alexnet"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{variant}_fp32.json").write_text("{ this is not valid json }")
    assert not is_mask_cached("alexnet", mask, "fp32")
