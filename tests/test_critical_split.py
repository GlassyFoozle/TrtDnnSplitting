"""
test_critical_split.py — Verify that critical_split is importable and returns
correct chunk counts for each supported model without running TensorRT.
"""

from __future__ import annotations

import pytest


def test_critical_split_importable():
    from src.splitting.critical_split import make_critical_full_chunks, available_critical_full_models
    assert callable(make_critical_full_chunks)
    models = available_critical_full_models()
    assert "alexnet" in models
    assert "resnet18" in models
    assert "vgg19" in models


def test_dag_aligned_split_importable():
    """dag_aligned_split imports critical_split at module level — verify no ModuleNotFoundError."""
    from src.splitting.dag_aligned_split import (
        make_dag_aligned_chunks, available_dag_aligned_models
    )
    assert callable(make_dag_aligned_chunks)


@pytest.mark.parametrize("model_name,expected_chunks", [
    ("alexnet", 5),
    ("vgg19", 7),
    ("resnet18", 14),
])
def test_chunk_count(model_name: str, expected_chunks: int):
    """Each model returns the documented number of critical-full chunks."""
    import torchvision.models as tv_models
    import torch

    loaders = {
        "alexnet":  lambda: tv_models.alexnet(weights=None),
        "vgg19":    lambda: tv_models.vgg19(weights=None),
        "resnet18": lambda: tv_models.resnet18(weights=None),
    }

    from src.splitting.critical_split import make_critical_full_chunks

    with torch.no_grad():
        model = loaders[model_name]().eval()
        chunks = make_critical_full_chunks(model_name, model)

    assert len(chunks) == expected_chunks, (
        f"{model_name}: expected {expected_chunks} chunks, got {len(chunks)}"
    )


def test_unknown_model_raises():
    from src.splitting.critical_split import make_critical_full_chunks
    import torch.nn as nn

    with pytest.raises(KeyError, match="No critical_full chunker"):
        make_critical_full_chunks("unknown_model", nn.Linear(1, 1))


def test_chunk_shapes_alexnet():
    """Verify AlexNet chunk input/output shapes match documented values."""
    import torchvision.models as tv_models
    import torch
    from src.splitting.critical_split import make_critical_full_chunks

    expected_shapes = [
        ((1, 3, 224, 224),  (1, 64, 27, 27)),
        ((1, 64, 27, 27),   (1, 192, 13, 13)),
        ((1, 192, 13, 13),  (1, 256, 6, 6)),
        ((1, 256, 6, 6),    (1, 9216)),
        ((1, 9216),         (1, 1000)),
    ]

    with torch.no_grad():
        model = tv_models.alexnet(weights=None).eval()
        chunks = make_critical_full_chunks("alexnet", model)

    for i, (chunk, (exp_in, exp_out)) in enumerate(zip(chunks, expected_shapes)):
        module, in_shape, out_shape, desc, target = chunk
        assert tuple(in_shape) == exp_in, f"chunk {i} input shape mismatch"
        assert tuple(out_shape) == exp_out, f"chunk {i} output shape mismatch"
