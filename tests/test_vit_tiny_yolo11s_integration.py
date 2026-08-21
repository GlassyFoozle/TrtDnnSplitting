"""Integration checks for the two additional experiment models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "model_name,input_shape,output_shape,n_chunks,param_count",
    [
        ("vit_tiny", (1, 3, 224, 224), (1, 1000), 14, 5_717_416),
        ("yolo11s", (1, 3, 640, 640), (1, 84, 8400), 24, 9_458_736),
    ],
)
def test_new_model_chunk_chain_matches_full_model(
    model_name: str,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    n_chunks: int,
    param_count: int,
) -> None:
    import torch

    from src.models.registry import build_model, get_model_info
    from src.splitting.dag_aligned_split import make_dag_aligned_chunks

    assert get_model_info(model_name).input_shape == input_shape
    model = build_model(model_name).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == param_count
    chunks = make_dag_aligned_chunks(model_name, model)
    assert len(chunks) == n_chunks

    x = torch.zeros(*input_shape)
    with torch.no_grad():
        reference = model(x)
        current = x
        for chunk in chunks:
            current = chunk.module(current)

    assert tuple(reference.shape) == output_shape
    assert tuple(current.shape) == output_shape
    torch.testing.assert_close(current, reference, rtol=0.0, atol=0.0)


def test_new_model_trt_fusion_safe_policies() -> None:
    from src.integration.split_point_policy import get_enabled_boundaries
    from src.splitting.selective_split import compute_merge_groups

    assert get_enabled_boundaries(
        "vit_tiny", "trt_fusion_safe", 13
    ) == list(range(13))
    yolo_safe = get_enabled_boundaries("yolo11s", "trt_fusion_safe", 23)
    assert yolo_safe == [
        *range(11), 13, 16, 19, 22
    ]
    mask = [1 if boundary in yolo_safe else 0 for boundary in range(23)]
    assert compute_merge_groups(mask) == [
        *[[i] for i in range(11)],
        [11, 12, 13],
        [14, 15, 16],
        [17, 18, 19],
        [20, 21, 22],
        [23],
    ]


@pytest.mark.parametrize(
    "model_name,n_chunks",
    [("vit_tiny", 14), ("yolo11s", 24)],
)
def test_new_model_versioned_base_config(model_name: str, n_chunks: int) -> None:
    path = REPO / "artifacts" / "split_configs" / model_name / "dag_aligned_full.json"
    config = json.loads(path.read_text())
    assert config["model"] == model_name
    assert config["n_chunks"] == n_chunks
    assert len(config["chunks"]) == n_chunks
    assert config["dag_aligned_summary"]["candidate_count"] == n_chunks
