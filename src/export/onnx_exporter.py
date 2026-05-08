"""
ONNX export utilities for full model and individual chunks.

All exports use:
  - opset 17 (last stable opset with good TRT 8.6 support)
  - fully static shapes (no dynamic_axes)
  - do_constant_folding=True for cleaner graphs
  - input_names=["input"], output_names=["output"]
    so TRT engines always use "input" / "output" as tensor names.
"""

import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import onnx


OPSET = 17
INPUT_NAME  = "input"
OUTPUT_NAME = "output"


def export_module(
    module: nn.Module,
    input_shape: Tuple[int, ...],
    out_path: str | Path,
    opset: int = OPSET,
    device: str = "cuda",
) -> None:
    """
    Export a single nn.Module to ONNX with a static input shape.

    Parameters
    ----------
    module      : eval-mode nn.Module
    input_shape : e.g. (1, 3, 224, 224)
    out_path    : destination .onnx file (parent dir must exist)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    module = module.to(device).eval()
    dummy  = torch.zeros(*input_shape, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            # No dynamic_axes: fully static for best TRT optimization
        )
    elapsed = time.perf_counter() - t0

    # Verify the exported graph
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    n_nodes  = len(onnx_model.graph.node)
    size_mb  = out_path.stat().st_size / 1e6
    print(f"  → {out_path.name}  nodes={n_nodes}  {size_mb:.1f} MB  ({elapsed:.2f}s)")


def get_output_shape(module: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Run a dummy forward to discover the output shape."""
    module.eval()
    with torch.no_grad():
        out = module(torch.zeros(*input_shape))
    return tuple(out.shape)
