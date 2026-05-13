"""
Experimental full-DAG-aligned split definitions.

For pure sequential torchvision models this variant materializes one chunk per
full-FX critical compute node wherever that node has a simple single-input,
single-output module/function representation.  This intentionally creates many
small chunks; the goal is maximum materializable granularity for later split
optimization, not low dispatch overhead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Tuple

import torch
import torch.nn as nn

from src.splitting.critical_split import make_critical_full_chunks


Shape = Tuple[int, ...]


@dataclass
class DagAlignedChunkSpec:
    module: nn.Module
    input_shape: Shape
    output_shape: Shape
    description: str
    chunk_id: int
    chunk_name: str
    source_graph: str
    covered_fx_nodes: List[str] = field(default_factory=list)
    covered_module_targets: List[str] = field(default_factory=list)
    source_fx_node_name: str = ""
    source_module_target: str = ""
    op_type: str = ""
    is_full_dag_critical_candidate: bool = True
    materialization_status: str = "materialized"
    boundary_reason: str = ""
    starts_at_critical_candidate: bool = True
    ends_at_critical_candidate: bool = True
    notes: str = ""


class _SequentialSlice(nn.Module):
    def __init__(self, layers: Iterable[nn.Module]):
        super().__init__()
        self.block = nn.Sequential(*list(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _SingleModule(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class _Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, 1)


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ViTPatchClassPos(nn.Module):
    """ViT image patch projection, class token prepend, positional add, dropout."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.conv_proj = model.conv_proj
        self.class_token = model.class_token
        self.pos_embedding = model.encoder.pos_embedding
        self.dropout = model.encoder.dropout
        self.image_size = model.image_size
        self.patch_size = model.patch_size
        self.hidden_dim = model.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        torch._assert(h == self.image_size, "Wrong image height")
        torch._assert(w == self.image_size, "Wrong image width")
        n_h = h // self.patch_size
        n_w = w // self.patch_size
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.pos_embedding
        return self.dropout(x)


class _ViTFinalNormHead(nn.Module):
    """ViT final encoder norm, class-token select, classifier head."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.ln = model.encoder.ln
        self.heads = model.heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = x[:, 0]
        return self.heads(x)


def _infer_output_shape(module: nn.Module, input_shape: Shape) -> Shape:
    module.eval()
    device, dtype = _module_device_dtype(module)
    with torch.no_grad():
        out = module(torch.zeros(*input_shape, device=device, dtype=dtype))
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Chunk returned non-tensor output: {type(out)!r}")
    return tuple(out.shape)


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    device = torch.device("cpu")
    dtype = torch.float32
    try:
        p = next(module.parameters())
        device = p.device
        dtype = p.dtype if p.is_floating_point() else torch.float32
    except StopIteration:
        try:
            b = next(module.buffers())
            device = b.device
            dtype = b.dtype if b.is_floating_point() else torch.float32
        except StopIteration:
            pass
    return device, dtype


def _make_spec(
    chunk_id: int,
    chunk_name: str,
    module: nn.Module,
    input_shape: Shape,
    description: str,
    covered_module_targets: List[str],
    covered_fx_nodes: List[str] | None = None,
    source_fx_node_name: str = "",
    source_module_target: str = "",
    op_type: str = "",
    output_shape: Shape | None = None,
    boundary_reason: str = "sequential module-DAG boundary",
    source_graph: str = "full_fx_dag",
    materialization_status: str = "materialized",
    notes: str = "",
) -> DagAlignedChunkSpec:
    module = module.eval()
    output_shape = output_shape or _infer_output_shape(module, input_shape)
    return DagAlignedChunkSpec(
        module=module,
        input_shape=input_shape,
        output_shape=output_shape,
        description=description,
        chunk_id=chunk_id,
        chunk_name=chunk_name,
        source_graph=source_graph,
        covered_fx_nodes=covered_fx_nodes or _fx_names_for_targets(covered_module_targets),
        covered_module_targets=covered_module_targets,
        source_fx_node_name=source_fx_node_name or (covered_fx_nodes or _fx_names_for_targets(covered_module_targets))[0],
        source_module_target=source_module_target or (covered_module_targets[0] if covered_module_targets else ""),
        op_type=op_type,
        is_full_dag_critical_candidate=True,
        materialization_status=materialization_status,
        boundary_reason=boundary_reason,
        notes=notes,
    )


def _fx_names_for_targets(targets: Iterable[str]) -> List[str]:
    names: List[str] = []
    for target in targets:
        if target == "flatten":
            names.append("flatten")
        else:
            names.append(target.replace(".", "_"))
    return names


def _module_op_name(module: nn.Module) -> str:
    return type(module).__name__


def _execute_chain_specs(
    chunks: List[tuple[str, nn.Module, str, List[str], List[str], str, str, str]],
    input_shape: Shape,
) -> List[DagAlignedChunkSpec]:
    specs: List[DagAlignedChunkSpec] = []
    first_device, first_dtype = _module_device_dtype(chunks[0][1])
    cur = torch.zeros(*input_shape, device=first_device, dtype=first_dtype)
    for idx, (name, module, desc, fx_nodes, targets, op_type, reason, notes) in enumerate(chunks):
        module = module.eval()
        device, dtype = _module_device_dtype(module)
        if cur.device != device or cur.dtype != dtype:
            cur = cur.to(device=device, dtype=dtype)
        with torch.no_grad():
            out = module(cur)
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Chunk {name} returned non-tensor output: {type(out)!r}")
        spec = _make_spec(
            chunk_id=idx,
            chunk_name=name,
            module=module,
            input_shape=tuple(cur.shape),
            description=desc,
            covered_fx_nodes=fx_nodes,
            covered_module_targets=targets,
            source_fx_node_name=fx_nodes[0] if fx_nodes else "",
            source_module_target=targets[0] if targets else "",
            op_type=op_type,
            output_shape=tuple(out.shape),
            boundary_reason=reason,
            notes=notes,
        )
        specs.append(spec)
        cur = out
    return specs


def _alexnet_dag_aligned(model: nn.Module) -> List[DagAlignedChunkSpec]:
    features = list(model.features.children())
    classifier = list(model.classifier.children())
    chunks: List[tuple[str, nn.Module, str, List[str], List[str], str, str, str]] = []

    for i, layer in enumerate(features):
        target = f"features.{i}"
        fx = f"features_{i}"
        op = _module_op_name(layer)
        chunks.append((
            fx,
            _SingleModule(layer),
            f"{target}/{op}",
            [fx],
            [target],
            op,
            "one full-FX critical call_module node in the sequential chain",
            "",
        ))

    chunks.append((
        "avgpool",
        _SingleModule(model.avgpool),
        "avgpool/AdaptiveAvgPool2d",
        ["avgpool"],
        ["avgpool"],
        _module_op_name(model.avgpool),
        "one full-FX critical call_module node in the sequential chain",
        "",
    ))
    chunks.append((
        "flatten",
        _Flatten(),
        "flatten/call_function",
        ["flatten"],
        ["flatten"],
        "call_function.flatten",
        "one full-FX critical call_function node materialized as an explicit wrapper",
        "No parameters; exported as a reshape/flatten-only ONNX graph.",
    ))

    for i, layer in enumerate(classifier):
        target = f"classifier.{i}"
        fx = f"classifier_{i}"
        op = _module_op_name(layer)
        notes = ""
        if isinstance(layer, nn.Dropout):
            notes = "Dropout is identity in eval mode; ONNX export may emit an identity/empty graph."
        chunks.append((
            fx,
            _SingleModule(layer),
            f"{target}/{op}",
            [fx],
            [target],
            op,
            "one full-FX critical call_module node in the sequential chain",
            notes,
        ))

    return _execute_chain_specs(chunks, (1, 3, 224, 224))


def _vgg19_dag_aligned(model: nn.Module) -> List[DagAlignedChunkSpec]:
    features = list(model.features.children())
    classifier = list(model.classifier.children())

    chunks: List[tuple[str, nn.Module, str, List[str], List[str], str, str, str]] = []

    for i, layer in enumerate(features):
        target = f"features.{i}"
        fx = f"features_{i}"
        op = _module_op_name(layer)
        chunks.append((
            fx,
            _SingleModule(layer),
            f"{target}/{op}",
            [fx],
            [target],
            op,
            "one full-FX critical call_module node in the sequential chain",
            "",
        ))

    chunks.append((
        "avgpool",
        _SingleModule(model.avgpool),
        "avgpool/AdaptiveAvgPool2d",
        ["avgpool"],
        ["avgpool"],
        _module_op_name(model.avgpool),
        "one full-FX critical call_module node in the sequential chain",
        "",
    ))
    chunks.append((
        "flatten",
        _Flatten(),
        "flatten/call_function",
        ["flatten"],
        ["flatten"],
        "call_function.flatten",
        "one full-FX critical call_function node materialized as an explicit wrapper",
        "No parameters; exported as a reshape/flatten-only ONNX graph.",
    ))

    for i, layer in enumerate(classifier):
        target = f"classifier.{i}"
        fx = f"classifier_{i}"
        op = _module_op_name(layer)
        notes = ""
        if isinstance(layer, nn.Dropout):
            notes = "Dropout is identity in eval mode; ONNX export may emit an identity/empty graph."
        chunks.append((
            fx,
            _SingleModule(layer),
            f"{target}/{op}",
            [fx],
            [target],
            op,
            "one full-FX critical call_module node in the sequential chain",
            notes,
        ))

    return _execute_chain_specs(chunks, (1, 3, 224, 224))


def _resnet18_dag_aligned(model: nn.Module) -> List[DagAlignedChunkSpec]:
    specs: List[DagAlignedChunkSpec] = []
    for idx, (module, in_shape, out_shape, desc, crit_target) in enumerate(
        make_critical_full_chunks("resnet18", model)
    ):
        specs.append(DagAlignedChunkSpec(
            module=module.eval(),
            input_shape=in_shape,
            output_shape=out_shape,
            description=desc,
            chunk_id=idx,
            chunk_name=desc.replace("[", "_").replace("]", "").replace("+", "_").replace(".", "_"),
            source_graph="module_only_dag",
            covered_fx_nodes=_fx_names_for_targets([crit_target]),
            covered_module_targets=[crit_target],
            source_fx_node_name=_fx_names_for_targets([crit_target])[0],
            source_module_target=crit_target,
            op_type=type(module).__name__,
            is_full_dag_critical_candidate=True,
            materialization_status="materialized",
            boundary_reason=(
                "module-DAG critical node; BasicBlock internals remain grouped "
                "because residual add requires multi-value liveness"
            ),
            notes=(
                "Equal to critical_full for safety. Finer BasicBlock splitting is "
                "deferred until multi-input/multi-output FX subgraph extraction exists."
            ),
        ))
    return specs


def _vit_l_16_dag_aligned(model: nn.Module) -> List[DagAlignedChunkSpec]:
    """
    ViT-L/16 split universe.

    Base chunks follow architecture-defined transformer units:
      0       patch projection + class token + positional embedding
      1..24   encoder.layers.encoder_layer_{0..23}
      25      final encoder norm + class-token classifier head

    Splitting inside an encoder block is intentionally avoided because MHSA,
    MLP, residual adds, and layer norms form a coupled semantic unit.
    """
    chunks: List[tuple[str, nn.Module, str, List[str], List[str], str, str, str]] = []
    chunks.append((
        "patch_class_pos",
        _ViTPatchClassPos(model),
        "conv_proj+class_token+pos_embedding",
        ["conv_proj", "class_token", "encoder_pos_embedding", "encoder_dropout"],
        ["conv_proj", "class_token", "encoder.pos_embedding", "encoder.dropout"],
        "ViTPatchClassPos",
        "ViT tokenization boundary before transformer encoder blocks",
        "Includes patch projection, class token prepend, positional embedding add, and eval-mode dropout.",
    ))

    for i, block in enumerate(model.encoder.layers):
        target = f"encoder.layers.encoder_layer_{i}"
        fx = f"encoder_layer_{i}"
        chunks.append((
            fx,
            _SingleModule(block),
            f"{target}/EncoderBlock",
            [fx],
            [target],
            "EncoderBlock",
            "ViT transformer encoder block boundary; block internals remain grouped",
            "Contains layer norm, multi-head self-attention, residual add, MLP, and residual add.",
        ))

    chunks.append((
        "final_norm_head",
        _ViTFinalNormHead(model),
        "encoder.ln+class_token_select+heads",
        ["encoder_ln", "getitem_class_token", "heads"],
        ["encoder.ln", "heads"],
        "ViTFinalNormHead",
        "ViT classifier boundary after the final transformer encoder block",
        "Applies final norm, selects class token, and runs the classifier head.",
    ))

    return _execute_chain_specs(chunks, (1, 3, 224, 224))


_DAG_ALIGNED_CHUNKERS: dict[str, Callable[[nn.Module], List[DagAlignedChunkSpec]]] = {
    "alexnet": _alexnet_dag_aligned,
    "resnet18": _resnet18_dag_aligned,
    "vit_l_16": _vit_l_16_dag_aligned,
    "vgg19": _vgg19_dag_aligned,
}


def make_dag_aligned_chunks(model_name: str, model: nn.Module) -> List[DagAlignedChunkSpec]:
    if model_name not in _DAG_ALIGNED_CHUNKERS:
        raise KeyError(
            f"No dag_aligned_full chunker for '{model_name}'. "
            f"Available: {sorted(_DAG_ALIGNED_CHUNKERS)}"
        )
    model.eval()
    return _DAG_ALIGNED_CHUNKERS[model_name](model)


def available_dag_aligned_models() -> List[str]:
    return sorted(_DAG_ALIGNED_CHUNKERS)
