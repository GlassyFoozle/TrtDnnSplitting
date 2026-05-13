"""
Model registry for Phase C split-point experiments.

Supported models:
  alexnet    — AlexNet (Krizhevsky 2012), torchvision
  resnet18   — ResNet18 (He 2016), torchvision
  vgg19      — VGG19 (Simonyan 2014), torchvision
  vit_l_16   — Vision Transformer large, patch 16, torchvision
  vit / vit_b_16 — Vision Transformer base, patch 16, torchvision
                (registry/full-model construction support only in this pass)
  inceptionv4 — NOT AVAILABLE: requires timm, which is not installed on this system
                (JetPack 6.0 / L4T R36.3 environment). Framework is ready; install
                timm and add an entry here when the dependency becomes available.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


@dataclass
class ModelInfo:
    name:        str
    constructor: Optional[Callable[[], nn.Module]]
    input_shape: Tuple[int, ...]
    notes:       str
    available:   bool


def _with_nonzero_classifier_head(model: nn.Module) -> nn.Module:
    """Keep weights=None synthetic, but avoid all-zero classifier pruning."""
    gen = torch.Generator()
    gen.manual_seed(12345)
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == 1000:
            if float(module.weight.detach().abs().max()) == 0.0:
                with torch.no_grad():
                    module.weight.normal_(mean=0.0, std=0.02, generator=gen)
                    if module.bias is not None:
                        module.bias.zero_()
    return model.eval()


def _make_registry() -> dict:
    return {
        "alexnet": ModelInfo(
            name="alexnet",
            constructor=lambda: tvm.alexnet(weights=None).eval(),
            input_shape=(1, 3, 224, 224),
            notes="AlexNet (Krizhevsky 2012); 5 conv + 3 FC; ~61 M params",
            available=True,
        ),
        "resnet18": ModelInfo(
            name="resnet18",
            constructor=lambda: tvm.resnet18(weights=None).eval(),
            input_shape=(1, 3, 224, 224),
            notes="ResNet18 (He 2016); skip connections; ~11 M params; manual 3-chunk split reused from Phase B",
            available=True,
        ),
        "vgg19": ModelInfo(
            name="vgg19",
            constructor=lambda: tvm.vgg19(weights=None).eval(),
            input_shape=(1, 3, 224, 224),
            notes="VGG19 (Simonyan 2014); 19-layer all-conv; ~138 M params",
            available=True,
        ),
        "vit_l_16": ModelInfo(
            name="vit_l_16",
            constructor=lambda: _with_nonzero_classifier_head(tvm.vit_l_16(weights=None)),
            input_shape=(1, 3, 224, 224),
            notes=(
                "Vision Transformer L/16, torchvision; 24 encoder blocks. "
                "Classifier head is deterministically initialized when weights=None "
                "to avoid TensorRT constant-output pruning."
            ),
            available=hasattr(tvm, "vit_l_16"),
        ),
        "vit_b_16": ModelInfo(
            name="vit_b_16",
            constructor=lambda: _with_nonzero_classifier_head(tvm.vit_b_16(weights=None)),
            input_shape=(1, 3, 224, 224),
            notes=(
                "Vision Transformer B/16, torchvision. Registry/full-model "
                "construction support only in this pass; dag_aligned_full "
                "chunking/export/profiling needs transformer-specific wrappers."
            ),
            available=hasattr(tvm, "vit_b_16"),
        ),
        "vit": ModelInfo(
            name="vit",
            constructor=lambda: tvm.vit_b_16(weights=None).eval(),
            input_shape=(1, 3, 224, 224),
            notes=(
                "Alias for torchvision vit_b_16. Registry/full-model construction "
                "support only in this pass; split materialization is pending."
            ),
            available=hasattr(tvm, "vit_b_16"),
        ),
        "inceptionv4": ModelInfo(
            name="inceptionv4",
            constructor=None,
            input_shape=(1, 3, 299, 299),
            notes=(
                "InceptionV4 (Szegedy 2016); requires timm — NOT AVAILABLE on this system "
                "(JetPack 6.0 / L4T R36.3). Install timm and register constructor here."
            ),
            available=False,
        ),
    }


_REGISTRY: dict = _make_registry()


def get_model_info(name: str) -> ModelInfo:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {list_models()}")
    return _REGISTRY[name]


def list_models() -> List[str]:
    return sorted(_REGISTRY)


def list_available_models() -> List[str]:
    return sorted(k for k, v in _REGISTRY.items() if v.available)


def build_model(name: str) -> nn.Module:
    info = get_model_info(name)
    if not info.available:
        raise RuntimeError(f"Model '{name}' is not available: {info.notes}")
    return info.constructor()
