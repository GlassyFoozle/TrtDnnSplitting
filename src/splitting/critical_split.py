"""
critical_full split definitions — paper-style "fully split" variant.

Strategy
--------
ResNet18
  The module-only DAG critical analysis gives 14 critical nodes:
  conv1, bn1, relu, maxpool, [8 BasicBlock post-add relus], avgpool, fc.
  Each node becomes the end of one sequential TRT chunk.
  → 14 chunks: single-op initial layers + one chunk per BasicBlock + avgpool + fc.

AlexNet / VGG19
  The module-only DAG shows all module nodes are critical (pure chain).
  Building one TRT engine per ReLU layer would drown in dispatch overhead.
  We therefore group the critical nodes at MaxPool boundaries, which are
  the natural architectural stage boundaries present in both models.
  → AlexNet: 5 chunks   (3 conv-group + avgpool+flatten + classifier)
  → VGG19:   7 chunks   (5 pool-group + avgpool+flatten + classifier)

Chunk shape reference
---------------------
AlexNet:
  0  features[0:3]   (1,3,224,224)→(1,64,27,27)
  1  features[3:6]   (1,64,27,27)→(1,192,13,13)
  2  features[6:13]  (1,192,13,13)→(1,256,6,6)
  3  avgpool+flatten (1,256,6,6)→(1,9216)
  4  classifier      (1,9216)→(1,1000)

VGG19:
  0  features[0:5]   (1,3,224,224)→(1,64,112,112)
  1  features[5:10]  (1,64,112,112)→(1,128,56,56)
  2  features[10:19] (1,128,56,56)→(1,256,28,28)
  3  features[19:28] (1,256,28,28)→(1,512,14,14)
  4  features[28:37] (1,512,14,14)→(1,512,7,7)
  5  avgpool+flatten (1,512,7,7)→(1,25088)
  6  classifier      (1,25088)→(1,1000)

ResNet18:
  0  conv1            (1,3,224,224)→(1,64,112,112)
  1  bn1              (1,64,112,112)→(1,64,112,112)
  2  relu             (1,64,112,112)→(1,64,112,112)
  3  maxpool          (1,64,112,112)→(1,64,56,56)
  4  layer1[0]        (1,64,56,56)→(1,64,56,56)
  5  layer1[1]        (1,64,56,56)→(1,64,56,56)
  6  layer2[0]        (1,64,56,56)→(1,128,28,28)
  7  layer2[1]        (1,128,28,28)→(1,128,28,28)
  8  layer3[0]        (1,128,28,28)→(1,256,14,14)
  9  layer3[1]        (1,256,14,14)→(1,256,14,14)
  10 layer4[0]        (1,256,14,14)→(1,512,7,7)
  11 layer4[1]        (1,512,7,7)→(1,512,7,7)
  12 avgpool          (1,512,7,7)→(1,512,1,1)
  13 flatten+fc       (1,512,1,1)→(1,1000)
"""

from typing import List, Tuple

import torch
import torch.nn as nn

# (module, input_shape, output_shape, description, critical_node_targets)
ChunkSpec = Tuple[nn.Module, Tuple[int, ...], Tuple[int, ...], str, str]


# ── AlexNet critical_full ─────────────────────────────────────────────────────

class _AlexNetCF_Chunk0(nn.Module):
    """features[0:3]: conv1+relu+pool1"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.block = nn.Sequential(*list(m.features.children())[:3])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _AlexNetCF_Chunk1(nn.Module):
    """features[3:6]: conv2+relu+pool2"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.block = nn.Sequential(*list(m.features.children())[3:6])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _AlexNetCF_Chunk2(nn.Module):
    """features[6:13]: conv3+relu+conv4+relu+conv5+relu+pool3"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.block = nn.Sequential(*list(m.features.children())[6:])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _AlexNetCF_Chunk3(nn.Module):
    """avgpool + flatten"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.avgpool = m.avgpool
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(x), 1)


class _AlexNetCF_Chunk4(nn.Module):
    """classifier"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.classifier = m.classifier
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def _alexnet_critical_full(model: nn.Module) -> List[ChunkSpec]:
    return [
        (_AlexNetCF_Chunk0(model).eval(),
         (1, 3, 224, 224), (1, 64, 27, 27),
         "features[0:3]/conv1+relu+pool1", "features.2"),
        (_AlexNetCF_Chunk1(model).eval(),
         (1, 64, 27, 27), (1, 192, 13, 13),
         "features[3:6]/conv2+relu+pool2", "features.5"),
        (_AlexNetCF_Chunk2(model).eval(),
         (1, 192, 13, 13), (1, 256, 6, 6),
         "features[6:13]/conv3+relu+conv4+relu+conv5+relu+pool3", "features.12"),
        (_AlexNetCF_Chunk3(model).eval(),
         (1, 256, 6, 6), (1, 9216),
         "avgpool+flatten", "avgpool"),
        (_AlexNetCF_Chunk4(model).eval(),
         (1, 9216), (1, 1000),
         "classifier", "classifier.6"),
    ]


# ── VGG19 critical_full ───────────────────────────────────────────────────────

class _VGG19CF_Chunk0(nn.Module):
    """features[0:5]: conv1_1+relu+conv1_2+relu+pool1"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.block = nn.Sequential(*list(m.features.children())[:5])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _VGG19CF_Chunk1(nn.Module):
    """features[5:10]: conv2_1+relu+conv2_2+relu+pool2"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.block = nn.Sequential(*list(m.features.children())[5:10])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _VGG19CF_Chunk2(nn.Module):
    """features[10:19]: conv3_x+pool3"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.block = nn.Sequential(*list(m.features.children())[10:19])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _VGG19CF_Chunk3(nn.Module):
    """features[19:28]: conv4_x+pool4"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.block = nn.Sequential(*list(m.features.children())[19:28])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _VGG19CF_Chunk4(nn.Module):
    """features[28:37]: conv5_x+pool5"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.block = nn.Sequential(*list(m.features.children())[28:])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _VGG19CF_Chunk5(nn.Module):
    """avgpool + flatten"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.avgpool = m.avgpool
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(x), 1)


class _VGG19CF_Chunk6(nn.Module):
    """classifier"""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.classifier = m.classifier
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def _vgg19_critical_full(model: nn.Module) -> List[ChunkSpec]:
    return [
        (_VGG19CF_Chunk0(model).eval(),
         (1, 3, 224, 224), (1, 64, 112, 112),
         "features[0:5]/pool1", "features.4"),
        (_VGG19CF_Chunk1(model).eval(),
         (1, 64, 112, 112), (1, 128, 56, 56),
         "features[5:10]/pool2", "features.9"),
        (_VGG19CF_Chunk2(model).eval(),
         (1, 128, 56, 56), (1, 256, 28, 28),
         "features[10:19]/pool3", "features.18"),
        (_VGG19CF_Chunk3(model).eval(),
         (1, 256, 28, 28), (1, 512, 14, 14),
         "features[19:28]/pool4", "features.27"),
        (_VGG19CF_Chunk4(model).eval(),
         (1, 512, 14, 14), (1, 512, 7, 7),
         "features[28:37]/pool5", "features.36"),
        (_VGG19CF_Chunk5(model).eval(),
         (1, 512, 7, 7), (1, 25088),
         "avgpool+flatten", "avgpool"),
        (_VGG19CF_Chunk6(model).eval(),
         (1, 25088), (1, 1000),
         "classifier", "classifier.6"),
    ]


# ── ResNet18 critical_full ────────────────────────────────────────────────────

class _ResNet18CF_Stem(nn.Module):
    """Single op wrapper for individual stem layers."""
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class _ResNet18CF_FlattenFC(nn.Module):
    """avgpool → flatten → fc"""
    def __init__(self, fc: nn.Module):
        super().__init__()
        self.fc = fc
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(x, 1))


def _resnet18_critical_full(model: nn.Module) -> List[ChunkSpec]:
    # The 14 critical nodes from module-only DAG analysis:
    # conv1, bn1, relu, maxpool, layer1[0..1], layer2[0..1],
    # layer3[0..1], layer4[0..1], avgpool, flatten+fc
    return [
        (_ResNet18CF_Stem(model.conv1).eval(),
         (1, 3, 224, 224), (1, 64, 112, 112),
         "conv1", "conv1"),
        (_ResNet18CF_Stem(model.bn1).eval(),
         (1, 64, 112, 112), (1, 64, 112, 112),
         "bn1", "bn1"),
        (_ResNet18CF_Stem(model.relu).eval(),
         (1, 64, 112, 112), (1, 64, 112, 112),
         "relu", "relu"),
        (_ResNet18CF_Stem(model.maxpool).eval(),
         (1, 64, 112, 112), (1, 64, 56, 56),
         "maxpool", "maxpool"),
        # BasicBlocks — each block's forward includes main path + skip + add + relu
        (model.layer1[0].eval(),
         (1, 64, 56, 56), (1, 64, 56, 56),
         "layer1[0]", "layer1.0.relu"),
        (model.layer1[1].eval(),
         (1, 64, 56, 56), (1, 64, 56, 56),
         "layer1[1]", "layer1.1.relu"),
        (model.layer2[0].eval(),
         (1, 64, 56, 56), (1, 128, 28, 28),
         "layer2[0]", "layer2.0.relu"),
        (model.layer2[1].eval(),
         (1, 128, 28, 28), (1, 128, 28, 28),
         "layer2[1]", "layer2.1.relu"),
        (model.layer3[0].eval(),
         (1, 128, 28, 28), (1, 256, 14, 14),
         "layer3[0]", "layer3.0.relu"),
        (model.layer3[1].eval(),
         (1, 256, 14, 14), (1, 256, 14, 14),
         "layer3[1]", "layer3.1.relu"),
        (model.layer4[0].eval(),
         (1, 256, 14, 14), (1, 512, 7, 7),
         "layer4[0]", "layer4.0.relu"),
        (model.layer4[1].eval(),
         (1, 512, 7, 7), (1, 512, 7, 7),
         "layer4[1]", "layer4.1.relu"),
        (_ResNet18CF_Stem(model.avgpool).eval(),
         (1, 512, 7, 7), (1, 512, 1, 1),
         "avgpool", "avgpool"),
        (_ResNet18CF_FlattenFC(model.fc).eval(),
         (1, 512, 1, 1), (1, 1000),
         "flatten+fc", "fc"),
    ]


# ── Dispatcher ────────────────────────────────────────────────────────────────

_CRITICAL_CHUNKERS = {
    "alexnet":  _alexnet_critical_full,
    "resnet18": _resnet18_critical_full,
    "vgg19":    _vgg19_critical_full,
}


def make_critical_full_chunks(model_name: str, model: nn.Module) -> List[ChunkSpec]:
    """
    Return critical_full chunks for the given model.
    Each chunk is (module, input_shape, output_shape, description, critical_target).
    """
    if model_name not in _CRITICAL_CHUNKERS:
        raise KeyError(
            f"No critical_full chunker for '{model_name}'. "
            f"Available: {sorted(_CRITICAL_CHUNKERS)}"
        )
    return _CRITICAL_CHUNKERS[model_name](model)


def available_critical_full_models() -> List[str]:
    return sorted(_CRITICAL_CHUNKERS)
