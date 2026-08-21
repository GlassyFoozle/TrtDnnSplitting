"""Self-contained YOLO11s inference graph used by the split experiments.

The repository only needs the inference architecture and deterministic weights
for TensorRT timing; it does not train the model.  Keeping this implementation
local avoids an Ultralytics runtime dependency on Jetson while matching the
YOLO11s backbone, PAN neck, three-scale Detect head, and output shape.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


def autopad(kernel_size: int, padding: int | None = None, dilation: int = 1) -> int:
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    return kernel_size // 2 if padding is None else padding


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU used throughout YOLO11."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=False) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True) -> None:
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Sequence[int] = (3, 3),
        e: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, int(k[0]), 1)
        self.cv2 = Conv(hidden, c2, int(k[1]), 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.cv3 = Conv(2 * hidden, c2, 1, 1)
        self.m = nn.Sequential(
            *(Bottleneck(hidden, hidden, shortcut, g, (k, k), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3k2(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.hidden, 1, 1)
        blocks: List[nn.Module] = []
        for _ in range(n):
            if c3k:
                blocks.append(C3k(self.hidden, self.hidden, n=2, shortcut=shortcut, g=g))
            else:
                # YOLO11's non-C3k branch keeps the Bottleneck's 0.5 inner
                # expansion (confirmed against the supplied yolo11s.onnx
                # convolution shapes, e.g. 32 -> 16 -> 32 in model.2).
                blocks.append(Bottleneck(self.hidden, self.hidden, shortcut, g, (3, 3), e=0.5))
        self.m = nn.ModuleList(blocks)
        self.cv2 = Conv((2 + n) * self.hidden, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.m:
            parts.append(block(parts[-1]))
        return self.cv2(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat((x, y1, y2, self.pool(y2)), dim=1))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        qkv_dim = dim + 2 * self.key_dim * num_heads
        self.qkv = Conv(dim, qkv_dim, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(x).view(batch, self.num_heads, self.key_dim * 2 + self.head_dim, tokens)
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attended = (v @ attn.transpose(-2, -1)).view(batch, channels, height, width)
        positional = self.pe(v.reshape(batch, channels, height, width))
        return self.proj(attended + positional)


class PSABlock(nn.Module):
    def __init__(self, channels: int, attn_ratio: float = 0.5, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = Attention(channels, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(channels, channels * 2, 1), Conv(channels * 2, channels, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError("C2PSA expects equal input/output channels")
        hidden = int(c1 * e)
        self.cv1 = Conv(c1, 2 * hidden, 1, 1)
        self.cv2 = Conv(2 * hidden, c1, 1, 1)
        self.m = nn.Sequential(*(PSABlock(hidden, num_heads=max(1, hidden // 64)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat((a, self.m(b)), dim=1))


class Detect(nn.Module):
    """Three-scale YOLO11 Detect head with DFL decode, no NMS."""

    def __init__(self, nc: int = 80, ch: Sequence[int] = (128, 256, 512), reg_max: int = 16) -> None:
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        box_channels = max(16, ch[0] // 4, reg_max * 4)
        cls_channels = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(c, box_channels, 3), Conv(box_channels, box_channels, 3), nn.Conv2d(box_channels, 4 * reg_max, 1))
            for c in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(c, c, 3), Conv(c, cls_channels, 1)),
                nn.Sequential(DWConv(cls_channels, cls_channels, 3), Conv(cls_channels, cls_channels, 1)),
                nn.Conv2d(cls_channels, nc, 1),
            )
            for c in ch
        )
        self.register_buffer("dfl_project", torch.arange(reg_max, dtype=torch.float32).view(1, 1, reg_max, 1))
        anchors, strides = self._make_anchors((80, 40, 20), (8.0, 16.0, 32.0))
        self.register_buffer("anchors", anchors)
        self.register_buffer("strides", strides)
        self._initialize_biases(ch)

    @staticmethod
    def _make_anchors(sizes: Sequence[int], stride_values: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = []
        strides = []
        for size, stride in zip(sizes, stride_values):
            axis = torch.arange(size, dtype=torch.float32) + 0.5
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            anchors.append(torch.stack((xx, yy), dim=0).reshape(2, -1))
            strides.append(torch.full((1, size * size), stride, dtype=torch.float32))
        return torch.cat(anchors, dim=1).unsqueeze(0), torch.cat(strides, dim=1).unsqueeze(0)

    def _initialize_biases(self, ch: Sequence[int]) -> None:
        for box, cls, stride, channels in zip(self.cv2, self.cv3, (8.0, 16.0, 32.0), ch):
            nn.init.constant_(box[-1].bias, 1.0)
            nn.init.constant_(cls[-1].bias, math.log(5.0 / self.nc / (640.0 / stride) ** 2))

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        scales = []
        for feature, box_head, cls_head in zip(features, self.cv2, self.cv3):
            scales.append(torch.cat((box_head(feature), cls_head(feature)), dim=1).flatten(2))
        prediction = torch.cat(scales, dim=2)
        box_logits, cls_logits = prediction.split((self.reg_max * 4, self.nc), dim=1)
        batch, _, anchors = box_logits.shape
        distances = box_logits.view(batch, 4, self.reg_max, anchors).softmax(dim=2)
        distances = (distances * self.dfl_project).sum(dim=2)
        top_left = self.anchors - distances[:, :2]
        bottom_right = self.anchors + distances[:, 2:]
        center = (top_left + bottom_right) * 0.5
        size = bottom_right - top_left
        boxes = torch.cat((center, size), dim=1) * self.strides
        return torch.cat((boxes, cls_logits.sigmoid()), dim=1)


class Yolo11s(nn.Module):
    """YOLO11s detection model with top-level modules numbered model.0..23."""

    def __init__(self, num_classes: int = 80) -> None:
        super().__init__()
        self.model = nn.ModuleList([
            Conv(3, 32, 3, 2),
            Conv(32, 64, 3, 2),
            C3k2(64, 128, n=1, c3k=False, e=0.25, shortcut=False),
            Conv(128, 128, 3, 2),
            C3k2(128, 256, n=1, c3k=False, e=0.25, shortcut=False),
            Conv(256, 256, 3, 2),
            C3k2(256, 256, n=1, c3k=True, e=0.5, shortcut=True),
            Conv(256, 512, 3, 2),
            C3k2(512, 512, n=1, c3k=True, e=0.5, shortcut=True),
            SPPF(512, 512, 5),
            C2PSA(512, 512, n=1),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Identity(),
            C3k2(768, 256, n=1, c3k=False, e=0.5, shortcut=False),
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Identity(),
            C3k2(512, 128, n=1, c3k=False, e=0.5, shortcut=False),
            Conv(128, 128, 3, 2),
            nn.Identity(),
            C3k2(384, 256, n=1, c3k=False, e=0.5, shortcut=False),
            Conv(256, 256, 3, 2),
            nn.Identity(),
            C3k2(768, 512, n=1, c3k=True, e=0.5, shortcut=True),
            Detect(num_classes, (128, 256, 512)),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: Dict[int, torch.Tensor] = {}
        for i, module in enumerate(self.model):
            if i == 12:
                x = torch.cat((x, outputs[6]), dim=1)
            elif i == 15:
                x = torch.cat((x, outputs[4]), dim=1)
            elif i == 18:
                x = torch.cat((x, outputs[13]), dim=1)
            elif i == 21:
                x = torch.cat((x, outputs[10]), dim=1)
            elif i == 23:
                return module((outputs[16], outputs[19], outputs[22]))
            else:
                x = module(x)
            outputs[i] = x
        raise RuntimeError("YOLO11s graph did not reach Detect")


# Live values after each top-level module.  Multiple differently shaped values
# are flattened and concatenated by Yolo11StateChunk so the existing one-input,
# one-output TensorRT pipeline can carry the DAG state without runtime changes.
YOLO11_LIVE_KEYS: List[List[int]] = [
    [0], [1], [2], [3], [4], [4, 5], [4, 6], [4, 6, 7], [4, 6, 8],
    [4, 6, 9], [4, 6, 10], [4, 6, 10, 11], [4, 10, 12], [4, 10, 13],
    [4, 10, 13, 14], [10, 13, 15], [10, 13, 16], [10, 13, 16, 17],
    [10, 16, 18], [10, 16, 19], [10, 16, 19, 20], [16, 19, 21],
    [16, 19, 22], [23],
]

YOLO11_FEATURE_SHAPES: Dict[int, tuple[int, ...]] = {
    0: (1, 32, 320, 320),
    1: (1, 64, 160, 160),
    2: (1, 128, 160, 160),
    3: (1, 128, 80, 80),
    4: (1, 256, 80, 80),
    5: (1, 256, 40, 40),
    6: (1, 256, 40, 40),
    7: (1, 512, 20, 20),
    8: (1, 512, 20, 20),
    9: (1, 512, 20, 20),
    10: (1, 512, 20, 20),
    11: (1, 512, 40, 40),
    12: (1, 768, 40, 40),
    13: (1, 256, 40, 40),
    14: (1, 256, 80, 80),
    15: (1, 512, 80, 80),
    16: (1, 128, 80, 80),
    17: (1, 128, 40, 40),
    18: (1, 384, 40, 40),
    19: (1, 256, 40, 40),
    20: (1, 256, 20, 20),
    21: (1, 768, 20, 20),
    22: (1, 512, 20, 20),
    23: (1, 84, 8400),
}


class Yolo11StateChunk(nn.Module):
    """One model.N step with packed single-tensor live-state I/O."""

    def __init__(self, model: Yolo11s, module_id: int) -> None:
        super().__init__()
        self.module_id = module_id
        self.op = model.model[module_id]
        self.input_keys = [-1] if module_id == 0 else YOLO11_LIVE_KEYS[module_id - 1]
        self.output_keys = YOLO11_LIVE_KEYS[module_id]

    @staticmethod
    def _numel(shape: Sequence[int]) -> int:
        result = 1
        for dim in shape[1:]:
            result *= int(dim)
        return result

    def _unpack(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        if self.input_keys == [-1]:
            return {-1: x}
        if len(self.input_keys) == 1:
            return {self.input_keys[0]: x}
        sizes = [self._numel(YOLO11_FEATURE_SHAPES[key]) for key in self.input_keys]
        values = torch.split(x, sizes, dim=1)
        return {
            key: value.reshape(value.shape[0], *YOLO11_FEATURE_SHAPES[key][1:])
            for key, value in zip(self.input_keys, values)
        }

    def _pack(self, state: Dict[int, torch.Tensor]) -> torch.Tensor:
        values = [state[key] for key in self.output_keys]
        if len(values) == 1:
            return values[0]
        return torch.cat([value.flatten(1) for value in values], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self._unpack(x)
        i = self.module_id
        if i == 0:
            current = self.op(state[-1])
        elif i in (12, 15, 18, 21):
            route_key = {12: 6, 15: 4, 18: 13, 21: 10}[i]
            current = torch.cat((state[i - 1], state[route_key]), dim=1)
        elif i == 23:
            current = self.op((state[16], state[19], state[22]))
        else:
            current = self.op(state[i - 1])
        state[i] = current
        return self._pack(state)


def build_yolo11s() -> Yolo11s:
    """Build deterministic synthetic weights for cross-device reproducibility."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11011)
        return Yolo11s().eval()
