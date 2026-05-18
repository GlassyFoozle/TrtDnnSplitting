# trt_fusion_safe 모델별 split policy 요약

작성일: 2026-05-15

참조 파일:

- `configs/split_point_policies.json`
- `artifacts/split_configs/<model>/dag_aligned_full.json`
- `src/models/registry.py`

Boundary 의미:

- `Boundary k`는 `chunk k`와 `chunk k+1` 사이 split point다.
- `Considered SPs`는 현재 `trt_fusion_safe` policy에서 실제 사용하는 boundary 개수다.
- `Total SPs`는 architecture와 TensorRT fusion을 고려하면서 더 쪼갰을 때 가능한 split point 개수다. 단, VGG19를 제외한 모델들은 현재 `trt_fusion_safe`가 이미 해당 기준의 충분히 granular한 단위라고 보고 `Total SPs = Considered SPs`로 둔다.

## Split Point 개수 요약

|  | alexnet | resnet18 | vgg19 | vit_b_16 | inception_v3 | mobilenet_v3_small |
|---|---:|---:|---:|---:|---:|---:|
| Total SPs | 10 | 10 | 20 | 13 | 16 | 15 |
| Considered SPs | 10 | 10 | 11 | 13 | 16 | 15 |

## Torchvision Label

| repo model key | torchvision label | weights enum label | input |
|---|---|---|---|
| `alexnet` | `torchvision.models.alexnet` | `AlexNet_Weights` | `1x3x224x224` |
| `resnet18` | `torchvision.models.resnet18` | `ResNet18_Weights` | `1x3x224x224` |
| `vgg19` | `torchvision.models.vgg19` | `VGG19_Weights` | `1x3x224x224` |
| `vit_b_16` | `torchvision.models.vit_b_16` | `ViT_B_16_Weights` | `1x3x224x224` |
| `inception_v3` | `torchvision.models.inception_v3` | `Inception_V3_Weights` | `1x3x299x299` |
| `mobilenet_v3_small` | `torchvision.models.mobilenet_v3_small` | `MobileNet_V3_Small_Weights` | `1x3x224x224` |

## AlexNet

Torchvision label: `torchvision.models.alexnet`, `AlexNet_Weights`

`trt_fusion_safe = [1, 2, 4, 5, 7, 9, 11, 14, 17, 20]`

### Node Table

| id | chunk | role | shape |
|---:|---|---|---|
| 0 | `features_0` | Conv2d | `1x3x224x224 -> 1x64x55x55` |
| 1 | `features_1` | ReLU | `1x64x55x55 -> 1x64x55x55` |
| 2 | `features_2` | MaxPool2d | `1x64x55x55 -> 1x64x27x27` |
| 3 | `features_3` | Conv2d | `1x64x27x27 -> 1x192x27x27` |
| 4 | `features_4` | ReLU | `1x192x27x27 -> 1x192x27x27` |
| 5 | `features_5` | MaxPool2d | `1x192x27x27 -> 1x192x13x13` |
| 6 | `features_6` | Conv2d | `1x192x13x13 -> 1x384x13x13` |
| 7 | `features_7` | ReLU | `1x384x13x13 -> 1x384x13x13` |
| 8 | `features_8` | Conv2d | `1x384x13x13 -> 1x256x13x13` |
| 9 | `features_9` | ReLU | `1x256x13x13 -> 1x256x13x13` |
| 10 | `features_10` | Conv2d | `1x256x13x13 -> 1x256x13x13` |
| 11 | `features_11` | ReLU | `1x256x13x13 -> 1x256x13x13` |
| 12 | `features_12` | MaxPool2d | `1x256x13x13 -> 1x256x6x6` |
| 13 | `avgpool` | AdaptiveAvgPool2d | `1x256x6x6 -> 1x256x6x6` |
| 14 | `flatten` | flatten | `1x256x6x6 -> 1x9216` |
| 15 | `classifier_0` | Dropout | `1x9216 -> 1x9216` |
| 16 | `classifier_1` | Linear | `1x9216 -> 1x4096` |
| 17 | `classifier_2` | ReLU | `1x4096 -> 1x4096` |
| 18 | `classifier_3` | Dropout | `1x4096 -> 1x4096` |
| 19 | `classifier_4` | Linear | `1x4096 -> 1x4096` |
| 20 | `classifier_5` | ReLU | `1x4096 -> 1x4096` |
| 21 | `classifier_6` | Linear | `1x4096 -> 1x1000` |

### Policy 도출 기준

AlexNet은 대부분 sequential `Conv -> ReLU -> Pool` 구조다. `Conv -> ReLU` 내부 boundary는 TensorRT activation fusion을 끊을 수 있으므로 닫고, ReLU 뒤 또는 MaxPool 뒤처럼 conv unit이나 spatial stage가 끝나는 지점을 split 후보로 둔다. `avgpool`은 shape 변화가 없고 바로 `flatten`이 따라오므로 `13`은 제외하고, rank가 바뀌는 `flatten` 뒤 `14`를 사용한다. Classifier는 `Linear -> ReLU` unit이 끝나는 `17`, `20`을 사용하고 eval-mode dropout만 따로 분리하는 boundary는 제외한다.

## ResNet18

Torchvision label: `torchvision.models.resnet18`, `ResNet18_Weights`

`trt_fusion_safe = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`

### Node Table

| id | chunk | role | shape |
|---:|---|---|---|
| 0 | `conv1` | stem conv | `1x3x224x224 -> 1x64x112x112` |
| 1 | `bn1` | stem batch norm | `1x64x112x112 -> 1x64x112x112` |
| 2 | `relu` | stem activation | `1x64x112x112 -> 1x64x112x112` |
| 3 | `maxpool` | stem maxpool | `1x64x112x112 -> 1x64x56x56` |
| 4 | `layer1_0` | BasicBlock | `1x64x56x56 -> 1x64x56x56` |
| 5 | `layer1_1` | BasicBlock | `1x64x56x56 -> 1x64x56x56` |
| 6 | `layer2_0` | downsample BasicBlock | `1x64x56x56 -> 1x128x28x28` |
| 7 | `layer2_1` | BasicBlock | `1x128x28x28 -> 1x128x28x28` |
| 8 | `layer3_0` | downsample BasicBlock | `1x128x28x28 -> 1x256x14x14` |
| 9 | `layer3_1` | BasicBlock | `1x256x14x14 -> 1x256x14x14` |
| 10 | `layer4_0` | downsample BasicBlock | `1x256x14x14 -> 1x512x7x7` |
| 11 | `layer4_1` | BasicBlock | `1x512x7x7 -> 1x512x7x7` |
| 12 | `avgpool` | global average pool | `1x512x7x7 -> 1x512x1x1` |
| 13 | `flatten_fc` | flatten + linear head | `1x512x1x1 -> 1x1000` |

### Policy 도출 기준

Stem의 `Conv -> BN -> ReLU -> MaxPool` 내부는 TensorRT의 folding/fusion 가능성이 크므로 `0`, `1`, `2` boundary를 닫는다. BasicBlock 내부에는 residual add와 multi-value liveness가 있으므로 현재 artifact는 block 전체를 하나의 안전한 chunk로 둔다. 따라서 `trt_fusion_safe`는 stem 이후 `3`, 각 BasicBlock 사이와 stage 종료 지점 `4..11`, avgpool 뒤 head 전환 `12`를 사용한다.

## VGG19

Torchvision label: `torchvision.models.vgg19`, `VGG19_Weights`

`trt_fusion_safe = [4, 9, 13, 18, 22, 27, 31, 36, 38, 41, 44]`

### Node Table

| id | chunk | role | shape |
|---:|---|---|---|
| 0 | `features_0` | Conv2d | `1x3x224x224 -> 1x64x224x224` |
| 1 | `features_1` | ReLU | `1x64x224x224 -> 1x64x224x224` |
| 2 | `features_2` | Conv2d | `1x64x224x224 -> 1x64x224x224` |
| 3 | `features_3` | ReLU | `1x64x224x224 -> 1x64x224x224` |
| 4 | `features_4` | MaxPool2d | `1x64x224x224 -> 1x64x112x112` |
| 5 | `features_5` | Conv2d | `1x64x112x112 -> 1x128x112x112` |
| 6 | `features_6` | ReLU | `1x128x112x112 -> 1x128x112x112` |
| 7 | `features_7` | Conv2d | `1x128x112x112 -> 1x128x112x112` |
| 8 | `features_8` | ReLU | `1x128x112x112 -> 1x128x112x112` |
| 9 | `features_9` | MaxPool2d | `1x128x112x112 -> 1x128x56x56` |
| 10 | `features_10` | Conv2d | `1x128x56x56 -> 1x256x56x56` |
| 11 | `features_11` | ReLU | `1x256x56x56 -> 1x256x56x56` |
| 12 | `features_12` | Conv2d | `1x256x56x56 -> 1x256x56x56` |
| 13 | `features_13` | ReLU | `1x256x56x56 -> 1x256x56x56` |
| 14 | `features_14` | Conv2d | `1x256x56x56 -> 1x256x56x56` |
| 15 | `features_15` | ReLU | `1x256x56x56 -> 1x256x56x56` |
| 16 | `features_16` | Conv2d | `1x256x56x56 -> 1x256x56x56` |
| 17 | `features_17` | ReLU | `1x256x56x56 -> 1x256x56x56` |
| 18 | `features_18` | MaxPool2d | `1x256x56x56 -> 1x256x28x28` |
| 19 | `features_19` | Conv2d | `1x256x28x28 -> 1x512x28x28` |
| 20 | `features_20` | ReLU | `1x512x28x28 -> 1x512x28x28` |
| 21 | `features_21` | Conv2d | `1x512x28x28 -> 1x512x28x28` |
| 22 | `features_22` | ReLU | `1x512x28x28 -> 1x512x28x28` |
| 23 | `features_23` | Conv2d | `1x512x28x28 -> 1x512x28x28` |
| 24 | `features_24` | ReLU | `1x512x28x28 -> 1x512x28x28` |
| 25 | `features_25` | Conv2d | `1x512x28x28 -> 1x512x28x28` |
| 26 | `features_26` | ReLU | `1x512x28x28 -> 1x512x28x28` |
| 27 | `features_27` | MaxPool2d | `1x512x28x28 -> 1x512x14x14` |
| 28 | `features_28` | Conv2d | `1x512x14x14 -> 1x512x14x14` |
| 29 | `features_29` | ReLU | `1x512x14x14 -> 1x512x14x14` |
| 30 | `features_30` | Conv2d | `1x512x14x14 -> 1x512x14x14` |
| 31 | `features_31` | ReLU | `1x512x14x14 -> 1x512x14x14` |
| 32 | `features_32` | Conv2d | `1x512x14x14 -> 1x512x14x14` |
| 33 | `features_33` | ReLU | `1x512x14x14 -> 1x512x14x14` |
| 34 | `features_34` | Conv2d | `1x512x14x14 -> 1x512x14x14` |
| 35 | `features_35` | ReLU | `1x512x14x14 -> 1x512x14x14` |
| 36 | `features_36` | MaxPool2d | `1x512x14x14 -> 1x512x7x7` |
| 37 | `avgpool` | AdaptiveAvgPool2d | `1x512x7x7 -> 1x512x7x7` |
| 38 | `flatten` | flatten | `1x512x7x7 -> 1x25088` |
| 39 | `classifier_0` | Linear | `1x25088 -> 1x4096` |
| 40 | `classifier_1` | ReLU | `1x4096 -> 1x4096` |
| 41 | `classifier_2` | Dropout | `1x4096 -> 1x4096` |
| 42 | `classifier_3` | Linear | `1x4096 -> 1x4096` |
| 43 | `classifier_4` | ReLU | `1x4096 -> 1x4096` |
| 44 | `classifier_5` | Dropout | `1x4096 -> 1x4096` |
| 45 | `classifier_6` | Linear | `1x4096 -> 1x1000` |

### Policy 도출 기준

VGG19의 가능한 architecture-level grouping은 `Conv-ReLU` 또는 `Conv-ReLU-MaxPool` unit을 기본 단위로 보면 21개 grouped chunk, 즉 20개 Total SPs가 된다:

`(0,1)`, `(2,3,4)`, `(5,6)`, `(7,8,9)`, `(10,11)`, `(12,13)`, `(14,15)`, `(16,17,18)`, `(19,20)`, `(21,22)`, `(23,24)`, `(25,26,27)`, `(28,29)`, `(30,31)`, `(32,33)`, `(34,35,36)`, `(37)`, `(38)`, `(39,40,41)`, `(42,43,44)`, `(45)`.

총 21개 grouped chunk이므로 가능한 split point는 20개다. 

 하지만 현재 `trt_fusion_safe`는 그 전체를 다 쓰지 않고 stage exit와 큰 stage의 midpoint만 선택한다. `4`, `9`, `18`, `27`, `36`은 MaxPool 뒤 stage exit다. `13`, `22`, `31`은 4-conv stage를 앞 2개 conv unit과 뒤 2개 conv unit으로 나누는 midpoint다. `38`은 flatten 뒤 classifier 전환이고, `41`, `44`는 classifier unit 종료 지점이다. 이렇게 하면 Conv/ReLU fusion은 보존하면서 20개 후보 중 11개만 사용한다.

## ViT-B-16

Torchvision label: `torchvision.models.vit_b_16`, `ViT_B_16_Weights`

`trt_fusion_safe = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`

### Node Table

| id | chunk | role | shape |
|---:|---|---|---|
| 0 | `patch_class_pos` | patch projection + class token + positional embedding | `1x3x224x224 -> 1x197x768` |
| 1 | `encoder_layer_0` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 2 | `encoder_layer_1` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 3 | `encoder_layer_2` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 4 | `encoder_layer_3` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 5 | `encoder_layer_4` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 6 | `encoder_layer_5` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 7 | `encoder_layer_6` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 8 | `encoder_layer_7` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 9 | `encoder_layer_8` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 10 | `encoder_layer_9` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 11 | `encoder_layer_10` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 12 | `encoder_layer_11` | transformer EncoderBlock | `1x197x768 -> 1x197x768` |
| 13 | `final_norm_head` | final norm + class token select + classifier head | `1x197x768 -> 1x1000` |

### Policy 도출 기준

ViT-B-16은 patch/class/position prep, 12개 encoder block, final norm/head로 나뉜다. Encoder block 내부에는 layer norm, attention, residual add, MLP가 같이 있으므로 내부 split은 liveness와 TensorRT engine IO가 복잡해진다. 따라서 block 내부는 묶고, block 사이 single-tensor handoff boundary는 모두 사용한다. ViT-B-16은 이 기준에서 가능한 split point가 13개이고 현재 `trt_fusion_safe`도 13개를 모두 사용한다.

## InceptionV3

Torchvision label: `torchvision.models.inception_v3`, `Inception_V3_Weights`

`trt_fusion_safe = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20]`

### Node Table

| id | chunk | role | shape |
|---:|---|---|---|
| 0 | `Conv2d_1a_3x3` | BasicConv2d | `1x3x299x299 -> 1x32x149x149` |
| 1 | `Conv2d_2a_3x3` | BasicConv2d | `1x32x149x149 -> 1x32x147x147` |
| 2 | `Conv2d_2b_3x3` | BasicConv2d | `1x32x147x147 -> 1x64x147x147` |
| 3 | `maxpool1` | MaxPool2d | `1x64x147x147 -> 1x64x73x73` |
| 4 | `Conv2d_3b_1x1` | BasicConv2d | `1x64x73x73 -> 1x80x73x73` |
| 5 | `Conv2d_4a_3x3` | BasicConv2d | `1x80x73x73 -> 1x192x71x71` |
| 6 | `maxpool2` | MaxPool2d | `1x192x71x71 -> 1x192x35x35` |
| 7 | `Mixed_5b` | InceptionA | `1x192x35x35 -> 1x256x35x35` |
| 8 | `Mixed_5c` | InceptionA | `1x256x35x35 -> 1x288x35x35` |
| 9 | `Mixed_5d` | InceptionA | `1x288x35x35 -> 1x288x35x35` |
| 10 | `Mixed_6a` | InceptionB reduction | `1x288x35x35 -> 1x768x17x17` |
| 11 | `Mixed_6b` | InceptionC | `1x768x17x17 -> 1x768x17x17` |
| 12 | `Mixed_6c` | InceptionC | `1x768x17x17 -> 1x768x17x17` |
| 13 | `Mixed_6d` | InceptionC | `1x768x17x17 -> 1x768x17x17` |
| 14 | `Mixed_6e` | InceptionC | `1x768x17x17 -> 1x768x17x17` |
| 15 | `Mixed_7a` | InceptionD reduction | `1x768x17x17 -> 1x1280x8x8` |
| 16 | `Mixed_7b` | InceptionE | `1x1280x8x8 -> 1x2048x8x8` |
| 17 | `Mixed_7c` | InceptionE | `1x2048x8x8 -> 1x2048x8x8` |
| 18 | `avgpool` | AdaptiveAvgPool2d | `1x2048x8x8 -> 1x2048x1x1` |
| 19 | `dropout` | Dropout | `1x2048x1x1 -> 1x2048x1x1` |
| 20 | `flatten` | flatten | `1x2048x1x1 -> 1x2048` |
| 21 | `fc` | Linear | `1x2048 -> 1x1000` |

### Policy 도출 기준

InceptionV3의 `BasicConv2d` 내부는 `Conv -> BN -> ReLU`가 하나의 fusion-friendly unit이고, `Mixed_*` 내부는 여러 branch와 concat을 포함한다. 그래서 각 block 내부는 split하지 않고 block output boundary를 사용한다. 초기 stem에서는 너무 작은 conv를 전부 나누지 않고 `2`, `3`, `5`, `6`처럼 stem sub-stage가 끝나는 지점을 사용한다. Inception block 구간은 `7..17`을 열어 block 단위 탐색을 허용한다. `avgpool/dropout` 주변은 별도 compute 의미가 약해 `20` flatten 뒤 classifier 전환만 사용한다.

## MobileNetV3-Small

Torchvision label: `torchvision.models.mobilenet_v3_small`, `MobileNet_V3_Small_Weights`

`trt_fusion_safe = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17]`

### Node Table

| id | chunk | role | shape |
|---:|---|---|---|
| 0 | `features_0` | stem Conv2dNormActivation | `1x3x224x224 -> 1x16x112x112` |
| 1 | `features_1` | InvertedResidual | `1x16x112x112 -> 1x16x56x56` |
| 2 | `features_2` | InvertedResidual | `1x16x56x56 -> 1x24x28x28` |
| 3 | `features_3` | InvertedResidual | `1x24x28x28 -> 1x24x28x28` |
| 4 | `features_4` | InvertedResidual | `1x24x28x28 -> 1x40x14x14` |
| 5 | `features_5` | InvertedResidual | `1x40x14x14 -> 1x40x14x14` |
| 6 | `features_6` | InvertedResidual | `1x40x14x14 -> 1x40x14x14` |
| 7 | `features_7` | InvertedResidual | `1x40x14x14 -> 1x48x14x14` |
| 8 | `features_8` | InvertedResidual | `1x48x14x14 -> 1x48x14x14` |
| 9 | `features_9` | InvertedResidual | `1x48x14x14 -> 1x96x7x7` |
| 10 | `features_10` | InvertedResidual | `1x96x7x7 -> 1x96x7x7` |
| 11 | `features_11` | InvertedResidual | `1x96x7x7 -> 1x96x7x7` |
| 12 | `features_12` | final Conv2dNormActivation | `1x96x7x7 -> 1x576x7x7` |
| 13 | `avgpool` | AdaptiveAvgPool2d | `1x576x7x7 -> 1x576x1x1` |
| 14 | `flatten` | flatten | `1x576x1x1 -> 1x576` |
| 15 | `classifier_0` | Linear | `1x576 -> 1x1024` |
| 16 | `classifier_1` | Hardswish | `1x1024 -> 1x1024` |
| 17 | `classifier_2` | Dropout | `1x1024 -> 1x1024` |
| 18 | `classifier_3` | Linear | `1x1024 -> 1x1000` |

### Policy 도출 기준

MobileNetV3-Small은 `InvertedResidual` 내부에 expansion, depthwise convolution, squeeze-excitation, projection, optional residual add가 묶인다. 이 내부를 나누면 TensorRT fusion과 residual/SE liveness를 해칠 수 있으므로 block 내부는 유지한다. 대신 stem과 각 feature block output `0..12`를 split 후보로 열고, `avgpool`은 바로 flatten으로 이어지므로 `14`를 rank-change boundary로 사용한다. Classifier는 `Linear -> Hardswish -> Dropout` unit을 묶고 `17` 뒤를 사용한다.

