# Paper Outline

## Title

Robust, Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification

## Abstract

Include:

- problem: tomato disease classification,
- limitation of clean datasets,
- models compared,
- clean vs corrupted evaluation,
- efficiency benchmark,
- Grad-CAM explainability,
- main findings.

## 1. Introduction

Cover:

- importance of plant disease detection,
- deep learning for leaf image classification,
- limitation of clean PlantVillage-style evaluation,
- need for lightweight and robust models,
- need for explainability,
- contribution statement.

## 2. Related Work

### 2.1 Plant Disease Classification

CNN and transfer learning studies on PlantVillage-style datasets.

### 2.2 Lightweight CNN Models

ResNet baseline, MobileNetV2, MobileNetV3-Small, EfficientNet-B0.

### 2.3 Modern Hybrid Lightweight Models

MobileViT-XS as optional exploratory model.

### 2.4 Explainable AI

Grad-CAM and visual explanation for image classifiers.

### 2.5 Robustness and Domain Shift

Clean dataset vs real-world-like image degradation.

## 3. Methodology

- dataset preparation,
- corrupted test generation,
- model architectures,
- training protocol,
- evaluation metrics,
- Grad-CAM protocol.

## 4. Results

Tables:

1. dataset distribution,
2. clean test performance,
3. efficiency comparison,
4. corrupted test performance,
5. robustness drop,
6. Grad-CAM qualitative analysis.

Figures:

- sample images,
- corrupted examples,
- confusion matrices,
- training curves,
- Grad-CAM examples.

## 5. Discussion

Answer RQs:

- lightweight vs baseline,
- accuracy-efficiency trade-off,
- robustness drop,
- Grad-CAM explanation stability,
- MobileViT-XS exploratory findings.

## 6. Limitations

- dataset is still PlantVillage-style,
- corrupted test simulates but does not fully replace real field data,
- Grad-CAM is qualitative,
- no real mobile app deployment,
- MobileViT-XS may be optional.

## 7. Future Work

- PlantDoc or real farm dataset,
- YOLO/segmentation with annotations,
- quantization and pruning,
- real mobile deployment,
- expert validation of Grad-CAM.
