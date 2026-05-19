# Grad-CAM Guide

This file defines how Grad-CAM must be implemented and analyzed.

## 1. Purpose

Grad-CAM is used to inspect whether the model focuses on disease-relevant leaf regions or irrelevant background regions.

This is not just for pretty visualization. It directly supports the research questions.

## 2. Library

Use:

```txt
pytorch-grad-cam
```

Do not write a custom Grad-CAM implementation unless the library fails.

## 3. Target Layers

Use the last convolutional layer of each model.

Suggested target layers:

| Model | Suggested Target Layer |
|---|---|
| ResNet50 | `model.layer4[-1]` |
| EfficientNet-B0 | final convolution block from timm model |
| MobileNetV2 | last feature block in `model.features` |
| MobileNetV3-Small | last feature block in `model.features` |

Codex should verify target layer names programmatically.

## 4. Required Grad-CAM Outputs

For each model:

```txt
results/gradcam/<model_name>/correct/
results/gradcam/<model_name>/wrong/
results/gradcam/<model_name>/per_class/
results/gradcam/<model_name>/cross_model/
```

## 5. Image Selection

Generate Grad-CAM for:

- 5 correct predictions per class.
- All wrong predictions if not too many.
- At least 1 representative image per class.
- Same selected images across all 4 models for cross-model comparison.

## 6. Naming Convention

Use filenames like:

```txt
true_<class>__pred_<class>__conf_<score>__idx_<image_id>.png
```

Example:

```txt
true_Tomato___Early_blight__pred_Tomato___Late_blight__conf_0.82__idx_0042.png
```

## 7. Qualitative Analysis Labels

For each heatmap, classify focus quality manually or semi-manually:

| Label | Meaning |
|---|---|
| Good | Focus mainly on visible diseased leaf regions |
| Partial | Focus partly on disease region and partly elsewhere |
| Poor | Focus mostly on background, border, or irrelevant area |
| Unclear | Heatmap is too diffuse or ambiguous |

Save notes to:

```txt
reports/gradcam_analysis.md
```

## 8. Important Warning

Grad-CAM is a post-hoc explanation method. It suggests influential regions but does not prove true causal reasoning.

The paper must not claim that Grad-CAM fully explains the model.
