# Grad-CAM Guide

## Purpose

Grad-CAM is used to inspect whether a model focuses on disease-relevant leaf regions or irrelevant background regions.

This is not just for visualization. It directly supports the explainability and failure analysis parts of the research.

## Library

Use:

```text
pytorch-grad-cam
```

Do not write a custom Grad-CAM implementation unless the library fails.

## Target Layers

Use the last suitable convolutional feature layer of each model.

Suggested target layers:

| Model | Suggested Target Layer |
|---|---|
| ResNet50 | `model.layer4[-1]` |
| EfficientNet-B0 | final convolution block from timm model |
| MobileNetV2 | last feature block in `model.features` |
| MobileNetV3-Small | last feature block in `model.features` |
| MobileViT-XS | appropriate convolutional feature layer before/around hybrid blocks |

Codex must verify target layer names programmatically.

## Required Grad-CAM Outputs

For each model:

```text
results/gradcam/<model_name>/correct/
results/gradcam/<model_name>/wrong/
results/gradcam/<model_name>/per_class/
results/gradcam/<model_name>/clean_vs_corrupted/
results/gradcam/<model_name>/cross_model/
```

## Image Selection

Generate Grad-CAM for:

- 5 correct predictions per class if possible,
- wrong predictions,
- at least 1 representative image per class,
- the same selected images across all models,
- clean image and corrupted version of the same image.

## Naming Convention

Use filenames like:

```text
true_<true_label>__pred_<pred_label>__conf_<confidence>__idx_<image_index>.png
```

Example:

```text
true_Tomato___Early_blight__pred_Tomato___Late_blight__conf_0.82__idx_0042.png
```

## Qualitative Analysis Labels

For each heatmap, classify focus quality:

| Label | Meaning |
|---|---|
| Good | Focus mainly on visible diseased leaf regions |
| Partial | Focus partly on disease region and partly elsewhere |
| Poor | Focus mostly on background, border, or irrelevant area |
| Unclear | Heatmap is too diffuse or ambiguous |

Save notes to:

```text
reports/gradcam_analysis.md
```

## Clean vs Corrupted Analysis

For selected images, compare:

- Grad-CAM on clean input,
- Grad-CAM on corrupted input,
- whether focus shifts from lesion/leaf to background/noise.

## Important Warning

Grad-CAM is a post-hoc explanation method.

It suggests influential regions but does not prove true causal reasoning. The report must not claim that Grad-CAM fully explains the model.
