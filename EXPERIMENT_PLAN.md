# Experiment Plan

This file defines the exact experimental protocol. All models must be trained and evaluated under identical conditions.

## Models

| Model | Source | Pretrained | Role |
|---|---|---|---|
| ResNet50 | torchvision | ImageNet | strong baseline / heavy CNN reference |
| EfficientNet-B0 | timm | ImageNet | efficient CNN candidate |
| MobileNetV2 | torchvision | ImageNet | lightweight CNN candidate |
| MobileNetV3-Small | torchvision | ImageNet | ultra-lightweight CNN candidate |
| MobileViT-XS | timm | ImageNet | optional exploratory modern hybrid model |

Do not train MobileViT-XS before the four required models are working.

## Training Protocol

Use the same protocol for all required models.

```text
Image size: 224 x 224
Batch size: 32
Epochs: 30 with early stopping
Optimizer: Adam
Initial learning rate: 1e-4
Fine-tuning learning rate: 1e-5
LR scheduler: ReduceLROnPlateau
Loss: Weighted CrossEntropyLoss
Seed: 42
Primary validation metric: macro F1-score
```

## Two-Phase Transfer Learning

### Phase 1 — Head Only

- Load ImageNet pretrained model.
- Replace classification head with a 10-class output layer.
- Freeze backbone.
- Train classification head only.

### Phase 2 — Fine-Tuning

- Unfreeze last blocks/layers.
- Continue training with smaller learning rate.
- Monitor validation macro F1-score.
- Save the best checkpoint.

## Data Augmentation

Training split only:

```text
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=15)
RandomResizedCrop(224, scale=(0.8, 1.0))
ColorJitter(brightness=0.2, contrast=0.2)
Normalize(ImageNet mean/std)
```

Validation/test:

```text
Resize(256)
CenterCrop(224)
Normalize(ImageNet mean/std)
```

Corrupted test:

- no random augmentation,
- only deterministic corruptions from `ROBUSTNESS_GUIDE.md`.

## Required Evaluations

For each model:

1. clean test evaluation,
2. corrupted/noisy test evaluation,
3. efficiency benchmark,
4. Grad-CAM analysis.

## Classification Metrics

- accuracy,
- macro precision,
- macro recall,
- macro F1-score,
- per-class precision,
- per-class recall,
- per-class F1-score,
- confusion matrix.

Primary metric:

```text
macro F1-score
```

## Efficiency Metrics

- number of parameters,
- FLOPs/MACs,
- model checkpoint size in MB,
- CPU inference latency,
- GPU inference latency if available,
- throughput images/second.

Optional:

- ONNX Runtime CPU latency.

## Robustness Metrics

- clean macro F1,
- corrupted macro F1,
- absolute robustness drop,
- relative robustness drop.

## Grad-CAM Protocol

Generate Grad-CAM for:

- correct predictions,
- wrong predictions,
- representative images per class,
- clean vs corrupted versions of the same image,
- cross-model comparison using the same images.

For MobileViT-XS, apply Grad-CAM to an appropriate convolutional feature layer.

## Output Files

```text
checkpoints/<model_name>/best.pth
checkpoints/<model_name>/last.pth
results/experiments.csv
results/model_comparison.csv
results/robustness_comparison.csv
results/confusion_matrices/<model_name>_cm.png
results/classification_reports/<model_name>_report.csv
results/training_curves/<model_name>_loss.png
results/training_curves/<model_name>_accuracy.png
results/gradcam/<model_name>/
```
