# Experiment Plan

This file defines the exact experimental protocol. All models must be trained and evaluated under **identical conditions** to ensure fair comparison.

---

## Models

| Model             | Source          | Pretrained | Params (approx) |
|-------------------|-----------------|------------|-----------------|
| ResNet50          | torchvision     | ImageNet   | ~25.6M          |
| EfficientNet-B0   | timm            | ImageNet   | ~5.3M           |
| MobileNetV2       | torchvision     | ImageNet   | ~3.4M           |
| MobileNetV3-Small | torchvision     | ImageNet   | ~2.5M           |

ResNet50 is the **baseline** (heavy, high accuracy). The others are the **lightweight candidates**.

---

## Training Protocol (Same for All Models)

```
Image size:       224 x 224
Batch size:       32
Epochs:           30 (with early stopping, patience=5)
Optimizer:        Adam
Learning rate:    1e-4
LR scheduler:     ReduceLROnPlateau (factor=0.5, patience=3)
Loss:             CrossEntropyLoss (with class weights for imbalance)
Seed:             42
```

### Two-phase training

**Phase 1 — Head only (5 epochs)**
- Freeze all backbone layers.
- Train only the final classification head.
- This warms up the new head without destroying pretrained features.

**Phase 2 — Fine-tune (remaining epochs)**
- Unfreeze the last 2 blocks of the backbone.
- Continue training with a lower LR (1e-5).
- Early stopping monitors validation F1-score (not accuracy).

> This two-phase approach is the same for all 4 models. Do not skip Phase 1.

---

## Data Augmentation (Training Split Only)

```
RandomHorizontalFlip(p=0.5)
RandomRotation(degrees=15)
ColorJitter(brightness=0.2, contrast=0.2)
RandomResizedCrop(224, scale=(0.8, 1.0))
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Validation and Test — No augmentation:**

```
Resize(256)
CenterCrop(224)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## Evaluation Metrics

### Classification (per model, per class)

| Metric           | Tool                        |
|------------------|-----------------------------|
| Accuracy         | torchmetrics / sklearn      |
| Precision        | macro average               |
| Recall           | macro average               |
| F1-score         | macro average (primary)     |
| Confusion matrix | sklearn + matplotlib        |

> Primary selection metric is **macro F1-score** to account for class imbalance.

### Efficiency (per model)

| Metric              | Method                                       |
|---------------------|----------------------------------------------|
| Number of parameters| `sum(p.numel() for p in model.parameters())` |
| Model file size     | `.pth` file size in MB                       |
| Inference time      | Average over 100 test images (CPU + GPU)     |

---

## Output Files Per Experiment

Each training run must produce:

```
checkpoints/<model_name>/
├── best.pth              # best checkpoint by val F1
└── last.pth              # last epoch checkpoint

results/training_curves/<model_name>/
├── accuracy.png
└── loss.png

results/confusion_matrices/
└── <model_name>_confusion_matrix.png

results/gradcam/<model_name>/
├── correct/              # correct prediction Grad-CAM samples
├── wrong/                # wrong prediction Grad-CAM samples
└── per_class/            # one representative per class
```

All numeric results are appended to `results/experiments.csv` with columns:

```
run_id, model, date, epochs_run, best_val_f1, test_accuracy, test_precision,
test_recall, test_f1, num_params, model_size_mb, inference_time_ms_cpu,
inference_time_ms_gpu, notes
```

---

## Grad-CAM Protocol

- Target layer: the last convolutional layer of each model's backbone.
- Use `pytorch-grad-cam` library (`GradCAM` class).
- Generate heatmaps for:
  - 5 correct predictions per class (50 total per model)
  - All misclassified images (logged, visualized selectively)
  - Cross-model comparison: same input image through all 4 models
- Save overlay images (original + heatmap superimposed).
- Qualitative analysis notes go in `reports/experiment_notes.md`.

---

## Reproducibility Checklist

Before starting any training run:

- [ ] `SEED = 42` set via `src/utils/seed.py`
- [ ] Config file confirmed correct for this model
- [ ] Dataset path verified pointing to `data/processed/tomato/`
- [ ] Previous checkpoint not overwriting accidentally (use `run_id`)
- [ ] GPU/CPU logged at start of training script

---

## Running an Experiment

```bash
# Single model
python scripts/train_model.py --config configs/mobilenet_v2.yaml

# All 4 models in sequence
python scripts/run_all_experiments.py

# Evaluate a saved checkpoint
python scripts/evaluate_model.py --config configs/mobilenet_v2.yaml --checkpoint checkpoints/mobilenet_v2/best.pth

# Generate Grad-CAM
python scripts/generate_gradcam.py --config configs/mobilenet_v2.yaml --checkpoint checkpoints/mobilenet_v2/best.pth
```
