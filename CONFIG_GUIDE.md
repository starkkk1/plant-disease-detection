# Config Guide

This file defines the required YAML configuration system. Codex must create these files before implementing training scripts.

## Required Config Files

```txt
configs/
├── default.yaml
├── resnet50.yaml
├── efficientnet_b0.yaml
├── mobilenet_v2.yaml
└── mobilenet_v3_small.yaml
```

## `configs/default.yaml`

```yaml
project:
  name: lightweight-plant-disease
  seed: 42
  task: tomato_leaf_disease_classification

data:
  raw_root: data/raw/plant-disease-dataset
  processed_root: data/processed/tomato
  train_dir: data/processed/tomato/train
  val_dir: data/processed/tomato/val
  test_dir: data/processed/tomato/test
  class_map: data/processed/tomato/class_to_idx.json
  image_size: 224
  batch_size: 32
  num_workers: 4

model:
  name: resnet50
  num_classes: 10
  pretrained: true

training:
  epochs: 30
  head_epochs: 5
  lr: 1.0e-4
  fine_tune_lr: 1.0e-5
  optimizer: adam
  scheduler: reduce_on_plateau
  scheduler_factor: 0.5
  scheduler_patience: 3
  early_stopping_patience: 5
  monitor_metric: val_f1_macro
  use_class_weights: true

augmentation:
  train:
    random_horizontal_flip: true
    rotation_degrees: 15
    color_jitter_brightness: 0.2
    color_jitter_contrast: 0.2
    random_resized_crop: true
    crop_scale_min: 0.8
    crop_scale_max: 1.0
  val_test:
    resize: 256
    center_crop: 224

normalization:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

output:
  checkpoint_dir: checkpoints
  results_dir: results
  reports_dir: reports
  log_dir: results/logs
```

## Model-Specific Configs

Each model-specific config should override only the model name unless there is a documented reason.

### `configs/resnet50.yaml`

```yaml
defaults: configs/default.yaml
model:
  name: resnet50
```

### `configs/efficientnet_b0.yaml`

```yaml
defaults: configs/default.yaml
model:
  name: efficientnet_b0
```

### `configs/mobilenet_v2.yaml`

```yaml
defaults: configs/default.yaml
model:
  name: mobilenet_v2
```

### `configs/mobilenet_v3_small.yaml`

```yaml
defaults: configs/default.yaml
model:
  name: mobilenet_v3_small
```

## Rules

- Do not hardcode hyperparameters in Python source files.
- Training scripts must receive only `--config`.
- If raw Kaggle extraction path differs, update `data.raw_root`.
- If class folder names differ, do not manually rename in code; update dataset preparation logic and document the mapping.
