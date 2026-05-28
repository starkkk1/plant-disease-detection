# Config Guide

All paths and hyperparameters must come from YAML config files.

## Required Config Files

```text
configs/default.yaml
configs/resnet50.yaml
configs/efficientnet_b0.yaml
configs/mobilenet_v2.yaml
configs/mobilenet_v3_small.yaml
configs/mobilevit_xs.yaml
```

## Default Config Template

```yaml
seed: 42

data:
  root: data/processed/tomato
  train_dir: data/processed/tomato/train
  val_dir: data/processed/tomato/val
  test_dir: data/processed/tomato/test
  corrupted_root: data/processed/tomato_corrupted
  class_map: data/processed/tomato/class_to_idx.json
  image_size: 224
  batch_size: 32
  num_workers: 4

model:
  name: mobilenet_v2
  num_classes: 10
  pretrained: true

training:
  epochs: 30
  head_epochs: 5
  lr: 1.0e-4
  fine_tune_lr: 1.0e-5
  optimizer: adam
  scheduler: reduce_on_plateau
  patience: 5
  use_class_weights: true
  selection_metric: val_f1_macro

augmentation:
  horizontal_flip: true
  rotation_degrees: 15
  random_resized_crop: true
  brightness: 0.2
  contrast: 0.2

output:
  checkpoint_dir: checkpoints
  results_dir: results
  reports_dir: reports
```

## Model Configs

Model-specific configs should override only:

```yaml
model:
  name: resnet50
```

Do not duplicate every hyperparameter unless necessary.

## Path Rule

Never hardcode paths inside `src/`.

All scripts must load paths from config.
