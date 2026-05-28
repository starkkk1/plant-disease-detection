# Dataset Guide

## Source

Primary dataset:

- Kaggle Plant Disease Dataset by rashidthihan
- Use tomato subset only.

Download command:

```bash
kaggle datasets download -d rashidthihan/plant-disease-dataset
unzip plant-disease-dataset.zip -d data/raw/
```

After extraction, inspect the exact top-level folder name. It may not be exactly `data/raw/plant-disease-dataset/`.

## Goal

Create this processed dataset:

```text
data/processed/tomato/
├── train/
├── val/
├── test/
└── class_to_idx.json
```

All training, evaluation, Grad-CAM, and robustness scripts must read from `data/processed/tomato/`.

## Expected Raw Structure

The dataset may be pre-split:

```text
data/raw/<extracted-folder>/
├── train/
├── valid/
└── test/
```

The raw folder may contain many plant species. Keep only class folders starting with:

```text
Tomato___
```

Expected tomato classes:

```text
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites
Tomato___Target_Spot
Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Tomato_mosaic_virus
Tomato___healthy
```

## Mapping `valid` to `val`

The raw dataset may use:

```text
valid/
```

This project standardizes it to:

```text
val/
```

Therefore:

- raw `train/` -> processed `train/`
- raw `valid/` -> processed `val/`
- raw `test/` -> processed `test/`

## Dataset Preparation Script

Implement:

```bash
python scripts/prepare_dataset.py --filter-tomato --stats
```

Expected outputs:

```text
data/processed/tomato/train/
data/processed/tomato/val/
data/processed/tomato/test/
data/processed/tomato/class_to_idx.json
reports/dataset_summary.md
```

## Dataset Summary

`reports/dataset_summary.md` must include:

- dataset source,
- raw folder path,
- processed folder path,
- number of classes,
- class names,
- train/val/test image counts per class,
- total images,
- class imbalance notes,
- corrupted test set generation status.

## Class Index Mapping

Use `torchvision.datasets.ImageFolder` to determine class indices.

Save the mapping to:

```text
data/processed/tomato/class_to_idx.json
```

Never hardcode class indices in evaluation or Grad-CAM scripts.

## Class Imbalance

After filtering, check image count per class.

Expected issue:

- some disease classes may have fewer images than others.

Decision:

- compute class weights from the training split,
- use weighted CrossEntropyLoss,
- report macro metrics and per-class F1.

## Corrupted Test Set

Generate a corrupted/noisy test set from the clean test set:

```text
data/processed/tomato_corrupted/
├── brightness/
├── contrast/
├── gaussian_noise/
├── blur/
├── jpeg/
└── shadow/
```

Each corruption folder should preserve the same class folder structure as the clean test set.

The corrupted test set must not be used for training.

## What Not To Do

- Do not train on non-tomato classes.
- Do not mix raw and processed paths.
- Do not re-split data if the Kaggle dataset already has train/valid/test.
- Do not apply random augmentation to validation or clean test folders.
- Do not include corrupted test images in training.
