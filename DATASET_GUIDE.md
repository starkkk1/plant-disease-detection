# Dataset Guide

---

## Source

**Kaggle:** https://www.kaggle.com/datasets/rashidthihan/plant-disease-dataset

Download command (requires Kaggle API key configured):

```bash
kaggle datasets download -d rashidthihan/plant-disease-dataset
unzip plant-disease-dataset.zip -d data/raw/
# After extraction, inspect the exact top-level folder name.
# If it is not data/raw/plant-disease-dataset/, update configs/default.yaml accordingly.
```

---

## Expected Raw Structure After Download

The dataset is expected to be pre-split, but the exact extracted top-level folder name may differ depending on Kaggle packaging. After extraction, verify it matches:

```
data/raw/plant-disease-dataset/
├── train/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___Leaf_Mold/
│   ├── Tomato___Septoria_leaf_spot/
│   ├── Tomato___Spider_mites/
│   ├── Tomato___Target_Spot/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   ├── Tomato___Tomato_mosaic_virus/
│   ├── Tomato___healthy/
│   └── [other plant classes — ignored]
├── valid/
│   └── [same structure]
└── test/
    └── [same structure]
```

> If the dataset is NOT pre-split (all images in one folder), run:
> `python scripts/prepare_dataset.py --split`
> This will create the train/valid/test split automatically using 70/15/15 ratio.

---

## Tomato Filter Step

The dataset contains images from many plant types (Apple, Corn, Grape, etc.). This project uses **Tomato only**.

The `scripts/prepare_dataset.py` script will:

1. Scan all class folders in `data/raw/`.
2. Keep only folders that start with `Tomato___`.
3. Copy them to `data/processed/tomato/train/`, `val/`, `test/`.

The raw dataset may use `valid/`, while this project standardizes the processed folder name to `val/`.

After this step, all downstream code reads from `data/processed/tomato/` only.

---

## Processed Structure (Used by All Code)

```
data/processed/tomato/
├── train/
│   ├── Tomato___Bacterial_spot/        (~2000 images)
│   ├── Tomato___Early_blight/          (~1000 images)
│   ├── Tomato___Late_blight/           (~1900 images)
│   ├── Tomato___Leaf_Mold/             (~950 images)
│   ├── Tomato___Septoria_leaf_spot/    (~1770 images)
│   ├── Tomato___Spider_mites/          (~1670 images)
│   ├── Tomato___Target_Spot/           (~1400 images)
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/ (~5350 images)
│   ├── Tomato___Tomato_mosaic_virus/   (~370 images)
│   └── Tomato___healthy/              (~1590 images)
├── val/
│   └── [same 10 folders, fewer images]
└── test/
    └── [same 10 folders, fewer images]
```

> Image counts above are approximate. Run `scripts/prepare_dataset.py --stats` to get exact counts and save to `reports/dataset_summary.md`.

---

## Class Index Mapping

The class index is determined alphabetically by `torchvision.datasets.ImageFolder`. The mapping will be auto-generated and saved to `data/processed/tomato/class_to_idx.json` during dataset preparation.

Approximate mapping:

```
0: Tomato___Bacterial_spot
1: Tomato___Early_blight
2: Tomato___Late_blight
3: Tomato___Leaf_Mold
4: Tomato___Septoria_leaf_spot
5: Tomato___Spider_mites
6: Tomato___Target_Spot
7: Tomato___Tomato_Yellow_Leaf_Curl_Virus
8: Tomato___Tomato_mosaic_virus
9: Tomato___healthy
```

> Always load `class_to_idx.json` for display labels. Never hardcode class indices in evaluation or Grad-CAM code.

---

## Class Imbalance Check

Run after preparation:

```bash
python scripts/prepare_dataset.py --stats
```

Expected imbalance: `Tomato___Tomato_mosaic_virus` has significantly fewer images than `Tomato___Tomato_Yellow_Leaf_Curl_Virus`.

Decision: Use `class_weights` in CrossEntropyLoss. This is computed automatically in `src/training/trainer.py`.

---

## Data Path Configuration

All code reads the dataset path from `configs/default.yaml`:

```yaml
data:
  root: data/processed/tomato
  train_dir: data/processed/tomato/train
  val_dir: data/processed/tomato/val
  test_dir: data/processed/tomato/test
  class_map: data/processed/tomato/class_to_idx.json
```

**Never hardcode paths in `src/` or `notebooks/`.**

---

## What NOT to Do

- Do not use images from non-Tomato classes.
- Do not mix raw and processed data paths.
- Do not re-split the data if it is already pre-split by the dataset source.
- Do not apply augmentation to `val/` or `test/` folders.
