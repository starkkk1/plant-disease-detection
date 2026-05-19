# Data Validation Checklist

Use this checklist immediately after downloading the Kaggle dataset.

## 1. Confirm Download

Expected command:

```bash
kaggle datasets download -d rashidthihan/plant-disease-dataset
unzip plant-disease-dataset.zip -d data/raw/
```

## 2. Inspect Top-Level Folder

Run:

```bash
find data/raw -maxdepth 3 -type d | head -50
```

Check whether the top-level folder is:

```txt
data/raw/plant-disease-dataset/
```

If not, update:

```txt
configs/default.yaml -> data.raw_root
```

## 3. Confirm Split Names

Expected raw split names:

```txt
train/
valid/
test/
```

The processed project standard is:

```txt
train/
val/
test/
```

So `prepare_dataset.py` must map:

```txt
valid -> val
```

## 4. Confirm Tomato Classes

Expected 10 total classes:

```txt
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

## 5. Count Images

After filtering tomato classes, generate:

```txt
reports/dataset_summary.md
```

It must include:

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|

## 6. Sanity Check Samples

For each class:

- Show 5 images.
- Confirm images are tomato leaves.
- Confirm no corrupted image files.
- Confirm labels match folder names.

## 7. Class Imbalance

Check whether some classes are much smaller.

If imbalance exists:

- Use macro F1 as primary metric.
- Use class weights in CrossEntropyLoss.
- Report class distribution in the paper.

## 8. Leakage Check

Make sure:

- No image exists in more than one split.
- Raw and processed folders are not mixed in training.
- Validation/test images are never augmented.
