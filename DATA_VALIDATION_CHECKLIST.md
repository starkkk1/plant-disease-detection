# Data Validation Checklist

Use this checklist before training any model.

## Raw Dataset Checks

- [ ] Kaggle dataset downloaded successfully.
- [ ] ZIP extracted into `data/raw/`.
- [ ] Exact extracted folder name identified.
- [ ] Raw folder has train/valid/test or equivalent structure.
- [ ] Tomato folders exist.

## Tomato Filtering Checks

- [ ] Only folders starting with `Tomato___` are copied.
- [ ] Non-tomato classes are excluded.
- [ ] Exactly 10 tomato classes are found.
- [ ] `valid/` is mapped to `val/`.

## Processed Dataset Checks

- [ ] `data/processed/tomato/train/` exists.
- [ ] `data/processed/tomato/val/` exists.
- [ ] `data/processed/tomato/test/` exists.
- [ ] Each split has the same class folders.
- [ ] `class_to_idx.json` exists.
- [ ] `reports/dataset_summary.md` exists.

## Class Count Checks

- [ ] Number of images per class is reported.
- [ ] Minority classes are identified.
- [ ] Class imbalance is documented.
- [ ] Decision on class weights is documented.

## Corrupted Test Checks

- [ ] Clean test set exists first.
- [ ] Corrupted test set generated from clean test only.
- [ ] Corrupted test set preserves class folder structure.
- [ ] Corrupted images are not used in training.
- [ ] Corruption types are documented.

## Dataloader Checks

- [ ] Train dataloader returns `(B, 3, 224, 224)`.
- [ ] Validation dataloader returns `(B, 3, 224, 224)`.
- [ ] Test dataloader returns `(B, 3, 224, 224)`.
- [ ] Labels are in correct range `0..9`.
- [ ] Class names display correctly from `class_to_idx.json`.
