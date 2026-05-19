# Risk Register

This file lists common project risks and how to handle them.

## Risk 1: Dataset Folder Name Differs

Problem:

The Kaggle zip may extract into a folder name different from `plant-disease-dataset`.

Mitigation:

- Inspect `data/raw/`.
- Update `configs/default.yaml -> data.raw_root`.
- Do not hardcode path assumptions in scripts.

## Risk 2: Raw Split Uses `valid`, Project Uses `val`

Problem:

Kaggle may use `valid/`, while this project standardizes to `val/`.

Mitigation:

- `prepare_dataset.py` must map `valid -> val`.

## Risk 3: Class Count Confusion

Problem:

Tomato subset has 10 total classes, not 10 disease classes plus healthy.

Mitigation:

- Use wording: "10 total classes: 9 disease + 1 healthy".
- Keep `model.num_classes = 10`.

## Risk 4: Class Imbalance

Problem:

Some diseases may have fewer images.

Mitigation:

- Use macro F1.
- Use class weights.
- Report class distribution.

## Risk 5: High Accuracy but Bad Grad-CAM

Problem:

Model may focus on background or leaf edge.

Mitigation:

- Analyze correct and wrong predictions.
- Discuss shortcut learning.
- Do not claim model is trustworthy based only on accuracy.

## Risk 6: Overfitting

Problem:

Models may overfit PlantVillage-style images.

Mitigation:

- Use validation F1 early stopping.
- Use augmentation.
- Report train/val gap.
- Mention external validation as future work.

## Risk 7: Codex Adds Unwanted Features

Problem:

Codex may add object detection, segmentation, LLM, web backend, or unrelated tools.

Mitigation:

- Always provide `CODEX_INSTRUCTIONS.md`.
- Use `CODEX_PROMPT.md`.
- Reject changes outside scope.
