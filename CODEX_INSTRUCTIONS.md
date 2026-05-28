# Codex Instructions

This file defines strict rules for any AI assistant working on this project. Read this file before generating code.

## Project Identity

This is a Research-Based Learning project in deep learning.

- Task: image classification only.
- Subject: tomato leaf disease.
- Dataset: Kaggle Plant Disease Dataset by rashidthihan, tomato subset only.
- Main models: ResNet50, EfficientNet-B0, MobileNetV2, MobileNetV3-Small.
- Optional exploratory model: MobileViT-XS, only if time and hardware allow.
- Explainability: Grad-CAM.
- Framework: PyTorch.
- Additional evaluation: clean test vs corrupted/noisy test.

## Non-Negotiable Rules

### Rule 1 — Do not change the task type

This is image classification only.

Do not add:

- object detection,
- YOLO,
- bounding boxes,
- semantic segmentation,
- treatment recommendation,
- LLM/RAG/NLP.

### Rule 2 — Do not change the dataset scope

Use only the tomato subset from the specified Kaggle Plant Disease Dataset.

Do not use non-tomato classes for model training.

Optional external datasets can be mentioned only in reports/future work unless the user explicitly approves.

### Rule 3 — Keep model list controlled

Required models:

- ResNet50,
- EfficientNet-B0,
- MobileNetV2,
- MobileNetV3-Small.

Optional exploratory model:

- MobileViT-XS.

Do not add extra models without explicit approval.

### Rule 4 — Use PyTorch only

Do not rewrite the project in TensorFlow/Keras.

### Rule 5 — No hardcoded paths

All paths must come from:

- `configs/default.yaml`, or
- `src/utils/paths.py`.

Never hardcode local absolute paths such as `/home/...` or `C:\Users\...`.

### Rule 6 — Augmentation split rule

Training split:

- data augmentation allowed.

Validation/test split:

- resize + normalize only.
- no random augmentation.

Corrupted test set:

- generated from clean test images.
- used only for robustness evaluation.
- never used for training.

### Rule 7 — Always set seed

Use `SEED = 42` and call `set_seed(42)` at the start of training/evaluation scripts.

### Rule 8 — Save outputs consistently

Save outputs to the correct folders:

- metrics: `results/experiments.csv`,
- model comparison: `results/model_comparison.csv`,
- clean vs corrupted comparison: `results/robustness_comparison.csv`,
- confusion matrices: `results/confusion_matrices/`,
- training curves: `results/training_curves/`,
- Grad-CAM: `results/gradcam/`,
- checkpoints: `checkpoints/<model_name>/best.pth`.

### Rule 9 — One model, one config

Each model has its own YAML config file.

The training script reads from config. It must not take inline hyperparameters except for the config path.

### Rule 10 — Do not over-engineer

Do not add:

- distributed training,
- Docker,
- MLflow,
- complicated backend APIs,
- mobile application code,
- database/authentication,
- custom CUDA kernels.

The priority is research correctness and reproducibility.

## Required Reading Before Code

| Module | Read First |
|---|---|
| Dataset preparation | `DATASET_GUIDE.md`, `DATA_VALIDATION_CHECKLIST.md` |
| Configs | `CONFIG_GUIDE.md` |
| Model setup | `EXPERIMENT_PLAN.md` |
| Training | `EXPERIMENT_PLAN.md`, `IMPLEMENTATION_ORDER.md` |
| Evaluation | `EVALUATION_GUIDE.md` |
| Robustness | `ROBUSTNESS_GUIDE.md`, `DATASET_GUIDE.md` |
| Grad-CAM | `GRADCAM_GUIDE.md` |
| Project structure | `PROJECT_STRUCTURE.md` |

## When Unsure

If a requested task conflicts with these rules:

1. Stop.
2. Explain the conflict.
3. Ask for confirmation before coding.
