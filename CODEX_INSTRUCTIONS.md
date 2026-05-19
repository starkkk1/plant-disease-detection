# Codex Instructions

This file defines strict rules for any AI assistant (Codex, Copilot, ChatGPT, Claude) working on this project. Read this file before generating any code.

---

## Identity of This Project

This is a **Research-Based Learning (RBL)** project in deep learning.

- Task: image classification (not detection, not segmentation)
- Subject: tomato leaf disease
- Dataset: Kaggle Plant Disease Dataset by rashidthihan (tomato subset only)
- Models: ResNet50, EfficientNet-B0, MobileNetV2, MobileNetV3-Small
- Explainability: Grad-CAM only
- Framework: PyTorch

---

## Non-Negotiable Rules

### Rule 1 — Do not change the dataset
- Only use the Kaggle dataset specified in `README.md`.
- Do not switch to ImageNet, CIFAR, or any other dataset.
- Do not download or suggest a different dataset.

### Rule 2 — Do not change the model list
- Only use: ResNet50, EfficientNet-B0, MobileNetV2, MobileNetV3-Small.
- Optional extras (DenseNet121, MobileViT) require explicit human approval.
- Do not suggest Vision Transformers, YOLO, or segmentation models.

### Rule 3 — Do not change the framework
- Use PyTorch only.
- Do not rewrite code in TensorFlow or Keras.

### Rule 4 — Do not change the task type
- This is image classification only.
- Do not add object detection heads.
- Do not add segmentation masks.
- Do not add LLM or NLP components.

### Rule 5 — Do not hardcode paths
- All paths must come from `configs/default.yaml` or `src/utils/paths.py`.
- Never hardcode `/home/user/...` or `C:\Users\...` inside source code.

### Rule 6 — Do not mix train/val/test augmentation
- Data augmentation is applied to training split only.
- Validation and test use only resize + normalize.

### Rule 7 — Always set random seed
- Use `SEED = 42` everywhere.
- Call `src/utils/seed.py → set_seed(42)` at the start of every training script.

### Rule 8 — Save all results to the correct folder
- Metrics → `results/experiments.csv`
- Confusion matrices → `results/confusion_matrices/`
- Training curves → `results/training_curves/`
- Grad-CAM images → `results/gradcam/`
- Checkpoints → `checkpoints/<model_name>/best.pth`

### Rule 9 — One model, one config
- Each model has its own config file in `configs/`.
- The training script reads from config. It does not take inline hyperparameters.

### Rule 10 — Do not over-engineer
- No distributed training setup unless explicitly requested.
- No custom CUDA kernels.
- No reinforcement learning or meta-learning.
- Keep each function under 50 lines when possible.

---

## When You Are Unsure

If a requested task is ambiguous or seems to conflict with these rules:

1. Do not guess and implement.
2. State the conflict clearly.
3. Ask for clarification before writing code.

---

## File You Must Read Before Writing Code

Before generating any code for a module, read the corresponding file:

| Module              | Read First                     |
|---------------------|--------------------------------|
| Dataset loading     | `DATASET_GUIDE.md`             |
| Model setup         | `EXPERIMENT_PLAN.md`           |
| Training loop       | `configs/default.yaml`         |
| Grad-CAM            | `EXPERIMENT_PLAN.md` and `GRADCAM_GUIDE.md` |
| Evaluation          | `EXPERIMENT_PLAN.md` and `EVALUATION_GUIDE.md` |
| Paths               | `PROJECT_STRUCTURE.md` and `CONFIG_GUIDE.md` |
