# Codex Prompt

Read these files first:

1. `CODEX_INSTRUCTIONS.md`
2. `PROJECT_STRUCTURE.md`
3. `DATASET_GUIDE.md`
4. `DATA_VALIDATION_CHECKLIST.md`
5. `IMPLEMENTATION_ORDER.md`
6. `EXPERIMENT_PLAN.md`
7. `ROBUSTNESS_GUIDE.md`
8. `EVALUATION_GUIDE.md`
9. `GRADCAM_GUIDE.md`

You are implementing a research project titled:

**Robust, Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification**

Strict rules:

- Use PyTorch only.
- Task is image classification only.
- Use tomato subset only.
- Do not add YOLO, detection, segmentation, LLM, RAG, or mobile app backend.
- Required models: ResNet50, EfficientNet-B0, MobileNetV2, MobileNetV3-Small.
- MobileViT-XS is optional after the required models work.
- Implement in the exact order from `IMPLEMENTATION_ORDER.md`.
- Do not hardcode paths.
- Save all outputs to the documented folders.

Start with Step 1 and Step 2 only:

1. verify/create project structure,
2. implement dataset preparation and validation.

Do not implement training before dataset preparation passes all checks.
