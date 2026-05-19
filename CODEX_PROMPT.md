# Codex Prompt

Use this prompt when starting a new Codex session.

```txt
Read these files first:
1. README.md
2. CODEX_INSTRUCTIONS.md
3. PROJECT_STRUCTURE.md
4. DATASET_GUIDE.md
5. CONFIG_GUIDE.md
6. EXPERIMENT_PLAN.md
7. RESEARCH_QUESTIONS.md
8. EVALUATION_GUIDE.md
9. GRADCAM_GUIDE.md

You are implementing a Research-Based Learning project:
Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification using Grad-CAM.

Strict rules:
- Use PyTorch only.
- Use the Kaggle dataset rashidthihan/plant-disease-dataset.
- Use only the Tomato subset.
- Use image classification only.
- Use only these models first: ResNet50, EfficientNet-B0, MobileNetV2, MobileNetV3-Small.
- Use Grad-CAM only for explainability.
- Do not add RAG, LLM, medical imaging, object detection, or segmentation.
- Do not hardcode paths or hyperparameters.
- Read configs from YAML.
- Save all outputs to the folders defined in PROJECT_STRUCTURE.md.

First task:
Create the folder structure, config files, requirements.txt, .gitignore, and dataset preparation script.

Dataset preparation must:
- Downloaded data is expected under data/raw/.
- Detect the exact extracted folder name if possible.
- Keep only folders starting with Tomato___.
- Convert raw split valid/ to processed split val/.
- Save class_to_idx.json.
- Generate reports/dataset_summary.md.
```
