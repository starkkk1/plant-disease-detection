# Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification

## Project Title

**Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification using Grad-CAM**

---

## Short Description

This project investigates lightweight CNN models for tomato leaf disease classification and applies Grad-CAM to explain model predictions. The goal is to evaluate the trade-off between accuracy, model size, inference speed, and interpretability — not just maximize accuracy.

---

## Research Direction (RBL)

Three focused pillars:

1. **Classification** — Classify tomato leaf images across 10 classes: 9 disease categories + 1 healthy class.
2. **Lightweight models** — Compare ResNet50, EfficientNet-B0, MobileNetV2, MobileNetV3-Small.
3. **Explainability** — Use Grad-CAM to verify that models focus on diseased leaf regions, not background.

---

## Dataset

**Primary:** Kaggle — [Plant Disease Dataset by rashidthihan](https://www.kaggle.com/datasets/rashidthihan/plant-disease-dataset)

This dataset is a curated split of PlantVillage. It is pre-organized into `train/`, `valid/`, `test/` folders. Only the **Tomato** subset is used in this project.

Target classes (10 total: 9 disease + 1 healthy):

| Index | Class Name                          |
|-------|-------------------------------------|
| 0     | Tomato___Bacterial_spot             |
| 1     | Tomato___Early_blight               |
| 2     | Tomato___Late_blight                |
| 3     | Tomato___Leaf_Mold                  |
| 4     | Tomato___Septoria_leaf_spot         |
| 5     | Tomato___Spider_mites               |
| 6     | Tomato___Target_Spot                |
| 7     | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| 8     | Tomato___Tomato_mosaic_virus        |
| 9     | Tomato___healthy                    |

> **Note:** Class folder names must match exactly as they appear in the downloaded dataset. Update `configs/default.yaml → classes` if folder names differ slightly.

**No external dataset** is used unless explicitly decided and documented in `reports/experiment_notes.md`.

---

## Research Problem

> How can lightweight deep learning models classify tomato leaf diseases effectively while remaining computationally efficient and interpretable?

---

## Scope Boundaries

### IN scope
- Tomato leaf disease image classification (10 total classes: 9 disease + 1 healthy)
- Transfer learning with pretrained CNNs
- Lightweight model comparison (accuracy vs. size vs. speed)
- Grad-CAM explainability analysis
- Research paper writing

### OUT of scope (do NOT drift here)
- Medical image classification
- Object detection or bounding box prediction
- Image segmentation (mention in Future Work only)
- LLMs, RAG, or any NLP task
- Training models from scratch
- Any dataset other than the one specified above
