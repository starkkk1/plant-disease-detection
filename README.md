# Robust, Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification

## Project Title

**Robust, Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification**

## Short Description

This project builds a research-based deep learning pipeline for tomato leaf disease classification. It compares a strong CNN baseline with lightweight models and one exploratory modern hybrid model. The project evaluates not only classification performance, but also computational efficiency, robustness under corrupted/noisy image conditions, and explainability using Grad-CAM.

The goal is not simply to maximize accuracy on clean PlantVillage-style images. The goal is to understand which model provides the best trade-off among:

- classification accuracy,
- macro F1-score,
- model size,
- FLOPs/MACs,
- inference latency,
- robustness under degraded images,
- Grad-CAM explanation quality.

## Research Direction

The project has four focused pillars:

1. **Classification** — classify tomato leaf images into 10 total classes: 9 disease categories + 1 healthy class.
2. **Lightweight comparison** — compare ResNet50 baseline with EfficientNet-B0, MobileNetV2, and MobileNetV3-Small.
3. **Robustness evaluation** — compare clean test performance with corrupted/noisy test performance.
4. **Explainability** — use Grad-CAM to check whether models focus on disease-relevant leaf regions or irrelevant background patterns.

Optional exploratory extension:

- **MobileViT-XS** as a modern lightweight CNN-Transformer hybrid model, if time and hardware allow.

## Dataset

Primary dataset:

- Kaggle Plant Disease Dataset by rashidthihan
- Project subset: tomato classes only
- Dataset style: PlantVillage-style folder structure

The raw dataset may contain many plant species. This project filters only folders starting with `Tomato___`.

Expected tomato classes:

| Index | Class Name |
|---:|---|
| 0 | Tomato___Bacterial_spot |
| 1 | Tomato___Early_blight |
| 2 | Tomato___Late_blight |
| 3 | Tomato___Leaf_Mold |
| 4 | Tomato___Septoria_leaf_spot |
| 5 | Tomato___Spider_mites |
| 6 | Tomato___Target_Spot |
| 7 | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| 8 | Tomato___Tomato_mosaic_virus |
| 9 | Tomato___healthy |

Before training, verify:

- exact extracted folder name,
- class folder names,
- train/valid/test split,
- number of images per class,
- whether `valid/` should be mapped to `val/`.

## Clean and Corrupted Test Sets

The project uses two evaluation settings:

1. **Clean test set** — original tomato test images.
2. **Corrupted/noisy test set** — generated from clean test images to simulate real-world-like image conditions.

Suggested corruptions:

- brightness change,
- contrast shift,
- Gaussian noise,
- motion blur,
- defocus blur,
- JPEG compression,
- shadow simulation,
- mild occlusion.

The corrupted test set must be used only for evaluation, not training.

## Main Models

| Model | Role |
|---|---|
| ResNet50 | Strong baseline / heavy CNN reference |
| EfficientNet-B0 | Efficient CNN with strong accuracy-efficiency balance |
| MobileNetV2 | Lightweight CNN for mobile/edge-oriented environments |
| MobileNetV3-Small | Ultra-lightweight CNN optimized for low-resource devices |
| MobileViT-XS | Optional exploratory modern lightweight CNN-Transformer hybrid |

## Research Questions

**RQ1:** Can lightweight CNN models achieve competitive classification performance compared to ResNet50 while significantly reducing computational cost?

**RQ2:** How does model performance degrade when evaluated on corrupted/noisy tomato leaf images?

**RQ3:** Do Grad-CAM explanations remain focused on disease-relevant regions under clean and corrupted image conditions?

**RQ4 optional:** As an exploratory comparison, does MobileViT-XS offer better robustness or more interpretable Grad-CAM heatmaps than traditional lightweight CNNs?

## In Scope

- Tomato leaf disease image classification.
- Transfer learning with pretrained models.
- Clean vs corrupted test evaluation.
- Efficiency benchmarking.
- Grad-CAM explainability.
- Research report and paper-style discussion.
- Optional Streamlit demo after the research pipeline is complete.

## Out of Scope

- YOLO-based lesion detection.
- Bounding box annotation.
- Semantic segmentation.
- Full mobile application development.
- Treatment recommendation system.
- Federated learning.
- Hyperspectral image analysis.
- Time-series monitoring.
- LLM/RAG/NLP components.
