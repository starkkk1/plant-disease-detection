# Robust, Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification

## Project Title

**Robust, Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification**

## Short Description

This project builds a research-based deep learning pipeline for tomato leaf disease classification. It explores the use of **Knowledge Distillation (KD)** to train highly efficient and lightweight models (EfficientNet-B0 and MobileNetV3-Small) under the guidance of a powerful teacher model (ConvNeXt-Tiny). 

The project evaluates not only classification performance but also computational efficiency (latency, model size), robustness under various data augmentations (MixUp, CutMix), and explainability using Grad-CAM. The goal is to find the optimal trade-off between high accuracy and deployability on low-resource edge devices.

## Research Direction

The project focuses on the following pillars:

1. **Classification**: Classifying tomato leaf images into 11 disease and healthy categories.
2. **Knowledge Distillation**: Transferring knowledge from a heavy, high-performing Teacher model to lightweight Student models.
3. **Efficiency Benchmarking**: Measuring model size (MB), parameter count (M), and CPU inference latency (ms).
4. **Explainability**: Using Grad-CAM to visualize whether the models are focusing on disease-relevant leaf symptoms (e.g., mosaic patterns, bacterial spots) or irrelevant backgrounds.

## Dataset

Primary dataset: Kaggle Plant Disease Dataset (Tomato subset).

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
| 10 | (Additional 11th class from recent dataset augmentation) |

## Main Models & Knowledge Distillation

| Model | Role |
|---|---|
| **ConvNeXt Tiny** | **Teacher Model**: A heavy, state-of-the-art vision model providing soft labels and feature representations. |
| **EfficientNet-B0** | **Student Model 1**: An efficient CNN balancing accuracy and computational cost. Trained both independently and via distillation. |
| **MobileNetV3-Small** | **Student Model 2**: An ultra-lightweight CNN optimized for mobile/low-resource devices. Trained both independently and via distillation. |

### Data Augmentation
Advanced augmentation techniques such as **MixUp** and **CutMix** are utilized during the distillation process to improve model robustness and generalization, ensuring the students learn effectively from the teacher.

## Project Structure & Scripts

The pipeline is fully automated through the following core scripts:

1. **Training & Distillation**
   - Controlled via YAML config files in the `configs/` directory (`convnext.yaml`, `efficientnet_b0.yaml`, `distillation_mobilenetv3.yaml`, etc.).
   
2. **Evaluation (`scripts/eval_external.py`)**
   - Automatically scans the `checkpoints/` directory for `.pth` files.
   - Evaluates models on the Test and Validation sets.
   - Computes comprehensive metrics: **Accuracy, Macro F1-Score, Loss, Precision, Recall**.
   - Automatically plots and saves **Confusion Matrices** into `results/confusion_matrix/`.

3. **Benchmarking (`scripts/benchmark.py`)**
   - Measures and compares the physical size (MB), total parameters (Millions), and inference latency (ms) on CPU.

4. **Explainability (`scripts/run_gradcam.py`)**
   - Generates visual heatmaps (Grad-CAM) for images to interpret the model's spatial focus.
   - Outputs visualizations grouped by model and prediction correctness into `results/gradcam/`.

## Key Findings

- **MobileNetV3-Small** achieves extreme efficiency (~7ms latency on CPU, <6MB size) making it ideal for mobile deployment.
- **Knowledge Distillation** combined with carefully tuned hyperparameters (Temperature, Alpha) can help lightweight models overcome architectural limitations.
- **Grad-CAM Analysis** reveals that models often struggle with diseases involving large-scale texture changes (e.g., *Tomato Mosaic Virus*), sometimes focusing on noisy backgrounds or misclassifying them as localized spot diseases.

## In Scope

- Tomato leaf disease image classification.
- Knowledge Distillation (Teacher-Student paradigm).
- Advanced Augmentation (MixUp, CutMix).
- Efficiency benchmarking (Latency, Parameters, Size).
- Grad-CAM explainability and misclassification analysis.

## Out of Scope

- Bounding box annotation & Object detection (e.g., YOLO).
- Semantic segmentation.
- Full mobile application development.
