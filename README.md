# Robust, Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification

## Local phone demo

The Sprint 01 MVP now includes a responsive Vietnamese phone UI and a standalone FastAPI CPU classification service. It runs without Qdrant and uses a verified real MobileNet checkpoint. From the repository root with the ML environment activated, start `python scripts/run_api.py --host 0.0.0.0 --port 8000`, then run `npm run dev` in `frontend/`. The API launcher resolves the project directory automatically. See [the run guide](RUN_GUIDE.md) for commands that also work from other directories, the verified Python environment, LAN demo, tests and configuration.

[API contract](docs/api-contract.md) · [Mobile plan](docs/mobile-plan.md) · [Architecture](docs/architecture.md) · [AWS readiness](docs/aws-readiness.md) · [Issues #4–#9 delivery](docs/sprint-01-app-delivery.md)

The local API/UI are verified with real weights; cloud readiness still needs an authenticated company AWS profile, permitted Region, named cost approver and budget. No AWS resources were created.

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
| 5 | Tomato___Spider_mites Two-spotted_spider_mite |
| 6 | Tomato___Target_Spot |
| 7 | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| 8 | Tomato___Tomato_mosaic_virus |
| 9 | Tomato___healthy |
| 10 | unknown |

The authoritative order is `data/new-data-removal/class_to_idx.json`, verified against the current training `ImageFolder.class_to_idx`. Checkpoints contain weights only, so retain the verified class map with every deployed artifact. See [the model audit](docs/sprint-01-model-audit.md).

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
   
2. **Evaluation (`scripts/evaluate.py`)**
   - Requires an explicit trained checkpoint and labeled dataset directory.
   - Reuses inference preprocessing and the training class map, including for subset datasets.
   - Saves **Accuracy, Macro F1, Precision, Recall**, confusion matrix counts and per-image predictions as JSON.

3. **Benchmarking (`scripts/benchmark.py`)**
   - Loads trained weights and measures checkpoint size (MiB), parameter count and p50/p95 CPU latency on real image files, including preprocessing.

4. **Explainability (`scripts/evaluate.py --gradcam`)**
   - Optional Grad-CAM++ pass with the same transform, outside classification's inference mode.
   - The original implementation remains in `src/models/inference_model.py`; the previous research evaluation workflow is preserved as `scripts/evaluate_gradcam_legacy.py`.

5. **Reproducible Sprint 01 baseline (`scripts/audit_models.py`)**
   - Verifies the current training class map, strictly loads all five checkpoints, benchmarks CPU inference and evaluates the complete `test` and external `eval` splits.
   - Saves evidence in `reports/sprint-01/`. See [the baseline report](docs/sprint-01-baseline.md) and [run commands](RUN_GUIDE.md).

## Key Findings

- **MobileNetV3-Small** checkpoints are approximately 5.97 MiB. Measured CPU latency depends on hardware, threads and image decoding; use the Sprint 01 p50/p95 baseline instead of the previous randomly initialized model benchmark.
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
