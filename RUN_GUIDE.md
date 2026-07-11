# Plant Disease Detection - Run Guide

This guide provides the quick commands needed to run the various scripts in this project. All commands should be run from the root directory of the project (`plant-disease-detection`).

## 1. Setup Environment
Ensure your virtual environment is activated and dependencies are installed.
```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 1.5. Training Models
To train the Teacher model (ConvNeXt):
```bash
python scripts/train_convnext.py
```

To run Knowledge Distillation for both Student models (MobileNetV3 and EfficientNet-B0) sequentially:
```bash
python scripts/train_distillation.py --config configs/distillation_mobilenetv3.yaml && python scripts/train_distillation.py --config configs/distillation_effnetb0.yaml
```
**Outputs:**
- Logs and metrics will be saved in `results/`.
- Best model checkpoints will be saved in `checkpoints/`.


## 2. Evaluation, Confusion Matrix & Explainability (Grad-CAM)
To evaluate all trained models on the test set, automatically generate Confusion Matrices, and generate visual heatmaps (Grad-CAM):
```bash
# Evaluate all models on test set (Default)
python scripts/evaluate.py

# Evaluate a specific model (e.g., convnext)
python scripts/evaluate.py --model convnext

# Evaluate a specific model on the 'eval' dataset instead of 'test'
python scripts/evaluate.py --dataset eval --model resnet
```
**Outputs:** 
- Metrics (Loss, Accuracy, F1, Precision, Recall) will be printed directly in the terminal.
- Confusion Matrix images (`.png`) are automatically saved in the `results/confusion_matrix/` directory.
- Grad-CAM Heatmap images are saved in `results/gradcam/`, categorized by model and separated into folders based on prediction correctness (e.g., `per_class`, `correct`, `wrong`).

## 3. Benchmarking (Latency & Size)
To measure model size (MB), parameter count (M), and inference latency (ms) on CPU:
```bash
python scripts/benchmark.py
```
**Outputs:** 
- A formatted table printed in the terminal comparing Teacher and Student models.

## 4. Build Qdrant Search Index (Vector Database)
To encode the dataset and push vectors into Qdrant for semantic search:
```bash
# For MobileNetV3 (Default)
python scripts/build_qdrant_index.py --model mobilenet

# For EfficientNet-B0
python scripts/build_qdrant_index.py --model efficientnet
```
**Outputs:** 
- Prints extraction progress and creates collections (`tomato_disease_multimodal` or `tomato_disease_efficientnet`) inside your Qdrant container.
