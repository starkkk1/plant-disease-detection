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

## 2. Evaluation & Confusion Matrix
To evaluate all trained models on the test set and automatically generate Confusion Matrices:
```bash
python scripts/eval_external.py
```
**Outputs:** 
- Metrics (Loss, Accuracy, F1, Precision, Recall) will be printed directly in the terminal.
- Confusion Matrix images (`.png`) are automatically saved in the `results/confusion_matrix/` directory.

## 3. Benchmarking (Latency & Size)
To measure model size (MB), parameter count (M), and inference latency (ms) on CPU:
```bash
python scripts/benchmark.py
```
**Outputs:** 
- A formatted table printed in the terminal comparing Teacher and Student models.

## 4. Explainability (Grad-CAM)
To generate visual heatmaps showing what the models are focusing on when making predictions:
```bash
python scripts/run_gradcam.py
```
**Outputs:** 
- Heatmap images saved in `results/gradcam/`.
- The images are categorized by model and separated into folders based on prediction correctness (e.g., `per_class`, `correct`, `wrong`).
