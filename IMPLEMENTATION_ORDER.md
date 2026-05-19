# Implementation Order

Codex should implement the project in this exact order.

## Step 1: Repository Skeleton

Create:

- folders from `PROJECT_STRUCTURE.md`
- `.gitignore`
- `requirements.txt`
- config YAML files

Do not implement models yet.

## Step 2: Dataset Preparation

Implement:

- `scripts/prepare_dataset.py`
- `src/data/dataset.py`
- `src/data/transforms.py`

Success check:

```bash
python scripts/prepare_dataset.py --filter-tomato --stats
```

Expected outputs:

- `data/processed/tomato/train`
- `data/processed/tomato/val`
- `data/processed/tomato/test`
- `data/processed/tomato/class_to_idx.json`
- `reports/dataset_summary.md`

## Step 3: Model Factory

Implement:

- `src/models/model_factory.py`

Success check:

- Each of 4 models can be instantiated with `num_classes=10`.
- Forward pass works with dummy tensor `(2, 3, 224, 224)`.

## Step 4: Training Pipeline

Implement:

- `src/training/trainer.py`
- `src/training/losses.py`
- `scripts/train_model.py`

Success check:

```bash
python scripts/train_model.py --config configs/mobilenet_v3_small.yaml
```

## Step 5: Evaluation Pipeline

Implement:

- `src/evaluation/metrics.py`
- `src/evaluation/evaluate.py`
- `src/evaluation/inference_speed.py`
- `scripts/evaluate_model.py`

Success check:

- Metrics saved to `results/experiments.csv`.
- Confusion matrix saved.

## Step 6: Grad-CAM Pipeline

Implement:

- `src/explainability/gradcam.py`
- `src/explainability/visualize.py`
- `scripts/generate_gradcam.py`

Success check:

- Overlay images saved to `results/gradcam/<model_name>/`.

## Step 7: Run All Experiments

Implement:

- `scripts/run_all_experiments.py`

Success check:

- All 4 models can be trained/evaluated using their configs.
- `results/model_comparison.csv` exists.

## Step 8: Paper Outputs

Generate:

- tables for paper
- figures for paper
- `reports/experiment_notes.md`
- `reports/gradcam_analysis.md`
