# Implementation Order

Codex must implement the project in this exact order. Do not skip steps.

## Step 1 — Repository Skeleton

Create or verify:

- folders from `PROJECT_STRUCTURE.md`,
- `.gitignore`,
- `requirements.txt`,
- YAML config files,
- placeholder `.gitkeep` files when needed.

Do not implement models yet.

## Step 2 — Dataset Preparation

Implement:

- `scripts/prepare_dataset.py`,
- `src/data/dataset.py`,
- `src/data/transforms.py`.

Main command:

```bash
python scripts/prepare_dataset.py --filter-tomato --stats
```

Expected outputs:

```text
data/processed/tomato/train
data/processed/tomato/val
data/processed/tomato/test
data/processed/tomato/class_to_idx.json
reports/dataset_summary.md
```

Success checks:

- exactly 10 tomato classes,
- no non-tomato classes,
- `valid` is mapped to `val`,
- dataset summary exists.

## Step 3 — Corrupted Test Set Generation

Implement:

- `scripts/generate_corrupted_test.py`,
- optional helper functions in `src/data/corruptions.py`.

Command:

```bash
python scripts/generate_corrupted_test.py --config configs/default.yaml
```

Expected output:

```text
data/processed/tomato_corrupted/
```

Do not use corrupted images for training.

## Step 4 — Model Factory

Implement:

- `src/models/model_factory.py`.

Required models:

- resnet50,
- efficientnet_b0,
- mobilenet_v2,
- mobilenet_v3_small.

Optional:

- mobilevit_xs.

Success check:

- each model can be instantiated with `num_classes=10`,
- forward pass works with dummy tensor `(2, 3, 224, 224)`.

## Step 5 — Training Pipeline

Implement:

- `src/training/trainer.py`,
- `src/training/losses.py`,
- `scripts/train_model.py`.

Command example:

```bash
python scripts/train_model.py --config configs/mobilenet_v3_small.yaml
```

Success checks:

- checkpoint saved,
- training curve saved,
- validation macro F1 logged.

## Step 6 — Evaluation Pipeline

Implement:

- `src/evaluation/metrics.py`,
- `src/evaluation/evaluate.py`,
- `src/evaluation/inference_speed.py`,
- `scripts/evaluate_model.py`.

Outputs:

- `results/experiments.csv`,
- `results/classification_reports/`,
- `results/confusion_matrices/`.

## Step 7 — Robustness Evaluation

Implement:

- `scripts/evaluate_robustness.py`.

Output:

```text
results/robustness_comparison.csv
```

Success checks:

- each model evaluated on clean test,
- each model evaluated on corrupted test,
- robustness drop calculated.

## Step 8 — Grad-CAM Pipeline

Implement:

- `src/explainability/gradcam.py`,
- `src/explainability/visualize.py`,
- `scripts/generate_gradcam.py`.

Outputs:

```text
results/gradcam/<model_name>/correct/
results/gradcam/<model_name>/wrong/
results/gradcam/<model_name>/clean_vs_corrupted/
results/gradcam/<model_name>/cross_model/
reports/gradcam_analysis.md
```

## Step 9 — Run All Experiments

Implement:

- `scripts/run_all_experiments.py`.

Do not include MobileViT-XS in automatic run until the four required models are complete.

## Step 10 — Paper Outputs

Generate:

- `results/model_comparison.csv`,
- `results/robustness_comparison.csv`,
- paper-ready tables,
- paper-ready figures,
- `reports/experiment_notes.md`,
- `reports/gradcam_analysis.md`.
