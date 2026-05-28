# Project Structure

This is the canonical folder layout. Do not create files outside this structure without updating this document.

```text
plant-disease-detection/
├── README.md
├── CODEX_INSTRUCTIONS.md
├── CODEX_PROMPT.md
├── CONFIG_GUIDE.md
├── DATASET_GUIDE.md
├── DATA_VALIDATION_CHECKLIST.md
├── EVALUATION_GUIDE.md
├── EXPERIMENT_PLAN.md
├── GRADCAM_GUIDE.md
├── IMPLEMENTATION_ORDER.md
├── PAPER_OUTLINE.md
├── PROJECT_STRUCTURE.md
├── RESEARCH_QUESTIONS.md
├── RISK_REGISTER.md
├── ROBUSTNESS_GUIDE.md
├── TECH_STACK.md
├── TODO.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── default.yaml
│   ├── resnet50.yaml
│   ├── efficientnet_b0.yaml
│   ├── mobilenet_v2.yaml
│   ├── mobilenet_v3_small.yaml
│   └── mobilevit_xs.yaml
│
├── data/
│   ├── raw/
│   │   └── plant-disease-dataset/
│   └── processed/
│       ├── tomato/
│       │   ├── train/
│       │   ├── val/
│       │   ├── test/
│       │   └── class_to_idx.json
│       └── tomato_corrupted/
│           ├── brightness/
│           ├── contrast/
│           ├── gaussian_noise/
│           ├── blur/
│           ├── jpeg/
│           └── shadow/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   ├── 03_evaluation_summary.ipynb
│   └── 04_gradcam_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── corruptions.py
│   ├── models/
│   │   ├── model_factory.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   └── inference_speed.py
│   ├── explainability/
│   │   ├── gradcam.py
│   │   └── visualize.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── seed.py
│   │   ├── logger.py
│   │   └── paths.py
│   └── demo/
│       └── streamlit_app.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── generate_corrupted_test.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── evaluate_robustness.py
│   ├── generate_gradcam.py
│   └── run_all_experiments.py
│
├── checkpoints/
│   ├── resnet50/
│   ├── efficientnet_b0/
│   ├── mobilenet_v2/
│   ├── mobilenet_v3_small/
│   └── mobilevit_xs/
│
├── results/
│   ├── experiments.csv
│   ├── model_comparison.csv
│   ├── robustness_comparison.csv
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── classification_reports/
│   ├── gradcam/
│   └── figures_for_paper/
│
├── reports/
│   ├── dataset_summary.md
│   ├── literature_review_table.md
│   ├── experiment_notes.md
│   ├── gradcam_analysis.md
│   └── paper_draft.md
│
└── tests/
    ├── test_dataset.py
    ├── test_models.py
    ├── test_metrics.py
    └── test_corruptions.py
```

## Key Rules

- `src/` contains reusable logic.
- `scripts/` contains CLI entry points.
- `notebooks/` are for exploration only.
- `configs/` is the single source of truth for hyperparameters.
- `data/` and `checkpoints/` are not committed.
- CSV results and paper figures can be committed.
