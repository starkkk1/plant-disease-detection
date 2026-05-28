# Research Questions

## Main Research Question

How can lightweight deep learning models classify tomato leaf diseases effectively while remaining computationally efficient, robust, and explainable?

## RQ1 — Lightweight vs Baseline

Can lightweight CNN models achieve competitive classification performance compared to ResNet50 while significantly reducing computational cost?

Answered by:

- clean test accuracy,
- macro F1-score,
- model size,
- parameters,
- FLOPs/MACs,
- inference latency.

## RQ2 — Robustness

How does model performance degrade when evaluated on corrupted/noisy tomato leaf images that simulate real-world-like agricultural conditions?

Answered by:

- clean macro F1,
- corrupted macro F1,
- absolute robustness drop,
- relative robustness drop.

## RQ3 — Explainability

Do Grad-CAM explanations remain focused on disease-relevant regions under clean and corrupted image conditions?

Answered by:

- Grad-CAM for correct and wrong predictions,
- clean vs corrupted heatmap comparison,
- qualitative focus labels: Good, Partial, Poor, Unclear.

## RQ4 — Optional Modern Model

As an exploratory comparison, does MobileViT-XS offer better robustness or more interpretable Grad-CAM heatmaps than traditional lightweight CNNs?

Answered by:

- MobileViT-XS macro F1,
- robustness drop,
- latency,
- Grad-CAM behavior.

## Contribution Statement

This project provides a unified benchmark of tomato leaf disease classification models across classification performance, computational efficiency, robustness under corrupted images, and Grad-CAM explainability. Unlike basic PlantVillage-style studies that mainly report clean accuracy, this project explicitly analyzes whether lightweight models remain reliable under more challenging image conditions.
