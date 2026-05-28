# Robustness Guide

## Purpose

Robustness evaluation checks whether models remain reliable when input images become less clean and more similar to real-world field conditions.

PlantVillage-style images are often clean. Real-world images may contain:

- poor lighting,
- blur,
- noise,
- shadows,
- compression artifacts,
- partial occlusion,
- more complex backgrounds.

## Core Idea

Train all models on the clean training set.

Evaluate each trained model on:

1. clean test set,
2. corrupted/noisy test set.

Then compare performance drop.

## Corruptions to Implement

Minimum required corruptions:

1. brightness change,
2. contrast shift,
3. Gaussian noise,
4. blur,
5. JPEG compression.

Optional:

6. shadow simulation,
7. mild occlusion.

## Recommended Severity

Start with one medium severity level to keep the project manageable.

Optional extension:

- severity 1: mild,
- severity 2: medium,
- severity 3: strong.

Do not add severity levels until the main pipeline works.

## Output Structure

```text
data/processed/tomato_corrupted/
├── brightness/
│   ├── Tomato___Bacterial_spot/
│   └── ...
├── contrast/
├── gaussian_noise/
├── blur/
├── jpeg/
└── shadow/
```

## Metrics

For each model, compute:

- clean accuracy,
- corrupted accuracy,
- clean macro F1,
- corrupted macro F1,
- absolute robustness drop,
- relative robustness drop.

Formula:

```text
Absolute Drop = Clean Macro F1 - Corrupted Macro F1
```

```text
Relative Drop (%) = ((Clean Macro F1 - Corrupted Macro F1) / Clean Macro F1) * 100
```

## Required Output File

```text
results/robustness_comparison.csv
```

Suggested columns:

```csv
model,corruption_type,clean_accuracy,corrupted_accuracy,clean_f1_macro,corrupted_f1_macro,absolute_drop,relative_drop_percent
```

## Discussion Questions

Use the results to answer:

- Which model performs best on clean images?
- Which model drops the least under corrupted images?
- Does the smallest model also become less robust?
- Does MobileViT-XS improve robustness compared with CNN-only lightweight models?
- Which corruption type causes the biggest performance drop?
