# Research Questions

This file defines the research questions that drive the project. Every experiment, analysis, and paper section must connect back to these questions.

---

## Main Research Question

> **RQ0:** How can lightweight deep learning models classify tomato leaf diseases effectively while remaining computationally efficient and interpretable?

---

## Sub-Questions

### RQ1 — Accuracy
> Which CNN architecture achieves the best classification accuracy on tomato leaf diseases, and is there a significant accuracy gap between lightweight models and the ResNet50 baseline?

**Answered by:** Test accuracy, F1-score, confusion matrix comparison across all 4 models.

---

### RQ2 — Efficiency
> What is the trade-off between model size, inference speed, and classification accuracy among the compared architectures?

**Answered by:** Comparison table of params / model size / inference time vs. F1-score.

---

### RQ3 — Explainability
> Do lightweight CNN models produce Grad-CAM heatmaps that focus on disease-relevant leaf regions, or do they rely on background shortcuts?

**Answered by:** Qualitative analysis of Grad-CAM outputs for correct and incorrect predictions, across all models.

---

### RQ4 — Accuracy vs. Explainability Alignment
> Does higher classification accuracy correlate with better Grad-CAM localization quality?

**Answered by:** Comparing Grad-CAM heatmaps of the best-accuracy model vs. the worst, and analyzing if the best model also produces the most interpretable heatmaps.

---

## Hypotheses

| Hypothesis | Expected Finding |
|------------|------------------|
| H1 | MobileNetV3-Small will be fastest with minor accuracy loss vs. ResNet50. |
| H2 | EfficientNet-B0 will best balance accuracy and size. |
| H3 | Models with high accuracy will tend to focus on leaf lesions rather than background. |
| H4 | Grad-CAM quality degrades on visually similar disease classes (e.g., Early Blight vs. Target Spot). |

---

## Research Gaps Identified (for Related Work section)

1. Most prior work benchmarks only on PlantVillage without analyzing Grad-CAM quality on misclassified samples.
2. Few papers compare Grad-CAM heatmap quality across different lightweight architectures.
3. Efficiency metrics (inference time, model size) are often reported separately from explainability analysis — rarely combined in one study.
4. Shortcut learning (background focus) in plant disease models is underexplored.

These gaps justify the contribution of this project.

---

## Paper Contribution Statement (draft)

> This paper presents a comparative study of lightweight CNN architectures for tomato leaf disease classification, evaluated not only by prediction accuracy but also by computational efficiency and Grad-CAM explainability. We systematically analyze whether model predictions are grounded in disease-relevant leaf features, and discuss implications for real-world agricultural deployment.
