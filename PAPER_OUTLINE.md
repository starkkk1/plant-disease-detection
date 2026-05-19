# Paper Outline

Target venue: conference paper or journal (IEEE Access / Applied Sciences / Computers and Electronics in Agriculture)

Target length: 8–12 pages

---

## Title

**Lightweight and Explainable Deep Learning for Tomato Leaf Disease Classification using Grad-CAM**

Alternative:

**A Comparative Study of Lightweight CNN Models with Grad-CAM for Tomato Leaf Disease Detection**

---

## Abstract (target: 200–250 words)

Cover:
- Problem: disease classification + need for lightweight + need for explainability
- Dataset: PlantVillage Tomato subset (via Kaggle)
- Method: 4 CNN models compared, Grad-CAM applied
- Key findings: [fill after experiments]
- Conclusion: best trade-off model identified, heatmaps confirm/challenge model reliability

---

## 1. Introduction

- Motivation: food security, agricultural losses from disease
- Gap: existing DL models often heavy + black box
- Contribution:
  - Systematic comparison of lightweight CNNs
  - Grad-CAM analysis on correct AND wrong predictions
  - Efficiency metrics alongside classification metrics
- Paper structure overview (1 paragraph)

Reference RQ0 from `RESEARCH_QUESTIONS.md`.

---

## 2. Related Work

### 2.1 Plant Disease Classification with Deep Learning
- Early CNN work on PlantVillage (Mohanty et al., 2016)
- Transfer learning approaches
- Recent lightweight model applications

### 2.2 Lightweight CNN Architectures
- MobileNet family (Howard et al.)
- EfficientNet (Tan & Le, 2019)
- Deployment considerations for edge/mobile

### 2.3 Explainable AI for Plant Disease
- Grad-CAM (Selvaraju et al., 2017)
- XAI applied to leaf images
- Shortcut learning in plant disease datasets

### 2.4 Research Gap
- Summarize the 4 gaps from `RESEARCH_QUESTIONS.md`.

---

## 3. Dataset and Preprocessing

- Dataset source and description
- Tomato class list (table with class names and image counts)
- Train/val/test split
- Class imbalance handling (class weights)
- Image preprocessing pipeline
- Data augmentation (training only)

Include: table of class distribution, sample image grid.

---

## 4. Methodology

### 4.1 Model Architectures
- Brief description of each model
- Transfer learning strategy (2-phase training)
- Classification head design

### 4.2 Training Setup
- Hyperparameters table (from `EXPERIMENT_PLAN.md`)
- Loss function with class weights
- Early stopping criterion

### 4.3 Evaluation Protocol
- Classification metrics (accuracy, precision, recall, macro F1)
- Efficiency metrics (params, size, inference time)
- Same conditions for all models

### 4.4 Grad-CAM Implementation
- Target layer for each model
- Overlay visualization method
- Analysis protocol (correct vs. wrong predictions)

---

## 5. Results

### 5.1 Classification Performance
- Table: accuracy, precision, recall, F1 for all 4 models
- Confusion matrices (appendix or inline)
- Training curves

### 5.2 Model Efficiency
- Table: params, size (MB), inference time (CPU + GPU)
- Scatter plot: F1 vs. inference time

### 5.3 Grad-CAM Analysis
- Heatmap figures: correct predictions (1–2 per class, best model)
- Heatmap figures: wrong predictions (discussion)
- Cross-model comparison: same image, 4 heatmaps
- Discussion: background focus vs. lesion focus

---

## 6. Discussion

- Answer each RQ from `RESEARCH_QUESTIONS.md`
- Best lightweight model recommendation with justification
- When heatmaps mislead (discuss shortcut learning)
- Dataset limitations (PlantVillage lab bias)
- Practical deployment considerations

---

## 7. Conclusion

- Summarize findings (3–4 sentences)
- Best model identified
- Grad-CAM as a reliability tool, not just visualization
- Broader implications for precision agriculture

---

## 8. Future Work

- Real-field dataset evaluation (PlantDoc)
- Quantization and pruning for further compression
- Segmentation-based localization for more precise XAI
- Mobile app deployment

---

## References (key papers to cite)

| Paper | Why |
|-------|-----|
| Mohanty et al. (2016) — PlantVillage | Original dataset paper |
| Selvaraju et al. (2017) — Grad-CAM | Core XAI method |
| Howard et al. (2017) — MobileNet | MobileNet family foundation |
| Sandler et al. (2018) — MobileNetV2 | MobileNetV2 |
| Howard et al. (2019) — MobileNetV3 | MobileNetV3 |
| Tan & Le (2019) — EfficientNet | EfficientNet-B0 |
| He et al. (2016) — ResNet | ResNet50 |
| Singh et al. (2021) — PlantDoc | Optional external dataset |
| Recent lightweight/XAI plant disease papers | Related work comparison |

---

## Figures and Tables Checklist

- [ ] Table 1: Dataset class distribution
- [ ] Figure 1: Sample images from each class
- [ ] Figure 2: Architecture overview diagram
- [ ] Table 2: Training hyperparameters
- [ ] Table 3: Classification results comparison
- [ ] Figure 3: Confusion matrices
- [ ] Figure 4: Training curves (loss + accuracy)
- [ ] Table 4: Efficiency comparison
- [ ] Figure 5: F1 vs. inference time scatter plot
- [ ] Figure 6: Grad-CAM correct predictions
- [ ] Figure 7: Grad-CAM wrong predictions
- [ ] Figure 8: Cross-model Grad-CAM comparison
