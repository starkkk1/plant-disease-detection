# Project TODO

Track progress here. Check off items as they are completed. Do not skip phases.

---

## Phase 0 — Setup

- [ ] Create GitHub repository named `lightweight-plant-disease`
- [ ] Add all `.md` files to repo root
- [ ] Create Python virtual environment (`python -m venv venv`)
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Confirm PyTorch + CUDA detected correctly
- [ ] Add `.gitignore` (exclude `data/`, `checkpoints/`, `__pycache__/`, `.env`)
- [ ] Configure Kaggle API (`~/.kaggle/kaggle.json`)

---

## Phase 1 — Literature Survey

- [ ] Read and summarize: Mohanty et al. (2016) — PlantVillage
- [ ] Read and summarize: Selvaraju et al. (2017) — Grad-CAM
- [ ] Read and summarize: Howard et al. (2017) — MobileNet, Sandler et al. (2018) — MobileNetV2, Howard et al. (2019) — MobileNetV3
- [ ] Read and summarize: Tan & Le (2019) — EfficientNet
- [ ] Read and summarize: He et al. (2016) — ResNet
- [ ] Survey 3–5 recent papers (2022–2026) on lightweight plant disease classification
- [ ] Survey 2–3 papers on XAI / Grad-CAM in agricultural image classification
- [ ] Fill `reports/literature_review_table.md`
- [ ] Identify and document 3+ research gaps
- [ ] Confirm research questions match gaps found

---

## Phase 2 — Dataset Preparation

- [ ] Download dataset from Kaggle (`rashidthihan/plant-disease-dataset`)
- [ ] Verify raw folder structure matches `DATASET_GUIDE.md`
- [ ] If the extracted folder name differs, update `configs/default.yaml → data.raw_root`
- [ ] Run `python scripts/prepare_dataset.py --filter-tomato`
- [ ] Verify 10 total tomato classes exist in `data/processed/tomato/train/`
- [ ] Run `python scripts/prepare_dataset.py --stats` → save to `reports/dataset_summary.md`
- [ ] Check class imbalance (count images per class)
- [ ] Verify `class_to_idx.json` is saved correctly
- [ ] Visualize 5 sample images per class (notebook 01)

---

## Phase 3 — Data Pipeline

- [ ] Implement `src/data/transforms.py` (train vs. val/test transforms)
- [ ] Implement `src/data/dataset.py` using `ImageFolder`
- [ ] Verify dataloader output: shape `(B, 3, 224, 224)`, labels correct
- [ ] Test batch loading speed
- [ ] Compute and log class weights for CrossEntropyLoss

---

## Phase 4 — Model Training

For each model: ResNet50 / EfficientNet-B0 / MobileNetV2 / MobileNetV3-Small

- [ ] Implement `src/models/model_factory.py` (load pretrained, replace head)
- [ ] Train ResNet50 (baseline)
- [ ] Train EfficientNet-B0
- [ ] Train MobileNetV2
- [ ] Train MobileNetV3-Small
- [ ] Verify each saves `best.pth` to `checkpoints/<model>/`
- [ ] Plot training curves for each model
- [ ] Check for overfitting (train vs. val gap)

---

## Phase 5 — Evaluation

- [ ] Run `scripts/evaluate_model.py` for each model
- [ ] Record: accuracy, precision, recall, F1 (macro)
- [ ] Generate confusion matrices for each model
- [ ] Measure: num_params, model_size_mb, inference_time_ms (CPU + GPU)
- [ ] Append all results to `results/experiments.csv`
- [ ] Build comparison table (all 4 models side by side)

---

## Phase 6 — Grad-CAM

- [ ] Implement `src/explainability/gradcam.py`
- [ ] Verify Grad-CAM output is correct shape and overlays properly
- [ ] Generate heatmaps: 5 correct predictions per class (per model)
- [ ] Generate heatmaps: all wrong predictions (per model)
- [ ] Generate cross-model comparison: same images, all 4 models
- [ ] Qualitative analysis: does the model focus on the diseased region?
- [ ] Document findings in `reports/experiment_notes.md`

---

## Phase 7 — Analysis

- [ ] Answer RQ1: which model is best by F1?
- [ ] Answer RQ2: what is the accuracy-efficiency trade-off?
- [ ] Answer RQ3: do Grad-CAM heatmaps focus on disease regions?
- [ ] Answer RQ4: does accuracy correlate with Grad-CAM quality?
- [ ] Identify commonly misclassified class pairs
- [ ] Discuss shortcut learning evidence (if any)

---

## Phase 8 — Paper Writing

- [ ] Abstract
- [ ] Introduction (motivation, gap, contribution)
- [ ] Related Work (3 subsections per `PAPER_OUTLINE.md`)
- [ ] Dataset and Preprocessing
- [ ] Methodology
- [ ] Results (tables + figures)
- [ ] Discussion (answer each RQ)
- [ ] Conclusion
- [ ] Future Work
- [ ] References (cite all papers from survey)
- [ ] Proofread and format

---

## Phase 9 — Optional Demo

- [ ] Build `src/demo/streamlit_app.py`
- [ ] Upload image → show class prediction + confidence
- [ ] Show Grad-CAM heatmap overlay
- [ ] Add model selector (dropdown for 4 models)
- [ ] Add research disclaimer footer
