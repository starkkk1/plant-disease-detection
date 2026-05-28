# TODO

## Phase 0 — Setup

- [ ] Create/verify GitHub repository.
- [ ] Replace root Markdown files with revised versions.
- [ ] Create Python environment.
- [ ] Install requirements.
- [ ] Configure Kaggle API.
- [ ] Confirm PyTorch works.

## Phase 1 — Literature and Scope

- [ ] Finalize research questions.
- [ ] Summarize papers on PlantVillage/plant disease classification.
- [ ] Summarize papers on MobileNet/EfficientNet/ResNet.
- [ ] Summarize papers on Grad-CAM/XAI.
- [ ] Summarize papers on robustness/domain shift.

## Phase 2 — Dataset

- [ ] Download Kaggle dataset.
- [ ] Extract to `data/raw/`.
- [ ] Inspect raw folder names.
- [ ] Filter tomato classes.
- [ ] Map `valid` to `val`.
- [ ] Generate dataset summary.
- [ ] Validate dataloaders.

## Phase 3 — Corrupted Test Set

- [ ] Implement corruption functions.
- [ ] Generate corrupted test set.
- [ ] Verify class folder structure.
- [ ] Save examples for report.

## Phase 4 — Model Factory

- [ ] Implement ResNet50.
- [ ] Implement EfficientNet-B0.
- [ ] Implement MobileNetV2.
- [ ] Implement MobileNetV3-Small.
- [ ] Add MobileViT-XS only after required models work.

## Phase 5 — Training

- [ ] Train ResNet50 baseline.
- [ ] Train EfficientNet-B0.
- [ ] Train MobileNetV2.
- [ ] Train MobileNetV3-Small.
- [ ] Train MobileViT-XS if time/hardware allow.

## Phase 6 — Clean Evaluation

- [ ] Evaluate all models on clean test set.
- [ ] Save classification reports.
- [ ] Save confusion matrices.
- [ ] Save model comparison table.

## Phase 7 — Efficiency Evaluation

- [ ] Count parameters.
- [ ] Measure FLOPs/MACs.
- [ ] Measure model size.
- [ ] Measure CPU latency.
- [ ] Measure GPU latency if available.

## Phase 8 — Robustness Evaluation

- [ ] Evaluate all models on corrupted test set.
- [ ] Calculate robustness drop.
- [ ] Save `results/robustness_comparison.csv`.
- [ ] Identify most damaging corruption type.

## Phase 9 — Grad-CAM

- [ ] Generate Grad-CAM for correct predictions.
- [ ] Generate Grad-CAM for wrong predictions.
- [ ] Generate clean vs corrupted Grad-CAM pairs.
- [ ] Generate cross-model comparison.
- [ ] Write `reports/gradcam_analysis.md`.

## Phase 10 — Report and Demo

- [ ] Write results discussion.
- [ ] Answer research questions.
- [ ] Prepare tables and figures.
- [ ] Build optional Streamlit demo only if main pipeline is complete.
