# Sprint 01 — ML audit and blockers

> Superseded by [the verified model audit](sprint-01-model-audit.md) and [measured baseline](sprint-01-baseline.md) on 2026-10-08. The historical blockers below describe the previous branch state; trained artifacts and data have now been found locally and exercised.

## Existing implementation

- `src/models/inference_model.py` already implements `PlantDiseasePredictor` with timm and optional Grad-CAM++; it hard-codes class names and catches checkpoint loading errors instead of stopping execution.
- `src/datasets/dataset.py` uses torchvision `ImageFolder` so the authoritative labels are `ImageFolder.class_to_idx`, not the ordering embedded in `inference_model.py`.
- `configs/default.yaml` points to `data/new-data-removal/class_to_idx.json`; this location is ignored by Git, and its actual contents must be supplied by the team.
- `src/datasets/transforms.py` uses `A.Resize(224,224)`, ImageNet normalization for validation; `scripts/evaluate.py` instead uses Resize(256) + CenterCrop(224), so metrics and production transforms are currently inconsistent.
- `scripts/benchmark.py` times freshly initialized models on synthetic tensors without loading model checkpoints. It does **not** establish latency or accuracy of production inference.
- `scripts/train_mobilenet_v3.py` saves a raw PyTorch state dict. Distilled checkpoint format must be verified separately.
- `frontend/` contains Next.js, not Flutter; mobile deployment strategy remains open.

## What this branch introduces

`src/inference/` exposes a CPU-ready, strict checkpoint-loading predictor separated from Grad-CAM and Qdrant. It requires a verified training `class_to_idx.json` rather than silently inventing an order. Loading raises explicit errors if weights are missing or incompatible.

## Mandatory verification before serving predictions

1. Locate trained MobileNetV3 checkpoint locally or in company-approved artifact storage; do not push large weights or secrets to Git.
2. Export the exact `ImageFolder.class_to_idx` mapping from the training dataset and confirm the class count matches classifier outputs. The source config points to `data/new-data-removal/class_to_idx.json` but existence and ordering are not verified.
3. Run CPU inference and compare predictions against a trusted evaluation run using the **same** resize/normalization preprocessing.
4. Measure median/p95 on real images, including preprocessing; benchmark script must be corrected in Issue #3.
5. Check external photos for distribution shift and decide if an unknown/low-confidence warning is needed.
6. Decide whether to use the current Next.js frontend responsively or implement a native application under Issue #7.

## Example local smoke test (requires real artifacts)

```python
from src.inference import PlantDiseasePredictor

predictor = PlantDiseasePredictor(
    checkpoint_path="checkpoints/mobilenetv3_small_100_best.pth",
    class_map_path="data/new-data-removal/class_to_idx.json",
    model_name="mobilenetv3_small_100",
    model_version="s01-candidate",
)
print(predictor.predict("path/to/test-leaf.jpg", top_k=3))
```

The checkpoint filename above is illustrative: use the **actual** saved artifact (possibly `*_distilled_best.pth`). This smoke test has not been executed because weights and local dataset artifacts are not available in the repository.

## Remaining Sprint 01 work

- #1: Validate true checkpoint and class map; verify training-vs-production preprocessing.
- #2: Add automated unit tests against a tiny fabricated timm-compatible state dict or mock and run one real-image inference with genuine weights.
- #3: Correct CPU benchmarking and evaluation before claiming performance metrics.
