# Sprint 01 model audit — issues #1–#3

Audit date: 2026-10-08 (Asia/Bangkok). Repository: `starkkk1/plant-disease-detection`.

Requirements: [#1 model audit](https://github.com/starkkk1/plant-disease-detection/issues/1), [#2 CPU predictor](https://github.com/starkkk1/plant-disease-detection/issues/2), [#3 trained-weight benchmarking](https://github.com/starkkk1/plant-disease-detection/issues/3).

## Artifacts and architectures

Paths below are relative to the repository root. All five local files are real trained checkpoints referenced by the training scripts and local training logs. The baseline runner checks strict state-dict compatibility, runs real-image inference, and records SHA-256 hashes in `reports/sprint-01/baseline.json`.

| Checkpoint in `checkpoints/` | timm architecture | Config in `configs/` | Size (MiB) | Role |
|---|---|---|---:|---|
| `convnext_tiny_best.pth` | `convnext_tiny` | `convnext.yaml` | 106.226 | Teacher |
| `efficientnet_b0_best.pth` | `efficientnet_b0` | `efficientnet_b0.yaml` | 15.626 | Independent student |
| `efficientnet_b0_distilled_best.pth` | `efficientnet_b0` | `distillation_effnetb0.yaml` | 15.630 | KD student |
| `mobilenetv3_small_100_best.pth` | `mobilenetv3_small_100` | `mobilenet_v3_small.yaml` | 5.964 | Independent student |
| `mobilenetv3_small_100_distilled_best.pth` | `mobilenetv3_small_100` | `distillation_mobilenetv3.yaml` | 5.967 | KD student |

The training scripts use `torch.save(model.state_dict(), ...)`; these artifacts do not embed class names or preprocessing metadata. The reusable predictor also supports wrappers under `model` or `state_dict`, loads with `torch.load(..., map_location="cpu", weights_only=True)`, and uses strict loading. No missing checkpoint is replaced with random or downloaded weights. Checkpoints and datasets are intentionally ignored by Git.

## Official labels and verification

The authoritative labels originate in `src/datasets/dataset.py`: `TomatoDataset` takes the training `torchvision.datasets.ImageFolder.class_to_idx`. The configured file is `data/new-data-removal/class_to_idx.json`; its local SHA-256 is `ff5c73e7a7e3e495593fa664b75ebc73de4c9a425f1d847351ae60bc1fbaa786`. Its mapping was compared with `ImageFolder("data/new-data-removal/train").class_to_idx` and matched exactly. A versioned copy of the verified mapping is supplied in `configs/class_to_idx.sprint01.json` (JSON formatting may change its byte hash).

| Index | Label |
|---:|---|
| 0 | Tomato___Bacterial_spot |
| 1 | Tomato___Early_blight |
| 2 | Tomato___Late_blight |
| 3 | Tomato___Leaf_Mold |
| 4 | Tomato___Septoria_leaf_spot |
| 5 | Tomato___Spider_mites Two-spotted_spider_mite |
| 6 | Tomato___Target_Spot |
| 7 | Tomato___Tomato_Yellow_Leaf_Curl_Virus |
| 8 | Tomato___Tomato_mosaic_virus |
| 9 | Tomato___healthy |
| 10 | unknown |

`unknown` is an explicitly trained class, not automatic out-of-distribution detection. The legacy hard-coded list in `src/models/inference_model.py` has a different order and must not be the source of labels for the new predictor. The README's unspecified eleventh class and abbreviated spider-mite label have been corrected.

The current directory and saved JSON agree, but the raw historical checkpoints have no embedded training manifest. Exact historical pairing cannot be proven solely from a raw state dict. Preserve the class map, config and checkpoint hash together for deployment; the audit records the evidence available locally without inventing historical metadata.

## Input, output and preprocessing

All audited configs inherit `configs/default.yaml`: image size 224, 11 output classes. Classification input is float32 RGB `[1, 3, 224, 224]`; logits are `[1, 11]`. Images are decoded using Pillow, converted to RGB, directly resized to 224×224 using OpenCV `INTER_LINEAR`, converted to float32 and normalized with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. No center crop, EXIF rotation or train augmentation is applied.

`src/inference/preprocess.py` numerically matches validation's `A.Resize` + `A.Normalize` + `ToTensorV2` in `src/datasets/transforms.py`; a non-square random image and custom mean/std verify this to `1e-6`. Using Pillow's resize would differ from training even with the same target size. The former evaluation's `Resize(256)` + `CenterCrop(224)` was also inconsistent and has been replaced.

`src/inference/predictor.py` loads the real weights on CPU, sets `eval()`, classifies under `torch.inference_mode()`, validates output shape and finite logits, and returns JSON containing a checkpoint-derived `model_version` and ranked `{class_id, label, confidence}` predictions. Confidence is a softmax score, not calibrated certainty. RGB, grayscale and RGBA input, tensor shape/dtype, trained-index ordering, checkpoint wrappers and errors are covered by unit tests.

Grad-CAM++ is optional in `scripts/evaluate.py --gradcam`, with lazy imports and gradients outside `inference_mode()`. The original model/Grad-CAM implementation remains intact. The older evaluation workflow is archived in `scripts/evaluate_gradcam_legacy.py`, including its historical crop and cleanup behavior; it is excluded from the new baseline.

## Reproduce and review evidence

From the repository root, use an interpreter with PyTorch installed. This workstation's verified interpreter is `D:\Code\python\.venv\Scripts\python.exe` (Python 3.14.5, PyTorch 2.12.0+cpu, torchvision 0.27.0, timm 1.0.27). The default MSYS Python does not contain PyTorch. For portable environments, install the project requirements, a compatible CPU torch/torchvision pair, and `requirements-dev.txt` for pytest.

```powershell
& D:\Code\python\.venv\Scripts\python.exe -m pytest tests -q
& D:\Code\python\.venv\Scripts\python.exe scripts/audit_models.py --threads 4 --iterations 100 --output-dir reports/sprint-01
```

The second command verifies the training mapping, loads all five models strictly and evaluates complete local `test` and external `eval` splits. `baseline.json` contains checkpoint hashes, sizes, parameter counts, shapes, environment, real-image predictions, latency samples and aggregate metrics. Each `<checkpoint-stem>_<split>.json` records all per-image predictions, confusion matrix counts, per-class counts and a dataset manifest hash.

For individual evaluation and benchmarks, see [RUN_GUIDE.md](../RUN_GUIDE.md). The quantitative results and dataset limitations are documented in [sprint-01-baseline.md](sprint-01-baseline.md).
