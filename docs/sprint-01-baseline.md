# Sprint 01 trained-model CPU baseline

Measured on 2026-10-08 (Asia/Bangkok) for GitHub issue #3, after the model audit and reusable predictor in issues #1 and #2. All five checkpoints loaded strictly and produced predictions on real local leaf images. No model was retrained or substituted with random weights.

## Measured results

Hardware: AMD Ryzen 5 220 with Radeon 740M Graphics; Windows 11; Python 3.14.5; PyTorch 2.12.0+cpu; torchvision 0.27.0; timm 1.0.27. CPU only, float32, four PyTorch threads. Exact package versions and artifact hashes are recorded in the JSON evidence.

| Model | Params (M) | Size (MiB) | CPU p50 (ms) | CPU p95 (ms) | Test accuracy | Test macro-F1 | External accuracy | External macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXt Tiny (teacher) | 27.829 | 106.226 | 67.56 | 82.79 | 99.56% | 0.9955 | 62.69% | 0.4464 |
| EfficientNet-B0 | 4.022 | 15.626 | 20.23 | 28.67 | 99.25% | 0.9916 | 52.24% | 0.3876 |
| EfficientNet-B0 KD | 4.022 | 15.630 | 20.89 | 26.08 | 98.94% | 0.9888 | 55.22% | 0.3961 |
| MobileNetV3-Small | 1.529 | 5.964 | 8.82 | 11.11 | 98.65% | 0.9846 | 55.22% | 0.3952 |
| MobileNetV3-Small KD | 1.529 | 5.967 | 9.37 | 13.94 | 98.37% | 0.9820 | 53.73% | 0.3750 |

The complete test split contains **3,195 images**. The external `eval` split contains **67 images**. No per-class limit was used. Macro metrics use all eleven training labels with `zero_division=0`; confusion matrices and sample counts use that same label order.

Both distilled students have lower test accuracy and macro-F1 than their independent counterparts. EfficientNet KD improves external accuracy and macro-F1 slightly, while MobileNet KD is lower on both splits. ConvNeXt is the strongest classifier on these data; MobileNet has the lowest measured CPU latency and checkpoint size. These are comparisons of the available artifacts, not a controlled retraining study.

## Latency protocol

The final latency measurements reload each audited checkpoint, call `eval()` and classify with `torch.inference_mode()`. They use ten warmup calls and 100 timed calls at batch size one. The 110-image pool selects ten actual test images per class in round-robin order. Each timed call includes file opening/decoding, RGB conversion, OpenCV resize, ImageNet normalization, tensor transfer, forward pass, finite-output validation, softmax and top-1 JSON serialization. The percentile method is NumPy percentile.

Model startup/loading, Grad-CAM++, API/network overhead and cold-machine latency are excluded. The final pass runs after evaluation, with warm filesystem caches. Power, temperature and CPU frequency were not controlled. These workstation measurements are a research baseline, not a production SLA or a prediction of phone performance. The earlier benchmark of random models on synthetic tensors must not be used as production evidence.

## Dataset coverage and limitations

| Training label | Test images | External images |
|---|---:|---:|
| Tomato___Bacterial_spot | 320 | 8 |
| Tomato___Early_blight | 200 | 9 |
| Tomato___Late_blight | 379 | 10 |
| Tomato___Leaf_Mold | 144 | 5 |
| Tomato___Septoria_leaf_spot | 267 | 10 |
| Tomato___Spider_mites Two-spotted_spider_mite | 252 | 1 |
| Tomato___Target_Spot | 212 | 1 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 805 | 5 |
| Tomato___Tomato_mosaic_virus | 57 | 9 |
| Tomato___healthy | 262 | 8 |
| unknown | 297 | 1 |

External performance is substantially below the held-out test results. The external split is small and imbalanced: spider mites, target spot and unknown each have only one image. Its macro-F1 is sensitive to individual mistakes; it is insufficient to establish field reliability. Dataset source descriptions and labels have not been independently re-annotated, and train/test leakage or near-duplicates were not separately audited here. The directory name `new-data-removal` alone is not evidence that leakage is absent.

`unknown` is a trained category, not a general rejection mechanism for arbitrary photos. Softmax confidence is not calibrated probability of diagnostic correctness. The current training directory agrees with the saved class map; raw historical checkpoints contain no embedded class-map manifest, so exact historical pairing has the limitation described in the [model audit](sprint-01-model-audit.md).

## Commands and evidence

Run from the repository root with the verified local environment:

```powershell
& D:\Code\python\.venv\Scripts\python.exe -m pytest tests -q
& D:\Code\python\.venv\Scripts\python.exe scripts/audit_models.py --threads 4 --iterations 100 --output-dir reports/sprint-01
```

**26 unit tests passed**. They cover RGB/grayscale/RGBA preprocessing, tensor size/dtype, numerical agreement with training validation, class-map validation and ordering, checkpoint loading/wrappers, inference mode, invalid inputs, malformed outputs and subset-dataset label remapping. Test-only fabricated weights are used for contract assertions; every number in the table above uses genuine local checkpoints.

The optional Grad-CAM++ path was also exercised on a real image using the distilled MobileNet checkpoint. Its explicit one-image-per-class smoke subset contains eleven images and is not used for the full baseline.

Evidence files:

- `reports/sprint-01/baseline.json`: all five checkpoint hashes, sizes, parameters, shapes, real-image predictions, timing samples, environment and complete-split summaries.
- `reports/sprint-01/*_test.json` and `*_eval.json`: ten complete evaluations, including every prediction, per-class counts, confusion matrix counts and SHA-256 dataset manifest hashes.
- `reports/sprint-01/gradcam-smoke.json` and `gradcam-smoke_gradcam/0_true_0_pred_0.png`: optional explanation smoke evidence.
- `reports/sprint-01/benchmark-cli-smoke.json`: successful standalone benchmark CLI with labeled external evaluation; its ten-iteration smoke latency is separate from the table above.

The verified class map is versioned in `configs/class_to_idx.sprint01.json`. Large checkpoints and datasets remain local ignored artifacts; retain them with the report hashes to reproduce the results. Individual CLI commands and optional Grad-CAM flags are in [RUN_GUIDE.md](../RUN_GUIDE.md).
