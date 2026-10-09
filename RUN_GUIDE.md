# Plant Disease Detection - Run Guide

This guide provides the quick commands needed to run the various scripts in this project. Unless a section states otherwise, run commands from the root directory of the project (`plant-disease-detection`).

## Local classification MVP (issues #4–#9)

Terminal 1 (this command also works from `D:\Code\python` or `frontend/`):

```powershell
# With the verified existing ML environment:
& D:\Code\python\.venv\Scripts\python.exe D:\Code\python\plant-disease-detection\scripts\run_api.py --host 0.0.0.0 --port 8000
```

For a new environment, install `backend/requirements.txt` with a supported CPython interpreter. It installs CPU-only torch/torchvision, Grad-CAM++ and the API dependencies without Qdrant or Streamlit. Keep the real `checkpoints/mobilenetv3_small_100_best.pth`; the verified class map is included in `configs/class_to_idx.sprint01.json`. The API fails readiness if the checkpoint is absent or incompatible. No retraining is needed.

Terminal 2:

```powershell
Set-Location D:\Code\python\plant-disease-detection\frontend
npm ci
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Open `http://localhost:3000`; Swagger is `http://localhost:8000/docs`, and readiness is `http://localhost:8000/health`. Choose or capture a JPEG/PNG photo, preview it and press the recognition button. The unchecked demo toggle uses the real API; explicitly checking it enables labeled frontend sample results. It is never enabled automatically when the API fails.

After a real result, press **Xem giải thích Grad-CAM++** to view the overlay for the top predicted class. The new `POST /explain` route also accepts an optional training `class_id`. Explanation can be cancelled/retried without discarding the prediction. It is unavailable in demo mode. Restart an existing backend process and restart/rebuild the frontend to load the new feature. See [explain.md](docs/explain.md) for requests, response images and CPU validation.

### If Python cannot import `backend`

`backend` belongs to `plant-disease-detection/`, not its parent folder or `frontend/`. The launcher above supplies Uvicorn's `--app-dir` using its own file location, so it resolves the project's module regardless of the terminal's current directory. An equivalent direct command is:

```powershell
& D:\Code\python\.venv\Scripts\python.exe -m uvicorn backend.main:app --app-dir D:\Code\python\plant-disease-detection --host 0.0.0.0 --port 8000
```

For a different checkout, replace the two absolute paths with its Python environment and project directory. When already in the repository root with the correct environment activated, `python scripts/run_api.py --port 8000` also works. Do not run `python backend/main.py` directly; its package imports require the repository on Python's module path. Check the interpreter with `python -c "import sys; print(sys.executable)"`: this workstation's default `python` is MSYS and does not contain the verified ML dependencies.

### If the web reports an invalid API response

Check `http://localhost:8000/health`: the classification service must return `status: "ok"`, `api_version: "1.0"` and a real `model_version`. A 404 here, combined with `/predict` returning `model_status` and `class_name`/`probability`, identifies the old search API running on the classification port. Stop that search process and start **`scripts/run_api.py`** using the command above. Refresh the web page afterward. The frontend rejects legacy predictions and identifies the API version mismatch instead of inventing missing class IDs/model versions.

**`scripts/run_search_api.py` is the research search launcher**, now using port **8001** by default. It does not replace the classification API on port **8000** and does not provide the new `/explain` endpoint. Both services can run separately when search is needed.

To set a hosted or different backend, put `NEXT_PUBLIC_API_URL=https://YOUR_APPROVED_API_ORIGIN` in `frontend/.env.local` or build environment, then restart/rebuild Next.js. This is a public endpoint setting, not a secret. If unset, the client uses the page hostname at port 8000. `backend/.env.example` documents backend variables; set them in the process environment rather than assuming the file is automatically loaded.

### Phone on the same Wi-Fi

Use the computer's LAN IPv4 address as `YOUR_LAN_IP`. Start the API with the phone page's exact origin allowed:

```powershell
$env:CORS_ORIGINS='http://localhost:3000,http://127.0.0.1:3000,http://YOUR_LAN_IP:3000'
& D:\Code\python\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Start Next.js on `0.0.0.0` as above and open `http://YOUR_LAN_IP:3000` on Android Chrome. Confirm `/health` is reachable at `http://YOUR_LAN_IP:8000/health`. Allow the development ports only on the trusted private network using the company's firewall rules. Camera/picker behavior is device-dependent; JPEG/PNG is supported, HEIC is not. Physical-phone testing remains a manual check; the recorded automated demo uses mobile browser emulation.

### Tests and evidence

```powershell
& D:\Code\python\.venv\Scripts\python.exe -m pytest tests -q
Set-Location frontend
npm run build
$env:PYTHON_EXECUTABLE='D:\Code\python\.venv\Scripts\python.exe'
npm run test:e2e
```

Playwright runs real-model and mocked-error scenarios on mobile and desktop and starts local services on ports 8000/3005 if needed. It uses installed Microsoft Edge on Windows; on other systems install Chromium with `npx playwright install chromium`. A missing checkpoint prevents the real E2E test from passing; the mock UI is a separate explicit scenario.

On this workstation, Windows temp-folder ACLs require a fresh workspace temp directory: the Grad-CAM++ test run used `-p no:cacheprovider --basetemp D:\Code\python\plant-disease-detection\.tmp\pytest-explain-01`. Choose a new directory if rerunning under restricted ACLs. Next.js 15.5.27 with patched PostCSS built successfully; Playwright started fresh local servers using `--output test-results-explain --reporter list`. Final results after `/explain` integration on 2026-10-09: **59 Python tests, 24 browser tests passed**, including real-model Grad-CAM++. Existing research-page image-optimization lint warnings remain non-fatal. The recorded runtime dependency audit from 2026-10-08 reports zero findings; remaining build/lint findings are in [frontend-dependency-review.md](docs/frontend-dependency-review.md).

Evidence: `reports/sprint-01/api-live-smoke.json`, `reports/sprint-01/frontend-mobile.png`, and [delivery status](docs/sprint-01-app-delivery.md). API details are in [api-contract.md](docs/api-contract.md), phone decision in [mobile-plan.md](docs/mobile-plan.md), cloud design in [architecture.md](docs/architecture.md), and company-access gaps in [aws-readiness.md](docs/aws-readiness.md). Docker and paid AWS deployment have not been run.

### Optional research search

The original frontend research UI is retained at `/research`. Its prediction reads the new classification API. To enable the old Qdrant search separately, start `python -m uvicorn src.search.api:app --host 127.0.0.1 --port 8001` with the existing research dependencies and Qdrant index. `NEXT_PUBLIC_SEARCH_API_URL` can override that origin. The old search service and its path-based image endpoint are for the local research workflow; they are not part of the new container/API.

## 1. Setup Environment
Ensure your virtual environment is activated and dependencies are installed.
```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

For tests, install `requirements-dev.txt` and run `python -m pytest tests -q`.
CPU inference works with a CPU PyTorch installation; CUDA is optional. The verified local interpreter is `D:\Code\python\.venv\Scripts\python.exe` (the default MSYS Python has no PyTorch). Activate that environment or substitute its full path for `python` in the commands below.

## 1.5. Training Models
To train the Teacher model (ConvNeXt):
```bash
python scripts/train_convnext.py
```

To run Knowledge Distillation for both Student models (MobileNetV3 and EfficientNet-B0) sequentially:
```bash
python scripts/train_distillation.py --config configs/distillation_mobilenetv3.yaml && python scripts/train_distillation.py --config configs/distillation_effnetb0.yaml
```
**Outputs:**
- Logs and metrics will be saved in `results/`.
- Best model checkpoints will be saved in `checkpoints/`.


## 2. Evaluation and optional Grad-CAM++

Run from the project root. Use the config matching the checkpoint; no training or pretrained-weight download occurs. Missing or incompatible artifacts stop execution.

```bash
python scripts/evaluate.py --config configs/distillation_mobilenetv3.yaml --checkpoint checkpoints/mobilenetv3_small_100_distilled_best.pth --dataset data/new-data-removal/test --device cpu --threads 4 --output reports/mobilenet_test.json
python scripts/evaluate.py --config configs/distillation_mobilenetv3.yaml --checkpoint checkpoints/mobilenetv3_small_100_distilled_best.pth --dataset data/new-data-removal/eval --device cpu --threads 4 --output reports/mobilenet_external.json
```

JSON reports include accuracy, macro-F1/precision/recall over all training classes, confusion matrix counts in training order, per-image predictions, class counts and artifact/dataset hashes. Add `--gradcam --gradcam-count 5` to save optional explanations next to the JSON. Add `--limit-per-class 1` for an explicitly labeled smoke subset; omit it for full evaluation. `--class-map` can override the config's training map.

The earlier research workflow is archived in `scripts/evaluate_gradcam_legacy.py`; its center-crop metrics and cleanup behavior are retained for reference. Use the new commands for inference-aligned metrics.

## 3. Benchmarking (Latency & Size)
To measure model size (MB), parameter count (M), and inference latency (ms) on CPU:
```bash
python scripts/benchmark.py --config configs/distillation_mobilenetv3.yaml --checkpoint checkpoints/mobilenetv3_small_100_distilled_best.pth --dataset data/new-data-removal/eval --device cpu --threads 4 --warmup 10 --iterations 100 --evaluate --output reports/mobilenet_benchmark.json
```
**Outputs:** 
- Trained checkpoint size (MiB), parameters, individual latency samples, p50/p95 and optional labeled accuracy/macro-F1 in JSON.
- Latency includes file decode, RGB conversion, resize, normalization, forward pass and top-1 output. Model loading, Grad-CAM and network overhead are excluded. `--image path/to/leaf.jpg` provides latency without labels; omit `--evaluate` in that case.

To reproduce the complete five-model audit and baseline on all local test/eval images:

```bash
python scripts/audit_models.py --threads 4 --iterations 100 --output-dir reports/sprint-01
```

This validates `class_to_idx.json` against the current training dataset and saves `baseline.json` plus ten detailed evaluation JSON files. See `docs/sprint-01-model-audit.md` and `docs/sprint-01-baseline.md` for findings and data limitations. Weights and datasets remain local artifacts ignored by Git.

## 4. Build Qdrant Search Index (Vector Database)
To encode the dataset and push vectors into Qdrant for semantic search:
```bash
# For MobileNetV3 (Default)
python scripts/build_qdrant_index.py --model mobilenet

# For EfficientNet-B0
python scripts/build_qdrant_index.py --model efficientnet
```
**Outputs:** 
- Prints extraction progress and creates collections (`tomato_disease_multimodal` or `tomato_disease_efficientnet`) inside your Qdrant container.
