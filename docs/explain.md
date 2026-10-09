# Grad-CAM++ API and phone UI

Implemented and verified locally on 2026-10-09. Uses [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), pinned as `grad-cam==1.5.5` in the backend requirements. No retraining, AWS service or image storage is required.

## Request

`POST /explain`, `multipart/form-data`:

| Field | Required | Meaning |
|---|---|---|
| `file` | Yes | Actual JPEG/PNG, with matching MIME type |
| `top_k` | No | 1–11, default 3, same prediction ranking as `/predict` |
| `class_id` | No | Training class index 0–10; omit to explain the highest-scoring class |

```powershell
curl.exe -X POST http://localhost:8000/explain -F "file=@frontend/tests/fixtures/tomato-leaf.jpg;type=image/jpeg" -F "top_k=3"
# Explain the healthy class even if it is outside the top three:
curl.exe -X POST http://localhost:8000/explain -F "file=@frontend/tests/fixtures/tomato-leaf.jpg;type=image/jpeg" -F "class_id=9"
```

Run these example requests from the repository root. Start the server using `scripts/run_api.py` as described in [RUN_GUIDE.md](../RUN_GUIDE.md), or try the endpoint directly at `/docs`. Restart a previously running API process to load the new route.

## Response

The response retains `/predict` fields: `request_id`, `model_version`, `is_mock=false`, `processing_time_ms`, and `predictions`. It also contains `explanation`:

| Field | Meaning |
|---|---|
| `method` | `gradcam++` |
| `target` | Class ID, training label and softmax confidence of the explained class |
| `target_layer` | Actual named model layer used for activations/gradients |
| `width`, `height` | Model input dimensions; 224 × 224 by default |
| `has_signal` | Whether the normalized map has any positive attribution above the numerical threshold |
| `overlay_png_base64` | RGB PNG containing the resized input plus color heatmap |
| `heatmap_png_base64` | Grayscale PNG with normalized attribution encoded as values 0–255 |

Display the overlay as `data:image/png;base64,` plus `overlay_png_base64`. Base64 fields contain encoded PNG bytes, not filesystem paths or remote URLs. The overlay uses the same direct square resize as the model input; it is not a map over a center crop or the original photo's full resolution. Heatmap values are normalized within the image and are not class probabilities or comparable magnitudes across images. A zero map is returned honestly with `has_signal=false`.

## Execution and errors

`/predict` remains under `torch.inference_mode()`. `/explain` runs a separate gradient-enabled forward/backward pass on the same loaded model and preprocessing. Both endpoints share one executor/CPU semaphore. Context-managed CAM hooks are released and parameter gradients are cleared after each request, including failures. A timed-out call keeps the slot until its actual computation ends.

The same upload limits, MIME checks, RGB conversion, request IDs, queue wait and CPU/browser deadlines apply to both routes. Additional errors are 503 `explanation_unavailable` for missing CAM dependencies or unconfigured architectures, and 500 `explanation_failed` for an internal explanation failure. Invalid `class_id` gives 422 `invalid_request`; timeout uses the existing 504 `prediction_timeout`. Internal exception details remain in server logs.

Supported checkpoint architectures and target layers:

| Architecture | Layer |
|---|---|
| MobileNetV3 Small (`mobilenetv3_small_100`) | `blocks.5` |
| EfficientNet-B0 (`efficientnet_b0`) | `blocks.6` |
| ConvNeXt Tiny (`convnext_tiny`) | `stages.3.blocks.2.conv_dw` |

The same layer selection supports both independent and distilled student weights. An unconfigured architecture is rejected explicitly.

## Frontend

After a real prediction, press **Xem giải thích Grad-CAM++**. The client resubmits the same selected image and top prediction's class ID, then validates the response and matching model version before rendering the overlay. Explanation loading/error/cancel/retry states preserve the prediction result. Resetting/changing the image clears its explanation and invalidates stale responses. Demo mode does not offer fabricated Grad-CAM maps.

The caption distinguishes model attribution from confirmed disease localization. Red/yellow indicate higher positive attribution for the requested class; a CAM is not a lesion segmentation or proof that a diagnosis is correct.

## Validation

- **59 Python tests passed**, covering class-specific maps on deterministic convolutional weights, PNG shapes, unchanged predictions/weights, gradient and hook cleanup, API validation/errors and timeout contention between `/explain` and `/predict`.
- All five real checkpoints generated valid 224 × 224 PNGs on CPU; predictions before/after explanation matched. An actual API request explicitly explaining class 9 passed, followed by healthy `/health` and `/predict` calls. See [explain-validation.json](../reports/sprint-01/explain-validation.json). Recorded per-call timings are smoke observations, including cold imports in the first call, not a controlled latency benchmark.
- **24 browser tests passed**, exercising the real overlay on mobile/desktop and explanation error/retry, malformed responses and cancellation. See `frontend/tests/prediction.spec.ts`; `frontend-explain-mobile.png` is captured from the real API.

For a fresh backend installation, install `backend/requirements.txt`. Grad-CAM's package declares `opencv-python`, so the API requirements use that wheel rather than installing both GUI and headless wheels. The Dockerfile includes its Linux `libgl1`/`libglib2.0-0` runtime libraries. Docker build remains unverified on this workstation with no running daemon; local native API tests are recorded separately.
