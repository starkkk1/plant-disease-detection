# Classification API contract v1 — issue #4

Implemented by `backend/main.py`, consumed by `frontend/src/lib/prediction-api.ts`. This is the local MVP contract chosen for the implementation; team review can use the examples and generated OpenAPI. It does not require Qdrant, a text encoder, image search or Grad-CAM.

## Existing backend audit

| Existing code | Finding | Decision |
|---|---|---|
| `scripts/run_search_api.py` | Runs `src.search.api` with reload on port 8000 | Keep the research service separate; run it on port 8001 when needed |
| `src/search/api.py` | Loads an encoder and predictor at startup; prediction has hard-coded label order and always attempts Grad-CAM; search needs Qdrant | Classification MVP uses the strictly loaded CPU predictor from issue #2 |
| `src/search/api.py` `/image` | Serves paths supplied by the client | The new API has no arbitrary-path file endpoint |
| `src/search/qdrant_engine.py` | Embedding retrieval dependency | Optional research functionality only |
| `src/demo/streamlit_app.py` | Imports a missing `src.models.multimodal_encoder`; search-oriented demo | Not used to start the MVP |
| Original Next.js `page.tsx` | Calls loopback URLs directly; expects `class_name`/`probability`; no timeout or file limits | New typed adapter, phone UI and environment-based origin; original research UI retained at `/research` |

## Request conventions

Local base URL: `http://localhost:8000`. Use HTTPS and set `NEXT_PUBLIC_API_URL` for hosting. The local browser fallback follows the page hostname at port 8000, so a phone uses the development computer's hostname/IP.

Requests may include `X-Request-ID` as a UUID. The server preserves valid UUIDs, generates one otherwise and returns it in `X-Request-ID` and JSON. No uploaded image or filename is logged. CORS allows explicitly configured origins, not a wildcard; browser credentials are not required. The MVP is for local demos and has no authentication or public abuse protection yet.

## GET /health

HTTP 200 means the real model loaded successfully. HTTP 503 means it failed to initialize. Health does not enqueue inference and remains responsive while classification is busy. It does not report Qdrant health or claim that the CPU queue is free.

```json
{
  "request_id": "ff7caa78-9d54-43ca-bbe0-5904cfe0ce36",
  "api_version": "1.0",
  "status": "ok",
  "model_version": "mobilenetv3_small_100:90b0a4bed0d2",
  "is_mock": false,
  "max_upload_bytes": 5242880,
  "max_image_pixels": 20000000,
  "supported_formats": ["image/jpeg", "image/png"]
}
```

On HTTP 503, `status` is `unavailable` and `model_version` is null. Internal paths and load-error details are not returned to clients. `/docs` and `/openapi.json` remain available for diagnosis.

## POST /predict

Send `multipart/form-data` with required binary `file` and optional integer `top_k` (default 3, range 1–11). Browser clients must let `fetch` set the multipart boundary; do not set `Content-Type` manually.

```powershell
curl.exe -F "file=@frontend/tests/fixtures/tomato-leaf.jpg;type=image/jpeg" -F "top_k=3" http://localhost:8000/predict
```

HTTP 200 contains ranked predictions:

```json
{
  "request_id": "12be436b-a27e-47e8-b780-7ad5407f8678",
  "model_version": "mobilenetv3_small_100:90b0a4bed0d2",
  "is_mock": false,
  "processing_time_ms": 127.83,
  "predictions": [
    {"class_id": 0, "label": "Tomato___Bacterial_spot", "confidence": 0.99999976},
    {"class_id": 6, "label": "Tomato___Target_Spot", "confidence": 0.000000227},
    {"class_id": 7, "label": "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "confidence": 0.0000000154}
  ]
}
```

The example is from the real local smoke run, rounded for readability. Timing includes handler upload read, queue wait, image validation and inference; it is not the controlled model latency from the baseline. Labels and class IDs follow the verified training mapping. `model_version` identifies the architecture and first twelve checkpoint SHA-256 characters. Confidence is softmax, not calibrated certainty; the returned top-k scores need not sum to one. The API never substitutes mock results. The frontend mock adapter uses `is_mock=true` and `demo-mock-v1` only after explicit user selection.

## Validation, limits and timeout

| Limit | Default | Behavior |
|---|---:|---|
| File size | 5 MiB / 5,242,880 bytes | Bounded read; larger file gives 413 |
| Complete multipart body | File limit + 65,536 bytes | ASGI receive counter also bounds chunked bodies without Content-Length |
| Decoded pixels | 20,000,000 | Larger/decompression-bomb images give 413 |
| Accepted types | JPEG, PNG | Actual decoder format must match media type; unsupported type gives 415 |
| CPU calls in flight | 1 per worker | One executor and semaphore, no unbounded inference queue |
| Queue wait | 1 second | 503 `service_busy` if occupied |
| Decode + prediction wait | 20 seconds | 504 `prediction_timeout` |
| Browser request | 25 seconds | Abort fetch and show retry; user can also cancel |

The server's timeout starts after acquiring the slot; upload parsing is bounded by bytes, not by this CPU deadline. A running native CPU operation cannot be forcibly killed safely: a timed-out call retains its slot until completion, while health stays responsive. Use one Uvicorn worker for the MVP; additional workers each load another model and allocate their own slot. Closing or cancelling a browser request does not imply the CPU computation stopped.

## Errors

Every handled HTTP error has the same envelope and request ID header:

```json
{
  "request_id": "bb7bc2ad-3eb5-4f21-89b7-613b29d5c907",
  "error": {"code": "invalid_image", "message": "The image is empty or cannot be decoded."}
}
```

| HTTP | Codes | Client action |
|---:|---|---|
| 400 | `invalid_image` | Select another image |
| 413 | `file_too_large`, `request_too_large`, `image_too_large` | Reduce file or image size |
| 415 | `unsupported_image` | Supply actual JPEG/PNG with matching MIME |
| 422 | `invalid_request` | Check file field and top_k |
| 503 | `model_unavailable`, `service_busy` | Retry after the service is ready |
| 504 | `prediction_timeout` | Retry; the old call may still occupy the worker |
| 500 | `prediction_failed` | Show a generic error; correlate server logs using request_id |

Exception details stay in server logs; they are not rendered in the product UI. Network failures, invalid response JSON/schema and browser aborts are separate frontend errors. No API error automatically switches to a mock result.

## Compatibility and evidence

The public schema uses `label` and `confidence`, not the older `class_name`/`probability` keys. The mobile UI consumes v1 directly; `/research` maps classification v1 to the legacy view. Legacy image search still requires the separate search service. Preserve these required fields and add optional fields for compatible changes; renamed fields, changed units or changed semantics require a new contract version and adapter update. Checkpoint replacements change `model_version`; class-map replacements must retain a verified training association.

The implementation is exercised by `tests/test_backend.py` and `frontend/tests/prediction.spec.ts`. Real HTTP evidence is in `reports/sprint-01/api-live-smoke.json`; OpenAPI is generated from `backend/schemas.py` and the actual endpoints.
