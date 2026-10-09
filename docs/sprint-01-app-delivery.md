# Sprint 01 local handoff — issues #1–#9

Implemented and verified locally on 2026-10-08 (Asia/Bangkok), following the completed model audit/predictor/baseline from issues #1–#3.

| Issue | Deliverable | Status and evidence |
|---|---|---|
| #1 model audit | `docs/sprint-01-model-audit.md` | All five real checkpoints, training class map, transforms, configs and model sizes audited; provenance limitations recorded |
| #2 reusable CPU inference | `src/inference/predictor.py`, `preprocess.py` | Strict checkpoint loading, shared preprocessing, reusable real inference with class/confidence/model version; inference tests passed |
| #3 baseline | `docs/sprint-01-baseline.md`, `reports/sprint-01/` | Real-checkpoint CPU p50/p95, full labeled test/external evaluation and reproducible commands delivered; no retraining |
| #4 API audit/contract | `docs/api-contract.md` | Existing backend/demo/search/frontend audited; implemented request/response/errors/limits/timeouts are consumed by the actual typed frontend adapter |
| #5 local FastAPI | `backend/main.py`, `schemas.py`, `settings.py`, `requirements.txt` | Ready locally: strict real CPU predictor, health/readiness, multipart validation, byte/pixel caps, request IDs, finite CPU slot, timeout and OpenAPI; 19 backend integration tests passed |
| #6 AWS readiness | `docs/architecture.md`, `docs/aws-readiness.md`, `Dockerfile`, `deploy/*.example.json` | Design and review artifacts delivered; account permissions/eligibility, Region, approved budget and named cost approver remain unverified. Docker build is blocked by an unavailable daemon. No paid AWS resources created |
| #7 mobile approach | `docs/mobile-plan.md` | Reuse responsive Next.js; compare against a new Flutter project using the repository evidence; record reuse, device target and subsequent work |
| #8 phone UI | `frontend/src/app/page.tsx`, `globals.css`, separate mock adapter | Gallery/camera input, preview, empty/loading/result/error/cancel/retry states, explicit demo mode; tested at Pixel 7 emulated mobile dimensions and desktop |
| #9 API integration | `frontend/src/lib/prediction-api.ts` | Real multipart prediction, validated label/confidence/model version, configurable URL, timeout/HTTP/network/invalid-file errors; real-model end-to-end flow passed |

GitHub issue state was not changed. No commits or pushes were made in this task. Human team agreement is not fabricated: the contract/architecture/choice are concrete implementation decisions available for review. Company-specific AWS readiness is a remaining external prerequisite, not an implemented or approved deployment.

## Validation

- Full Python suite after Grad-CAM++ integration (2026-10-09): **59 passed** (26 inference/class-map tests, 28 API integration tests, 5 explanation tests). Final command used a workspace temp directory because of workstation Temp ACLs.
- Next.js **15.5.27** production build: **passed**, including TypeScript and lint checks; inherited research-page `<img>` optimization warnings are non-fatal. Clean installation from the regenerated lockfile passed.
- Playwright: **24 passed** across mobile and desktop after Grad-CAM++ integration, starting fresh local API/frontend servers and completing normally (52.5 seconds). Covers actual CPU prediction/explanation, explicit mock mode, unsupported/oversized files, HTTP retry, unavailable API, malformed responses, cancellation and timeout. Explanation failures preserve the prediction. Backend artifacts are required for the real-image case.
- Live API: `/health`, `/docs`, `/openapi.json` and actual JPEG `/predict` all succeeded. `reports/sprint-01/api-live-smoke.json` includes the request ID/header and real model version.
- API launcher: `scripts/run_api.py` supplies the project import directory explicitly. Startup and actual JPEG prediction passed from both the parent workspace and `frontend/`, using the ML interpreter. Evidence: `reports/sprint-01/api-launcher-smoke.json`; test servers were stopped afterward.
- UI screenshot: `reports/sprint-01/frontend-mobile.png`, captured after real inference and visually inspected; no horizontal page overflow at the tested phone width.
- Grad-CAM++: all five real checkpoints generated valid heatmap/overlay PNGs without changing subsequent predictions. `/explain` defaults to the top prediction or accepts a training class ID. See [explain.md](explain.md), `reports/sprint-01/explain-validation.json`, and `reports/sprint-01/frontend-explain-mobile.png`.
- Docker/AWS: **not run**. Docker client has no usable daemon; no AWS CLI/SSO profile or authenticated connection for the supplied Console name `stark1` was available.
- Dependency audit after updates: **0 runtime findings**, 9 remaining build/lint dependency findings (0 critical). Raw reports and migration details are linked in [frontend-dependency-review.md](frontend-dependency-review.md).

Camera capture input is implemented and tested for the correct attributes. Its actual native camera/picker behavior is device-dependent and has not been falsely reported as physical-phone testing. Browser emulation uses Chromium/Edge on Windows, not Android OS emulation.

## Local API mismatch recovery — 2026-10-09

The web's "invalid response" error was reproduced against the live service: `scripts/run_search_api.py` was listening on 8000, returning HTTP 200 with legacy `model_status` and `class_name`/`probability`, while `/health` returned 404. The classification frontend correctly rejected that incompatible schema.

The verified legacy process was replaced with `scripts/run_api.py` on 8000. The search launcher's default port was changed to 8001, and the frontend now identifies this legacy response as an API version mismatch. Two new targeted browser regression cases passed on mobile and desktop; the production build passed. These targeted cases are additional to the earlier full 24-case run, not a claim that the full expanded suite was rerun.

The existing local frontend was restarted on 3000 and checked directly in Chromium: upload, actual model prediction, and the Grad-CAM++ overlay all succeeded without a UI error. Evidence: `reports/sprint-01/inference-server-fix.json` and `reports/sprint-01/inference-live-fixed.png`. Classification API and frontend were left running for the user at `http://localhost:8000` and `http://localhost:3000`. Recovery-process logs are in `.tmp/backend-live.stderr.log` and `.tmp/frontend-live.stdout.log`/`frontend-live.stderr.log`.

## Boundaries for hosting

AWS preparation records the App Runner new-customer cutoff and an ECS Express alternative with review-only templates. Neither option is selected as permitted for the company without its access/approval record. Images are classified locally; optional Qdrant remains a separate research service. Grad-CAM++ is now available through the optional `/explain` action using the same classification model. The classification service is unauthenticated for the local MVP; public hosting requires company controls. Runtime dependency advisories were addressed; remaining build/lint findings are documented for a subsequent toolchain migration. No public deployment was performed. AWS-dependent work is deferred at the user's request; the local implementation does not require AWS.

Follow [RUN_GUIDE.md](../RUN_GUIDE.md) to reproduce the local/LAN demo and tests. Follow [aws-readiness.md](aws-readiness.md) to finish company-specific readiness without sharing credentials or creating unapproved paid resources.
