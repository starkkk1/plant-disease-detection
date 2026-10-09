# Lá — tomato leaf phone UI

Responsive Next.js frontend for the standalone classification API. The main route `/` supports camera/gallery JPEG/PNG selection, preview, processing/cancellation/retry and ranked results with model version. Demo results are explicitly labeled and require selecting the demo checkbox. The previous research/search UI is retained at `/research`.

Real predictions now offer **Xem giải thích Grad-CAM++**, calling `/explain` with the selected image and predicted class. The overlay has separate loading/cancel/error/retry states and preserves the classification result. Mock mode does not fabricate explanations. See [explain.md](../docs/explain.md).

```powershell
npm ci
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Run the backend in another terminal using the ML environment. From `frontend/`, run `python ../scripts/run_api.py --host 0.0.0.0 --port 8000`; the launcher resolves the backend's project directory automatically. Use the verified ML Python environment and real checkpoint as described in [RUN_GUIDE.md](../RUN_GUIDE.md).

The API origin is `NEXT_PUBLIC_API_URL`, which can be placed in `.env.local`. If unset, local demos follow the page hostname at port 8000, including LAN phone demos. For a hosted deployment, set the actual HTTPS API origin at build time and allow the frontend origin in backend `CORS_ORIGINS`. No AWS credentials or secrets belong in public frontend variables. `NEXT_PUBLIC_SEARCH_API_URL` is optional for the separate local research search service on port 8001.

```powershell
npm run build
$env:PYTHON_EXECUTABLE='D:\Code\python\.venv\Scripts\python.exe'
npm run test:e2e
```

E2E checks use mobile and desktop Chromium, start services if needed, test real CPU prediction and Grad-CAM++ overlays, and separately intercept errors and explicit mock mode. Windows uses installed Edge; other platforms need `npx playwright install chromium`. See `playwright.config.ts`. Final run after `/explain` integration: 24 passed. Screenshots are `../reports/sprint-01/frontend-mobile.png` and `../reports/sprint-01/frontend-explain-mobile.png`; phone OS camera behavior still needs a physical-device check.

No service worker, offline classification or installable PWA is claimed. See [mobile-plan.md](../docs/mobile-plan.md) for the choice against starting a new Flutter app. Next.js is updated to 15.5.27 with patched PostCSS and a regenerated lockfile: the runtime audit reports zero findings; remaining build/lint findings are recorded in [frontend-dependency-review.md](../docs/frontend-dependency-review.md). This task verified a local MVP and did not deploy it publicly.
