# Lá — tomato leaf phone UI

## Docker (production build)

The GitHub Actions workflow in `.github/workflows/ci.yml` also builds the frontend
image and checks its healthcheck, non-root runtime, pages and static assets.
Pull requests run these checks without publishing. Successful master builds
publish the verified frontend image to
`ghcr.io/starkkk1/plant-disease-detection-frontend:latest` and a full commit-SHA tag,
alongside the existing backend image. CI does not need a real checkpoint for the
frontend smoke test; real model integration is tested separately. Publishing an
image does not automatically deploy it to the local machine or AWS.

From the repository root, with Docker Desktop running:

```powershell
docker compose up -d --build
docker compose ps
docker compose logs --tail 50 frontend
```

Compose keeps the existing backend image/checkpoint mount and adds a separate
frontend container. The backend image `plant-disease-api:v1.0.0` must already be
available locally or in an accessible registry. Open `http://localhost:3000` on
the computer, or `http://<computer-LAN-IP>:3000` on a phone on the same network.
The default frontend calls that same hostname on port 8000; backend CORS must
allow the actual frontend origin. The optional `/research` search service is
separate and is not started by this Compose file.

To build/run only the frontend with an already running backend:

```powershell
docker compose up -d --build frontend
```

Do not run a second backend on host port 8000 at the same time. Stop a native
Next.js process using port 3000 before starting the frontend container.

Standalone Docker commands (also from the repository root):

```powershell
docker build -t plant-disease-frontend:local ./frontend
docker run -d --name plant-disease-web -p 3000:3000 plant-disease-frontend:local
```

The multi-stage Dockerfile builds Next.js standalone output, then runs only the
production server and its assets as the non-root `node` user. The frontend build
context is `./frontend`, independent of the backend's root `.dockerignore`.

`NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_SEARCH_API_URL` are optional **build-time**
arguments. They are public browser URLs: do not set them to Docker-only service
names such as `http://backend:8000`. Leave the API URL unset for the LAN default,
or build with a URL reachable from the browser:

```powershell
docker build --build-arg NEXT_PUBLIC_API_URL=http://192.168.2.48:8000 -t plant-disease-frontend:local ./frontend
```

For Compose, set `$env:NEXT_PUBLIC_API_URL` before `docker compose up -d --build frontend`.
Changing these values requires rebuilding the frontend; `docker run -e` alone
does not change the browser bundle. Never put credentials in these variables.
To publish the frontend on a different host port, set `$env:FRONTEND_PORT='3001'`
before starting Compose and add the corresponding frontend origin to backend
`CORS_ORIGINS`.

Stop the Compose services with `docker compose stop`; restart existing containers
with `docker compose start`. Use `docker compose up -d --build frontend` after
frontend code changes so the image and container are updated.

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
