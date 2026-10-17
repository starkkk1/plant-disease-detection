# Plant Disease Detection - Run Guide

Updated **2026-10-16**. This is the operational guide for the local application and the personal-account AWS demo in **Sydney (`ap-southeast-2`)**. Run project commands from the repository root unless stated otherwise.

| Terminal | Purpose |
|---|---|
| Windows PowerShell | Local development and SSH into EC2 |
| EC2 SSH (Bash) | S3 downloads, Docker containers, application checks and Agent management |
| AWS CloudShell (Bash) | EC2 start/stop using the signed-in operator's AWS permissions |

No AWS access keys or private SSH keys belong in Git. EC2 uses an instance role for S3 and CloudWatch. Replace placeholders with actual values; public IPv4 addresses are not permanent deployment identifiers.

[AWS operations](#aws-operations-sydney) · [Local Docker](#local-docker-compose) · [Native development](#local-native-development) · [Tests](#tests-and-validation) · [Research](#research-commands)

## AWS operations (Sydney)

### Current release and prerequisites

The operator verified real backend inference on EC2 and frontend page access from desktop and phone. S3 artifact hashes match local files. CloudWatch Agent namespace/disk ingestion was shown in Console. Dashboard/RAM-alarm configuration and SNS email delivery must be verified in Console; a configured alarm is not proof that its notification was delivered. No fresh AWS access is performed by this document.

- Amazon Linux 2023, **x86_64**, with Docker and AWS CLI v2; project cloned into `~/plant-disease-detection`.
- EC2 instance role with `s3:GetObject` on the actual model prefix and `CloudWatchAgentServerPolicy` for logs/metrics. The deployed role is named `plantdiseaseec2role`.
- To use `aws logs tail`, add `logs:FilterLogEvents` scoped to the application's two log groups. The log-group resource ARN uses the form `arn:aws:logs:ap-southeast-2:ACCOUNT_ID:log-group:/plant-disease/backend:*`; include the frontend group as well.
- Private S3 bucket, prefix **`mobilenet-distilled-v1/`**, with `model.pth` and `class_to_idx.json`. If source filenames differ, use the exact S3 object names in download commands.
- CloudWatch Standard log groups **`/plant-disease/backend`** and **`/plant-disease/frontend`**, retention **7 days**.
- Public subnet with an Internet Gateway route, public IPv4, and security-group access to TCP 22/3000/8000 from approved client public IPs. A phone using a different network may need a separate source-IP rule.
- This is an HTTP demo. Domain/HTTPS, authentication and multi-instance availability have not been implemented by this runbook.

CI publishes both images to GHCR after successful `master` checks. Frontend image health, pages, static assets and non-root execution are checked in CI; Python tests use fixture weights. Pulling images and replacing EC2 containers remain manual.

### Connect and select the deployed release

On **Windows**, use the instance's current Public IPv4 from the EC2 Console:

```powershell
$ec2Ip = Read-Host 'Current EC2 public IPv4'
ssh -i "D:\Code\python\plant-disease-demo.pem" "ec2-user@$ec2Ip"
```

The private key path is workstation-specific. If SSH reports an unprotected private key, restrict its Windows ACL to the operator; never paste key contents or commit the key. Amazon Linux uses `ec2-user`.

In **EC2 SSH**, set variables again after each reconnect:

```bash
cd ~/plant-disease-detection
export AWS_DEFAULT_REGION=ap-southeast-2
read -r -p "Current EC2 public IPv4: " PUBLIC_IP
export PUBLIC_IP

export BACKEND_IMAGE='ghcr.io/starkkk1/plant-disease-detection-api@sha256:f61be1570a834f1819614ad041f79e6114e00a0503852a121a564725cad4c27c'
export FRONTEND_IMAGE='ghcr.io/starkkk1/plant-disease-detection-frontend@sha256:ec211d13241680f208b29f8282c7fb12a19735e9048d1f8dab9b6a27499a7f85'

aws sts get-caller-identity
sudo docker pull "$BACKEND_IMAGE"
sudo docker pull "$FRONTEND_IMAGE"
```

These digests identify the operator-confirmed release from source commit `ff6b75bf3a625db124cc6449f52fa224bf76631e`. For a later release, record both new digests after successful publication. `latest` and commit tags are names; digest references fix the image content. A private GHCR package requires `sudo docker login ghcr.io` with read access before pulling.

### Download and verify S3 artifacts

Skip downloading when the verified files already exist. Do not overwrite model files mounted by a running service; prepare another version directory for a new model release.

```bash
read -r -p "S3 bucket name (without s3://): " MODEL_BUCKET
mkdir -p checkpoints/v1

aws s3 cp \
  "s3://$MODEL_BUCKET/mobilenet-distilled-v1/model.pth" \
  checkpoints/v1/model.pth
aws s3 cp \
  "s3://$MODEL_BUCKET/mobilenet-distilled-v1/class_to_idx.json" \
  checkpoints/v1/class_to_idx.json

chmod 755 checkpoints checkpoints/v1
chmod 644 checkpoints/v1/model.pth checkpoints/v1/class_to_idx.json

printf '%s\n' \
  '05e72798a74193499f17c4c8267c1436e8bc65cc1e8b8037245664a17644efa5  checkpoints/v1/model.pth' \
  'ff5c73e7a7e3e495593fa664b75ebc73de4c9a425f1d847351ae60bc1fbaa786  checkpoints/v1/class_to_idx.json' \
  | sha256sum -c -
```

Both files must report **OK** for this release. A `403 HeadObject` can mean missing `s3:GetObject` permission or an incorrect object key when the role lacks `s3:ListBucket`. Copy the exact S3 URI from Console and check the role's resource prefix; keep the bucket private.

### Create containers on EC2

Use these blocks only when the corresponding container name is absent. Check with `sudo docker ps -a`; for an existing stopped container, use the start commands below. Ensure `PUBLIC_IP`, image variables, artifacts and CloudWatch log groups are ready before creating containers.

Backend image has no checkpoint. Mount the verified directory read-only and explicitly select both artifacts:

```bash
: "${PUBLIC_IP:?Set the current EC2 public IPv4 first}"
: "${BACKEND_IMAGE:?Select the backend image digest first}"
sudo docker run -d \
  --name plant-disease-api \
  --restart unless-stopped \
  -p 8000:8000 \
  --mount "type=bind,source=$PWD/checkpoints/v1,target=/app/artifacts,readonly" \
  -e MODEL_CHECKPOINT=/app/artifacts/model.pth \
  -e MODEL_CLASS_MAP=/app/artifacts/class_to_idx.json \
  -e MODEL_NAME=mobilenetv3_small_100 \
  -e MODEL_THREADS=2 \
  -e "CORS_ORIGINS=http://$PUBLIC_IP:3000" \
  --log-driver awslogs \
  --log-opt awslogs-region=ap-southeast-2 \
  --log-opt awslogs-group=/plant-disease/backend \
  "$BACKEND_IMAGE"
```

Wait for model initialization and verify `/health` before starting the frontend:

```bash
curl --fail http://127.0.0.1:8000/health
```

Expected: `status: "ok"`, `api_version: "1.0"`, `is_mock: false`, model version **`mobilenetv3_small_100:05e72798a741`**. Retry after startup if the first connection is refused; a persistent 503 needs investigation.

```bash
: "${FRONTEND_IMAGE:?Select the frontend image digest first}"
sudo docker run -d \
  --name plant-disease-frontend \
  --restart unless-stopped \
  -p 3000:3000 \
  --log-driver awslogs \
  --log-opt awslogs-region=ap-southeast-2 \
  --log-opt awslogs-group=/plant-disease/frontend \
  "$FRONTEND_IMAGE"
```

The frontend runs the existing standalone production build (`node server.js`). With no compiled API URL override, the browser calls its page hostname on port 8000. `NEXT_PUBLIC_API_URL` is a **build-time** setting; adding it to `docker run -e` does not rewrite the browser bundle.

### Validate and inspect

```bash
sudo docker ps
curl -I http://127.0.0.1:3000

curl --fail-with-body \
  -H "Origin: http://$PUBLIC_IP:3000" \
  -F "file=@frontend/tests/fixtures/tomato-leaf.jpg;type=image/jpeg" \
  -F "top_k=3" \
  http://127.0.0.1:8000/predict

aws logs tail /plant-disease/backend --since 10m
aws logs tail /plant-disease/frontend --since 10m
sudo docker stats --no-stream
```

Open `http://CURRENT_PUBLIC_IP:3000` in a browser, upload a JPEG/PNG up to 5 MiB, request prediction, then Grad-CAM++. Confirm real model version and UI results. Opening the page alone does not validate API access. For browser CORS diagnosis, inspect response headers:

```bash
curl -i -H "Origin: http://$PUBLIC_IP:3000" http://127.0.0.1:8000/health
```

`Access-Control-Allow-Origin` must match the page origin. The variable is **`CORS_ORIGINS`**, plural. Changing a shell variable or YAML file does not change an existing container; recreate it to apply new environment values.

### Stop, start and restart the application

These commands run in **EC2 SSH** and keep the containers, images and model files:

```bash
# Stop application traffic, then inference.
sudo docker stop plant-disease-frontend
sudo docker stop plant-disease-api

# To resume later, start the backend and wait for health status ok.
sudo docker start plant-disease-api
curl --fail http://127.0.0.1:8000/health
# Run this only after the health request succeeds:
sudo docker start plant-disease-frontend
```

Retry the health request after initialization if needed. To restart an existing container without changing its image/environment:

```bash
sudo docker restart plant-disease-api
# Wait for /health to succeed before restarting/resuming the frontend.
sudo docker restart plant-disease-frontend
```

`docker start`/`restart` reuse the existing configuration. They do not deploy a pulled image or update CORS. Stopping containers does **not** stop EC2 compute billing.

### EC2 stop/start and public IP changes

Use the EC2 Console or the following **AWS CloudShell** commands with your operator identity. The application instance role does not need permission to start/stop EC2. Set `INSTANCE_ID` to the instance shown in your Console:

```bash
export AWS_DEFAULT_REGION=ap-southeast-2
read -r -p "EC2 instance ID: " INSTANCE_ID

# End a session: stop the machine and wait for completion.
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
```

In a later CloudShell session, set the region and instance ID again, then:

```bash
aws ec2 start-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID"
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

Public IPv4 usually changes after stop/start without an Elastic IP. SSH to the new address. On EBS-backed EC2, images, container definitions and checkpoint files remain on disk. In **EC2 SSH**:

```bash
sudo systemctl start docker
sudo systemctl start amazon-cloudwatch-agent
sudo docker ps -a
sudo docker start plant-disease-api
curl --fail http://127.0.0.1:8000/health
# Run after model readiness succeeds:
sudo docker start plant-disease-frontend
```

With `unless-stopped`, containers explicitly stopped by the operator need a manual start. If public IPv4 changed, recreate the backend to update CORS:

```bash
cd ~/plant-disease-detection
read -r -p "New EC2 public IPv4: " PUBLIC_IP
export PUBLIC_IP
: "${PUBLIC_IP:?Public IPv4 is required}"
export BACKEND_IMAGE=$(sudo docker inspect --format '{{.Config.Image}}' plant-disease-api)
: "${BACKEND_IMAGE:?Read the existing backend image before removing its container}"
sudo docker stop plant-disease-api
sudo docker rm plant-disease-api
```

Then rerun the **backend creation block** above from the repository root. It uses the retained image and model directory with the updated origin. This interrupts inference briefly. The default frontend needs no rebuild for a hostname change. Recheck security-group source IPs when client networks change.

Stopping EC2 does not delete it. EBS, S3 objects/versions, CloudWatch storage, alarms and any retained Elastic IP can continue to incur charges. Check credit balance/expiry and budget notifications; notifications are not a spending cap.

### Update and rollback

1. Merge changes and verify the complete GitHub Actions run, including publication of both images.
2. Record the running image references (`sudo docker inspect --format '{{.Name}} {{.Config.Image}}' plant-disease-api plant-disease-frontend`), artifact hashes and configuration before changing anything.
3. Pull new image digests. Prepare model changes in a new version directory and verify the matching map/preprocessing.
4. Validate a candidate on unused ports before replacing production containers. A candidate frontend on port 3001 needs its exact origin added to backend CORS for a browser test; the default frontend still calls API port 8000.
5. During a maintenance window, stop frontend/backend, remove only their container definitions with `docker rm`, select the new image variables and rerun creation blocks with the intended artifact paths. Keep old images/artifacts available. This method has downtime.
6. Repeat health, prediction, explanation and browser checks. Rollback uses the recorded old images **and** compatible model/map/config, recreating containers with those values.

`git pull` updates source; `docker pull` downloads images; neither changes running containers. Updating a model file on disk also does not reload the predictor already in memory. There is no automatic EC2 deployment or rollback in the current workflow.

### CloudWatch operations and RAM alarm

- Container logging driver: `awslogs`, the two `/plant-disease/...` groups, retention target 7 days. No manual copying of Docker log files is required.
- Host metrics: EC2 CPU in `AWS/EC2`; Agent RAM/disk in **`PlantDisease/Host`**, sampled every 60 seconds.
- Dashboard: **`plant-disease-demo`**; CPU Average / 5 minutes for basic monitoring, RAM and disk Average / 1 minute.
- Only application alarm in scope: **`plant-disease-high-memory`**, `mem_used_percent` Average > **85**, period 1 minute, **3 of 3** datapoints, missing data treated as **missing**.
- Notification: **`plant-disease-alerts`** SNS topic, confirmed email subscription, action on transition to ALARM. API-error alarms are not part of the chosen scope.

Agent lifecycle commands run in **EC2 SSH** after the Agent is installed/configured:

```bash
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a status
sudo systemctl enable --now amazon-cloudwatch-agent
```

For troubleshooting, use `sudo systemctl restart amazon-cloudwatch-agent`; to stop metric collection deliberately, use `sudo systemctl stop amazon-cloudwatch-agent`. Docker logs continue independently while containers run. Missing Agent data is not the same as healthy RAM usage.

The installed metrics configuration can be kept at `/opt/aws/amazon-cloudwatch-agent/etc/plant-metrics.json`:

```json
{
  "agent": {"metrics_collection_interval": 60},
  "metrics": {
    "namespace": "PlantDisease/Host",
    "append_dimensions": {"InstanceId": "${aws:InstanceId}"},
    "metrics_collected": {
      "mem": {"measurement": ["mem_used_percent"]},
      "disk": {"measurement": ["used_percent"], "resources": ["/"]}
    }
  }
}
```

After editing that configuration on EC2, load it explicitly:

```bash
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/plant-metrics.json -s
```

To validate notification once, temporarily set the RAM threshold to 0, wait for ALARM and email, then restore **85** and confirm recovery to OK when RAM is below threshold. Check alarm History and SNS subscription confirmation if email fails. Keep the saved threshold at 85 after testing. This monitors host RAM, not endpoint availability or model accuracy.

Useful AWS references: [EC2 stop/start](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html), [S3 HeadObject permissions](https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html), [CloudWatch Agent](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Install-CloudWatch-Agent.html).

## Domain and HTTPS on EC2

The domain setup targets `https://caycachua.id.vn`. It requires a working public DNS A record to the EC2 public IPv4 and inbound TCP 80/443. These are separate from the existing HTTP demo on port 3000. Keep the IP demo available until browser prediction and Grad-CAM++ pass on HTTPS.

At the DNS provider, create the root A record (`@`) pointing to the current EC2 public IPv4. Leave unrelated mail/TXT records intact. Confirm the authoritative nameservers return the correct address; a saved record or SERVFAIL is not proof of propagation. If EC2 public IPv4 changes after stop/start, update this A record before using the domain again.

The frontend defaults to `https://PAGE_HOST/api` on HTTPS, while existing HTTP demos retain API port 8000. `deploy/Caddyfile` terminates TLS, forwards `/` to `127.0.0.1:3000` and strips `/api` before forwarding requests to `127.0.0.1:8000`. Browser calls stay on the same origin, so the HTTPS path does not need cross-origin CORS permission. Explicit `NEXT_PUBLIC_API_URL` build settings still take precedence.

The following commands run in **EC2 SSH**, after DNS and ports have been verified and the HTTPS-capable frontend image has been deployed. This recipe uses Linux host networking and the existing host-published frontend/API ports. Port 80/443 must not already belong to another service.

```bash
cd ~/plant-disease-detection
export APP_DOMAIN=caycachua.id.vn
# Official Caddy 2 image digest verified during setup; upgrade intentionally.
export CADDY_IMAGE='caddy@sha256:d8542f48d34a9cf4e4c11a478865229840e87e4c96ea3f439101f31a5d35f75f'
sudo docker pull "$CADDY_IMAGE"
sudo docker run --rm --network none \
  -e "APP_DOMAIN=$APP_DOMAIN" \
  --mount "type=bind,source=$PWD/deploy/Caddyfile,target=/etc/caddy/Caddyfile,readonly" \
  "$CADDY_IMAGE" caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile

sudo docker volume create plant-disease-caddy-data
sudo docker volume create plant-disease-caddy-config
sudo docker run -d --name plant-disease-proxy \
  --restart unless-stopped --network host \
  -e "APP_DOMAIN=$APP_DOMAIN" \
  --mount "type=bind,source=$PWD/deploy/Caddyfile,target=/etc/caddy/Caddyfile,readonly" \
  --mount type=volume,source=plant-disease-caddy-data,target=/data \
  --mount type=volume,source=plant-disease-caddy-config,target=/config \
  --log-driver awslogs \
  --log-opt awslogs-region=ap-southeast-2 \
  --log-opt awslogs-group=/plant-disease/frontend \
  --log-opt awslogs-stream=plant-disease-proxy \
  "$CADDY_IMAGE"
```

Caddy requests and renews the public certificate automatically once DNS and network access are correct. Keep its data volume: it contains certificate/account keys and is not a Git artifact. If the proxy container already exists, inspect it before running the creation command again. To reload a changed Caddyfile, use `sudo docker exec plant-disease-proxy caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile`.

```bash
curl --fail https://caycachua.id.vn/api/health
curl -I http://caycachua.id.vn
```

Check the browser at `https://caycachua.id.vn`: upload a real image, analyze it and request the explanation. There must be no mixed-content/CORS errors and no connection to port 8000 from the browser. After these checks pass, remove public inbound access to ports 3000/8000 from the Security Group; the local proxy can still reach those host ports. Keep SSH restricted to the operator's source IP. Keep TCP 80/443 accessible for normal HTTPS operation and certificate renewal.

To stop/start the proxy use `sudo docker stop plant-disease-proxy` / `sudo docker start plant-disease-proxy`. Stopping it makes the domain unavailable while the frontend/API containers can remain running. This setup uses the same EC2 instance and the existing DNS provider; no Route 53 or load balancer is required.

## Local Docker Compose

The current local `compose.yaml` builds `frontend/Dockerfile` and references backend image **`plant-disease-api:v1.0.0`**. Build that image first if it is absent. The root `Dockerfile` includes the local distilled checkpoint; `Dockerfile.deploy` is the separate checkpoint-free GHCR recipe.

In **Windows PowerShell**, with Docker Desktop running and the local checkpoint present:

```powershell
Set-Location D:\Code\python\plant-disease-detection
Test-Path .\checkpoints\mobilenetv3_small_100_distilled_best.pth
docker build -f Dockerfile -t plant-disease-api:v1.0.0 .
docker compose config --quiet
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://localhost:8000/health
```

The Compose file also mounts the checkpoint read-only. Its CORS values contain the previously observed LAN IP; adjust them for your current frontend origin. `FRONTEND_PORT` changes the published frontend port, so update backend CORS accordingly. Change compiled `NEXT_PUBLIC_*` settings through Compose build args and rebuild the frontend.

```powershell
# Stop without removing container definitions:
docker compose stop
# Resume existing containers:
docker compose start
# Remove this Compose project's containers/network, retaining host artifacts/images:
docker compose down
```

If a manually created `plant-disease-api` already owns port 8000 or the same container name, inspect it before switching to Compose. Local Compose does not manage the manually deployed AWS containers. The root Dockerfile requires a local checkpoint and is not suitable for a clean CI checkout without that artifact.

## Local native development

Terminal 1 (this command also works from `D:\Code\python` or `frontend/`):

```powershell
# With the verified existing ML environment:
$env:MODEL_CHECKPOINT='D:\Code\python\plant-disease-detection\checkpoints\mobilenetv3_small_100_distilled_best.pth'
& D:\Code\python\.venv\Scripts\python.exe D:\Code\python\plant-disease-detection\scripts\run_api.py --host 0.0.0.0 --port 8000
```

For a new environment, use CPython 3.12 and install `backend/requirements.txt`. It installs CPU-only torch/torchvision, Grad-CAM++ and the API dependencies without Qdrant or Streamlit. The command above explicitly selects the distilled checkpoint used on AWS. Without `MODEL_CHECKPOINT`, `backend/settings.py` defaults to `checkpoints/mobilenetv3_small_100_best.pth`. The verified class map is included in `configs/class_to_idx.sprint01.json`. Readiness returns 503 if the checkpoint is absent or incompatible; no retraining is required.

Terminal 2:

```powershell
Set-Location D:\Code\python\plant-disease-detection\frontend
npm ci
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Open `http://localhost:3000`; Swagger is `http://localhost:8000/docs`, and readiness is `http://localhost:8000/health`. Choose or capture a JPEG/PNG photo, preview it and press the recognition button. The unchecked demo toggle uses the real API; explicitly checking it enables labeled frontend sample results. It is never enabled automatically when the API fails.

After a real result, press **Explain prediction** to compare the original image and Grad-CAM++ overlay for the top predicted class. Model identity is under **Analysis details**, and sample mode is opt-in under **Preview options → Use sample results**. The `POST /explain` route also accepts an optional training `class_id`. Explanation can be cancelled/retried without discarding the prediction. It is unavailable in demo mode. Restart an existing backend process and restart/rebuild the frontend to load the feature. See [explain.md](docs/explain.md) for requests, response images and CPU validation.

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

For the existing Docker image, see [container-lan.md](docs/container-lan.md). The API container serves port 8000; the Next.js frontend must also run on `0.0.0.0:3000`. The Wi-Fi address observed on 2026-10-10 is `192.168.2.48`, so the phone URL is `http://192.168.2.48:3000`. Container port publishing, LAN CORS and the Windows network profile all need to match.

Use the computer's LAN IPv4 address as `YOUR_LAN_IP`. Start the API with the phone page's exact origin allowed:

```powershell
$env:CORS_ORIGINS='http://localhost:3000,http://127.0.0.1:3000,http://YOUR_LAN_IP:3000'
& D:\Code\python\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Start Next.js on `0.0.0.0` as above and open `http://YOUR_LAN_IP:3000` on the phone. Confirm `/health` is reachable at `http://YOUR_LAN_IP:8000/health`. Allow development ports only on the trusted private network. In `frontend/next.config.mjs`, update `allowedDevOrigins` when the LAN hostname/IP changes, then restart the dev server; this is separate from FastAPI CORS. Camera/picker behavior is device-dependent; JPEG/PNG is supported, HEIC is not. Automated mobile-browser emulation and physical-device checks are separate evidence.

## Tests and validation

```powershell
& D:\Code\python\.venv\Scripts\python.exe -m pytest tests -q
Set-Location frontend
npm run build
$env:PYTHON_EXECUTABLE='D:\Code\python\.venv\Scripts\python.exe'
npm run test:e2e
```

Playwright runs real-model and mocked-error scenarios on mobile and desktop and starts local services on ports 8000/3005 if needed. It uses installed Microsoft Edge on Windows; on other systems install Chromium with `npx playwright install chromium`. A missing checkpoint prevents the real E2E test from passing; the mock UI is a separate explicit scenario.

On this workstation, Windows temp-folder ACLs can require a fresh workspace temp directory; use `-p no:cacheprovider --basetemp PATH_TO_A_NEW_WORKSPACE_TEMP_DIRECTORY` if needed. Historical results after `/explain` integration on 2026-10-09 were **59 Python tests and 24 browser tests passed**; the 59 Python tests were rerun successfully on 2026-10-12. These are dated results, not proof of a new test run. Frontend image CI now smoke-tests runtime health, non-root user, pages and static assets; full Playwright real-model E2E is not part of the current workflow. See [frontend-dependency-review.md](docs/frontend-dependency-review.md) for the historical dependency audit; rerun audits when assessing current dependencies.

Historical local evidence: `reports/sprint-01/api-live-smoke.json`, `reports/sprint-01/frontend-mobile.png`, and [Sprint 01 delivery](docs/sprint-01-app-delivery.md). API details are in [api-contract.md](docs/api-contract.md). Earlier [architecture](docs/architecture.md) and [AWS readiness](docs/aws-readiness.md) documents predate the personal-account EC2 deployment; their company-account/App Runner plans are not the current runbook. AWS evidence supplied by the operator includes matching artifact hashes, container health, a real `/predict` response and desktop/phone page access; fresh AWS verification must be performed on the deployed instance.

### Optional research search

The original frontend research UI is retained at `/research`. Its prediction reads the new classification API. To enable the old Qdrant search separately, start `python -m uvicorn src.search.api:app --host 127.0.0.1 --port 8001` with the existing research dependencies and Qdrant index. `NEXT_PUBLIC_SEARCH_API_URL` can override that origin. The old search service and its path-based image endpoint are for the local research workflow; they are not part of the new container/API.

## Research commands

These workflows are separate from serving the application and are not automatically run on EC2 or by the container CI pipeline. Datasets/checkpoints are required locally and remain outside Git.

### Setup environment

Ensure your virtual environment is activated and dependencies are installed.

```powershell
# Reuse the verified workstation environment; adjust this path on another machine.
& D:\Code\python\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For tests, install `requirements-dev.txt` and run `python -m pytest tests -q`.
CPU inference works with a CPU PyTorch installation; CUDA is optional. The verified local interpreter is `D:\Code\python\.venv\Scripts\python.exe` (the default MSYS Python has no PyTorch). Activate that environment or substitute its full path for `python` in the commands below.

### Training models

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


### Evaluation and optional Grad-CAM++

Run from the project root. Use the config matching the checkpoint; no training or pretrained-weight download occurs. Missing or incompatible artifacts stop execution.

```bash
python scripts/evaluate.py --config configs/distillation_mobilenetv3.yaml --checkpoint checkpoints/mobilenetv3_small_100_distilled_best.pth --dataset data/new-data-removal/test --device cpu --threads 4 --output reports/mobilenet_test.json
python scripts/evaluate.py --config configs/distillation_mobilenetv3.yaml --checkpoint checkpoints/mobilenetv3_small_100_distilled_best.pth --dataset data/new-data-removal/eval --device cpu --threads 4 --output reports/mobilenet_external.json
```

JSON reports include accuracy, macro-F1/precision/recall over all training classes, confusion matrix counts in training order, per-image predictions, class counts and artifact/dataset hashes. Add `--gradcam --gradcam-count 5` to save optional explanations next to the JSON. Add `--limit-per-class 1` for an explicitly labeled smoke subset; omit it for full evaluation. `--class-map` can override the config's training map.

The earlier research workflow is archived in `scripts/evaluate_gradcam_legacy.py`; its center-crop metrics and cleanup behavior are retained for reference. Use the new commands for inference-aligned metrics.

### Benchmarking (latency and size)
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

### Build Qdrant search index
To encode the dataset and push vectors into Qdrant for semantic search:
```bash
# For MobileNetV3 (Default)
python scripts/build_qdrant_index.py --model mobilenet

# For EfficientNet-B0
python scripts/build_qdrant_index.py --model efficientnet
```
**Outputs:** 
- Prints extraction progress and creates collections (`tomato_disease_multimodal` or `tomato_disease_efficientnet`) inside your Qdrant container.
