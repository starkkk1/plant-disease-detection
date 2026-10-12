# CI/CD cho Plant Disease Detection (máy cá nhân, chưa dùng AWS)

## 1. Kiến trúc

- Pull request vào `master`: Python tests, Next.js production build, Docker image build.
- Push/merge vào `master`: chạy lại toàn bộ; chỉ publish image lên GitHub Container Registry (GHCR) khi các bước đều qua.
- Image phát hành **không chứa checkpoint**. Checkpoint `.pth` vẫn nằm trên máy host và được bind mount **read-only** khi chạy.
- Máy Windows tự quyết định thời điểm pull/restart. GitHub-hosted runner **không có quyền vào máy cá nhân**, không cài self-hosted runner hay mở SSH/Docker API cho Internet.
- Frontend Next.js tiếp tục chạy riêng trên Windows; workflow chỉ kiểm tra `npm run build`, chưa deploy Next.js.

## 2. Yêu cầu trước khi triển khai

1. Docker Desktop đang chạy, checkpoint thực tồn tại trên máy Windows.
2. Kiểm tra đúng `MODEL_NAME`, class map; ví dụ MobileNetV3-Small distilled dùng `mobilenetv3_small_100`.
3. Vào GitHub Actions kiểm tra workflow xanh sau merge. Vào GitHub Packages kiểm tra image đã xuất bản. Nếu GHCR package ở chế độ private, dùng `docker login ghcr.io` với token chỉ cấp `read:packages` trên máy host. Không ghi token vào repo.
4. Nếu cấu hình repo chặn publish packages, quản trị viên phải bật quyền thích hợp cho GITHUB_TOKEN hoặc điều chỉnh package visibility.
5. Container này chưa có authentication; chỉ phục vụ trong LAN tin cậy, kiểm soát firewall. Đừng public port 8000 ra Internet.

## 3. Windows PowerShell: triển khai thủ công bản mới đã được kiểm thử

Chạy tại **thư mục gốc repository**. Đảm bảo file checkpoint tồn tại:

```powershell
Test-Path .\checkpoints\mobilenetv3_small_100_distilled_best.pth
docker pull ghcr.io/starkkk1/plant-disease-detection-api:latest
```

**Nếu đang chạy container cũ**, `docker stop plant-disease-api` sẽ làm gián đoạn service vài giây. Khi chưa chắc image mới hoạt động, hãy chạy canary trên port 8001 trước:

```powershell
$root = (Get-Location).Path
$ckpt = (Resolve-Path '.\checkpoints\mobilenetv3_small_100_distilled_best.pth').Path
docker run -d --rm --name plant-disease-canary -p 127.0.0.1:8001:8000 `
  --mount "type=bind,source=$ckpt,target=/app/checkpoints/model.pth,readonly" `
  -e MODEL_NAME=mobilenetv3_small_100 `
  ghcr.io/starkkk1/plant-disease-detection-api:latest
Invoke-RestMethod http://127.0.0.1:8001/health
```

Chỉ chuyển sang container chính nếu `status` bằng `ok`, rồi dừng canary:

```powershell
docker stop plant-disease-canary
docker stop plant-disease-api
docker run -d --rm --name plant-disease-api -p 8000:8000 `
  --mount "type=bind,source=$ckpt,target=/app/checkpoints/model.pth,readonly" `
  -e MODEL_NAME=mobilenetv3_small_100 `
  -e CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://YOUR_LAN_IP:3000" `
  ghcr.io/starkkk1/plant-disease-detection-api:latest
Invoke-RestMethod http://localhost:8000/health
```

Thay `YOUR_LAN_IP` bằng IPv4 thật. Máy host cần giữ Docker Desktop chạy. Nếu đổi model sang ConvNeXt/EfficientNet, sửa `MODEL_NAME` và kiểm tra lớp đầu ra/class map.

`latest` tiện cho demo nhưng để rollback cần lưu một image tag theo commit SHA (workflow publish cả SHA). Dùng một tag cố định để triển khai lại phiên bản cũ.

## 4. Giới hạn và bước tiếp theo

- Image trong GHCR không có checkpoint, nên `/health` sẽ trả 503 nếu chưa mount model. Docker HEALTHCHECK cũng sẽ không pass cho đến khi mount đúng checkpoint.
- CI dùng test weights giả lập để kiểm tra hợp đồng API; **không thay thế smoke test inference bằng checkpoint thật** trên host.
- Workflow **chưa tự động triển khai** lên Windows. Cách này giữ máy cá nhân không phải nhận lệnh từ các PR trên GitHub. Khi đã có log, quyền truy cập và rollback ổn định, có thể tạo tác vụ Windows định kỳ pull image đã duyệt, kiểm tra canary rồi chuyển phiên bản. Không gắn self-hosted runner Docker có quyền cao với PR không tin cậy.
- Nếu bạn muốn CI/CD cho frontend nữa, cần pipeline build và cơ chế cập nhật Next.js riêng.
