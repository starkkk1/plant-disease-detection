# Dataset Preparation & Modifications Changelog

Tài liệu này ghi chú lại toàn bộ các chỉnh sửa và cập nhật đã được thực hiện trong quá trình chuẩn bị dữ liệu (Dataset Preparation) cho dự án nhận diện bệnh trên lá cà chua.

## 1. Chuẩn hóa định dạng tên file (Standardization)
- **Vấn đề:** Các tập dữ liệu khác nhau (train, val, test, test-multileaves, v.v.) có thể có tên file trùng lặp, gây lỗi khi gom nhóm hoặc di chuyển dữ liệu trên hệ điều hành Windows.
- **Giải pháp:** Cập nhật script `scripts/standardize_filenames.py`. Tên file hiện tại được chuẩn hóa theo định dạng `[class]_[split]_[uuid].jpg`. Việc sử dụng UUID (Unique Identifier) đảm bảo 100% không có sự trùng lặp tên file giữa các tập dữ liệu.

## 2. Lọc dữ liệu tập `test-multileaves`
- **Vấn đề:** Tập dữ liệu gốc chứa nhiều loại lá khác nhau, nhưng mục tiêu dự án chỉ tập trung vào lá cà chua.
- **Giải pháp:** Đã chỉnh sửa script `scripts/extract_multileaves.py` để lọc và chỉ giữ lại 3 lớp (classes) chính liên quan đến cà chua:
  - `Tomato___Bacterial_spot`
  - `Tomato___Late_blight`
  - `Tomato___healthy`

## 3. Tích hợp tập dữ liệu Bangladesh (Bangladesh Tomato Leaf Dataset)
- **Thông tin dữ liệu:** Tập dữ liệu thu thập tại Bangladesh gồm 1.028 ảnh (thực tế trích xuất khoảng 1.035 ảnh tùy nguồn), chia thành 2 nhãn chính: Healthy (H - Khỏe mạnh) và Diseased (D - Bệnh). Ảnh có bối cảnh phức tạp (complex background).
- **Giải pháp:** 
  - Tạo mới script `scripts/extract_bangladesh.py` để xử lý định dạng nhãn YOLO.
  - Script tự động chuyển đổi tọa độ YOLO chuẩn hóa về tọa độ pixel (`x1 = int((x_c - box_w/2) * w)`).
  - Dữ liệu đầu ra được lưu trữ gọn gàng tại thư mục `data/test-bangladesh`.

## 4. Cải tiến Pipeline Augmentation với Albumentations
- **Vấn đề:** Cần một thư viện Data Augmentation mạnh mẽ, đa dạng hiệu ứng (màu sắc, hình học, thời tiết, nhiễu) để tăng cường tính khái quát hóa cho mô hình, thay thế/bổ sung cho torchvision.transforms mặc định.
- **Giải pháp:**
  - Cài đặt `albumentations==1.3.1` (Phiên bản 1.3.1 được chọn để tương thích tối đa với Windows mà không yêu cầu cài đặt C++ Build Tools phức tạp như các bản 2.x).
  - Cập nhật `src/data/transforms.py` để xây dựng pipeline với Albumentations (bao gồm: Flips, Rotations, ColorJitter, CoarseDropout, GaussianBlur, ImageCompression, v.v.).
  - Sửa đổi lớp `TomatoDataset` trong `src/data/dataset.py` để xử lý việc chuyển đổi từ `PIL Image` sang `Numpy Array` (định dạng bắt buộc của Albumentations).

## 5. Offline Data Augmentation (Tăng cường dữ liệu ngoại tuyến)
- **Vấn đề:** Thay vì chỉ Augment on-the-fly (trong lúc train), cần sinh ra sẵn một tập dữ liệu augmented lưu trên ổ cứng để kiểm tra và huấn luyện cố định.
- **Giải pháp:**
  - Viết mới script `scripts/augment_dataset_offline.py`.
  - Chạy augmentation trên **toàn bộ tập training gốc** (3.599 ảnh). Mỗi ảnh gốc sinh ra thêm 1 biến thể với các điều kiện môi trường ngẫu nhiên tối đa.
  - Lưu 3.599 ảnh mới vào thư mục `data/augmented/[label]`. Thư mục `data/` được đưa vào `.gitignore` để tránh đẩy dữ liệu lớn lên git, nhưng mã nguồn augmentation (script) vẫn được theo dõi (track).

## 6. Cập nhật Cấu hình (Configuration)
- Sửa đổi file `configs/default.yaml` để trỏ đường dẫn dữ liệu về đúng vị trí thực tế trên máy (`D:\Code\python\data\...`), bao gồm cả cấu hình đọc từ thư mục `augmented` cho quá trình huấn luyện sắp tới.
- Pipeline `DataLoader` đã được verify hoạt động trơn tru (đầu ra tensor kích thước chuẩn `(32, 3, 224, 224)`).
