import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from rembg import remove, new_session
import concurrent.futures

def process_image(input_path, output_path, session):
    try:
        # Mở ảnh
        input_image = Image.open(input_path).convert("RGBA")
        
        # Tách nền (trả về ảnh RGBA với nền trong suốt)
        output_image_rgba = remove(input_image, session=session)
        
        # Tạo một nền đen trơn để giống với ảnh Lab (PlantVillage)
        background = Image.new("RGBA", output_image_rgba.size, (0, 0, 0, 255))
        
        # Dán chiếc lá đã tách nền lên nền đen
        background.paste(output_image_rgba, mask=output_image_rgba)
        
        # Chuyển về hệ màu RGB (3 kênh) chuẩn cho train mô hình
        final_image = background.convert("RGB")
        
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Lưu ảnh
        final_image.save(output_path, format="JPEG", quality=95)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Tách nền ảnh và thay bằng nền đen (Lab style)")
    parser.add_argument('--input_dir', type=str, default='data/new-data', help='Thư mục chứa ảnh gốc')
    parser.add_argument('--output_dir', type=str, default='data/new-data-removal', help='Thư mục lưu ảnh đã tách nền')
    parser.add_argument('--workers', type=int, default=8, help='Số luồng xử lý đồng thời')
    args = parser.parse_args()

    # Lấy đường dẫn gốc của project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    input_dir = Path(os.path.join(base_dir, args.input_dir))
    output_dir = Path(os.path.join(base_dir, args.output_dir))
    
    if not input_dir.exists():
        print(f"Cannot find directory {input_dir}")
        return

    # Tìm tất cả file ảnh
    print("Scanning for image files...")
    image_paths = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    for ext in extensions:
        image_paths.extend(list(input_dir.rglob(ext)))
    
    # Xóa trùng lặp
    image_paths = list(set(image_paths))
    
    print(f"Found {len(image_paths)} images to process.")
    
    # Khởi tạo session của rembg một lần để chạy nhanh hơn
    print("Loading U^2-Net model with CPU (Multi-threading)...")
    session = new_session("u2net")
    
    def worker(img_path):
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        out_path = out_path.with_suffix('.jpg')
        if out_path.exists():
            return
        process_image(str(img_path), str(out_path), session)
        
    print(f"Removing backgrounds using {args.workers} concurrent workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(worker, image_paths), total=len(image_paths), desc="Removing backgrounds"))
        
    print(f"\nDone! All images saved to: {output_dir}")

if __name__ == '__main__':
    main()
