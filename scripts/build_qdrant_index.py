import os
import sys
import glob
import argparse
from tqdm import tqdm

# Đảm bảo import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multimodal_encoder import MultimodalEncoder
from src.search.qdrant_engine import MultimodalSearchEngine

def build_index(data_dir: str, collection_name: str, batch_size: int = 64):
    print(f"Initializing models...")
    encoder = MultimodalEncoder()
    engine = MultimodalSearchEngine()
    
    # Vector size của OpenAI CLIP (ViT-B/32) là 512
    engine.create_collection(collection_name=collection_name, vector_size=512)
    
    # Lấy danh sách tất cả các ảnh trong thư mục train
    # Cấu trúc: data/train/class_name/image.jpg
    image_paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    
    print(f"Found {len(image_paths)} images in {data_dir}. Extracting features with batch size {batch_size}...")
    
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        
        # Xử lý theo batch trên GPU
        embs = encoder.encode_images_batch(batch_paths)
        
        valid_embs = []
        valid_payloads = []
        
        if embs is None:
            continue
            
        for j, emb in enumerate(embs):
            if emb is not None:
                img_path = batch_paths[j]
                class_name = os.path.basename(os.path.dirname(img_path))
                disease_name = class_name.replace("Tomato___", "").replace("_", " ")
                
                valid_embs.append(emb)
                valid_payloads.append({
                    "image_path": img_path,
                    "class_name": class_name,
                    "disease_name": disease_name
                })
        
        # Push batch lên Qdrant
        if valid_embs:
            engine.insert_embeddings(
                collection_name=collection_name,
                embeddings=valid_embs,
                payloads=valid_payloads
            )

    print("Finished building index!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant Index for Plant Disease Detection")
    parser.add_argument("--data_dir", type=str, default=r"d:\Code\python\data\new-data", 
                        help="Đường dẫn đến thư mục chứa dữ liệu hình ảnh")
    parser.add_argument("--collection", type=str, default="tomato_disease_multimodal", 
                        help="Tên Qdrant collection")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Kích thước batch để xử lý song song trên GPU")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
    else:
        build_index(args.data_dir, args.collection, args.batch_size)
