import os
import sys
import glob
from tqdm import tqdm

# Đảm bảo import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multimodal_encoder import MultimodalEncoder
from src.search.qdrant_engine import MultimodalSearchEngine

def build_index(data_dir: str, collection_name: str, batch_size: int = 32):
    print(f"Initializing models...")
    encoder = MultimodalEncoder()
    engine = MultimodalSearchEngine()
    
    # Vector size của OpenAI CLIP (ViT-B/32) là 512
    engine.create_collection(collection_name=collection_name, vector_size=512)
    
    # Lấy danh sách tất cả các ảnh trong thư mục train
    # Cấu trúc: data/train/class_name/image.jpg
    image_paths = glob.glob(os.path.join(data_dir, "*", "*.jpg")) + \
                  glob.glob(os.path.join(data_dir, "*", "*.png"))
    
    print(f"Found {len(image_paths)} images in {data_dir}. Extracting features...")
    
    current_batch_embs = []
    current_batch_payloads = []
    
    for i, img_path in enumerate(tqdm(image_paths)):
        # Lấy tên class từ tên thư mục (ví dụ: Tomato___Early_blight)
        class_name = os.path.basename(os.path.dirname(img_path))
        
        # Có thể parse thêm tên bệnh để thân thiện hơn
        disease_name = class_name.replace("Tomato___", "").replace("_", " ")
        
        emb = encoder.encode_image(img_path)
        if emb is not None:
            current_batch_embs.append(emb)
            current_batch_payloads.append({
                "image_path": img_path,
                "class_name": class_name,
                "disease_name": disease_name
            })
            
        # Push batch lên Qdrant
        if len(current_batch_embs) >= batch_size or i == len(image_paths) - 1:
            if len(current_batch_embs) > 0:
                engine.insert_embeddings(
                    collection_name=collection_name,
                    embeddings=current_batch_embs,
                    payloads=current_batch_payloads
                )
            current_batch_embs = []
            current_batch_payloads = []

    print("Finished building index!")

if __name__ == "__main__":
    # Sử dụng thư mục data của user
    DATA_DIR = r"d:\Code\python\data\train"
    COLLECTION_NAME = "tomato_disease_multimodal"
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
    else:
        build_index(DATA_DIR, COLLECTION_NAME)
