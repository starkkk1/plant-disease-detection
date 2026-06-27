import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.search.qdrant_engine import MultimodalSearchEngine

def build_index(model_name: str, data_dir: str, collection_name: str, batch_size: int = 64):
    print(f"Initializing model '{model_name}'...")
    
    if model_name == "mobilenet":
        from src.models.mobilenet_encoder import MobilenetEncoder
        encoder = MobilenetEncoder()
        vector_size = 1024
        if not collection_name: collection_name = "tomato_disease_multimodal"
    elif model_name == "efficientnet":
        from src.models.efficientnet_encoder import EfficientNetEncoder
        encoder = EfficientNetEncoder()
        vector_size = 1280
        if not collection_name: collection_name = "tomato_disease_efficientnet"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    engine = MultimodalSearchEngine()
    engine.create_collection(collection_name=collection_name, vector_size=vector_size)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        if "eval" in root.split(os.sep):
            continue
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    
    device_name = str(encoder.device).upper()
    print(f"Found {len(image_paths)} images in {data_dir}. Extracting features with batch size {batch_size} on {device_name}...")
    
    pbar = tqdm(range(0, len(image_paths), batch_size), desc=f"Indexing ({device_name})")
    for i in pbar:
        batch_paths = image_paths[i:i + batch_size]
        embs = encoder.encode_images_batch(batch_paths)
        
        valid_embs = []
        valid_payloads = []
        if embs is None: continue
            
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
        
        if valid_embs:
            engine.insert_embeddings(
                collection_name=collection_name,
                embeddings=valid_embs,
                payloads=valid_payloads
            )

    print(f"Finished building index! Collection '{collection_name}' contains {len(image_paths)} points.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Qdrant Index for Plant Disease Detection")
    parser.add_argument("--model", type=str, choices=["mobilenet", "efficientnet"], default="mobilenet", 
                        help="Model to use for encoding (mobilenet or efficientnet)")
    parser.add_argument("--data_dir", type=str, default=r"data\new-data-removal", 
                        help="Đường dẫn đến thư mục chứa dữ liệu hình ảnh (có thể để mặc định)")
    parser.add_argument("--collection", type=str, default="", 
                        help="Tên Qdrant collection (để trống sẽ tự lấy tên mặc định)")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Kích thước batch để xử lý song song trên GPU")
    
    args = parser.parse_args()
    
    # Format data_dir to absolute if it's relative
    if not os.path.isabs(args.data_dir):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        args.data_dir = os.path.join(base_dir, args.data_dir)
        
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found at {args.data_dir}")
    else:
        build_index(args.model, args.data_dir, args.collection, args.batch_size)
