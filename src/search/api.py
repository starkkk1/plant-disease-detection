import os
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import tempfile
import uuid

# Đảm bảo import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.mobilenet_encoder import MobilenetEncoder
from src.search.qdrant_engine import MultimodalSearchEngine
from src.models.inference_model import PlantDiseasePredictor
from fastapi.middleware.cors import CORSMiddleware
from src.models.efficientnet_encoder import EfficientNetEncoder

app = FastAPI(title="Plant Disease Search API", description="API tìm kiếm bệnh lá cây bằng hình ảnh và văn bản")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên sửa thành domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables cho models
encoder = None
engine = None
predictor = None
COLLECTION_NAME = "tomato_disease_efficientnet"

@app.on_event("startup")
async def startup_event():
    import shutil
    tmp_dir = os.path.join("data", "tmp")
    if os.path.exists(tmp_dir):
        print(f"Cleaning up old temp files in {tmp_dir}...")
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"Failed to clean up: {e}")
    os.makedirs(tmp_dir, exist_ok=True)

    global encoder, engine, predictor
    print("Loading models... (This might take a few seconds)")
    encoder = EfficientNetEncoder()
    
    try:
        engine = MultimodalSearchEngine()
        print("Qdrant Search Engine loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize Qdrant Engine (Search will be disabled). Error: {e}")
        engine = None
    
    # Khởi tạo Inference Model
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    predictor = PlantDiseasePredictor(
        model_path=os.path.join(base_dir, "checkpoints", "efficientnet_b0_distilled_best.pth"),
        model_name="efficientnet_b0"
    )
    
    print("Models loaded successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    import shutil
    tmp_dir = os.path.join("data", "tmp")
    if os.path.exists(tmp_dir):
        print(f"Cleaning up {tmp_dir} during shutdown...")
        try:
            shutil.rmtree(tmp_dir)
            print("Cleanup complete.")
        except Exception as e:
            print(f"Failed to clean up: {e}")


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), limit: int = Form(5)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPG images are supported")
        
    try:
        # Save temp file
        _, ext = os.path.splitext(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        emb = encoder.encode_image(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if emb is None:
            raise HTTPException(status_code=500, detail="Failed to encode image")
            
        results = engine.search(collection_name=COLLECTION_NAME, query_vector=emb, limit=limit)
        
        formatted_results = []
        for point in results:
            formatted_results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload
            })
            
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...), top_k: int = Form(3)):
    """
    Endpoint dự đoán bệnh từ ảnh sử dụng mô hình đã train (ví dụ MobileNetV3).
    Trả về Top K bệnh có xác suất cao nhất.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPG images are supported")
        
    try:
        # Save temp file
        _, ext = os.path.splitext(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Gọi hàm dự đoán
        results = predictor.predict(temp_path, top_k=top_k)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "model_status": "loaded" if predictor.model is not None else "not_loaded_properly",
            "predictions": results
        }
    except RuntimeError as re:
        # Lỗi nếu model chưa được config path
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=str(re))
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/image")
async def get_image(path: str):
    """
    Endpoint trả về file ảnh thật dựa trên đường dẫn tuyệt đối (hoặc tương đối) 
    để Frontend có thể hiển thị được kết quả tìm kiếm.
    """
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)
