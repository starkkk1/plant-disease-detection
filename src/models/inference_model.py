import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

class PlantDiseasePredictor:
    def __init__(self, model_path: str = "", model_name: str = "", num_classes: int = 11):
        """
        Khởi tạo mô hình dự đoán.
        Lưu ý: Bạn cần truyền model_path và model_name chính xác khi sử dụng, 
        hoặc sửa trực tiếp ở file api.py.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        self.model = None
        self.class_names = [
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___healthy",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "unknown"
        ] # TODO: Sửa lại list này nếu thứ tự index lúc train của bạn khác
        
        # Tiền xử lý ảnh (Chuẩn ImageNet mặc định cho các mô hình timm)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if model_path and model_name:
            self.load_model(model_path, model_name)
            
    def load_model(self, model_path: str, model_name: str):
        print(f"Loading inference model '{model_name}' from {model_path} on {self.device}...")
        try:
            self.model = timm.create_model(model_name, pretrained=False, num_classes=self.num_classes)
            # Load weights (tùy thuộc vào cách bạn lưu state_dict lúc train)
            state_dict = torch.load(model_path, map_location=self.device)
            # Nếu lưu cả dict có key 'state_dict' hay 'model'
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Inference model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            
    def predict(self, image_path: str, top_k: int = 3):
        """
        Dự đoán ảnh và trả về top_k class có xác suất cao nhất.
        """
        if self.model is None:
            raise RuntimeError("Model chưa được load. Vui lòng config model_path và model_name!")
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Không thể mở ảnh: {e}")
            
        # Thêm batch dimension [1, C, H, W]
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            # Chuyển logits thành probabilities (phần trăm)
            probs = F.softmax(outputs, dim=1)[0]
            
            # Lấy top k
            top_probs, top_indices = torch.topk(probs, top_k)
            
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            results = []
            for i in range(top_k):
                idx = int(top_indices[i])
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
                if class_name == "unknown":
                    class_name = "Mô hình chưa được học về bệnh này"
                results.append({
                    "class_name": class_name,
                    "probability": float(top_probs[i])
                })
                
            return results
