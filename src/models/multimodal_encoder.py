import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

class MultimodalEncoder:
    def __init__(self, 
                 image_model_name="openai/clip-vit-base-patch32",
                 text_model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1",
                 device=None):
        """
        Khởi tạo Multimodal Encoder.
        - Image Encoder: Sử dụng mô hình CLIP gốc.
        - Text Encoder: Sử dụng Multilingual CLIP để hỗ trợ Tiếng Việt (và 50+ ngôn ngữ khác).
        Lưu ý: sentence-transformers/clip-ViT-B-32-multilingual-v1 đã được align với openai/clip-vit-base-patch32.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MultimodalEncoder on {self.device}...")
        
        # Load Image Encoder (OpenAI CLIP)
        self.clip_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
        
        # Load Text Encoder (Multilingual SentenceTransformer)
        # We load it via sentence_transformers to get the exact aligned embeddings easily
        from sentence_transformers import SentenceTransformer
        self.text_model = SentenceTransformer(text_model_name).to(self.device)
        
        self.clip_model.eval()
        self.text_model.eval()

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode hình ảnh thành vector embedding.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None
            
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output  # pooler_output
            image_features = self.clip_model.visual_projection(pooled_output)
            # Normalize vector
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()[0]

    def encode_images_batch(self, image_paths: list) -> np.ndarray:
        """
        Encode một batch hình ảnh để tăng tốc độ trên GPU.
        """
        images = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_indices.append(i)
            except Exception as e:
                print(f"Error opening image {path}: {e}")
                
        if not images:
            return None
            
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.clip_model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_output = vision_outputs.pooler_output
            image_features = self.clip_model.visual_projection(pooled_output)
            # Normalize vector
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
        embeddings = image_features.cpu().numpy()
        
        # Trả về mảng cùng kích thước ban đầu, None cho ảnh lỗi
        result = [None] * len(image_paths)
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb
            
        return result

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode văn bản (Tiếng Việt, Tiếng Anh...) thành vector embedding.
        """
        with torch.no_grad():
            # sentence_transformers tự động trả về numpy array đã được normalize (nếu config chuẩn)
            # nhưng ta vẫn đảm bảo normalize thủ công để an toàn tính cosine similarity
            text_features = self.text_model.encode([text], convert_to_tensor=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()[0]

    def encode_multimodal(self, image_path: str, text: str, alpha: float = 0.5) -> np.ndarray:
        """
        Kết hợp vector ảnh và vector văn bản (Late Fusion).
        - alpha: trọng số của ảnh (0 -> 1). 
        - alpha = 0.5: chia đều ảnh và văn bản.
        """
        image_emb = self.encode_image(image_path)
        text_emb = self.encode_text(text)
        
        if image_emb is None:
            return text_emb
            
        # Kết hợp tuyến tính
        fused_emb = alpha * image_emb + (1 - alpha) * text_emb
        
        # Normalize lại
        fused_emb = fused_emb / np.linalg.norm(fused_emb)
        return fused_emb
