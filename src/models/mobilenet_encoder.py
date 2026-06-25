import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
from torchvision import transforms

class MobilenetEncoder:
    def __init__(self, 
                 model_path="checkpoints/mobilenetv3_small_100_distilled_best.pth",
                 model_name="mobilenetv3_small_100",
                 num_classes=11,
                 device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MobilenetEncoder on {self.device}...")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        self.model.load_state_dict(state_dict, strict=False)
        
        # Thay thế lớp classifier cuối cùng bằng Identity để lấy vector đặc trưng
        self.model.classifier = nn.Identity()
        
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None
            
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(input_tensor)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
        return features.cpu().numpy()[0]

    def encode_images_batch(self, image_paths: list) -> list:
        images = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                images.append(self.transform(Image.open(path).convert("RGB")))
                valid_indices.append(i)
            except Exception as e:
                print(f"Error opening image {path}: {e}")
                
        if not images:
            return None
            
        input_tensor = torch.stack(images).to(self.device)
        with torch.no_grad():
            features = self.model(input_tensor)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
        embeddings = features.cpu().numpy()
        
        result = [None] * len(image_paths)
        for idx, emb in zip(valid_indices, embeddings):
            result[idx] = emb
            
        return result
