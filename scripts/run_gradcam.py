import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def get_target_layer(model, model_name):
    """
    Returns the target layer for Grad-CAM based on the model name.
    """
    model_name = model_name.lower()
    if 'convnext' in model_name:
        return [model.stages[-1].blocks[-1]]
    elif 'resnet' in model_name:
        return [model.layer4[-1]]
    elif 'efficientnet' in model_name:
        return [model.blocks[-1]]
    elif 'mobilenet' in model_name:
        return [model.blocks[-1]]
    elif 'mobilevit' in model_name:
        # For mobilevit, usually the last stage before the classifier
        return [model.stages[-1]]
    else:
        # Default fallback: try to find the last layer dynamically or just raise error
        raise ValueError(f"Target layer not defined for model {model_name}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'defaults' in config:
        default_path = os.path.join(os.path.dirname(config_path), os.path.basename(config['defaults']))
        with open(default_path, 'r') as f:
            default_config = yaml.safe_load(f)
        merged = {**default_config, **config}
        for k, v in config.items():
            if isinstance(v, dict) and k in default_config and isinstance(default_config[k], dict):
                merged[k] = {**default_config[k], **v}
        return merged
    return config

def generate_cam(model, target_layers, img_tensor):
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    # You can pass targets=None to get the CAM for the highest scoring class
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    return grayscale_cam[0, :]

def main():
    parser = argparse.ArgumentParser(description="Run Grad-CAM++ on test dataset")
    parser.add_argument('--config', type=str, required=True, help="Path to config file (e.g., configs/resnet50.yaml)")
    parser.add_argument('--weights', type=str, default=None, help="Path to trained weights (.pth file). If None, uses untrained/pretrained weights.")
    parser.add_argument('--test_dir', type=str, default=None, help="Path to test images directory.")
    parser.add_argument('--num_correct', type=int, default=5, help="Number of correct predictions to save per class")
    parser.add_argument('--num_wrong', type=int, default=5, help="Number of wrong predictions to save per class")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = load_config(args.config)
    
    model_name = config.get('model', {}).get('name', 'resnet50')
    num_classes = config.get('model', {}).get('num_classes', 10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model: {model_name}")
    # Load model
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print("WARNING: No trained weights provided. Using default pretrained/untrained weights for demo!")
        
    model = model.to(device)
    model.eval()

    target_layers = get_target_layer(model, model_name)
    
    # Dataset path
    test_dir = args.test_dir
    if not test_dir:
        test_dir = config.get('data', {}).get('test_dir', 'data/new_processed/test')
        test_dir = os.path.join(base_dir, test_dir)
        
    if not os.path.exists(test_dir):
        # Fallback to the user's latest path if data/new_processed/test doesn't exist
        test_dir = os.path.join(base_dir, '..', 'data+source', 'data+source', 'new-data', 'test')
        if not os.path.exists(test_dir):
            print(f"Error: Test directory not found at {test_dir}")
            return
            
    print(f"Scanning test directory: {test_dir}")
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    # Required Output Dirs
    out_base = Path(base_dir) / 'results' / 'gradcam' / model_name
    dir_correct = out_base / 'correct'
    dir_wrong = out_base / 'wrong'
    dir_per_class = out_base / 'per_class'
    dir_cross_model = out_base / 'cross_model'
    
    for d in [dir_correct, dir_wrong, dir_per_class, dir_cross_model]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Standard ImageNet norms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Trackers for number of saved images
    saved_correct = {c: 0 for c in classes}
    saved_wrong = {c: 0 for c in classes}

    print("Generating Grad-CAM++ heatmaps...")
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(test_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            if saved_correct[class_name] >= args.num_correct and saved_wrong[class_name] >= args.num_wrong:
                break # Move to next class if we got enough samples
                
            img_path = os.path.join(class_dir, img_name)
            
            # Load image as numpy float32 [0, 1] for visualization
            img_rgb = np.array(Image.open(img_path).convert('RGB'))
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_viz = np.float32(img_resized) / 255.0
            
            # Prepare tensor for model
            input_tensor = preprocess_image(img_viz, mean=mean, std=std).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                pred_conf, pred_idx = torch.max(probs, dim=1)
                pred_conf = pred_conf.item()
                pred_idx = pred_idx.item()
                
            pred_class = classes[pred_idx] if pred_idx < len(classes) else "Unknown"
            
            is_correct = (pred_class == class_name)
            
            if is_correct and saved_correct[class_name] >= args.num_correct:
                continue
            if not is_correct and saved_wrong[class_name] >= args.num_wrong:
                continue
                
            # Generate CAM
            try:
                grayscale_cam = generate_cam(model, target_layers, input_tensor)
                cam_image = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
            except Exception as e:
                print(f"Error generating CAM for {img_name}: {e}")
                continue
                
            # Formatting filename: true_<class>__pred_<class>__conf_<score>__idx_<image_id>.png
            img_id = os.path.splitext(img_name)[0]
            out_filename = f"true_{class_name}__pred_{pred_class}__conf_{pred_conf:.2f}__idx_{img_id}.png"
            
            # Save using OpenCV
            cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            
            if is_correct:
                cv2.imwrite(str(dir_correct / out_filename), cam_image_bgr)
                saved_correct[class_name] += 1
            else:
                cv2.imwrite(str(dir_wrong / out_filename), cam_image_bgr)
                saved_wrong[class_name] += 1
                
            # Also save at least 1 to per_class and cross_model
            if saved_correct[class_name] == 1:
                cv2.imwrite(str(dir_per_class / out_filename), cam_image_bgr)
                cv2.imwrite(str(dir_cross_model / out_filename), cam_image_bgr)

    print(f"Done! Results saved in {out_base}")

if __name__ == '__main__':
    main()
