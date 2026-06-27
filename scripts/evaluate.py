import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import timm

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# ----------------- UTILS -----------------
def infer_model_name(ckpt_name):
    ckpt_name = ckpt_name.lower()
    if 'convnext' in ckpt_name:
        return 'convnext_tiny'
    elif 'efficientnet' in ckpt_name:
        return 'efficientnet_b0'
    elif 'mobilenet' in ckpt_name:
        return 'mobilenetv3_small_100'
    else:
        return 'resnet50'

def infer_config_name(ckpt_name):
    ckpt_name = ckpt_name.lower()
    if 'convnext' in ckpt_name:
        return 'convnext'
    elif 'efficientnet' in ckpt_name:
        if 'distilled' in ckpt_name:
            return 'distillation_effnetb0'
        return 'efficientnet_b0'
    elif 'mobilenet' in ckpt_name:
        if 'distilled' in ckpt_name:
            return 'distillation_mobilenetv3'
        return 'mobilenet_v3_small'
    return None

def get_target_layer(model, model_name):
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
        return [model.stages[-1]]
    else:
        raise ValueError(f"Target layer not defined for model {model_name}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'defaults' in config:
        default_path = os.path.join(os.path.dirname(config_path), os.path.basename(config['defaults']))
        if os.path.exists(default_path):
            with open(default_path, 'r') as f:
                default_config = yaml.safe_load(f)
            merged = {**default_config, **config}
            for k, v in config.items():
                if isinstance(v, dict) and k in default_config and isinstance(default_config[k], dict):
                    merged[k] = {**default_config[k], **v}
            return merged
    return config

# ----------------- EVALUATION -----------------
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    if len(dataloader.dataset) == 0:
        return 0, 0, 0, 0, 0, [], []

    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, acc, f1, precision, recall, all_targets, all_preds

def generate_cam(model, target_layers, img_tensor):
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    return grayscale_cam[0, :]

# ----------------- RUN PIPELINE -----------------
def run_evaluation_and_gradcam(ckpt_filename, base_dir, device, args):
    ckpt_path = os.path.join(base_dir, 'checkpoints', ckpt_filename)
    model_cfg_name = infer_config_name(ckpt_filename)
    timm_model_name = infer_model_name(ckpt_filename)
    
    if not model_cfg_name:
        print(f"Skipping {ckpt_filename}, could not infer config.")
        return
        
    print(f"\n[{model_cfg_name.upper()}] Loading model...")
    config_path = os.path.join(base_dir, 'configs', f"{model_cfg_name}.yaml")
    config = load_config(config_path) if os.path.exists(config_path) else {}
    num_classes = config.get('model', {}).get('num_classes', 11)
    
    try:
        model = timm.create_model(timm_model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model {timm_model_name}: {e}")
        return
        
    criterion = nn.CrossEntropyLoss()
    
    # Dataset setup
    test_dir = os.path.join(base_dir, 'data', 'new-data-removal', 'test')
    if args.dataset == 'eval':
        test_dir = os.path.join(base_dir, 'data', 'new-data-removal', 'eval')
        
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
        
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 1. EVALUATION
    print(f"--- Evaluating {model_cfg_name} on {args.dataset.upper()} set ---")
    test_loss, test_acc, test_f1, test_prec, test_rec, test_targets, test_preds = validate(model, test_loader, criterion, device)
    print(f"{model_cfg_name} Test -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Thực tế (Actual)')
    plt.xlabel('Dự đoán (Predicted)')
    plt.title(f'Confusion Matrix - {model_cfg_name}')
    plt.tight_layout()
    
    cm_dir = os.path.join(base_dir, 'results', 'confusion_matrix')
    os.makedirs(cm_dir, exist_ok=True)
    cm_path = os.path.join(cm_dir, f"confusion_matrix_{model_cfg_name}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[*] Đã lưu Confusion Matrix: {cm_path}")
    
    # 2. GRAD-CAM
    if args.skip_gradcam:
        return
        
    print(f"--- Generating Grad-CAM for {model_cfg_name} ---")
    try:
        target_layers = get_target_layer(model, timm_model_name)
    except Exception as e:
        print(f"Error getting Grad-CAM target layers: {e}")
        return
        
    out_base = Path(base_dir) / 'results' / 'gradcam' / model_cfg_name
    dir_correct = out_base / 'correct'
    dir_wrong = out_base / 'wrong'
    dir_per_class = out_base / 'per_class'
    
    for d in [dir_correct, dir_wrong, dir_per_class]:
        d.mkdir(parents=True, exist_ok=True)
        
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    saved_correct = {c: 0 for c in classes}
    saved_wrong = {c: 0 for c in classes}
    
    pbar = tqdm(classes, desc="Grad-CAM Progress", position=0)
    for class_name in pbar:
        pbar.set_postfix({'Current': class_name})
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in images:
            if saved_correct[class_name] >= args.num_correct and saved_wrong[class_name] >= args.num_wrong:
                break
                
            img_path = os.path.join(class_dir, img_name)
            img_rgb = np.array(Image.open(img_path).convert('RGB'))
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_viz = np.float32(img_resized) / 255.0
            
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
                
            try:
                grayscale_cam = generate_cam(model, target_layers, input_tensor)
                cam_image = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
            except Exception as e:
                continue
                
            img_id = os.path.splitext(img_name)[0]
            out_filename = f"true_{class_name}__pred_{pred_class}__conf_{pred_conf:.2f}__idx_{img_id}.png"
            cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            
            if is_correct:
                cv2.imwrite(str(dir_correct / out_filename), cam_image_bgr)
                saved_correct[class_name] += 1
            else:
                cv2.imwrite(str(dir_wrong / out_filename), cam_image_bgr)
                saved_wrong[class_name] += 1
                
            if saved_correct[class_name] == 1:
                cv2.imwrite(str(dir_per_class / out_filename), cam_image_bgr)
    print(f"[*] Đã lưu Grad-CAM cho {model_cfg_name}.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and generate Grad-CAM for models.")
    parser.add_argument('--dataset', type=str, default='test', choices=['test', 'eval'], help="Dataset to evaluate on")
    parser.add_argument('--num_correct', type=int, default=5, help="Grad-CAM: number of correct predictions to save per class")
    parser.add_argument('--num_wrong', type=int, default=5, help="Grad-CAM: number of wrong predictions to save per class")
    parser.add_argument('--skip_gradcam', action='store_true', help="Skip Grad-CAM generation")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chk_dir = os.path.join(base_dir, 'checkpoints')
    
    # Cleanup old results
    import shutil
    cm_dir = os.path.join(base_dir, 'results', 'confusion_matrix')
    if os.path.exists(cm_dir): shutil.rmtree(cm_dir, ignore_errors=True)
    
    if not args.skip_gradcam:
        gradcam_dir = os.path.join(base_dir, 'results', 'gradcam')
        if os.path.exists(gradcam_dir): shutil.rmtree(gradcam_dir, ignore_errors=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(chk_dir):
        print("Error: Checkpoints directory not found!")
        return
        
    ckpt_files = [f for f in os.listdir(chk_dir) if f.endswith('.pth') and 'cyclegan' not in f.lower()]
    
    print(f"\n[*] Bắt đầu đánh giá {len(ckpt_files)} models...")
    for idx, f in enumerate(ckpt_files):
        print(f"\n{'='*80}\n[{idx+1}/{len(ckpt_files)}] Đang xử lý: {f}\n{'='*80}")
        run_evaluation_and_gradcam(f, base_dir, device, args)
        
    print("\n[+] HOÀN TẤT! Kết quả đã được lưu trong thư mục results/")

if __name__ == '__main__':
    main()
