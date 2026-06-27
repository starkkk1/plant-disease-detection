"""
Script to evaluate trained models without retraining, with correct label mapping for external datasets.
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import timm

def infer_config_name(ckpt_name):
    """
    Tự động suy luận tên file config dựa vào tên file .pth
    """
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
        return 0, 0, 0, 0, 0

    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, acc, f1, precision, recall, all_targets, all_preds

def get_mapped_dataset(eval_dir, train_class_to_idx, transform):
    dataset = datasets.ImageFolder(eval_dir, transform=transform)
    
    valid_samples = []
    for path, idx in dataset.samples:
        class_name = dataset.classes[idx]
        if class_name in train_class_to_idx:
            mapped_idx = train_class_to_idx[class_name]
            valid_samples.append((path, mapped_idx))
            
    dataset.samples = valid_samples
    dataset.targets = [s[1] for s in valid_samples]
    return dataset

def evaluate_model(model_cfg_name, base_dir, device, train_class_to_idx, test_transform, ckpt_path=None):
    config_path = os.path.join(base_dir, 'configs', f"{model_cfg_name}.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
        
    config = load_config(config_path)
    model_name = config.get('model', {}).get('name', model_cfg_name)
    checkpoint_dir = os.path.join(base_dir, config.get('output', {}).get('checkpoint_dir', 'checkpoints'))
    
    if ckpt_path:
        checkpoint_path = os.path.join(base_dir, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
    else:
        if 'distillation' in model_cfg_name:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_distilled_best.pth')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    num_classes = config.get('model', {}).get('num_classes', 10)
    
    print(f"Loading model {model_name}...")
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    tv_dir = os.path.join(base_dir, 'data', 'new-data-removal', 'eval')
    if os.path.exists(tv_dir):
        print(f"--- Evaluating {model_name} on new-data-removal Eval ---")
        tv_dataset = get_mapped_dataset(tv_dir, train_class_to_idx, test_transform)
        if len(tv_dataset) > 0:
            tv_loader = DataLoader(tv_dataset, batch_size=32, shuffle=False)
            tv_loss, tv_acc, tv_f1, tv_prec, tv_rec, _, _ = validate(model, tv_loader, criterion, device)
            print(f"{model_name} new-data Eval -> Loss: {tv_loss:.4f} | Acc: {tv_acc:.4f} | F1: {tv_f1:.4f}\n")
        else:
            print("No valid classes found to evaluate.\n")
            
    test_dir = os.path.join(base_dir, config.get('data', {}).get('test_dir', 'data/new_processed/test'))
    if os.path.exists(test_dir):
        print(f"--- Evaluating {model_name} on Test set ---")
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_loss, test_acc, test_f1, test_prec, test_rec, test_targets, test_preds = validate(model, test_loader, criterion, device)
        print(f"{model_name} Test -> Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}\n")
        
        # Vẽ và lưu Confusion Matrix
        cm = confusion_matrix(test_targets, test_preds)
        plt.figure(figsize=(10, 8))
        
        # Lấy tên class từ dict train_class_to_idx
        idx_to_class = {v: k for k, v in train_class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Thực tế (Actual)')
        plt.xlabel('Dự đoán (Predicted)')
        plt.title(f'Confusion Matrix - {model_cfg_name}')
        plt.tight_layout()
        
        cm_filename = f"confusion_matrix_{model_cfg_name}.png"
        cm_dir = os.path.join(base_dir, 'results', 'confusion_matrix')
        os.makedirs(cm_dir, exist_ok=True)
        cm_path = os.path.join(cm_dir, cm_filename)
        plt.savefig(cm_path)
        plt.close()
        print(f"[*] Đã lưu Confusion Matrix tại: {cm_path}\n")
        
    print("="*60)

def main():
    # ==========================================
    # QUẢN LÝ NHANH BẰNG CODE (QUICK CONFIG)
    # ==========================================
    USE_CUSTOM_CONFIG = True
    SCAN_ALL_CHECKPOINTS = True # Đổi thành True để tự động quét tất cả file .pth trong thư mục checkpoints
    
    # Nếu SCAN_ALL_CHECKPOINTS = False, nó sẽ chạy cấu hình thủ công dưới đây:
    CUSTOM_CONFIG_NAME = 'distillation_effnetb0' 
    CUSTOM_CKPT_PATH = 'checkpoints/efficientnet_b0_distilled_best.pth'  
    # ==========================================

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument('--models', type=str, nargs='+', help="List of model config names (e.g., mobilenet_v3_small resnet50)")
    parser.add_argument('--all', action='store_true', help="Evaluate all 5 standard models")
    parser.add_argument('--convnext', action='store_true', help="Evaluate the convnext model")
    parser.add_argument('--ckpt', type=str, help="Optional: Path to a specific .pth checkpoint file to evaluate")
    args = parser.parse_args()

    models_to_run = []
    # ckpt_to_use is now a list to support multiple ckpts in SCAN mode
    ckpt_to_use_list = []

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    chk_dir = os.path.join(base_dir, 'checkpoints')

    if USE_CUSTOM_CONFIG:
        if SCAN_ALL_CHECKPOINTS:
            print(f"[*] Quick Config: Tự động quét TẤT CẢ model trong thư mục checkpoints...")
            if os.path.exists(chk_dir):
                for f in os.listdir(chk_dir):
                    if f.endswith('.pth') and 'cyclegan' not in f.lower():
                        cfg_name = infer_config_name(f)
                        if cfg_name:
                            models_to_run.append(cfg_name)
                            ckpt_to_use_list.append(os.path.join('checkpoints', f))
                            print(f"  -> Tìm thấy: {f} (Config: {cfg_name})")
            else:
                print("Lỗi: Không tìm thấy thư mục checkpoints!")
        else:
            print(f"[*] Quick Config is ON. Loading file: {CUSTOM_CKPT_PATH}")
            models_to_run = [CUSTOM_CONFIG_NAME]
            ckpt_to_use_list = [CUSTOM_CKPT_PATH]
    else:
        if args.all:
            models_to_run = ['convnext', 'distillation_mobilenetv3', 'distillation_effnetb0', 'efficientnet_b0', 'mobilenet_v3_small']
            ckpt_to_use_list = [None] * len(models_to_run)
        elif args.models:
            models_to_run = args.models
            ckpt_to_use_list = [args.ckpt] * len(models_to_run)
        elif args.convnext:
            models_to_run = ['convnext']
            ckpt_to_use_list = [args.ckpt]
        else:
            print("Please provide models using --models, --all, or --convnext (or set USE_CUSTOM_CONFIG = True).")
            return

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load default config just to get data paths for mapping
    default_config_path = os.path.join(base_dir, 'configs', 'default.yaml')
    config = load_config(default_config_path) if os.path.exists(default_config_path) else {}
    
    train_dir = os.path.join(base_dir, config.get('data', {}).get('train_dir', 'data/new_processed_augmented/train_augmented'))
    train_dataset = datasets.ImageFolder(train_dir)
    train_class_to_idx = train_dataset.class_to_idx
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Found {len(train_class_to_idx)} training classes.")
    print("="*60)
    
    # Xoá toàn bộ ảnh cũ trong thư mục confusion_matrix trước khi chạy mới
    cm_dir = os.path.join(base_dir, 'results', 'confusion_matrix')
    if os.path.exists(cm_dir):
        import shutil
        shutil.rmtree(cm_dir)
    os.makedirs(cm_dir, exist_ok=True)
    
    for idx, model_name in enumerate(models_to_run):
        ckpt_path = ckpt_to_use_list[idx]
        print(f"\n[{idx+1}/{len(models_to_run)}] ĐÁNH GIÁ MÔ HÌNH: {model_name} (File: {os.path.basename(ckpt_path) if ckpt_path else 'Default'})")
        evaluate_model(model_name, base_dir, device, train_class_to_idx, test_transform, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
