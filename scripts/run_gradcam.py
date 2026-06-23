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

def run_gradcam_for_model(model_name, weights_path, out_name, test_dir, classes, device, base_dir, args):
    num_classes = len(classes)
    print(f"Loading model: {model_name} with {num_classes} classes")
    
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    except Exception as e:
        print(f"Error creating model {model_name}: {e}")
        return
        
    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("WARNING: No trained weights provided. Using default pretrained/untrained weights for demo!")
        
    model = model.to(device)
    model.eval()

    try:
        target_layers = get_target_layer(model, model_name)
    except Exception as e:
        print(f"Error getting target layers for {model_name}: {e}")
        return
    
    # Required Output Dirs
    out_base = Path(base_dir) / 'results' / 'gradcam' / out_name
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

    print(f"\n[>] Ti\u1ebfn h\u00e0nh t\u1ea1o b\u1ea3n \u0111\u1ed3 nhi\u1ec7t cho: {out_name}")
    
    # Thêm thanh tr\u1ea1ng thái tổng quát cho các class
    pbar_classes = tqdm(classes, desc="Ti\u1ebfn \u0111\u1ed9 các l\u1edbp b\u1ec7nh", position=0)
    
    for class_idx, class_name in enumerate(pbar_classes):
        pbar_classes.set_postfix({'Current': class_name})
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
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

    print(f"Done for {out_name}! Results saved in {out_base}")

def main():
    # ==========================================
    # QUẢN LÝ NHANH BẰNG CODE (QUICK CONFIG)
    # ==========================================
    USE_CUSTOM_CONFIG = True
    CUSTOM_DATASET = 'eval' # Chọn 'test' (ảnh sạch lab) hoặc 'eval' (ảnh hoang dã)
    SCAN_ALL_CHECKPOINTS = True # Bật True để tự động quét toàn bộ thư mục checkpoints
    
    # Nếu SCAN_ALL_CHECKPOINTS = False, nó sẽ chạy cấu hình thủ công dưới đây:
    CUSTOM_MODELS = [
        ('convnext_tiny', 'checkpoints/convnext_tiny_best.pth', 'convnext_tiny_teacher'),
        ('efficientnet_b0', 'checkpoints/efficientnet_b0_distilled_best.pth', 'efficientnet_b0_distilled'),
        ('mobilenetv3_small_100', 'checkpoints/mobilenetv3_small_100_distilled_best.pth', 'mobilenetv3_small_distilled'),
    ]
    # ==========================================

    parser = argparse.ArgumentParser(description="Run Grad-CAM++ on test dataset")
    parser.add_argument('--config', type=str, default=None, help="Path to config file (e.g., configs/resnet50.yaml).")
    parser.add_argument('--weights', type=str, default=None, help="Path to trained weights (.pth file).")
    parser.add_argument('--test_dir', type=str, default=None, help="Path to test images directory.")
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help="Directory containing model checkpoints to scan.")
    parser.add_argument('--all', action='store_true', help="Scan checkpoints_dir and run on all checkpoints.")
    parser.add_argument('--num_correct', type=int, default=5, help="Number of correct predictions to save per class")
    parser.add_argument('--num_wrong', type=int, default=5, help="Number of wrong predictions to save per class")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Xóa sạch thư mục results/gradcam cũ trước khi chạy
    gradcam_results_dir = os.path.join(base_dir, 'results', 'gradcam')
    if os.path.exists(gradcam_results_dir):
        print(f"[*] Đang dọn dẹp dữ liệu cũ tại: {gradcam_results_dir}...")
        import shutil
        shutil.rmtree(gradcam_results_dir, ignore_errors=True)
    os.makedirs(gradcam_results_dir, exist_ok=True)
    
    models_to_run = []
    
    if USE_CUSTOM_CONFIG:
        if SCAN_ALL_CHECKPOINTS:
            print(f"[*] Quick Config: Tự động quét TẤT CẢ model trong thư mục checkpoints để tạo bản đồ nhiệt...")
            chk_dir = os.path.join(base_dir, 'checkpoints')
            if os.path.exists(chk_dir):
                for f in os.listdir(chk_dir):
                    if f.endswith('.pth') and 'cyclegan' not in f.lower():
                        model_name = infer_model_name(f)
                        weights = os.path.join('checkpoints', f)
                        out_name = os.path.splitext(f)[0]
                        models_to_run.append((model_name, weights, out_name))
                        print(f"  -> Tìm thấy: {f} (Model: {model_name})")
            else:
                print("Lỗi: Không tìm thấy thư mục checkpoints!")
        else:
            print(f"[*] Quick Config is ON. Generating Grad-CAM for {len(CUSTOM_MODELS)} models on '{CUSTOM_DATASET}' dataset.")
            models_to_run = CUSTOM_MODELS
    else:
        if args.config:
            config = load_config(args.config)
            model_name = config.get('model', {}).get('name', 'resnet50')
            weights = args.weights
            out_name = model_name
            models_to_run.append((model_name, weights, out_name))
        elif args.weights and not args.all:
            model_name = infer_model_name(os.path.basename(args.weights))
            out_name = os.path.splitext(os.path.basename(args.weights))[0]
            models_to_run.append((model_name, args.weights, out_name))
        elif args.all:
            # Scan checkpoints dir
            chk_dir = os.path.join(base_dir, args.checkpoints_dir)
            if not os.path.exists(chk_dir):
                print(f"Error: Checkpoints directory not found at {chk_dir}")
                return
                
            print(f"Scanning checkpoints directory: {chk_dir}")
            for f in os.listdir(chk_dir):
                if f.endswith('.pth'):
                    model_name = infer_model_name(f)
                    weights = os.path.join(chk_dir, f)
                    out_name = os.path.splitext(f)[0]
                    models_to_run.append((model_name, weights, out_name))
        else:
            print("Please provide --config, --weights, or use the --all flag.")
            return

    # Dataset path
    test_dir = args.test_dir
    if USE_CUSTOM_CONFIG:
        if CUSTOM_DATASET == 'eval':
            test_dir = os.path.join(base_dir, 'data', 'new-data-removal', 'eval')
        else:
            test_dir = os.path.join(base_dir, 'data', 'new-data-removal', 'test')
    elif not test_dir:
        test_dir = os.path.join(base_dir, 'data', 'new-data-removal', 'test')
            
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        return
            
    print(f"Scanning test directory: {test_dir}")
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    if not classes:
        print("Error: No classes found in test directory.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    for idx, (model_name, weights, out_name) in enumerate(models_to_run):
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(models_to_run)}] MÔ HÌNH: {model_name.upper()} (File: {os.path.basename(weights)})")
        print(f"{'='*60}")
        run_gradcam_for_model(model_name, weights, out_name, test_dir, classes, device, base_dir, args)
        
    print("\n[+] HOÀN TẤT TẤT CẢ CÁC MÔ HÌNH! HÃY MỞ THƯ MỤC results/gradcam ĐỂ XEM KẾT QUẢ.")

if __name__ == '__main__':
    main()
