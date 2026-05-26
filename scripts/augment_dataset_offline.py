import os
import cv2
import shutil
import numpy as np
import albumentations as A
from src.utils.config import load_config

# Paths
CONFIG_PATH = "configs/default.yaml"
OUTPUT_DIR = r"D:\Code\python\data\augmented"

def main():
    config = load_config(CONFIG_PATH)
    train_dir = config['data']['train_dir']
    image_size = config['data'].get('image_size', 224)
    
    if not os.path.exists(train_dir):
        print(f"Error: Train directory '{train_dir}' does not exist.")
        return
        
    print(f"Starting offline dataset augmentation...")
    print(f"Source: {train_dir}")
    print(f"Destination: {OUTPUT_DIR}")
    
    # 1. Define the rich Albumentations pipeline with as many conditions as possible
    aug_pipeline = A.Compose([
        # Geometric transformations
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, p=0.5),
        
        # Color & Light adjustments (weather, time of day)
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        
        # Blur & Noise (camera focus, wind movement, sensor noise)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        
        # Obstruction & weather effects (shadows, fog)
        A.CoarseDropout(max_holes=10, max_height=24, max_width=24, fill_value=0, p=0.5), # Shadows
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.2), # Light fog
    ])
    
    # 2. Re-create output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing augmented folder...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    total_originals = 0
    total_augmented = 0
    
    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not img_files:
            continue
            
        dest_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        
        print(f"Processing class: {class_name} ({len(img_files)} original images)...")
        
        for idx, filename in enumerate(img_files):
            src_path = os.path.join(class_path, filename)
            img = cv2.imread(src_path)
            if img is None:
                continue
                
            # Resize original image to standard size for consistency
            orig_resized = cv2.resize(img, (image_size, image_size))
            
            # Save original image
            orig_name = f"{class_name}_{idx+1:05d}_original.jpg"
            cv2.imwrite(os.path.join(dest_class_dir, orig_name), orig_resized)
            total_originals += 1
            
            # Generate and save 1 augmented variant
            # Convert BGR (OpenCV) to RGB for Albumentations
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = aug_pipeline(image=img_rgb)
            augmented_rgb = augmented['image']
            # Convert back to BGR for saving
            augmented_bgr = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)
            
            aug_name = f"{class_name}_{idx+1:05d}_augmented.jpg"
            cv2.imwrite(os.path.join(dest_class_dir, aug_name), augmented_bgr)
            total_augmented += 1
            
    print(f"\n--- Augmentation Summary ---")
    print(f"  Classes processed: {len(classes)}")
    print(f"  Original images copied (resized): {total_originals}")
    print(f"  Augmented images generated: {total_augmented}")
    print(f"  Total images saved to {OUTPUT_DIR}: {total_originals + total_augmented}")

if __name__ == "__main__":
    main()
