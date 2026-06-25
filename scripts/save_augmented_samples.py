import os
import cv2
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
        
    print(f"Loading sample images from: {train_dir}")
    print(f"Saving augmented samples to: {OUTPUT_DIR}")
    
    # Define the exact train augmentations but WITHOUT normalization/ToTensorV2 for visualization
    aug_pipeline = A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=24, max_width=24, p=0.5), 
    ])
    
    # Re-create output directory
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Scan class directories in train folder
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not img_files:
            continue
            
        # Select up to 2 images per class to demonstrate
        samples = img_files[:2]
        
        dest_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)
        
        for idx, filename in enumerate(samples):
            src_path = os.path.join(class_path, filename)
            img = cv2.imread(src_path)
            if img is None:
                continue
                
            # Resize original for standard size comparison
            original_resized = cv2.resize(img, (image_size, image_size))
            
            # Save original resized
            orig_name = f"sample_{idx+1}_original.jpg"
            cv2.imwrite(os.path.join(dest_class_dir, orig_name), original_resized)
            
            # Generate 3 augmented versions
            for aug_idx in range(3):
                # Convert BGR (OpenCV) to RGB for Albumentations
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented = aug_pipeline(image=img_rgb)
                augmented_rgb = augmented['image']
                # Convert back to BGR for saving
                augmented_bgr = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)
                
                aug_name = f"sample_{idx+1}_augmented_{aug_idx+1}.jpg"
                cv2.imwrite(os.path.join(dest_class_dir, aug_name), augmented_bgr)
                
        print(f"  Class '{class_name}': Generated original and augmented comparison images.")
        
    print(f"\nCompleted! Samples generated successfully in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
