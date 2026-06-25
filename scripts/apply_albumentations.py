import os
import cv2
import albumentations as A
from tqdm import tqdm

# Configuration
SOURCE_DIR = r"D:\Code\python\plant-disease-detection\data\new_processed\train"
OUTPUT_DIR = r"D:\Code\python\plant-disease-detection\data\new_processed_augmented\train_augmented"

# How many augmented copies to generate per original image
NUM_AUGMENTATIONS_PER_IMAGE = 2
IMAGE_SIZE = 224

def get_augmentation_pipeline():
    """
    Define your Albumentations pipeline here.
    """
    return A.Compose([
        A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], p=0.5),
        A.CoarseDropout(max_holes=8, max_height=int(IMAGE_SIZE/8), max_width=int(IMAGE_SIZE/8), fill_value=0, p=0.5),
    ])

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        print("Please check the SOURCE_DIR path.")
        return
        
    print(f"Starting Albumentations offline augmentation...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {OUTPUT_DIR}")
    
    pipeline = get_augmentation_pipeline()
    
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    total_originals = 0
    total_augmented = 0
    
    for class_name in classes:
        class_path = os.path.join(SOURCE_DIR, class_name)
        dest_class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(dest_class_path, exist_ok=True)
        
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing class: {class_name} ({len(img_files)} original images)...")
        
        for filename in tqdm(img_files, desc=f"  Augmenting {class_name}", leave=False):
            src_path = os.path.join(class_path, filename)
            img = cv2.imread(src_path)
            if img is None:
                continue
            
            # Save the original image (resized to target shape to ensure uniformity)
            orig_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            
            base_name, ext = os.path.splitext(filename)
            
            # Save original
            orig_out_path = os.path.join(dest_class_path, f"{base_name}_orig.jpg")
            cv2.imwrite(orig_out_path, orig_resized)
            total_originals += 1
            
            # Albumentations expects RGB images
            img_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB)
            
            for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
                augmented = pipeline(image=img_rgb)
                aug_rgb = augmented['image']
                
                # Convert back to BGR to save with OpenCV
                aug_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
                
                aug_out_path = os.path.join(dest_class_path, f"{base_name}_aug_{i+1}.jpg")
                cv2.imwrite(aug_out_path, aug_bgr)
                total_augmented += 1

    print(f"\n--- Augmentation Complete ---")
    print(f"Originals saved: {total_originals}")
    print(f"Augmented generated: {total_augmented}")
    print(f"Total images in output: {total_originals + total_augmented}")

if __name__ == "__main__":
    main()
