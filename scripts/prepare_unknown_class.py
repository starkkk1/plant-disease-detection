import os
import random
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

def main():
    random.seed(42)
    
    base_dir = Path(r"d:\Code\python\plant-disease-detection")
    eval_dir = base_dir / "data" / "new_processed" / "eval" / "Tomato-Village"
    
    ood_classes = [
        "Leaf Miner",
        "Magnesium Deficiency",
        "Nitrogen Deficiency",
        "Pottassium Deficiency",
        "Spotted Wilt Virus"
    ]
    
    # 1. Collect all OOD images
    all_images = []
    for cls in ood_classes:
        cls_dir = eval_dir / cls
        if cls_dir.exists():
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    all_images.append(img_path)
                    
    print(f"Total OOD images found: {len(all_images)}")
    if len(all_images) == 0:
        return
        
    random.shuffle(all_images)
    
    train_count = int(0.8 * len(all_images))
    val_count = int(0.1 * len(all_images))
    
    train_imgs = all_images[:train_count]
    val_imgs = all_images[train_count:train_count + val_count]
    test_imgs = all_images[train_count + val_count:]
    
    print(f"Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
    
    # 2. Create target directories
    out_train_dir = base_dir / "data" / "new_processed" / "train" / "Unknown_Disease"
    out_val_dir = base_dir / "data" / "new_processed" / "val" / "Unknown_Disease"
    out_test_dir = base_dir / "data" / "new_processed" / "test" / "Unknown_Disease"
    out_train_aug_dir = base_dir / "data" / "new_processed_augmented" / "train_augmented" / "Unknown_Disease"
    
    for d in [out_train_dir, out_val_dir, out_test_dir, out_train_aug_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 3. Copy files to train, val, test
    def copy_files(imgs, target_dir, prefix=""):
        for i, img_path in enumerate(imgs):
            ext = img_path.suffix
            new_name = f"{prefix}_{i:04d}{ext}"
            shutil.copy2(img_path, target_dir / new_name)
            
    print("Copying to val and test...")
    copy_files(val_imgs, out_val_dir, "ood_val")
    copy_files(test_imgs, out_test_dir, "ood_test")
    
    print("Copying to train...")
    copy_files(train_imgs, out_train_dir, "ood_train")
    
    # 4. Augment and copy to train_augmented
    print("Copying and augmenting for train_augmented...")
    
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])
    
    for i, img_path in enumerate(train_imgs):
        ext = img_path.suffix
        # Original
        orig_name = f"ood_train_{i:04d}_orig{ext}"
        dest_orig = out_train_aug_dir / orig_name
        shutil.copy2(img_path, dest_orig)
        
        # Augmented 1
        try:
            img = Image.open(img_path).convert('RGB')
            aug_img = aug_transform(img)
            aug_name = f"ood_train_{i:04d}_aug1{ext}"
            aug_img.save(out_train_aug_dir / aug_name)
        except Exception as e:
            print(f"Error augmenting {img_path}: {e}")
            
    print(f"Done! Created {len(list(out_train_aug_dir.glob('*.*')))} images in {out_train_aug_dir.name}")

if __name__ == '__main__':
    main()
