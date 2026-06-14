import cv2
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Global variable to hold background paths for workers
g_bg_paths = []

def process_image(args):
    img_path, out_path, bg_path = args
    if out_path.exists():
        return
        
    # Read foreground (the leaf on black background)
    fg = cv2.imread(str(img_path))
    if fg is None:
        return
        
    h, w = fg.shape[:2]
    
    # Load chosen background
    bg = cv2.imread(str(bg_path))
    if bg is None:
        return
        
    # Resize background to match foreground
    bg = cv2.resize(bg, (w, h))
    
    # Heavy blur to destroy any disease patterns in the background, leaving just the texture/color
    bg = cv2.GaussianBlur(bg, (31, 31), 0)
    
    # Create mask from foreground. Since it's JPEG, black might not be exactly 0.
    # We sum the channels and threshold
    fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(fg_gray, 5, 255, cv2.THRESH_BINARY)
    
    # Convert mask to 3 channels
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Normalize mask to 0-1
    mask_norm = mask_3d.astype(np.float32) / 255.0
    fg_norm = fg.astype(np.float32)
    bg_norm = bg.astype(np.float32)
    
    # Alpha blending
    result = fg_norm * mask_norm + bg_norm * (1.0 - mask_norm)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)

def main():
    global g_bg_paths
    random.seed(42)
    
    base_dir = Path(r"d:\Code\python\plant-disease-detection")
    
    # The source is the background-removed images
    in_dir = base_dir / "data" / "new_processed_bg_removed" / "train_augmented"
    # The destination is the final copy-pasted augmented dataset
    out_dir = base_dir / "data" / "new_processed_copy_paste" / "train_augmented"
    
    # We use Tomato-Village as a source of backgrounds!
    bg_dir = base_dir / "data" / "new_processed" / "eval" / "Tomato-Village"
    
    if not in_dir.exists():
        print(f"Input directory does not exist: {in_dir}")
        return
        
    g_bg_paths = list(bg_dir.rglob("*.JPG")) + list(bg_dir.rglob("*.jpg"))
    if not g_bg_paths:
        print("No background images found!")
        return
        
    print(f"Loaded {len(g_bg_paths)} background images.")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
        for img_path in in_dir.rglob(ext):
            rel_path = img_path.relative_to(in_dir)
            out_path = out_dir / rel_path
            bg_path = random.choice(g_bg_paths)
            tasks.append((img_path, out_path, bg_path))
            
    print(f"Found {len(tasks)} images to process. Starting Copy-Paste Augmentation...")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc="Copy-Pasting"))
        
    print(f"Copy-Paste complete. Results saved to {out_dir}")

if __name__ == '__main__':
    main()
