import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_image(args):
    img_path, out_path = args
    if out_path.exists():
        return
        
    img = cv2.imread(str(img_path))
    if img is None:
        return
        
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for green/yellow/brown colors of leaves
    lower_bound = np.array([10, 20, 20])
    upper_bound = np.array([90, 255, 255])
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask=mask)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), res)

def main():
    base_dir = Path(r"d:\Code\python\plant-disease-detection")
    in_dir = base_dir / "data" / "new_processed_augmented" / "train_augmented"
    out_dir = base_dir / "data" / "new_processed_bg_removed" / "train_augmented"
    
    if not in_dir.exists():
        print(f"Input directory does not exist: {in_dir}")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png']:
        for img_path in in_dir.rglob(ext):
            rel_path = img_path.relative_to(in_dir)
            out_path = out_dir / rel_path
            tasks.append((img_path, out_path))
            
    print(f"Found {len(tasks)} images to process. Starting background removal...")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks), desc="Processing"))
        
    print(f"Background removal complete. Results saved to {out_dir}")

if __name__ == '__main__':
    main()
