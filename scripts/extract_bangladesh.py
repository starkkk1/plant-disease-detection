import os
import cv2
import shutil

# Paths
SOURCE_DIR = r"D:\Code\python\Tomato_Leaves"
OUTPUT_DIR = r"D:\Code\python\data\processed_custom\test-bangladesh"

# Class mapping based on Bangladesh dataset description
# Class 0: Healthy, Class 1: Diseased
class_mapping = {
    0: "Tomato___healthy",
    1: "Tomato___diseased"
}

def extract_crops():
    print(f"Starting extraction from Bangladesh dataset: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Clean output directory if it exists
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Counter for filenames per class to ensure clean numbering
    class_counters = {cls: 1 for cls in class_mapping.values()}
    total_cropped = 0

    splits = ["train", "val"]
    for split in splits:
        image_dir = os.path.join(SOURCE_DIR, "Images", split)
        yolo_dir = os.path.join(SOURCE_DIR, "labels", split)
        
        if not os.path.exists(image_dir) or not os.path.exists(yolo_dir):
            print(f"Warning: Directory missing for split: {split}")
            continue
            
        print(f"Processing split '{split}'...")
        yolo_files = [f for f in os.listdir(yolo_dir) if f.endswith(".txt")]
        
        for filename in yolo_files:
            base_name = os.path.splitext(filename)[0]
            label_path = os.path.join(yolo_dir, filename)
            
            # Find the corresponding image file
            img_name = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                potential_name = base_name + ext
                if os.path.exists(os.path.join(image_dir, potential_name)):
                    img_name = potential_name
                    break
                    
            if not img_name:
                continue
                
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w, _ = img.shape
            
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        # Parse class_id and box coordinates
                        class_id = int(float(parts[0]))
                        x_c = float(parts[1])
                        y_c = float(parts[2])
                        box_w = float(parts[3])
                        box_h = float(parts[4])
                    except ValueError:
                        continue
                    
                    class_name = class_mapping.get(class_id)
                    if not class_name:
                        continue
                        
                    # Convert to pixel coordinates
                    x1 = int((x_c - box_w/2) * w)
                    y1 = int((y_c - box_h/2) * h)
                    x2 = int((x_c + box_w/2) * w)
                    y2 = int((y_c + box_h/2) * h)
                    
                    # Clip boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                        continue
                        
                    crop_img = img[y1:y2, x1:x2]
                    
                    # Save crop with standardized naming
                    dest_class_dir = os.path.join(OUTPUT_DIR, class_name)
                    os.makedirs(dest_class_dir, exist_ok=True)
                    
                    idx = class_counters[class_name]
                    # Format: classname_bangladesh_index.jpg
                    crop_name = f"{class_name}_bangladesh_{idx:05d}.jpg"
                    dest_path = os.path.join(dest_class_dir, crop_name)
                    
                    cv2.imwrite(dest_path, crop_img)
                    class_counters[class_name] += 1
                    total_cropped += 1
                    
    print("\n--- Extraction Summary ---")
    for cls, counter in class_counters.items():
        print(f"  Class '{cls}': Extracted {counter - 1} images.")
    print(f"Total images saved to {OUTPUT_DIR}: {total_cropped}")

if __name__ == "__main__":
    extract_crops()
