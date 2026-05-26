import os
import cv2
import shutil

# Paths
OD_DIR = r"D:\Code\python\Tomato-Village\Variant-c(Object Detection)"
OUTPUT_DIR = r"D:\Code\python\data\processed_custom\test-multileaves"

# Class Mapping from Varient-C Labels.txt
class_mapping = {
    0: "Tomato___Early_blight",
    1: "Tomato___healthy",
    2: "Tomato___Late_blight",
    3: "Tomato___Leaf_Miner",
    4: "Tomato___Magnesium_Deficiency",
    5: "Tomato___Nitrogen_Deficiency",
    6: "Tomato___Pottassium_Deficiency",
    7: "Tomato___Spotted_Wilt_Virus"
}

def extract_crops_from_split(split_name):
    split_dir = os.path.join(OD_DIR, split_name)
    image_dir = os.path.join(split_dir, "images")
    yolo_dir = os.path.join(split_dir, "yolo")
    
    if not os.path.exists(image_dir) or not os.path.exists(yolo_dir):
        print(f"Directory missing for split: {split_name}")
        return 0
        
    print(f"Processing split '{split_name}' from: {split_dir}")
    
    total_cropped = 0
    yolo_files = [f for f in os.listdir(yolo_dir) if f.endswith(".txt")]
    
    for filename in yolo_files:
        base_name = os.path.splitext(filename)[0]
        # Bounding boxes label file
        label_path = os.path.join(yolo_dir, filename)
        
        # Try to find corresponding image file (.jpg, .png, .jpeg, or uppercase)
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
            for idx, line in enumerate(f.readlines()):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                except ValueError:
                    continue
                
                # Check class id validity
                class_name = class_mapping.get(class_id)
                if not class_name:
                    continue
                    
                # Convert normalized YOLO coordinates to pixel coordinates
                x1 = int((x_c - box_w/2) * w)
                y1 = int((y_c - box_h/2) * h)
                x2 = int((x_c + box_w/2) * w)
                y2 = int((y_c + box_h/2) * h)
                
                # Clip to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Make sure crop is valid
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue
                    
                crop_img = img[y1:y2, x1:x2]
                
                # Save crop to target directory
                dest_class_dir = os.path.join(OUTPUT_DIR, class_name)
                os.makedirs(dest_class_dir, exist_ok=True)
                
                crop_name = f"crop_{split_name}_{base_name}_{idx}.jpg"
                dest_path = os.path.join(dest_class_dir, crop_name)
                
                cv2.imwrite(dest_path, crop_img)
                total_cropped += 1
                
    print(f"Extracted {total_cropped} leaf crops from split '{split_name}'.")
    return total_cropped

def main():
    if not os.path.exists(OD_DIR):
        print(f"Error: Object Detection directory '{OD_DIR}' does not exist.")
        return
        
    # Clear existing output directory if it exists
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total = 0
    # Process both train and val splits of Object Detection dataset
    for split in ["train", "val"]:
        total += extract_crops_from_split(split)
        
    print(f"\nCompleted! Total cropped images saved: {total}")
    print(f"Crops stored at: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
