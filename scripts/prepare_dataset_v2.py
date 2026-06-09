from PIL import Image
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths
PV_RAW = r"d:\Code\python\data+source\data+source\PlantVillage-Dataset\raw\color"
TV_DIR = r"d:\Code\python\data+source\data+source\Tomato-Village\Variant-a(Multiclass Classification)"
TL_DIR = r"d:\Code\python\data+source\data+source\Tomato_Leaves"
OUT_DIR = r"d:\Code\python\plant-disease-detection\data\new_processed"

def process_plantvillage():
    print("Processing PlantVillage...")
    pv_out_train = os.path.join(OUT_DIR, "train")
    pv_out_val = os.path.join(OUT_DIR, "val")
    pv_out_test = os.path.join(OUT_DIR, "test")
    
    if not os.path.exists(PV_RAW):
        print(f"Warning: {PV_RAW} does not exist.")
        return
        
    classes = [d for d in os.listdir(PV_RAW) if os.path.isdir(os.path.join(PV_RAW, d)) and d.startswith("Tomato___")]
    
    for cls in tqdm(classes, desc="PlantVillage Classes"):
        cls_dir = os.path.join(PV_RAW, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) == 0:
            continue
            
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        # Create out dirs
        os.makedirs(os.path.join(pv_out_train, cls), exist_ok=True)
        os.makedirs(os.path.join(pv_out_val, cls), exist_ok=True)
        os.makedirs(os.path.join(pv_out_test, cls), exist_ok=True)
        
        for img in train_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(pv_out_train, cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(pv_out_val, cls, img))
        for img in test_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(pv_out_test, cls, img))

def process_tomatovillage():
    print("Processing Tomato-Village...")
    tv_out = os.path.join(OUT_DIR, "eval", "Tomato-Village")
    
    if not os.path.exists(TV_DIR):
        print(f"Warning: {TV_DIR} does not exist.")
        return
        
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(TV_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        for cls in tqdm(classes, desc=f"Tomato-Village {split}"):
            cls_dir = os.path.join(split_dir, cls)
            out_cls_dir = os.path.join(tv_out, cls)
            os.makedirs(out_cls_dir, exist_ok=True)
            
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src = os.path.join(cls_dir, img)
                    dst = os.path.join(out_cls_dir, f"{split}_{img}")
                    shutil.copy(src, dst)

def process_tomatoleaves():
    print("Processing Tomato_Leaves...")
    tl_out = os.path.join(OUT_DIR, "eval", "Tomato_Leaves")
    
    if not os.path.exists(TL_DIR):
        print(f"Warning: {TL_DIR} does not exist.")
        return
        
    for split in ["train", "val"]:
        img_dir = os.path.join(TL_DIR, "Images", split)
        lbl_dir = os.path.join(TL_DIR, "labels", split)
        
        if not os.path.exists(img_dir):
            continue
            
        images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in tqdm(images, desc=f"Tomato_Leaves {split}"):
            base_name = os.path.splitext(img_name)[0]
            img_path = os.path.join(img_dir, img_name)
            txt_path = os.path.join(lbl_dir, base_name + ".txt")
            
            if not os.path.exists(txt_path):
                # Copy the whole image if no label
                out_cls_dir = os.path.join(tl_out, "no_label")
                os.makedirs(out_cls_dir, exist_ok=True)
                shutil.copy(img_path, os.path.join(out_cls_dir, f"{split}_{img_name}"))
                continue
            
            try:
                img = Image.open(img_path)
                img.verify()
                img = Image.open(img_path)
            except Exception:
                continue
                
            w, h = img.size
            
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = parts[0]
                    try:
                        x_c, y_c, box_w, box_h = map(float, parts[1:5])
                    except ValueError:
                        continue
                        
                    x_center = x_c * w
                    y_center = y_c * h
                    width = box_w * w
                    height = box_h * h
                    
                    x_min = int(max(0, x_center - width / 2))
                    y_min = int(max(0, y_center - height / 2))
                    x_max = int(min(w, x_center + width / 2))
                    y_max = int(min(h, y_center + height / 2))
                    
                    if y_max <= y_min or x_max <= x_min:
                        continue
                        
                    crop_img = img.crop((x_min, y_min, x_max, y_max))
                    
                    if crop_img.size[0] > 0 and crop_img.size[1] > 0:
                        out_cls_dir = os.path.join(tl_out, str(class_id))
                        os.makedirs(out_cls_dir, exist_ok=True)
                        out_path = os.path.join(out_cls_dir, f"{split}_{base_name}_crop_{i}.jpg")
                        try:
                            # Convert to RGB before saving to prevent errors with some formats
                            if crop_img.mode != 'RGB':
                                crop_img = crop_img.convert('RGB')
                            crop_img.save(out_path)
                        except Exception:
                            pass

if __name__ == "__main__":
    # Create main output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)
    
    process_plantvillage()
    process_tomatovillage()
    process_tomatoleaves()
    print(f"Dataset preparation completed! Data saved to {OUT_DIR}")
