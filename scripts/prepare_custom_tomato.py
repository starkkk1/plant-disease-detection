import os
import shutil
import random
import argparse
import json

# Seed for reproducibility
random.seed(42)

# Source directories defaults
PLANT_VILLAGE_DEFAULT = r"D:\Code\python\PlantVillage-Dataset"
TOMATO_VILLAGE_DEFAULT = r"D:\Code\python\Tomato-Village"
OUTPUT_DIR_DEFAULT = r"D:\Code\python\data\processed_custom"

# Class name normalization & mapping
PLANTVILLAGE_TOMATO_CLASSES = {
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___Leaf_Mold": "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot": "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy": "Tomato___healthy"
}

TOMATO_VILLAGE_CLASSES = {
    "Early_blight": "Tomato___Early_blight",
    "Healthy": "Tomato___healthy",
    "Late_blight": "Tomato___Late_blight",
    "Leaf Miner": "Tomato___Leaf_Miner",
    "Magnesium Deficiency": "Tomato___Magnesium_Deficiency",
    "Nitrogen Deficiency": "Tomato___Nitrogen_Deficiency",
    "Pottassium Deficiency": "Tomato___Pottassium_Deficiency",
    "Spotted Wilt Virus": "Tomato___Spotted_Wilt_Virus"
}

COMMON_CLASSES = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]

def scan_images(directory):
    images = []
    if not os.path.exists(directory):
        return images
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, f))
    return images

def copy_images(src_paths, dest_dir, prefix=""):
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for path in src_paths:
        name = os.path.basename(path)
        if prefix:
            name = f"{prefix}_{name}"
        dest_path = os.path.join(dest_dir, name)
        # Avoid overwriting and duplicate names
        if os.path.exists(dest_path):
            name = f"{prefix}_{random.randint(1000, 9999)}_{os.path.basename(path)}"
            dest_path = os.path.join(dest_dir, name)
        shutil.copy2(path, dest_path)
        copied += 1
    return copied

def main():
    parser = argparse.ArgumentParser(description="Split and merge PlantVillage and Tomato-Village datasets")
    parser.add_argument("--pv-path", default=PLANT_VILLAGE_DEFAULT, help="Path to PlantVillage-Dataset")
    parser.add_argument("--tv-path", default=TOMATO_VILLAGE_DEFAULT, help="Path to Tomato-Village")
    parser.add_argument("--out-path", default=OUTPUT_DIR_DEFAULT, help="Output destination folder")
    parser.add_argument("--mode", choices=["all", "common_only"], default="all",
                        help="all: use all 15 classes; common_only: use only 3 overlapping classes")
    args = parser.parse_args()

    pv_color_dir = os.path.join(args.pv_path, "raw", "color")
    tv_multiclass_dir = os.path.join(args.tv_path, "Variant-a(Multiclass Classification)")

    # Validate directories
    if not os.path.exists(pv_color_dir):
        print(f"Error: PlantVillage tomato directory not found at: {pv_color_dir}")
        return
    if not os.path.exists(tv_multiclass_dir):
        print(f"Error: Tomato-Village Multiclass directory not found at: {tv_multiclass_dir}")
        return

    print(f"Starting dataset preparation with mode: {args.mode.upper()}")
    print(f"Source PV: {pv_color_dir}")
    print(f"Source TV: {tv_multiclass_dir}")
    print(f"Output path: {args.out_path}")

    # Remove target directory if exists to start fresh
    if os.path.exists(args.out_path):
        print("Cleaning previous output folder...")
        shutil.rmtree(args.out_path)
    os.makedirs(args.out_path, exist_ok=True)

    # Class lists based on mode
    if args.mode == "common_only":
        active_pv_classes = {k: v for k, v in PLANTVILLAGE_TOMATO_CLASSES.items() if v in COMMON_CLASSES}
        active_tv_classes = {k: v for k, v in TOMATO_VILLAGE_CLASSES.items() if v in COMMON_CLASSES}
    else:
        active_pv_classes = PLANTVILLAGE_TOMATO_CLASSES
        active_tv_classes = TOMATO_VILLAGE_CLASSES

    stats = {
        "train": {},
        "val": {},
        "test": {}
    }

    # Step 1: Process PlantVillage (Controlled environment)
    # Split PlantVillage classes into 80% train / 20% val for active classes.
    # For classes in "all" mode that are NOT in Tomato-Village test set (i.e. the 7 non-overlapping classes), 
    # we split them 70% train / 15% val / 15% test so we still have test images.
    print("\n--- Processing PlantVillage-Dataset ---")
    for pv_folder, standard_class in active_pv_classes.items():
        src_dir = os.path.join(pv_color_dir, pv_folder)
        images = scan_images(src_dir)
        random.shuffle(images)
        total = len(images)
        
        if total == 0:
            print(f"Warning: No images found in {src_dir}")
            continue

        # Decide split ratios based on whether this class has a test set in Tomato-Village
        if args.mode == "all" and standard_class not in COMMON_CLASSES:
            # Non-overlapping class: needs test set from PV
            train_idx = int(total * 0.70)
            val_idx = train_idx + int(total * 0.15)
            
            train_imgs = images[:train_idx]
            val_imgs = images[train_idx:val_idx]
            test_imgs = images[val_idx:]
        else:
            # Overlapping or common class: Tomato-Village will provide the primary test set
            # Split PV as 80% train / 20% val
            train_idx = int(total * 0.80)
            train_imgs = images[:train_idx]
            val_imgs = images[train_idx:]
            test_imgs = []

        # Copy PV images
        t_copied = copy_images(train_imgs, os.path.join(args.out_path, "train", standard_class), prefix="pv")
        v_copied = copy_images(val_imgs, os.path.join(args.out_path, "val", standard_class), prefix="pv")
        te_copied = 0
        if test_imgs:
            te_copied = copy_images(test_imgs, os.path.join(args.out_path, "test", standard_class), prefix="pv")

        stats["train"][standard_class] = stats["train"].get(standard_class, 0) + t_copied
        stats["val"][standard_class] = stats["val"].get(standard_class, 0) + v_copied
        if te_copied > 0:
            stats["test"][standard_class] = stats["test"].get(standard_class, 0) + te_copied

        print(f"Class '{standard_class}': copied {t_copied} train, {v_copied} val, {te_copied} test images from PlantVillage.")

    # Step 2: Process Tomato-Village (Real-world environment)
    # The folders train/val/test are already separated. We copy them into our dataset.
    print("\n--- Processing Tomato-Village ---")
    tv_splits = ["train", "val", "test"]
    for split in tv_splits:
        split_dir = os.path.join(tv_multiclass_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for tv_folder, standard_class in active_tv_classes.items():
            src_dir = os.path.join(split_dir, tv_folder)
            images = scan_images(src_dir)
            total = len(images)
            
            if total == 0:
                continue

            # In common_only mode, we only use test images from Tomato-Village, as per user requirement.
            # However, for the 5 unique real-world classes in 'all' mode, we need their train/val images from TV too!
            if args.mode == "common_only" and split != "test":
                # In common_only, we only want Tomato-Village for test. Skip train/val.
                continue

            # Copy TV images
            copied = copy_images(images, os.path.join(args.out_path, split, standard_class), prefix="tv")
            stats[split][standard_class] = stats[split].get(standard_class, 0) + copied
            print(f"Split '{split}' Class '{standard_class}': copied {copied} images from Tomato-Village.")

    # Write summary report
    print("\n--- Dataset Summary ---")
    all_classes = sorted(list(set(list(stats["train"].keys()) + list(stats["val"].keys()) + list(stats["test"].keys()))))
    
    print(f"{'Class Name':<45} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Total':<6}")
    print("-" * 75)
    for cls in all_classes:
        tr = stats["train"].get(cls, 0)
        va = stats["val"].get(cls, 0)
        te = stats["test"].get(cls, 0)
        tot = tr + va + te
        print(f"{cls:<45} | {tr:<6} | {va:<6} | {te:<6} | {tot:<6}")
    
    # Save class mapping to json
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    mapping_path = os.path.join(args.out_path, "class_to_idx.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"\nSaved class index mapping to {mapping_path}")
    print("Dataset division completed successfully!")

if __name__ == "__main__":
    main()
