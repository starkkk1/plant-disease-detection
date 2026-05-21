import os
import shutil
import random
import argparse
import json
import yaml
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger

TARGET_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def normalize_class_name(name: str) -> str:
    return name.lower().replace("___", "_").replace(" ", "_").replace("-", "_")

def get_target_class_mapping():
    return {normalize_class_name(cls): cls for cls in TARGET_CLASSES}

def scan_for_images(directory: str):
    """
    Scans a directory for images, returning a list of paths.
    """
    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))
    return image_paths

def copy_image(src_path, dest_dir, new_name=None):
    os.makedirs(dest_dir, exist_ok=True)
    if new_name is None:
        new_name = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, new_name)
    shutil.copy2(src_path, dest_path)

def generate_report(processed_root, output_report_path, logger):
    """
    Generates reports/dataset_summary.md with stats.
    """
    splits = ['train', 'val', 'test']
    stats = {split: {cls: 0 for cls in TARGET_CLASSES} for split in splits}
    
    for split in splits:
        split_dir = os.path.join(processed_root, split)
        if not os.path.exists(split_dir):
            continue
        for cls in TARGET_CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            if os.path.exists(cls_dir):
                stats[split][cls] = len([
                    f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                
    # Calculate totals
    total_per_class = {cls: sum(stats[split][cls] for split in splits) for cls in TARGET_CLASSES}
    total_per_split = {split: sum(stats[split][cls] for cls in TARGET_CLASSES) for split in splits}
    total_images = sum(total_per_split.values())
    
    # Write to markdown
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Summary Report\n\n")
        f.write("This report is auto-generated during dataset preparation.\n\n")
        
        f.write("## Image Counts by Class and Split\n\n")
        f.write("| Class Name | Train | Val | Test | Total |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for cls in TARGET_CLASSES:
            f.write(f"| {cls} | {stats['train'][cls]} | {stats['val'][cls]} | {stats['test'][cls]} | {total_per_class[cls]} |\n")
        f.write(f"| **Total** | **{total_per_split['train']}** | **{total_per_split['val']}** | **{total_per_split['test']}** | **{total_images}** |\n\n")
        
        f.write("## Class Imbalance Analysis\n\n")
        if total_images > 0:
            max_class = max(total_per_class, key=total_per_class.get)
            min_class = min(total_per_class, key=total_per_class.get)
            ratio = total_per_class[max_class] / max(total_per_class[min_class], 1)
            f.write(f"- **Majority Class**: `{max_class}` ({total_per_class[max_class]} images)\n")
            f.write(f"- **Minority Class**: `{min_class}` ({total_per_class[min_class]} images)\n")
            f.write(f"- **Imbalance Ratio (Max/Min)**: {ratio:.2f}x\n\n")
            f.write("Use `class_weights` in CrossEntropyLoss during training to mitigate this imbalance.\n")
            
    logger.info(f"Dataset stats report saved to {output_report_path}")
    
    # Also log stats to console
    logger.info("Dataset split summary:")
    for split in splits:
        logger.info(f"  {split}: {total_per_split[split]} images")

def prepare_dataset(config_path: str, filter_tomato: bool, force_split: bool, run_stats: bool):
    config = load_config(config_path)
    
    # Setup logger
    log_dir = config['output'].get('log_dir', 'results/logs')
    logger = setup_logger("prepare_dataset", log_dir=log_dir)
    
    # Set seed
    set_seed(config['project'].get('seed', 42))
    
    raw_root = config['data']['raw_root']
    processed_root = config['data']['processed_root']
    class_map_path = config['data']['class_map']
    
    logger.info(f"Starting dataset preparation. Raw root: {raw_root}, Processed root: {processed_root}")
    
    if not os.path.exists(raw_root):
        logger.error(f"Raw dataset path '{raw_root}' does not exist.")
        logger.error("Please download the dataset from Kaggle and extract it:")
        logger.error("  kaggle datasets download -d rashidthihan/plant-disease-dataset")
        logger.error("  unzip plant-disease-dataset.zip -d data/raw/")
        return
        
    # Check if raw_root contains pre-split folders (train/valid/test or train/val/test)
    raw_subdirs = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
    
    # We identify splits (case-insensitive checks)
    train_names = {'train', 'training'}
    val_names = {'val', 'valid', 'validation'}
    test_names = {'test', 'testing', 'test_set'}
    
    has_train = any(d.lower() in train_names for d in raw_subdirs)
    has_val = any(d.lower() in val_names for d in raw_subdirs)
    has_test = any(d.lower() in test_names for d in raw_subdirs)
    
    is_pre_split = has_train and has_val and has_test
    
    # Target class mapping
    class_mapping = get_target_class_mapping()
    
    # Clear processed root directory if it exists to avoid mixing old data
    if os.path.exists(processed_root):
        logger.info(f"Cleaning existing processed directory: {processed_root}")
        shutil.rmtree(processed_root)
    os.makedirs(processed_root, exist_ok=True)
    
    if is_pre_split and not force_split:
        logger.info("Detected pre-split folders in raw dataset. Copying files...")
        
        # Determine actual folder names for splits
        raw_train_dir = next(os.path.join(raw_root, d) for d in raw_subdirs if d.lower() in train_names)
        raw_val_dir = next(os.path.join(raw_root, d) for d in raw_subdirs if d.lower() in val_names)
        raw_test_dir = next(os.path.join(raw_root, d) for d in raw_subdirs if d.lower() in test_names)
        
        splits = [
            (raw_train_dir, 'train'),
            (raw_val_dir, 'val'),
            (raw_test_dir, 'test')
        ]
        
        for raw_split_path, split_name in splits:
            logger.info(f"Processing split '{split_name}' from '{raw_split_path}'...")
            for item in os.listdir(raw_split_path):
                raw_class_path = os.path.join(raw_split_path, item)
                if not os.path.isdir(raw_class_path):
                    continue
                    
                norm_name = normalize_class_name(item)
                
                # Check filter
                if filter_tomato and not norm_name.startswith("tomato"):
                    continue
                    
                # Match to target class name
                if norm_name in class_mapping:
                    target_class_name = class_mapping[norm_name]
                    dest_class_dir = os.path.join(processed_root, split_name, target_class_name)
                    
                    # Copy images
                    img_files = [f for f in os.listdir(raw_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    for img_name in img_files:
                        copy_image(os.path.join(raw_class_path, img_name), dest_class_dir)
                        
    else:
        # We need to split the dataset ourselves
        logger.info("Dataset will be merged and split using 70/15/15 ratio...")
        
        # Dictionary: target_class_name -> list of (src_img_path, new_filename)
        class_images = {cls: [] for cls in TARGET_CLASSES}
        
        # Scan raw_root for any folder matching target classes
        # This scans recursively so it will find classes whether they are in subfolders or not
        for root, dirs, files in os.walk(raw_root):
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                continue
                
            folder_name = os.path.basename(root)
            norm_name = normalize_class_name(folder_name)
            
            if filter_tomato and not norm_name.startswith("tomato"):
                continue
                
            if norm_name in class_mapping:
                target_class_name = class_mapping[norm_name]
                
                # To distinguish sources if we are merging from multiple dataset folders
                rel_path = os.path.relpath(root, raw_root)
                parts = rel_path.split(os.sep)
                source_prefix = parts[0] if len(parts) > 1 else "raw"
                
                for img_name in images:
                    img_path = os.path.join(root, img_name)
                    new_img_name = f"{source_prefix}_{img_name}"
                    class_images[target_class_name].append((img_path, new_img_name))
                    
        # Perform splits
        splits = ['train', 'val', 'test']
        train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
        
        for target_class_name, images in class_images.items():
            if not images:
                logger.warning(f"No images found for class: {target_class_name}")
                continue
                
            random.shuffle(images)
            total_images = len(images)
            
            train_idx = int(total_images * train_ratio)
            val_idx = train_idx + int(total_images * val_ratio)
            
            train_imgs = images[:train_idx]
            val_imgs = images[train_idx:val_idx]
            test_imgs = images[val_idx:]
            
            # Helper to copy images
            for img_list, split_name in [(train_imgs, 'train'), (val_imgs, 'val'), (test_imgs, 'test')]:
                dest_class_dir = os.path.join(processed_root, split_name, target_class_name)
                for src_path, new_name in img_list:
                    copy_image(src_path, dest_class_dir, new_name=new_name)
                    
            logger.info(f"Processed class '{target_class_name}': {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")
            
    # Save class_to_idx.json in the processed root directory
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(TARGET_CLASSES))}
    os.makedirs(os.path.dirname(class_map_path), exist_ok=True)
    with open(class_map_path, 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, indent=4)
    logger.info(f"Saved class-to-index mapping to {class_map_path}")
    
    # Run stats if requested
    if run_stats:
        reports_dir = config['output']['reports_dir']
        output_report_path = os.path.join(reports_dir, 'dataset_summary.md')
        generate_report(processed_root, output_report_path, logger)
        
    logger.info("Dataset preparation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and split plant disease dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--filter-tomato", action="store_true", help="Only process Tomato classes")
    parser.add_argument("--force-split", action="store_true", help="Force custom merging/splitting of raw data")
    parser.add_argument("--stats", action="store_true", help="Generate dataset summary stats report")
    args = parser.parse_args()
    
    prepare_dataset(args.config, args.filter_tomato, args.force_split, args.stats)
