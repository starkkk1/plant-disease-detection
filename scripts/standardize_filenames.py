import os

# Root directory of processed dataset
ROOT_DIR = r"D:\Code\python\data\processed_custom"
SPLITS = ["train", "val", "test", "test-multileaves"]

def main():
    if not os.path.exists(ROOT_DIR):
        print(f"Error: Processed root directory '{ROOT_DIR}' does not exist.")
        return
        
    print(f"Standardizing filenames in: {ROOT_DIR}")
    
    for split in SPLITS:
        split_path = os.path.join(ROOT_DIR, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"\nProcessing split: {split}")
        classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            
            # Get list of image files
            files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files.sort()
            
            total_files = len(files)
            if total_files == 0:
                continue
                
            # Phase 1: Rename to temporary names to prevent naming collisions (using a unique UUID prefix)
            import uuid
            unique_prefix = f"temp_std_{uuid.uuid4().hex}"
            
            temp_list = []
            for idx, filename in enumerate(files):
                ext = os.path.splitext(filename)[1].lower() # Normalize extension to lowercase
                src_path = os.path.join(class_path, filename)
                temp_name = f"{unique_prefix}_{idx}{ext}"
                temp_dest = os.path.join(class_path, temp_name)
                
                os.rename(src_path, temp_dest)
                temp_list.append((temp_dest, ext))
                
            # Phase 2: Rename to final standardized format: class_split_index.extension
            for idx, (temp_src, ext) in enumerate(temp_list):
                # Using 5-digit padding for index (e.g. 00001)
                final_name = f"{class_name}_{split}_{idx+1:05d}{ext}"
                final_dest = os.path.join(class_path, final_name)
                os.rename(temp_src, final_dest)
                
            print(f"  Class '{class_name}': Renamed {total_files} images.")
            
    print("\nFilename standardization completed successfully!")

if __name__ == "__main__":
    main()
