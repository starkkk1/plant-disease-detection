import os
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    removed_dir = Path(os.path.join(base_dir, 'data', 'new-data-removal'))
    original_dir = Path(os.path.join(base_dir, 'data', 'new-data'))
    
    if not removed_dir.exists():
        print(f"Directory {removed_dir} does not exist.")
        return

    # Find all images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(list(removed_dir.rglob(ext)))
        
    print(f"Found {len(image_paths)} processed images. Checking for environment images...")
    
    restored_count = 0
    for img_path in tqdm(image_paths, desc="Checking and restoring"):
        try:
            name_lower = img_path.name.lower()
            if '_train_' in name_lower or '_val_' in name_lower or '_test_' in name_lower:
                rel_path = img_path.relative_to(removed_dir)
                orig_parent = original_dir / rel_path.parent
                orig_stem = rel_path.stem
                
                # Find original file regardless of extension
                found_orig = None
                if orig_parent.exists():
                    for f in orig_parent.iterdir():
                        if f.stem == orig_stem:
                            found_orig = f
                            break
                            
                if found_orig:
                    # Remove the bad output
                    if found_orig.suffix.lower() != img_path.suffix.lower() or found_orig.suffix != img_path.suffix:
                        pass
                        
                    dest_path = removed_dir / rel_path.parent / found_orig.name
                    if img_path != dest_path and img_path.exists():
                        img_path.unlink()
                        
                    shutil.copy2(found_orig, dest_path)
                    restored_count += 1
                    
        except Exception as e:
            pass # Skip broken images
            
    print(f"\nDone! Restored {restored_count} environment images back to their original state.")

if __name__ == '__main__':
    main()
