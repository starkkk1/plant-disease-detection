"""
Generate Synthetic Field Data using trained CycleGAN Generator.
It translates all images from Domain A (Lab) to Domain B (Field) style,
preserving the original class subdirectory structure.
"""

import os
import sys
import yaml
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.cyclegan_generator import ResnetGenerator

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data using trained CycleGAN")
    parser.add_argument('--config', type=str, default='configs/cyclegan.yaml', help='Path to config file')
    parser.add_argument('--weights', type=str, required=True, help='Path to netG_A weights (Lab to Field)')
    parser.add_argument('--output_dir', type=str, default='data/new_processed_augmented/train_synthetic', help='Where to save synthetic images')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(base_dir, args.config)
    config = load_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading Generator A (Lab -> Field) from {args.weights}")
    netG_A = ResnetGenerator().to(device)
    netG_A.load_state_dict(torch.load(os.path.join(base_dir, args.weights), map_location=device))
    netG_A.eval()

    # Transforms
    image_size = config['data']['image_size']
    transform_in = transforms.Compose([
        transforms.Resize((image_size, image_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # De-normalization for saving
    def denormalize(tensor):
        return tensor * 0.5 + 0.5

    domain_a_dir = os.path.join(base_dir, config['data']['domain_a_dir'])
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning source directory: {domain_a_dir}")
    print(f"Output directory: {output_dir}")

    total_images = sum([len(files) for r, d, files in os.walk(domain_a_dir)])
    pbar = tqdm(total=total_images, desc="Generating")

    with torch.no_grad():
        for root, _, fnames in os.walk(domain_a_dir):
            for fname in fnames:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    # Path to source image
                    src_path = os.path.join(root, fname)
                    
                    # Compute relative path to maintain class folder structure
                    rel_path = os.path.relpath(root, domain_a_dir)
                    target_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    target_path = os.path.join(target_dir, fname)

                    # Process
                    img = Image.open(src_path).convert('RGB')
                    img_tensor = transform_in(img).unsqueeze(0).to(device)
                    
                    # Generate fake field image
                    fake_field_tensor = netG_A(img_tensor)
                    
                    # Save image
                    save_image(denormalize(fake_field_tensor.squeeze(0)), target_path)
                    
                    pbar.update(1)

    pbar.close()
    print("Done! All synthetic images generated.")

if __name__ == '__main__':
    main()
