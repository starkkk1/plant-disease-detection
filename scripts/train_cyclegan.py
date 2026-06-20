"""
Training script for CycleGAN.
Translates images from Domain A to Domain B and vice versa.
"""

import os
import sys
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.cyclegan_dataset import UnalignedDataset
from src.models.cyclegan_generator import ResnetGenerator
from src.models.cyclegan_discriminator import NLayerDiscriminator

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(log_file):
    logger = logging.getLogger('cyclegan')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

class GANLoss(nn.Module):
    """Define different GAN objectives."""
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(base_dir, 'configs', 'cyclegan.yaml')
    config = load_config(config_path)

    checkpoint_dir = os.path.join(base_dir, config['output']['checkpoint_dir'])
    results_dir = os.path.join(base_dir, config['output']['results_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    log_file = os.path.join(results_dir, 'training_log.txt')
    logger = setup_logger(log_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Dataset
    domain_a_dir = os.path.join(base_dir, config['data']['domain_a_dir'])
    domain_b_dir = os.path.join(base_dir, config['data']['domain_b_dir'])
    
    dataset = UnalignedDataset(domain_a_dir, domain_b_dir, image_size=config['data']['image_size'], is_train=True)
    dataloader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data'].get('num_workers', 4))
    
    logger.info(f"Dataset size: {len(dataset)}")

    # Models
    logger.info("Initializing models...")
    netG_A = ResnetGenerator().to(device)
    netG_B = ResnetGenerator().to(device)
    netD_A = NLayerDiscriminator().to(device)
    netD_B = NLayerDiscriminator().to(device)

    # Losses
    criterionGAN = GANLoss().to(device)
    criterionCycle = nn.L1Loss()
    criterionIdt = nn.L1Loss()

    # Optimizers
    lr = config['training']['lr']
    beta1 = config['training']['beta1']
    optimizer_G = optim.Adam(list(netG_A.parameters()) + list(netG_B.parameters()), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(list(netD_A.parameters()) + list(netD_B.parameters()), lr=lr, betas=(beta1, 0.999))

    # Pools
    fake_A_pool = ImagePool(config['training']['pool_size'])
    fake_B_pool = ImagePool(config['training']['pool_size'])

    epochs = config['training']['epochs']
    lambda_A = config['training']['lambda_A']
    lambda_B = config['training']['lambda_B']
    lambda_idt = config['training']['lambda_identity']

    # Torch AMP for mixed precision
    scaler_G = torch.amp.GradScaler(device.type, enabled=device.type=='cuda')
    scaler_D = torch.amp.GradScaler(device.type, enabled=device.type=='cuda')

    logger.info("Starting training loop...")
    for epoch in range(epochs):
        epoch_iter = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, data in enumerate(pbar):
            real_A = data['A'].to(device)
            real_B = data['B'].to(device)

            # ------------------
            # Train Generators
            # ------------------
            optimizer_G.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                # Identity loss
                idt_A = netG_A(real_B)
                loss_idt_A = criterionIdt(idt_A, real_B) * lambda_B * lambda_idt
                idt_B = netG_B(real_A)
                loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_idt

                # GAN loss D_A(G_A(A))
                fake_B = netG_A(real_A)
                pred_fake = netD_A(fake_B)
                loss_G_A = criterionGAN(pred_fake, True)

                # GAN loss D_B(G_B(B))
                fake_A = netG_B(real_B)
                pred_fake = netD_B(fake_A)
                loss_G_B = criterionGAN(pred_fake, True)

                # Forward cycle loss || G_B(G_A(A)) - A||
                rec_A = netG_B(fake_B)
                loss_cycle_A = criterionCycle(rec_A, real_A) * lambda_A

                # Backward cycle loss || G_A(G_B(B)) - B||
                rec_B = netG_A(fake_A)
                loss_cycle_B = criterionCycle(rec_B, real_B) * lambda_B

                # combined loss
                loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
                
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # ------------------
            # Train Discriminators
            # ------------------
            optimizer_D.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                # D_A
                fake_B_p = fake_B_pool.query(fake_B.detach())
                pred_real_A = netD_A(real_B)
                loss_D_real_A = criterionGAN(pred_real_A, True)
                pred_fake_A = netD_A(fake_B_p)
                loss_D_fake_A = criterionGAN(pred_fake_A, False)
                loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

                # D_B
                fake_A_p = fake_A_pool.query(fake_A.detach())
                pred_real_B = netD_B(real_A)
                loss_D_real_B = criterionGAN(pred_real_B, True)
                pred_fake_B = netD_B(fake_A_p)
                loss_D_fake_B = criterionGAN(pred_fake_B, False)
                loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

                loss_D = loss_D_A + loss_D_B
                
            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            pbar.set_postfix({
                'G_loss': f"{loss_G.item():.4f}", 
                'D_loss': f"{loss_D.item():.4f}",
                'cyc_A': f"{loss_cycle_A.item():.4f}"
            })

        logger.info(f"End of epoch {epoch+1} / {epochs}")
        
        # Save checkpoints
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save(netG_A.state_dict(), os.path.join(checkpoint_dir, f'netG_A_epoch_{epoch+1}.pth'))
            torch.save(netG_B.state_dict(), os.path.join(checkpoint_dir, f'netG_B_epoch_{epoch+1}.pth'))
            torch.save(netD_A.state_dict(), os.path.join(checkpoint_dir, f'netD_A_epoch_{epoch+1}.pth'))
            torch.save(netD_B.state_dict(), os.path.join(checkpoint_dir, f'netD_B_epoch_{epoch+1}.pth'))
            logger.info(f"Saved checkpoints at epoch {epoch+1}")

if __name__ == '__main__':
    main()
