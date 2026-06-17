import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from src.datasets.transforms import get_transforms

class TomatoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_folder = ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.image_folder.classes
        self.class_to_idx = self.image_folder.class_to_idx

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        img_path, label = self.image_folder.samples[idx]
        
        # Read image with OpenCV (BGR) and convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

def get_dataloaders(config):
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']
    test_dir = config.get('data', {}).get('test_dir')
    
    batch_size = config.get('data', {}).get('batch_size', 32)
    num_workers = config.get('data', {}).get('num_workers', 4)
    
    train_transform = get_transforms(config, split="train")
    val_transform = get_transforms(config, split="val")
    
    train_dataset = TomatoDataset(train_dir, transform=train_transform)
    val_dataset = TomatoDataset(val_dir, transform=val_transform)
    
    # Calculate class weights if needed
    class_weights = None
    if config.get('training', {}).get('use_class_weights', False):
        targets = [s[1] for s in train_dataset.image_folder.samples]
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = torch.FloatTensor(class_weights)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    if test_dir and os.path.exists(test_dir):
        test_dataset = TomatoDataset(test_dir, transform=val_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        loaders['test'] = test_loader
        
    return loaders, class_weights
