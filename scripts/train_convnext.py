"""
Training script for ConvNeXt model with Mixup/CutMix.
Robust against domain shift.
"""

import os
import sys
import yaml
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.dataset import get_dataloaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'defaults' in config:
        default_path = os.path.join(os.path.dirname(config_path), os.path.basename(config['defaults']))
        with open(default_path, 'r') as f:
            default_config = yaml.safe_load(f)
        merged = {**default_config, **config}
        for k, v in config.items():
            if isinstance(v, dict) and k in default_config and isinstance(default_config[k], dict):
                merged[k] = {**default_config[k], **v}
        return merged
    return config

def setup_logger(log_file):
    logger = logging.getLogger('convnext')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, mixup_fn):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
            
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, acc, f1, precision, recall

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(base_dir, 'configs', 'convnext.yaml')
    
    config = load_config(config_path)
    
    results_dir = os.path.join(base_dir, config.get('output', {}).get('results_dir', 'results/new_processed'))
    checkpoint_dir = os.path.join(base_dir, config.get('output', {}).get('checkpoint_dir', 'checkpoints'))
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_name = config.get('model', {}).get('name', 'convnext_tiny')
    log_file = os.path.join(results_dir, f'{model_name}_training.log')
    metrics_file = os.path.join(results_dir, f'{model_name}_metrics.json')
    
    logger = setup_logger(log_file)
    logger.info(f"Starting training for {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    for key in ['train_dir', 'val_dir', 'test_dir']:
        if key in config.get('data', {}):
            config['data'][key] = os.path.join(base_dir, config['data'][key])
            
    loaders, class_weights = get_dataloaders(config)
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test', val_loader)
    
    num_classes = config.get('model', {}).get('num_classes', 10)
    pretrained = config.get('model', {}).get('pretrained', True)
    
    logger.info(f"Creating model: {model_name}")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = model.to(device)
    
    lr = float(config.get('training', {}).get('lr', 1e-4))
    epochs = int(config.get('training', {}).get('epochs', 30))
    
    # Setup Mixup
    mixup_args = {
        'mixup_alpha': config.get('training', {}).get('mixup_alpha', 0.8),
        'cutmix_alpha': config.get('training', {}).get('cutmix_alpha', 1.0),
        'prob': config.get('training', {}).get('prob', 1.0),
        'switch_prob': config.get('training', {}).get('switch_prob', 0.5),
        'mode': 'batch',
        'label_smoothing': 0.1,
        'num_classes': num_classes
    }
    mixup_fn = Mixup(**mixup_args)
    logger.info(f"Using Mixup/CutMix with args: {mixup_args}")
    
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type=='cuda')
    
    best_f1 = 0.0
    patience_counter = 0
    early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 5)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, train_criterion, optimizer, device, scaler, mixup_fn)
        val_loss, val_acc, val_f1, val_prec, val_rec = validate(model, val_loader, val_criterion, device)
        
        logger.info(f"Train - Loss: {train_loss:.4f}")
        logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved new best model with F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{early_stopping_patience} epoch(s).")
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs!")
                break
            
    # Final test evaluation
    best_model_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_f1, test_prec, test_rec = validate(model, test_loader, val_criterion, device)
    logger.info(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

    final_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'history': history,
    }

    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")

if __name__ == '__main__':
    main()
