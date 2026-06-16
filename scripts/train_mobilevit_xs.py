"""
Training script for MobileViT-XS model.
Logs metrics during and after training.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import get_dataloaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if 'defaults' in config:
        # Resolve default config path
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
    logger = logging.getLogger('mobilevit_xs')
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

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return epoch_loss, acc, f1

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
    config_path = os.path.join(base_dir, 'configs', 'mobilevit_xs.yaml')
    
    config = load_config(config_path)
    
    results_dir = os.path.join(base_dir, config.get('output', {}).get('results_dir', 'results/new_processed'))
    checkpoint_dir = os.path.join(base_dir, config.get('output', {}).get('checkpoint_dir', 'checkpoints'))
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_name = config.get('model', {}).get('name', 'mobilevit_xs')
    log_file = os.path.join(results_dir, f'{model_name}_training.log')
    metrics_file = os.path.join(results_dir, f'{model_name}_metrics.json')
    
    logger = setup_logger(log_file)
    logger.info(f"Starting training for {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Adjust config paths if needed
    for key in ['train_dir', 'val_dir', 'test_dir']:
        if key in config.get('data', {}):
            config['data'][key] = os.path.join(base_dir, config['data'][key])
            
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    num_classes = config.get('model', {}).get('num_classes', 10)
    pretrained = config.get('model', {}).get('pretrained', True)
    
    logger.info(f"Creating model: {model_name}")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = model.to(device)
    
    lr = float(config.get('training', {}).get('lr', 1e-4))
    epochs = int(config.get('training', {}).get('epochs', 30))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type=='cuda')
    
    best_f1 = 0.0
    patience_counter = 0
    early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 5)
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_f1, val_prec, val_rec = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
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
    test_loss, test_acc, test_f1, test_prec, test_rec = validate(model, test_loader, criterion, device)
    logger.info(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
    
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tv_dir = os.path.join(base_dir, 'data', 'new-data', 'eval')
    tl_dir = os.path.join(base_dir, 'data', 'new_processed', 'eval', 'Tomato_Leaves')

    final_metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'history': history,
        'eval_tomato_village': {},
        'eval_tomato_leaves': {}
    }

    if os.path.exists(tv_dir):
        tv_dataset = ImageFolder(tv_dir, transform=test_transform)
        tv_loader = DataLoader(tv_dataset, batch_size=config.get('data', {}).get('batch_size', 32), shuffle=False)
        tv_loss, tv_acc, tv_f1, tv_prec, tv_rec = validate(model, tv_loader, criterion, device)
        logger.info(f"Eval (new-data) - Loss: {tv_loss:.4f}, Acc: {tv_acc:.4f}, F1: {tv_f1:.4f}, Precision: {tv_prec:.4f}, Recall: {tv_rec:.4f}")
        final_metrics['eval_tomato_village'] = {
            'loss': tv_loss, 'accuracy': tv_acc, 'f1': tv_f1, 'precision': tv_prec, 'recall': tv_rec
        }

    if os.path.exists(tl_dir):
        tl_dataset = ImageFolder(tl_dir, transform=test_transform)
        tl_loader = DataLoader(tl_dataset, batch_size=config.get('data', {}).get('batch_size', 32), shuffle=False)
        tl_loss, tl_acc, tl_f1, tl_prec, tl_rec = validate(model, tl_loader, criterion, device)
        logger.info(f"Tomato_Leaves Eval - Loss: {tl_loss:.4f}, Acc: {tl_acc:.4f}, F1: {tl_f1:.4f}, Precision: {tl_prec:.4f}, Recall: {tl_rec:.4f}")
        final_metrics['eval_tomato_leaves'] = {
            'loss': tl_loss, 'accuracy': tl_acc, 'f1': tl_f1, 'precision': tl_prec, 'recall': tl_rec
        }

    # Save final metrics
    with open(metrics_file, 'w') as f:
        json.dump(final_metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")

if __name__ == '__main__':
    main()
