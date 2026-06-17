"""
Knowledge Distillation Training Script.
Trains a lightweight student model using a larger, robust teacher model.
"""

import os
import sys
import yaml
import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import timm
from timm.data import Mixup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse

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
    logger = logging.getLogger('distillation')
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

class DistillationLoss(nn.Module):
    """
    Computes the Knowledge Distillation loss.
    Loss = (1 - alpha) * CE(student_logits, true_labels) + (alpha * T^2) * KL(student_logits/T, teacher_logits/T)
    Note: If Mixup is used, the true_labels are soft labels, so we use CrossEntropy or SoftTargetCrossEntropy accordingly.
    """
    def __init__(self, alpha=0.5, temperature=3.0, base_criterion=None):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.base_criterion = base_criterion

    def forward(self, student_logits, teacher_logits, targets):
        # Student loss on true labels (which might be mixup soft labels)
        base_loss = self.base_criterion(student_logits, targets)
        
        # Distillation loss
        student_log_probs = F.log_softmax(student_logits / self.T, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.T, dim=1)
        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.T ** 2)
        
        return (1.0 - self.alpha) * base_loss + self.alpha * distillation_loss

def train_one_epoch(student, teacher, dataloader, criterion, optimizer, device, scaler, mixup_fn):
    student.train()
    teacher.eval() # Teacher is always in eval mode
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)
            
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            
            # Student forward pass
            student_logits = student(inputs)
            
            # Compute Distillation Loss
            loss = criterion(student_logits, teacher_logits, targets)
            
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
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument('--config', type=str, default='configs/distillation_mobilenetv3.yaml', help='Path to config file')
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(base_dir, args.config)
    
    config = load_config(config_path)
    
    results_dir = os.path.join(base_dir, config.get('output', {}).get('results_dir', 'results/new_processed'))
    checkpoint_dir = os.path.join(base_dir, config.get('output', {}).get('checkpoint_dir', 'checkpoints'))
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    student_name = config.get('model', {}).get('name', 'mobilenet_v3_small')
    teacher_name = config.get('distillation', {}).get('teacher_model', 'convnext_tiny')
    teacher_ckpt = os.path.join(base_dir, config.get('distillation', {}).get('teacher_checkpoint', 'checkpoints/convnext_tiny_best.pth'))
    
    log_file = os.path.join(results_dir, f'{student_name}_distilled_training.log')
    metrics_file = os.path.join(results_dir, f'{student_name}_distilled_metrics.json')
    
    logger = setup_logger(log_file)
    logger.info(f"Starting Knowledge Distillation.")
    logger.info(f"Teacher: {teacher_name}")
    logger.info(f"Student: {student_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    for key in ['train_dir', 'val_dir', 'test_dir']:
        if key in config.get('data', {}):
            config['data'][key] = os.path.join(base_dir, config['data'][key])
            
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    num_classes = config.get('model', {}).get('num_classes', 10)
    
    # Init Teacher
    logger.info(f"Loading Teacher model...")
    teacher = timm.create_model(teacher_name, pretrained=False, num_classes=num_classes)
    if not os.path.exists(teacher_ckpt):
        logger.error(f"Teacher checkpoint not found at {teacher_ckpt}. Please train the teacher first.")
        return
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Init Student
    logger.info(f"Loading Student model...")
    student = timm.create_model(student_name, pretrained=config.get('model', {}).get('pretrained', True), num_classes=num_classes)
    student = student.to(device)
    
    lr = float(config.get('training', {}).get('lr', 1e-4))
    epochs = int(config.get('training', {}).get('epochs', 30))
    
    # Setup Mixup
    from timm.loss import SoftTargetCrossEntropy
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
    
    base_train_criterion = SoftTargetCrossEntropy()
    kd_criterion = DistillationLoss(
        alpha=config.get('distillation', {}).get('alpha', 0.5),
        temperature=config.get('distillation', {}).get('temperature', 3.0),
        base_criterion=base_train_criterion
    )
    val_criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type=='cuda')
    
    best_f1 = 0.0
    patience_counter = 0
    early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 5)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(student, teacher, train_loader, kd_criterion, optimizer, device, scaler, mixup_fn)
        val_loss, val_acc, val_f1, val_prec, val_rec = validate(student, val_loader, val_criterion, device)
        
        logger.info(f"Train - KD Loss: {train_loss:.4f}")
        logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = os.path.join(checkpoint_dir, f'{student_name}_distilled_best.pth')
            torch.save(student.state_dict(), checkpoint_path)
            logger.info(f"Saved new best model with F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter}/{early_stopping_patience} epoch(s).")
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs!")
                break
            
    # Final test evaluation
    best_model_path = os.path.join(checkpoint_dir, f'{student_name}_distilled_best.pth')
    if os.path.exists(best_model_path):
        student.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_f1, test_prec, test_rec = validate(student, test_loader, val_criterion, device)
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
