"""
Модуль для проверки overfit на 1-2 картах.
Используется для быстрой проверки корректности обучения.
"""

import os
import json
from typing import Dict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics.metrics import MetricsCalculator
from visualization.visualize import visualize_training_results
from settings import settings


def overfit_check(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,  # CombinedLoss
    optimizer: torch.optim.Optimizer,
    device: str = None,
    n_epochs: int = 100,
    save_path: str = './logs/overfit_check/'
) -> None:
    """
    Проверка обучения на overfit на 1-2 картах без валидации.
    
    Args:
        model: Модель
        train_loader: DataLoader с 1-2 картами
        criterion: Функция потерь
        optimizer: Оптимизатор
        device: Устройство
        n_epochs: Количество эпох
        save_path: Путь для сохранения логов
    """
    device = device or settings.DEVICE
    os.makedirs(save_path, exist_ok=True)
    
    print("=" * 60)
    print("OVERFIT CHECK: Training on 1-2 maps without validation")
    print("=" * 60)
    
    model.train()
    
    history = {
        'loss': [],
        'dice': [],
        'iou': []
    }
    
    metrics_calc = MetricsCalculator()
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask_map = batch.get('mask_map', None)
            if mask_map is not None:
                mask_map = mask_map.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(x)
            
            # Loss
            loss, loss_metrics = criterion(predictions, y, mask_map=mask_map)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            metrics = metrics_calc.compute_all(predictions, y, mask_map)
            
            epoch_loss += loss.item()
            epoch_dice += metrics['dice']
            epoch_iou += metrics['iou']
            n_batches += 1
        
        # Средние значения за эпоху
        avg_loss = epoch_loss / n_batches
        avg_dice = epoch_dice / n_batches
        avg_iou = epoch_iou / n_batches
        
        history['loss'].append(avg_loss)
        history['dice'].append(avg_dice)
        history['iou'].append(avg_iou)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}, IoU={avg_iou:.4f}")
            
            # Визуализация
            with torch.no_grad():
                batch_example = next(iter(train_loader))
                x_example = batch_example['x'].to(device)
                preds_example = model(x_example)
                visualize_training_results(
                    batch_example, 
                    preds_example, 
                    epoch+1, 
                    save_path=save_path,
                    n_samples=2
                )
    
    # Сохраняем историю
    with open(os.path.join(save_path, 'overfit_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # График истории
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['dice'])
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['iou'])
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'overfit_history.png'), dpi=150)
    plt.close()
    
    print(f"\nOverfit check completed. Results saved to {save_path}")
    print(f"Final metrics: Loss={history['loss'][-1]:.4f}, Dice={history['dice'][-1]:.4f}, IoU={history['iou'][-1]:.4f}")
