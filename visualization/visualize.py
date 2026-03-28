"""
Модуль для визуализации результатов обучения и тестирования.
Визуализация: RGB карта, y_traps, prediction, overlay.
"""

import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import wandb


def visualize_training_results(
    batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    epoch: int,
    save_path: str = './logs/visualizations/',
    n_samples: int = 4
) -> None:
    """
    Визуализирует результаты обучения: оригинальную RGB карту, y_traps, 
    результат модели и наложение результата на RGB.
    
    Args:
        batch: Батч данных
        predictions: Предсказания модели
        epoch: Номер эпохи
        save_path: Путь для сохранения
        n_samples: Количество семплов для визуализации
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Извлекаем данные из батча
    x_rgb = batch['x'][:, :3, :, :]  # Первые 3 канала - RGB
    y_traps = batch['y']
    mask_map = batch.get('mask_map', None)
    
    # Применяем сигмоиду к предсказаниям
    preds_prob = torch.sigmoid(predictions)
    
    n_samples = min(n_samples, x_rgb.shape[0])
    
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Оригинальная RGB карта (без изолиний)
        rgb_img = x_rgb[i].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # Ground truth traps
        gt_traps = y_traps[i, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[i].cpu().numpy()
        
        # Предсказание модели
        pred_traps = preds_prob[i, 0, :, :].cpu().detach().numpy()
        
        # Наложение предсказания на RGB
        overlay = rgb_img.copy()
        overlay_pred = np.zeros_like(overlay)
        overlay_pred[:, :, 0] = pred_traps  # Красный канал для предсказаний
        overlay = cv2.addWeighted(overlay, 0.7, overlay_pred, 0.3, 0)
        
        # Отображаем
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f'RGB Map (No Isolines)\nSample {i}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_traps, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth Traps\n(Sample {i})')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_traps, cmap='gray')
        axes[i, 2].set_title(f'Predicted Traps\n(Sample {i})')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Prediction Overlay on RGB\n(Sample {i})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    filename = f'epoch_{epoch:03d}_training_visualization.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {filepath}")
    
    # Логируем в wandb если активен
    if wandb.run is not None:
        wandb.log({
            'training_visualization': wandb.Image(filepath),
            'epoch': epoch
        })


def visualize_test_results(
    batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    sample_indices: List[int],
    save_path: str = './logs/test_visualizations/',
    alpha: float = 0.4
) -> None:
    """
    Визуализирует результаты на тестовых данных: наложение результата модели
    (карта ловушек) в прозрачности на RGB карту без изолиний.
    
    Args:
        batch: Батч данных
        predictions: Предсказания модели
        sample_indices: Индексы семплов для визуализации
        save_path: Путь для сохранения
        alpha: Прозрачность наложения
    """
    os.makedirs(save_path, exist_ok=True)
    
    x_rgb = batch['x'][:, :3, :, :]
    y_traps = batch['y']
    preds_prob = torch.sigmoid(predictions)
    
    for idx in sample_indices:
        if idx >= x_rgb.shape[0]:
            continue
        
        rgb_img = x_rgb[idx].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)
        
        gt_traps = y_traps[idx, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[idx].cpu().numpy()
        pred_traps = preds_prob[idx, 0, :, :].cpu().detach().numpy()
        
        # Создаем overlay с предсказанием
        overlay = rgb_img.copy()
        heatmap = cv2.applyColorMap((pred_traps * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = cv2.addWeighted(overlay, 1.0 - alpha, heatmap, alpha, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(rgb_img)
        axes[0].set_title(f'RGB Map (No Isolines)\nTest Sample {idx}')
        axes[0].axis('off')
        
        axes[1].imshow(gt_traps, cmap='gray')
        axes[1].set_title(f'Ground Truth Traps')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Predicted Traps Overlay (alpha={alpha})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        filename = f'test_sample_{idx:03d}_results.png'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved test visualization to {filepath}")
        
        if wandb.run is not None:
            wandb.log({
                f'test_visualization_sample_{idx}': wandb.Image(filepath)
            })
