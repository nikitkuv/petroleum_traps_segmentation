"""
Модуль для оценки модели на тестовых данных.
Проверка дообученной модели и визуализация результатов.
"""

import os
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.metrics import MetricsCalculator
from visualization.visualize import visualize_test_results
from settings import settings


def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    criterion,  # CombinedLoss
    device: str = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Проверка дообученной модели на тестовых данных.
    
    Args:
        model: Модель
        test_loader: Тестовый DataLoader
        criterion: Функция потерь
        device: Устройство
        threshold: Порог бинаризации
    
    Returns:
        Словарь с метриками
    """
    device = device or settings.DEVICE
    model.eval()
    
    metrics_calc = MetricsCalculator(threshold=threshold)
    
    all_predictions = []
    all_targets = []
    all_masks = []
    
    total_loss = 0.0
    n_batches = 0
    
    print("=" * 60)
    print("EVALUATING ON TEST DATA")
    print("=" * 60)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask_map = batch.get('mask_map', None)
            if mask_map is not None:
                mask_map = mask_map.to(device)
            
            predictions = model(x)
            
            # Loss
            loss, _ = criterion(predictions, y, mask_map=mask_map)
            total_loss += loss.item()
            
            # Metrics
            metrics = metrics_calc.compute_all(predictions, y, mask_map)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())
            if mask_map is not None:
                all_masks.append(mask_map.cpu())
            
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{metrics["dice"]:.4f}',
                'iou': f'{metrics["iou"]:.4f}'
            })
    
    # Агрегируем метрики по всем батчам
    avg_metrics = {
        'test_loss': total_loss / n_batches,
        'test_dice': 0.0,
        'test_iou': 0.0,
        'test_recall': 0.0,
        'test_precision': 0.0,
        'test_f1': 0.0,
        'test_fp_area': 0.0,
        'test_fn_area': 0.0
    }
    
    # Пересчитываем метрики на всех данных сразу для большей точности
    all_preds = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0) if all_masks else None
    
    final_metrics = metrics_calc.compute_all(all_preds, all_targets, all_masks)
    
    avg_metrics.update(final_metrics)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Loss:    {avg_metrics['test_loss']:.4f}")
    print(f"Dice:    {avg_metrics['test_dice']:.4f}")
    print(f"IoU:     {avg_metrics['test_iou']:.4f}")
    print(f"Recall:  {avg_metrics['test_recall']:.4f}")
    print(f"Precision: {avg_metrics['test_precision']:.4f}")
    print(f"F1:      {avg_metrics['test_f1']:.4f}")
    print(f"FP Area: {avg_metrics['test_fp_area']:.4f}")
    print(f"FN Area: {avg_metrics['test_fn_area']:.4f}")
    print("=" * 60)
    
    return avg_metrics


def visualize_test_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = None,
    sample_indices: List[int] = None,
    save_path: str = './logs/test_visualizations/',
    alpha: float = 0.4
) -> None:
    """
    Визуализация выборочных результатов тестовых данных:
    наложение результата модели (карта ловушек) в прозрачности на RGB карту.
    
    Args:
        model: Модель
        test_loader: Тестовый DataLoader
        device: Устройство
        sample_indices: Индексы семплов для визуализации
        save_path: Путь для сохранения
        alpha: Прозрачность наложения
    """
    device = device or settings.DEVICE
    model.eval()
    
    print("\nVisualizing test predictions...")
    
    with torch.no_grad():
        # Берем несколько случайных батчей для визуализации
        if sample_indices is None:
            sample_indices = [0, 1, 2, 3]
        
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 1:  # Достаточно одного батча
                break
            
            x = batch['x'].to(device)
            predictions = model(x)
            
            visualize_test_results(
                batch=batch,
                predictions=predictions,
                sample_indices=sample_indices,
                save_path=save_path,
                alpha=alpha
            )
    
    print(f"Test visualizations saved to {save_path}")
