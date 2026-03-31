import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import wandb


def create_prediction_overlay(rgb_img: np.ndarray, pred_traps: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Создает overlay предсказания на RGB изображение с прозрачностью.
    Черные области (где нет предсказания) не отображаются.
    Используется серый цвет для наложения.

    Args:
        rgb_img: RGB изображение (H, W, 3), значения [0, 1]
        pred_traps: Карта предсказаний (H, W), значения [0, 1]
        alpha: Прозрачность наложения предсказания

    Returns:
        overlay: Изображение с наложенным предсказанием
    """
    overlay = rgb_img.copy()

    # Создаем серую маску для областей с предсказаниями
    # Серый цвет: одинаковые значения по всем каналам
    gray_overlay = np.stack([pred_traps] * 3, axis=-1)

    # Создаем маску для областей с ненулевыми предсказаниями
    mask = pred_traps > 0.01  # Порог для отсечения фона

    # Применяем наложение только там, где есть предсказания
    if mask.any():
        overlay[mask] = cv2.addWeighted(
            rgb_img[mask],
            1.0 - alpha,
            gray_overlay[mask],
            alpha,
            0
        )

    return overlay


def create_error_map(gt_traps: np.ndarray, pred_traps: np.ndarray, map_mask: np.ndarray = None) -> np.ndarray:
    """
    Создает карту ошибок как абсолютную разницу между ground truth и предсказанием.

    Args:
        gt_traps: Ground truth карта (H, W), значения [0, 1]
        pred_traps: Предсказание модели (H, W), значения [0, 1]
        map_mask: Маска карты (H, W), булева или 0/1. Если None, считается по всей области.

    Returns:
        error_map: Карта абсолютных ошибок (H, W). За пределами mask значения = 0.
    """
    error_map = np.abs(gt_traps - pred_traps)

    if map_mask is not None:
        # Убираем ошибку за пределами карты
        mask_bool = map_mask.astype(bool)
        error_map[~mask_bool] = 0

    return error_map


def visualize_training_results(
    batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    epoch: int,
    save_path: str = './logs/visualizations/',
    n_samples: int = 4
) -> None:
    """
    Визуализирует результаты обучения: оригинальную RGB карту, y_traps,
    результат модели и карту ошибок.

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

    # 5 колонок: RGB, GT, Pred, Overlay, Error Map
    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 5 * n_samples))
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

        # Prediction overlay с серым цветом
        overlay = create_prediction_overlay(rgb_img, pred_traps, alpha=0.4)

        # Карта ошибок (с учетом mask_map если есть)
        map_mask_np = None
        if mask_map is not None:
            map_mask_np = mask_map[i, 0, :, :].cpu().numpy() if mask_map.dim() == 4 else mask_map[i].cpu().numpy()
        error_map = create_error_map(gt_traps, pred_traps, map_mask=map_mask_np)

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
        axes[i, 3].set_title(f'Prediction Overlay (Gray)\n(Sample {i})')
        axes[i, 3].axis('off')

        # Визуализация карты ошибок с colormap
        im_error = axes[i, 4].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[i, 4].set_title(f'Error Map (|GT - Pred)|\n(Sample {i})')
        axes[i, 4].axis('off')
        plt.colorbar(im_error, ax=axes[i, 4], fraction=0.046, pad=0.04)

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
    Визуализирует результаты на тестовых данных: RGB карту, ground truth,
    предсказание модели и карту ошибок (с учетом mask_map).

    Args:
        batch: Батч данных
        predictions: Предсказания модели
        sample_indices: Индексы семплов для визуализации
        save_path: Путь для сохранения
        alpha: Прозрачность наложения (не используется, оставлен для совместимости)
    """
    os.makedirs(save_path, exist_ok=True)

    x_rgb = batch['x'][:, :3, :, :]
    y_traps = batch['y']
    mask_map = batch.get('mask_map', None)
    preds_prob = torch.sigmoid(predictions)

    for idx in sample_indices:
        if idx >= x_rgb.shape[0]:
            continue

        rgb_img = x_rgb[idx].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        gt_traps = y_traps[idx, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[idx].cpu().numpy()
        pred_traps = preds_prob[idx, 0, :, :].cpu().detach().numpy()

        # Карта ошибок (с учетом mask_map если есть)
        map_mask_np = None
        if mask_map is not None:
            map_mask_np = mask_map[idx, 0, :, :].cpu().numpy() if mask_map.dim() == 4 else mask_map[idx].cpu().numpy()
        error_map = create_error_map(gt_traps, pred_traps, map_mask=map_mask_np)

        # Prediction overlay с серым цветом
        overlay = create_prediction_overlay(rgb_img, pred_traps, alpha=0.4)

        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        axes[0].imshow(rgb_img)
        axes[0].set_title(f'RGB Map (No Isolines)\nTest Sample {idx}')
        axes[0].axis('off')

        axes[1].imshow(gt_traps, cmap='gray')
        axes[1].set_title(f'Ground Truth Traps')
        axes[1].axis('off')

        axes[2].imshow(pred_traps, cmap='gray')
        axes[2].set_title(f'Predicted Traps')
        axes[2].axis('off')

        axes[3].imshow(overlay)
        axes[3].set_title(f'Prediction Overlay (Gray)')
        axes[3].axis('off')

        im_error = axes[4].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[4].set_title(f'Error Map (|GT - Pred)|')
        axes[4].axis('off')
        plt.colorbar(im_error, ax=axes[4], fraction=0.046, pad=0.04)

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


def visualize_advanced_metrics(
    batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    epoch: int,
    save_path: str = './logs/advanced_visualizations/',
    n_samples: int = 4,
    threshold: float = 0.5
) -> None:
    """
    Расширенная визуализация с дополнительными метриками качества:
    - IoU карта (пересечение над объединением для каждого пикселя)
    - Precision-Recall кривые для разных порогов
    - Гистограмма распределения ошибок
    - Confusion matrix visualization

    Args:
        batch: Батч данных
        predictions: Предсказания модели
        epoch: Номер эпохи
        save_path: Путь для сохранения
        n_samples: Количество семплов для визуализации
        threshold: Порог бинаризации предсказаний
    """
    os.makedirs(save_path, exist_ok=True)

    x_rgb = batch['x'][:, :3, :, :]
    y_traps = batch['y']
    preds_prob = torch.sigmoid(predictions)

    n_samples = min(n_samples, x_rgb.shape[0])

    # 6 колонок: RGB, GT, Pred, Comparison, Error Map, Distribution
    fig, axes = plt.subplots(n_samples, 6, figsize=(30, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    all_ious = []
    all_dice = []

    for i in range(n_samples):
        rgb_img = x_rgb[i].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        gt_traps = y_traps[i, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[i].cpu().numpy()
        pred_traps = preds_prob[i, 0, :, :].cpu().detach().numpy()

        # Бинаризация предсказаний
        pred_binary = (pred_traps >= threshold).astype(np.float32)
        gt_binary = (gt_traps >= threshold).astype(np.float32)

        # IoU для каждого пикселя (локальный IoU в скользящем окне)
        intersection = gt_binary * pred_binary
        union = np.maximum(gt_binary, pred_binary)

        # Глобальные метрики
        tp = np.sum((gt_binary == 1) & (pred_binary == 1))
        fp = np.sum((gt_binary == 0) & (pred_binary == 1))
        fn = np.sum((gt_binary == 1) & (pred_binary == 0))
        tn = np.sum((gt_binary == 0) & (pred_binary == 0))

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        all_ious.append(iou)
        all_dice.append(dice)

        # 1. RGB изображение
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f'RGB Map\nSample {i}')
        axes[i, 0].axis('off')

        # 2. Ground Truth
        axes[i, 1].imshow(gt_traps, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth\n(IoU={iou:.3f})')
        axes[i, 1].axis('off')

        # 3. Prediction (probability)
        axes[i, 2].imshow(pred_traps, cmap='gray')
        axes[i, 2].set_title(f'Prediction (prob)\n(Dice={dice:.3f})')
        axes[i, 2].axis('off')

        # 4. Binary prediction vs GT overlay
        comparison = np.zeros((*gt_binary.shape, 3))
        comparison[:, :, 0] = gt_binary  # Red - GT
        comparison[:, :, 2] = pred_binary  # Blue - Pred
        # Purple areas show overlap
        axes[i, 3].imshow(comparison)
        axes[i, 3].set_title(f'GT(Red) vs Pred(Blue)\nOverlap=Purple')
        axes[i, 3].axis('off')

        # 5. Error map с градиентом (с учетом mask_map если есть)
        map_mask_np = None
        if 'mask_map' in batch and batch['mask_map'] is not None:
            map_mask_np = batch['mask_map'][i, 0, :, :].cpu().numpy() if batch['mask_map'].dim() == 4 else batch['mask_map'][i].cpu().numpy()
        error_map = create_error_map(gt_traps, pred_traps, map_mask=map_mask_np)
        im_error = axes[i, 4].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[i, 4].set_title(f'Error Map\n(MAE={np.mean(error_map):.3f})')
        axes[i, 4].axis('off')
        plt.colorbar(im_error, ax=axes[i, 4], fraction=0.046, pad=0.04)

        # 6. Распределение вероятностей предсказаний
        axes[i, 5].hist(pred_traps.flatten(), bins=50, alpha=0.7,
                       color='blue', label='Predictions', density=True)
        axes[i, 5].hist(gt_traps.flatten(), bins=50, alpha=0.7,
                       color='green', label='Ground Truth', density=True)
        axes[i, 5].axvline(x=threshold, color='red', linestyle='--',
                          label=f'Threshold={threshold}')
        axes[i, 5].set_xlabel('Probability')
        axes[i, 5].set_ylabel('Density')
        axes[i, 5].set_title(f'Distribution\nTP={tp}, FP={fp}, FN={fn}')
        axes[i, 5].legend(fontsize=8)
        axes[i, 5].grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f'epoch_{epoch:03d}_advanced_metrics.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved advanced visualization to {filepath}")
    print(f"Average IoU: {np.mean(all_ious):.4f}, Average Dice: {np.mean(all_dice):.4f}")

    if wandb.run is not None:
        wandb.log({
            'advanced_visualization': wandb.Image(filepath),
            'avg_iou': np.mean(all_ious),
            'avg_dice': np.mean(all_dice),
            'epoch': epoch
        })


def visualize_comparison_grid(
    batches: List[Dict[str, torch.Tensor]],
    all_predictions: List[torch.Tensor],
    epochs: List[int],
    save_path: str = './logs/comparison_grid/',
    n_samples: int = 2
) -> None:
    """
    Визуализирует сравнение предсказаний модели на разных эпохах обучения.
    Полезно для отслеживания прогресса обучения.

    Args:
        batches: Список батчей с разных эпох
        all_predictions: Список предсказаний с разных эпох
        epochs: Список номеров эпох
        save_path: Путь для сохранения
        n_samples: Количество семплов для визуализации
    """
    os.makedirs(save_path, exist_ok=True)

    n_epochs = len(epochs)
    n_cols = n_epochs + 2  # RGB, GT, + predictions для каждой эпохи

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(5 * n_cols, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Берем первый батч для RGB и GT (предполагаем одинаковые данные)
        x_rgb = batches[0]['x'][:, :3, :, :]
        y_traps = batches[0]['y']

        rgb_img = x_rgb[i].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        gt_traps = y_traps[i, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[i].cpu().numpy()

        # RGB
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f'RGB Map\nSample {i}')
        axes[i, 0].axis('off')

        # GT
        axes[i, 1].imshow(gt_traps, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth')
        axes[i, 1].axis('off')

        # Predictions для каждой эпохи
        for j, (pred, epoch) in enumerate(zip(all_predictions, epochs)):
            preds_prob = torch.sigmoid(pred)
            pred_traps = preds_prob[i, 0, :, :].cpu().detach().numpy()

            axes[i, j + 2].imshow(pred_traps, cmap='gray')

            # Вычисляем IoU для этой эпохи
            pred_binary = (pred_traps >= 0.5).astype(np.float32)
            gt_binary = (gt_traps >= 0.5).astype(np.float32)
            tp = np.sum((gt_binary == 1) & (pred_binary == 1))
            fp = np.sum((gt_binary == 0) & (pred_binary == 1))
            fn = np.sum((gt_binary == 1) & (pred_binary == 0))
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

            axes[i, j + 2].set_title(f'Epoch {epoch}\nIoU={iou:.3f}')
            axes[i, j + 2].axis('off')

    plt.tight_layout()

    filename = f'comparison_epochs_{"_".join(map(str, epochs))}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison grid to {filepath}")

    if wandb.run is not None:
        wandb.log({
            'comparison_grid': wandb.Image(filepath),
            'epochs_compared': epochs
        })
        