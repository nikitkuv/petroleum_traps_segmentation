import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import wandb

from settings import settings


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
    save_path: str = settings.LOGS_TRAIN_VIZ_DIR,
    dataset=None,
    n_samples: int = 4
) -> None:
    """
    Визуализирует результаты обучения: оригинальную RGB карту, y_traps,
    результат модели и карту ошибок.

    Args:
        batch: Батч данных
        predictions: Предсказания модели
        epoch: Номер эпохи
        dataset: Объект GeologyTrapsDataset для получения имен семплов
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
        # Получаем имя семпла из датасета если доступен
        if dataset is not None and hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
            sample_idx = batch['sample_idx'][i].item() if 'sample_idx' in batch else i
            # samples - это список словарей путей, ключи имеют формат {number}_{name}
            # Нам нужно получить этот ключ. Поскольку samples[sample_idx] - это dict с путями,
            # мы можем извлечь номер и имя из любого пути в этом словаре
            sample_paths = dataset.samples[sample_idx]
            # Берем первый ключ (например, 'rgb') и извлекаем имя из пути
            first_path = list(sample_paths.values())[0]
            # Извлекаем basename без расширения
            from pathlib import Path
            filename = Path(first_path).stem
            # Паттерн: {number}_{x|y}_{type}_{name}, извлекаем number и name
            import re
            match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
            if match:
                number = match.group(1)
                name = match.group(2)
                sample_name = f"{number}_{name}"
            else:
                sample_name = f"sample_{sample_idx}"
        else:
            sample_name = f"Sample {i}"

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
        axes[i, 0].set_title(f'RGB Map (No Isolines)\n{sample_name}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_traps, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth Traps\n({sample_name})')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_traps, cmap='gray')
        axes[i, 2].set_title(f'Predicted Traps\n({sample_name})')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Prediction Overlay (Gray)\n({sample_name})')
        axes[i, 3].axis('off')

        # Визуализация карты ошибок с colormap
        im_error = axes[i, 4].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[i, 4].set_title(f'Error Map (|GT - Pred)|\n({sample_name})')
        axes[i, 4].axis('off')
        plt.colorbar(im_error, ax=axes[i, 4], fraction=0.046, pad=0.04)

    plt.tight_layout()

    name_of_image = "validation" if save_path == settings.LOGS_VAL_VIZ_DIR else "training"
    filename = f'epoch_{epoch:03d}_{name_of_image}_visualization.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization to {filepath}")

    # Логируем в wandb если активен
    if wandb.run is not None:
        wandb.log({
            f'{name_of_image}_visualization': wandb.Image(filepath),
            'epoch': epoch
        })


def visualize_test_results(
    batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    sample_indices: List[int],
    dataset=None,
    save_path: str = settings.LOGS_TEST_VIZ_DIR,
    alpha: float = 0.4
) -> None:
    """
    Визуализирует результаты на тестовых данных: RGB карту, ground truth,
    предсказание модели и карту ошибок (с учетом mask_map).

    Args:
        batch: Батч данных
        predictions: Предсказания модели
        sample_indices: Индексы семплов для визуализации
        dataset: Объект GeologyTrapsDataset для получения имен семплов
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

        # Получаем имя семпла из датасета если доступен
        if dataset is not None and hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
            # samples - это список словарей путей, ключи имеют формат {number}_{name}
            sample_paths = dataset.samples[idx]
            # Берем первый ключ (например, 'rgb') и извлекаем имя из пути
            first_path = list(sample_paths.values())[0]
            # Извлекаем basename без расширения
            from pathlib import Path
            filename = Path(first_path).stem
            # Паттерн: {number}_{x|y}_{type}_{name}, извлекаем number и name
            import re
            match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
            if match:
                number = match.group(1)
                name = match.group(2)
                sample_name = f"{number}_{name}"
            else:
                sample_name = f"sample_{idx}"
        else:
            sample_name = f"Test Sample {idx}"

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
        axes[0].set_title(f'RGB Map (No Isolines)\n{sample_name}')
        axes[0].axis('off')

        axes[1].imshow(gt_traps, cmap='gray')
        axes[1].set_title(f'Ground Truth Traps\n({sample_name})')
        axes[1].axis('off')

        axes[2].imshow(pred_traps, cmap='gray')
        axes[2].set_title(f'Predicted Traps\n({sample_name})')
        axes[2].axis('off')

        axes[3].imshow(overlay)
        axes[3].set_title(f'Prediction Overlay (Gray)\n({sample_name})')
        axes[3].axis('off')

        im_error = axes[4].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[4].set_title(f'Error Map (|GT - Pred)|\n({sample_name})')
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


def visualize_comparison_grid(
    batches: List[Dict[str, torch.Tensor]],
    all_predictions: List[torch.Tensor],
    epochs: List[int],
    dataset=None,
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
        dataset: Объект GeologyTrapsDataset для получения имен семплов
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
        # Получаем имя семпла из датасета если доступен
        if dataset is not None and hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
            sample_idx = batches[0]['sample_idx'][i].item() if 'sample_idx' in batches[0] else i
            # samples - это список словарей путей, ключи имеют формат {number}_{name}
            sample_paths = dataset.samples[sample_idx]
            # Берем первый ключ (например, 'rgb') и извлекаем имя из пути
            first_path = list(sample_paths.values())[0]
            # Извлекаем basename без расширения
            from pathlib import Path
            filename = Path(first_path).stem
            # Паттерн: {number}_{x|y}_{type}_{name}, извлекаем number и name
            import re
            match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
            if match:
                number = match.group(1)
                name = match.group(2)
                sample_name = f"{number}_{name}"
            else:
                sample_name = f"sample_{sample_idx}"
        else:
            sample_name = f"Sample {i}"

        # Берем первый батч для RGB и GT (предполагаем одинаковые данные)
        x_rgb = batches[0]['x'][:, :3, :, :]
        y_traps = batches[0]['y']

        rgb_img = x_rgb[i].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        gt_traps = y_traps[i, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[i].cpu().numpy()

        # RGB
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f'RGB Map\n{sample_name}')
        axes[i, 0].axis('off')

        # GT
        axes[i, 1].imshow(gt_traps, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth\n({sample_name})')
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

            axes[i, j + 2].set_title(f'Epoch {epoch}\nIoU={iou:.3f}\n({sample_name})')
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
        