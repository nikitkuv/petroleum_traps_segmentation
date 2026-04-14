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
    Используется инвертированная карта ловушек (черные ловушки на прозрачном фоне),
    которые при наложении становятся темно-серыми на RGB изображении.

    Args:
        rgb_img: RGB изображение (H, W, 3), значения [0, 1]
        pred_traps: Карта предсказаний (H, W), значения [0, 1]
        alpha: Прозрачность наложения предсказания

    Returns:
        overlay: Изображение с наложенным предсказанием (ловушки отображаются темно-серым)
    """
    overlay = rgb_img.copy()

    # Инвертируем карту ловушек: ловушки становятся черными (0), фон белым (1)
    # Затем используем это как маску для создания темного overlay
    inverted_pred = 1.0 - pred_traps

    # Создаем темную маску для областей с предсказаниями
    # Используем инвертированные значения: где были ловушки (близко к 1), теперь близко к 0 (темный)
    dark_overlay = np.stack([inverted_pred] * 3, axis=-1)

    # Создаем маску для областей с ненулевыми предсказаниями
    mask = pred_traps > 0.01  # Порог для отсечения фона

    # Применяем наложение только там, где есть предсказания
    if mask.any():
        overlay[mask] = cv2.addWeighted(
            rgb_img[mask],
            1.0 - alpha,
            dark_overlay[mask],
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
    Визуализирует результаты обучения: оригинальную RGB карту, карту изолиний, y_traps,
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
    x_isolines = batch['x'][:, 4:5, :, :] if batch['x'].shape[1] >= 5 else None  # Канал 4 - изолинии (для cps_tiles)
    y_traps = batch['y']
    mask_map = batch.get('mask_map', None)
    data_source = batch.get('data_source', 'png')

    # Применяем сигмоиду к предсказаниям
    preds_prob = torch.sigmoid(predictions)

    n_samples = min(n_samples, x_rgb.shape[0])

    # 6 колонок: RGB, Isolines, GT, Pred, Overlay, Error Map
    fig, axes = plt.subplots(n_samples, 6, figsize=(24, 5 * n_samples))
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

        # Карта изолиний (если доступна)
        isolines_img = None
        if x_isolines is not None:
            isolines_img = x_isolines[i, 0, :, :].cpu().numpy()

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
        axes[i, 0].set_title(f'RGB Input\n{sample_name}')
        axes[i, 0].axis('off')

        # Карта изолиний
        if isolines_img is not None:
            axes[i, 1].imshow(isolines_img, cmap='gray')
            axes[i, 1].set_title(f'Isolines\n({sample_name})')
        else:
            axes[i, 1].text(0.5, 0.5, 'No isolines', ha='center', va='center', transform=axes[i, 1].transAxes)
            axes[i, 1].set_title(f'Isolines\n({sample_name})')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(gt_traps, cmap='gray')
        axes[i, 2].set_title(f'Ground Truth Traps\n({sample_name})')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(pred_traps, cmap='gray')
        axes[i, 3].set_title(f'Predicted Traps\n({sample_name})')
        axes[i, 3].axis('off')

        axes[i, 4].imshow(overlay)
        axes[i, 4].set_title(f'Prediction Overlay (Inverted Traps)\n({sample_name})')
        axes[i, 4].axis('off')

        # Визуализация карты ошибок с colormap
        im_error = axes[i, 5].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[i, 5].set_title(f'Error Map (|GT - Pred)|\n({sample_name})')
        axes[i, 5].axis('off')
        plt.colorbar(im_error, ax=axes[i, 5], fraction=0.046, pad=0.04)

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
    alpha: float = 0.4,
    metrics_by_sample: Dict = None
) -> None:
    """
    Визуализирует результаты на тестовых данных: RGB карту, карту изолиний, ground truth,
    предсказание модели и карту ошибок (с учетом mask_map).

    Args:
        batch: Батч данных
        predictions: Предсказания модели
        sample_indices: Индексы семплов для визуализации
        dataset: Объект GeologyTrapsDataset для получения имен семплов
        save_path: Путь для сохранения
        alpha: Прозрачность наложения (не используется, оставлен для совместимости)
        metrics_by_sample: Словарь с метриками по имени семпла для отображения в title
    """
    os.makedirs(save_path, exist_ok=True)

    x_rgb = batch['x'][:, :3, :, :]
    x_isolines = batch['x'][:, 4:5, :, :] if batch['x'].shape[1] >= 5 else None
    y_traps = batch['y']
    mask_map = batch.get('mask_map', None)
    preds_prob = torch.sigmoid(predictions)

    # Проходим по каждому индексу в батче (от 0 до batch_size-1)
    for batch_idx in range(x_rgb.shape[0]):
        # Получаем реальный индекс семпла из переданного списка
        real_idx = sample_indices[batch_idx]

        # Получаем имя семпла из датасета используя реальный индекс
        if dataset is not None and hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
            sample_paths = dataset.samples[real_idx]
            first_path = list(sample_paths.values())[0]
            from pathlib import Path
            filename = Path(first_path).stem
            import re
            match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
            if match:
                number = match.group(1)
                name = match.group(2)
                sample_name = f"{number}_{name}"
            else:
                sample_name = f"sample_{real_idx}"
        else:
            sample_name = f"Test Sample {real_idx}"

        rgb_img = x_rgb[batch_idx].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        # Карта изолиний (если доступна)
        isolines_img = None
        if x_isolines is not None:
            isolines_img = x_isolines[batch_idx, 0, :, :].cpu().numpy()

        gt_traps = y_traps[batch_idx, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[batch_idx].cpu().numpy()
        pred_traps = preds_prob[batch_idx, 0, :, :].cpu().detach().numpy()

        # Получаем метрики для этого семпла если доступны
        metrics = None
        if metrics_by_sample is not None:
            metrics = metrics_by_sample.get(sample_name, None)

        # Карта ошибок (с учетом mask_map если есть)
        map_mask_np = None
        if mask_map is not None:
            map_mask_np = mask_map[batch_idx, 0, :, :].cpu().numpy() if mask_map.dim() == 4 else mask_map[batch_idx].cpu().numpy()
        error_map = create_error_map(gt_traps, pred_traps, map_mask=map_mask_np)

        # Prediction overlay с серым цветом
        overlay = create_prediction_overlay(rgb_img, pred_traps, alpha=0.4)

        fig, axes = plt.subplots(1, 6, figsize=(24, 5))

        # Формируем title с метриками если они есть
        if metrics is not None:
            rgb_title = f"RGB Input\n{sample_name}\nDice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}"
            isolines_title = f"Isolines\nDice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}"
            gt_title = f"Ground Truth Traps\nDice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}"
            pred_title = f"Predicted Traps\nDice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}"
            overlay_title = f"Prediction Overlay\nDice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}"
            error_title = f"Error Map (|GT - Pred)|\nDice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}"
        else:
            rgb_title = f'RGB Input\n{sample_name}'
            isolines_title = f'Isolines\n({sample_name})'
            gt_title = f'Ground Truth Traps\n({sample_name})'
            pred_title = f'Predicted Traps\n({sample_name})'
            overlay_title = f'Prediction Overlay (Inverted Traps)\n({sample_name})'
            error_title = f'Error Map (|GT - Pred)|\n({sample_name})'

        axes[0].imshow(rgb_img)
        axes[0].set_title(rgb_title)
        axes[0].axis('off')

        # Карта изолиний
        if isolines_img is not None:
            axes[1].imshow(isolines_img, cmap='gray')
            axes[1].set_title(isolines_title)
        else:
            axes[1].text(0.5, 0.5, 'No isolines', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(isolines_title)
        axes[1].axis('off')

        axes[2].imshow(gt_traps, cmap='gray')
        axes[2].set_title(gt_title)
        axes[2].axis('off')

        axes[3].imshow(pred_traps, cmap='gray')
        axes[3].set_title(pred_title)
        axes[3].axis('off')

        axes[4].imshow(overlay)
        axes[4].set_title(overlay_title)
        axes[4].axis('off')

        im_error = axes[5].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[5].set_title(error_title)
        axes[5].axis('off')
        plt.colorbar(im_error, ax=axes[5], fraction=0.046, pad=0.04)

        plt.tight_layout()

        filename = f'{sample_name}_results.png'
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved test visualization to {filepath}")

        if wandb.run is not None:
            wandb.log({
                f'test_visualization_{sample_name}': wandb.Image(filepath)
            })
