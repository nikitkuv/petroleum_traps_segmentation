import os
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import wandb
from pathlib import Path

from settings import settings
from utils.cps_utils import (
    read_cps_grid, 
    cps_to_rgb, 
    cps_to_isolines,
    cps_to_closed_mask,
    cps_to_binary_mask
)


def overlay_isolines_on_rgb_from_cps(
    cps_path: str,
    isoline_step: float = 5.0,
    cmap_name: str = 'purple_jet',
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует RGB карту и изолинии из CPS грида и накладывает их друг на друга.

    Args:
        cps_path: Путь к CPS файлу (например, x_structuralNOisoline_*)
        isoline_step: Шаг изолиний в метрах (по умолчанию 5.0)
        cmap_name: Название цветовой палитры для RGB карты (по умолчанию 'purple_jet')
        alpha: Прозрачность наложения изолиний (0.0 - полностью прозрачные, 1.0 - полностью видимые)
    """
    # 1. Загружаем CPS грид
    grid, _ = read_cps_grid(cps_path)
    
    # 2. Генерируем RGB и изолинии
    rgb_img = cps_to_rgb(grid, cmap_name=cmap_name)
    isolines_img = cps_to_isolines(grid, step=isoline_step)

    # 43. Нормализуем RGB к [0, 1]
    rgb_float = rgb_img.astype(np.float32) / 255.0

    # 4. Инвертируем изолинии: черный фон (0) → белый (1), белые линии (255) → черные (0)
    isolines_inverted = 1.0 - (isolines_img.astype(np.float32) / 255.0)

    # Создаем маску для линий (где изолинии черные после инверсии, т.е. близки к 0)
    # isolines_inverted: 1.0 = фон, 0.0 = линии
    line_mask = 1.0 - isolines_inverted  # Теперь: 1.0 = линии, 0.0 = фон

    # 5. Накладываем изолинии на RGB (затемняем области с линиями)
    overlay = rgb_float.copy()
    # Используем alpha для контроля интенсивности затемнения (0.7 - множитель затемнения)
    overlay = overlay * (1.0 - line_mask[:, :, np.newaxis] * alpha * 0.7)

    # Ограничиваем значения к [0, 1]
    overlay = np.clip(overlay, 0, 1)

    # 6. Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RGB изображение
    axes[0].imshow(rgb_float)
    axes[0].set_title('RGB Map (from CPS)')
    axes[0].axis('off')

    # Инвертированные изолинии
    axes[1].imshow(isolines_inverted, cmap='gray')
    axes[1].set_title('Isolines (Inverted)\n(white background, black lines)')
    axes[1].axis('off')

    # Результат наложения
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\n(alpha={alpha})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_closed_isolines(rgb_cps_paht: str, traps_cps_path: str, isoline_step: float):
    # Проверка существования файлов
    for path in [rgb_cps_paht, traps_cps_path]:
        if not Path(path).exists():
            print(f"Ошибка: Файл не найден - {path}")
            return

    print("Загрузка CPS grids...")
    # Загружаем структурный грид (карта глубин)
    structural_grid, _ = read_cps_grid(rgb_cps_paht)
    # Загружаем грид ловушек
    traps_grid, _ = read_cps_grid(traps_cps_path)

    print("Генерация изолиний...")
    # Генерируем изолинии из структурного грида
    isolines_img = cps_to_isolines(structural_grid, step=isoline_step)

    print("Генерация маски замкнутых изолиний...")
    # Генерируем маску замкнутых контуров
    closed_mask = cps_to_closed_mask(structural_grid, step=isoline_step)

    print("Генерация маски ловушек (GT)...")
    # Генерируем бинарную маску ловушек
    traps_mask = cps_to_binary_mask(traps_grid)

    # Визуализация 1x3 (1 ряд, 3 колонки)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Левая картинка: Ground Truth ловушки
    axes[0].imshow(traps_mask, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title("Ground Truth Traps (from y_traps)", fontsize=14)
    axes[0].axis('off')

    # Центральная картинка: Исходные изолинии
    axes[1].imshow(isolines_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Original Isolines (from structural)", fontsize=14)
    axes[1].axis('off')

    # Правая картинка: Замкнутые изолинии
    axes[2].imshow(closed_mask, cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].set_title("Closed Isolines Mask (from structural)", fontsize=14)
    axes[2].axis('off')

    plt.suptitle("Direct CPS Grid Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()


def create_rgb_isolines_overlay(rgb_img: np.ndarray, isolines_img: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Создает overlay изолиний на RGB изображение.
    Изолинии отображаются черными линиями на полупрозрачном фоне поверх RGB.

    Args:
        rgb_img: RGB изображение (H, W, 3), значения [0, 1]
        isolines_img: Карта изолиний (H, W), значения [0, 1] (1 - линия, 0 - фон)
        alpha: Интенсивность затемнения линий изолиний

    Returns:
        overlay: Изображение с наложенными изолиниями
    """
    overlay = rgb_img.copy()
    
    if isolines_img is not None:
        # Маска где есть линии (значение близко к 1)
        line_mask = isolines_img > 0.5
        
        # Затемняем пиксели RGB изображения там, где проходят линии
        overlay[line_mask] = overlay[line_mask] * (1.0 - alpha)
    
    return overlay


def create_prediction_overlay(rgb_img: np.ndarray, pred_traps: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Создает overlay предсказания на RGB изображение с прозрачностью.
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
    inverted_pred = 1.0 - pred_traps

    # Создаем темную маску для областей с предсказаниями
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
    Визуализирует результаты обучения.
    Колонки: RGB+Isolines, Closed Isolines, GT, Pred, Overlay, Error Map
    """
    os.makedirs(save_path, exist_ok=True)

    # Извлекаем данные из батча
    x_rgb = batch['x'][:, :3, :, :]  # Каналы 0-2: RGB
    x_isolines = batch['x'][:, 4:5, :, :] if batch['x'].shape[1] >= 5 else None  # Канал 4: изолинии
    x_closed_isolines = batch['x'][:, 5:6, :, :] if batch['x'].shape[1] >= 6 else None  # Канал 5: замкнутые изолинии
    y_traps = batch['y']
    mask_map = batch.get('mask_map', None)

    # Применяем сигмоиду к предсказаниям
    preds_prob = torch.sigmoid(predictions)

    n_samples = min(n_samples, x_rgb.shape[0])

    fig, axes = plt.subplots(n_samples, 6, figsize=(24, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Получаем имя семпла из датасета если доступен
        if dataset is not None and hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
            sample_idx = batch['sample_idx'][i].item() if 'sample_idx' in batch else i
            sample_paths = dataset.samples[sample_idx]
            first_path = list(sample_paths.values())[0]
            from pathlib import Path
            filename = Path(first_path).stem
            import re
            match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
            sample_name = f"{match.group(1)}_{match.group(2)}" if match else f"sample_{sample_idx}"
        else:
            sample_name = f"Sample {i}"

        # Оригинальная RGB карта
        rgb_img = x_rgb[i].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        # Карта изолиний
        isolines_img = x_isolines[i, 0, :, :].cpu().numpy() if x_isolines is not None else None
        
        # Замкнутые изолинии
        closed_isolines_img = x_closed_isolines[i, 0, :, :].cpu().numpy() if x_closed_isolines is not None else None

        # Ground truth traps
        gt_traps = y_traps[i, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[i].cpu().numpy()

        # Предсказание модели
        pred_traps = preds_prob[i, 0, :, :].cpu().detach().numpy()

        # Маска карты
        map_mask_np = None
        if mask_map is not None:
            map_mask_np = mask_map[i, 0, :, :].cpu().numpy() if mask_map.dim() == 4 else mask_map[i].cpu().numpy()

        # Применяем map_mask за пределами карты
        if map_mask_np is not None:
            pred_traps_masked = pred_traps * map_mask_np
        else:
            pred_traps_masked = pred_traps

        # Создаем оверлеи
        rgb_iso_overlay = create_rgb_isolines_overlay(rgb_img, isolines_img, alpha=0.6)
        pred_overlay = create_prediction_overlay(rgb_img, pred_traps_masked, alpha=0.4)
        error_map = create_error_map(gt_traps, pred_traps_masked, map_mask=map_mask_np)

        # Отображаем
        # 1. RGB + Isolines Overlay
        axes[i, 0].imshow(rgb_iso_overlay)
        axes[i, 0].set_title(f'RGB + Isolines\n{sample_name}')
        axes[i, 0].axis('off')

        # 2. Closed Isolines
        if closed_isolines_img is not None:
            axes[i, 1].imshow(closed_isolines_img, cmap='gray', vmin=0.0, vmax=1.0)
            axes[i, 1].set_title(f'Closed Isolines\n({sample_name})')
        else:
            axes[i, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[i, 1].transAxes)
            axes[i, 1].set_title(f'Closed Isolines\n({sample_name})')
        axes[i, 1].axis('off')

        # 3. Ground Truth Traps
        axes[i, 2].imshow(gt_traps, cmap='gray')
        axes[i, 2].set_title(f'Ground Truth Traps\n({sample_name})')
        axes[i, 2].axis('off')

        # 4. Predicted Traps (Masked)
        axes[i, 3].imshow(pred_traps_masked, cmap='gray')
        axes[i, 3].set_title(f'Predicted Traps\n({sample_name})')
        axes[i, 3].axis('off')

        # 5. Prediction Overlay
        axes[i, 4].imshow(pred_overlay)
        axes[i, 4].set_title(f'Prediction Overlay\n({sample_name})')
        axes[i, 4].axis('off')

        # 6. Error Map
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
    Визуализирует результаты на тестовых данных.
    Колонки: RGB+Isolines, Closed Isolines, GT, Pred, Overlay, Error Map
    """
    os.makedirs(save_path, exist_ok=True)

    x_rgb = batch['x'][:, :3, :, :]
    x_isolines = batch['x'][:, 4:5, :, :] if batch['x'].shape[1] >= 5 else None
    x_closed_isolines = batch['x'][:, 5:6, :, :] if batch['x'].shape[1] >= 6 else None
    y_traps = batch['y']
    mask_map = batch.get('mask_map', None)
    preds_prob = torch.sigmoid(predictions)

    for batch_idx in range(x_rgb.shape[0]):
        real_idx = sample_indices[batch_idx]

        if dataset is not None and hasattr(dataset, 'samples') and isinstance(dataset.samples, list):
            sample_paths = dataset.samples[real_idx]
            first_path = list(sample_paths.values())[0]
            from pathlib import Path
            filename = Path(first_path).stem
            import re
            match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
            sample_name = f"{match.group(1)}_{match.group(2)}" if match else f"sample_{real_idx}"
        else:
            sample_name = f"Test Sample {real_idx}"

        rgb_img = x_rgb[batch_idx].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        isolines_img = x_isolines[batch_idx, 0, :, :].cpu().numpy() if x_isolines is not None else None
        closed_isolines_img = x_closed_isolines[batch_idx, 0, :, :].cpu().numpy() if x_closed_isolines is not None else None
        
        gt_traps = y_traps[batch_idx, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[batch_idx].cpu().numpy()
        pred_traps = preds_prob[batch_idx, 0, :, :].cpu().detach().numpy()

        map_mask_np = None
        if mask_map is not None:
            map_mask_np = mask_map[batch_idx, 0, :, :].cpu().numpy() if mask_map.dim() == 4 else mask_map[batch_idx].cpu().numpy()

        if map_mask_np is not None:
            pred_traps_masked = pred_traps * map_mask_np
        else:
            pred_traps_masked = pred_traps

        metrics = None
        if metrics_by_sample is not None:
            metrics = metrics_by_sample.get(sample_name, None)

        rgb_iso_overlay = create_rgb_isolines_overlay(rgb_img, isolines_img, alpha=0.6)
        pred_overlay = create_prediction_overlay(rgb_img, pred_traps_masked, alpha=0.4)
        error_map = create_error_map(gt_traps, pred_traps_masked, map_mask=map_mask_np)

        fig, axes = plt.subplots(1, 6, figsize=(24, 5))

        if metrics is not None:
            metrics_str = f"Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}"
            titles = [
                f"RGB + Isolines\n{sample_name}\n{metrics_str}",
                f"Closed Isolines\n{metrics_str}",
                f"Ground Truth Traps\n{metrics_str}",
                f"Predicted Traps\n{metrics_str}",
                f"Prediction Overlay\n{metrics_str}",
                f"Error Map (|GT - Pred)|\n{metrics_str}"
            ]
        else:
            titles = [
                f'RGB + Isolines\n{sample_name}',
                f'Closed Isolines\n({sample_name})',
                f'Ground Truth Traps\n({sample_name})',
                f'Predicted Traps\n({sample_name})',
                f'Prediction Overlay\n({sample_name})',
                f'Error Map (|GT - Pred)|\n({sample_name})'
            ]

        # 1. RGB + Isolines
        axes[0].imshow(rgb_iso_overlay)
        axes[0].set_title(titles[0])
        axes[0].axis('off')

        # 2. Closed Isolines
        if closed_isolines_img is not None:
            axes[1].imshow(closed_isolines_img, cmap='gray', vmin=0.0, vmax=1.0)
            axes[1].set_title(titles[1])
        else:
            axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(titles[1])
        axes[1].axis('off')

        # 3. GT Traps
        axes[2].imshow(gt_traps, cmap='gray')
        axes[2].set_title(titles[2])
        axes[2].axis('off')

        # 4. Predicted Traps (Masked)
        axes[3].imshow(pred_traps_masked, cmap='gray')
        axes[3].set_title(titles[3])
        axes[3].axis('off')

        # 5. Prediction Overlay (Masked)
        axes[4].imshow(pred_overlay)
        axes[4].set_title(titles[4])
        axes[4].axis('off')

        # 6. Error Map
        im_error = axes[5].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[5].set_title(titles[5])
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
