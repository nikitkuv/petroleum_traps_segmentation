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
    cps_to_grayscale,
    cps_to_isolines,
    cps_to_binary_mask
)
from utils.images_utils import create_map_mask


def overlay_isolines_on_rgb_from_cps(
    cps_path: str,
    isoline_step: float = 5.0,
    cmap_name: str = 'purple_jet',
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid, _ = read_cps_grid(cps_path)
    rgb_img = cps_to_rgb(grid, cmap_name=cmap_name)
    isolines_img = cps_to_isolines(grid, step=isoline_step)

    rgb_float = rgb_img.astype(np.float32) / 255.0
    isolines_inverted = 1.0 - (isolines_img.astype(np.float32) / 255.0)
    line_mask = 1.0 - isolines_inverted

    overlay = rgb_float.copy()
    overlay = overlay * (1.0 - line_mask[:, :, np.newaxis] * alpha * 0.7)
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb_float)
    axes[0].set_title('RGB Map (from CPS)')
    axes[0].axis('off')

    axes[1].imshow(isolines_inverted, cmap='gray')
    axes[1].set_title('Isolines (Inverted)\n(white background, black lines)')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\n(alpha={alpha})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_closed_isolines(rgb_cps_path: str, traps_cps_path: str, isoline_step: float):
    # Функция оставлена для ручного использования, если понадобится анализ изолиний
    for path in [rgb_cps_path, traps_cps_path]:
        if not Path(path).exists():
            print(f"Ошибка: Файл не найден - {path}")
            return

    print("Загрузка CPS grids...")
    structural_grid, _ = read_cps_grid(rgb_cps_path)
    traps_grid, _ = read_cps_grid(traps_cps_path)

    print("Генерация изолиний...")
    isolines_img = cps_to_isolines(structural_grid, step=isoline_step)

    print("Генерация маски ловушек (GT)...")
    traps_mask = cps_to_binary_mask(traps_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(traps_mask, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0].set_title("Ground Truth Traps (from y_traps)", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(isolines_img, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Original Isolines (from structural)", fontsize=14)
    axes[1].axis('off')

    plt.suptitle("Direct CPS Grid Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()


def create_rgb_isolines_overlay(rgb_img: np.ndarray, isolines_img: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    overlay = rgb_img.copy()
    if isolines_img is not None:
        line_mask = isolines_img > 0.5
        overlay[line_mask] = overlay[line_mask] * (1.0 - alpha)
    return overlay


def create_prediction_overlay(rgb_img: np.ndarray, pred_traps: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay = rgb_img.copy()
    inverted_pred = 1.0 - pred_traps
    dark_overlay = np.stack([inverted_pred] * 3, axis=-1)
    mask = pred_traps > 0.01

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
    error_map = np.abs(gt_traps - pred_traps)
    if map_mask is not None:
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
    Колонки: RGB+Isolines, GT, Pred, Overlay, Error Map
    """
    os.makedirs(save_path, exist_ok=True)

    x_rgb = batch['x'][:, :3, :, :]  # 0-2: RGB
    x_isolines = batch['x'][:, 4:5, :, :] if batch['x'].shape[1] >= 5 else None  # 4: Isolines (3=Depth)
    y_traps = batch['y']
    mask_map = batch.get('mask_map', None)

    preds_prob = torch.sigmoid(predictions)
    n_samples = min(n_samples, x_rgb.shape[0])

    fig, axes = plt.subplots(n_samples, 5, figsize=(20, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
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

        rgb_img = x_rgb[i].cpu().permute(1, 2, 0).numpy()
        rgb_img = np.clip(rgb_img, 0, 1)

        isolines_img = x_isolines[i, 0, :, :].cpu().numpy() if x_isolines is not None else None

        gt_traps = y_traps[i, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[i].cpu().numpy()
        pred_traps = preds_prob[i, 0, :, :].cpu().detach().numpy()

        map_mask_np = None
        if mask_map is not None:
            map_mask_np = mask_map[i, 0, :, :].cpu().numpy() if mask_map.dim() == 4 else mask_map[i].cpu().numpy()

        pred_traps_masked = pred_traps * map_mask_np if map_mask_np is not None else pred_traps

        rgb_iso_overlay = create_rgb_isolines_overlay(rgb_img, isolines_img, alpha=0.6)
        pred_overlay = create_prediction_overlay(rgb_img, pred_traps_masked, alpha=0.4)
        error_map = create_error_map(gt_traps, pred_traps_masked, map_mask=map_mask_np)

        # 1. RGB + Isolines Overlay
        axes[i, 0].imshow(rgb_iso_overlay)
        axes[i, 0].set_title(f'RGB + Isolines\n{sample_name}')
        axes[i, 0].axis('off')

        # 2. Ground Truth Traps
        axes[i, 1].imshow(gt_traps, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth Traps\n({sample_name})')
        axes[i, 1].axis('off')

        # 3. Predicted Traps (Masked)
        axes[i, 2].imshow(pred_traps_masked, cmap='gray')
        axes[i, 2].set_title(f'Predicted Traps\n({sample_name})')
        axes[i, 2].axis('off')

        # 4. Prediction Overlay
        axes[i, 3].imshow(pred_overlay)
        axes[i, 3].set_title(f'Prediction Overlay\n({sample_name})')
        axes[i, 3].axis('off')

        # 5. Error Map
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
    Колонки: RGB+Isolines, GT, Pred, Overlay, Error Map
    """
    os.makedirs(save_path, exist_ok=True)

    x_rgb = batch['x'][:, :3, :, :]
    x_isolines = batch['x'][:, 4:5, :, :] if batch['x'].shape[1] >= 5 else None
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
        
        gt_traps = y_traps[batch_idx, 0, :, :].cpu().numpy() if y_traps.dim() == 4 else y_traps[batch_idx].cpu().numpy()
        pred_traps = preds_prob[batch_idx, 0, :, :].cpu().detach().numpy()

        map_mask_np = None
        if mask_map is not None:
            map_mask_np = mask_map[batch_idx, 0, :, :].cpu().numpy() if mask_map.dim() == 4 else mask_map[batch_idx].cpu().numpy()

        pred_traps_masked = pred_traps * map_mask_np if map_mask_np is not None else pred_traps

        metrics = None
        if metrics_by_sample is not None:
            metrics = metrics_by_sample.get(sample_name, None)

        rgb_iso_overlay = create_rgb_isolines_overlay(rgb_img, isolines_img, alpha=0.6)
        pred_overlay = create_prediction_overlay(rgb_img, pred_traps_masked, alpha=0.4)
        error_map = create_error_map(gt_traps, pred_traps_masked, map_mask=map_mask_np)

        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        if metrics is not None:
            metrics_str = f"Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}"
            titles = [
                f"RGB + Isolines\n{sample_name}\n{metrics_str}",
                f"Ground Truth Traps\n{metrics_str}",
                f"Predicted Traps\n{metrics_str}",
                f"Prediction Overlay\n{metrics_str}",
                f"Error Map (|GT - Pred)|\n{metrics_str}"
            ]
        else:
            titles = [
                f'RGB + Isolines\n{sample_name}',
                f'Ground Truth Traps\n({sample_name})',
                f'Predicted Traps\n({sample_name})',
                f'Prediction Overlay\n({sample_name})',
                f'Error Map (|GT - Pred)|\n({sample_name})'
            ]

        axes[0].imshow(rgb_iso_overlay)
        axes[0].set_title(titles[0])
        axes[0].axis('off')

        axes[1].imshow(gt_traps, cmap='gray')
        axes[1].set_title(titles[1])
        axes[1].axis('off')

        axes[2].imshow(pred_traps_masked, cmap='gray')
        axes[2].set_title(titles[2])
        axes[2].axis('off')

        axes[3].imshow(pred_overlay)
        axes[3].set_title(titles[3])
        axes[3].axis('off')

        im_error = axes[4].imshow(error_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[4].set_title(titles[4])
        axes[4].axis('off')
        plt.colorbar(im_error, ax=axes[4], fraction=0.046, pad=0.04)

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


def visualize_full_cps_analysis(
    rgb_cps_path: str, 
    traps_cps_path: str, 
    isoline_step: float = 5.0,
    overlay_alpha: float = 0.4
) -> None:
    for path in [rgb_cps_path, traps_cps_path]:
        if not Path(path).exists():
            print(f"Ошибка: Файл не найден - {path}")
            return

    print("Загрузка и обработка CPS грида...")
    structural_grid, _ = read_cps_grid(rgb_cps_path)
    
    rgb_img = cps_to_rgb(structural_grid)
    depth_img = cps_to_grayscale(structural_grid, invert=False)
    isolines_img = cps_to_isolines(structural_grid, step=isoline_step)

    traps_grid, _ = read_cps_grid(traps_cps_path)
    traps_mask = cps_to_binary_mask(traps_grid)

    map_mask = create_map_mask(rgb_img)

    rgb_float = rgb_img.astype(np.float32) / 255.0
    depth_norm = depth_img.astype(np.float32) / 255.0
    isolines_norm = isolines_img.astype(np.float32) / 255.0

    rgb_isolines_overlay = rgb_float.copy()
    line_mask = isolines_norm > 0.5
    rgb_isolines_overlay[line_mask] = rgb_isolines_overlay[line_mask] * (1.0 - 0.6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1
    axes[0, 0].imshow(rgb_float)
    axes[0, 0].set_title("RGB Map", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(depth_norm, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Depth Norm Map", fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(traps_mask, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("Ground Truth Traps", fontsize=14)
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(rgb_isolines_overlay)
    axes[1, 0].set_title("RGB + Isolines Overlay", fontsize=14)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(map_mask, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Map Mask (1=inside, 0=outside)", fontsize=14)
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')

    plt.suptitle("Full CPS Data Analysis", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    