"""
Визуализация результата инференса на полную карту (одна фигура на горизонт).

В отличие от visualize_test_results (пертайловая), здесь — единая склеенная карта,
что нагляднее для инференса. Переиспользуются хелперы наложения из visualization.visualize.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from visualization.visualize import (
    create_rgb_isolines_overlay,
    create_prediction_overlay,
    create_error_map,
)


def visualize_full_map(
    rgb_img: np.ndarray,
    depth_img: np.ndarray,
    isolines_img: np.ndarray,
    pred_mask: np.ndarray,
    valid_mask: np.ndarray,
    horizon: str,
    out_path: str,
    gt_traps: np.ndarray = None,
    metrics: dict = None,
    alpha: float = 0.4,
) -> None:
    """
    Сохраняет мультепанельную визуализацию предсказания на полную карту.

    Панели: RGB+изолинии, карта глубин, предсказанные ловушки, наложение предсказания,
    и (если есть GT) ловушки GT + карта ошибок.

    Args:
        rgb_img: полная RGB-карта uint8 (h, w, 3) — разломы уже обнулены.
        depth_img: нормализованная глубина float32 [0, 1] (h, w).
        isolines_img: изолинии uint8 (h, w).
        pred_mask: бинарная маска предсказанных ловушек float [0, 1] (h, w).
        valid_mask: маска валидной области карты (h, w).
        horizon: имя горизонта (для заголовка/имени файла).
        out_path: куда сохранить PNG.
        gt_traps: полная маска ловушек GT uint8 (h, w), если есть.
        metrics: словарь метрик горизонта (для подписи), опционально.
        alpha: прозрачность наложения предсказания.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rgb_float = np.clip(rgb_img.astype(np.float32) / 255.0, 0, 1)
    isolines_float = isolines_img.astype(np.float32) / 255.0
    on_map = valid_mask.astype(bool)
    pred_masked = (pred_mask * on_map).astype(np.float32)

    metrics_str = ""
    if metrics is not None:
        metrics_str = (
            f"   Dice={metrics.get('dice', 0):.3f}  IoU={metrics.get('iou', 0):.3f}"
            f"  Recall={metrics.get('recall', 0):.3f}  Prec={metrics.get('precision', 0):.3f}"
        )

    panels = []
    panels.append((
        create_rgb_isolines_overlay(rgb_float, isolines_float, alpha=0.6),
        f"RGB + изолинии\n{horizon}{metrics_str}", None, None,
    ))
    panels.append((depth_img, f"Карта глубин (норм.)\n{horizon}", 'gray', (0.0, 1.0)))
    panels.append((pred_masked, f"Предсказанные ловушки\n{horizon}", 'gray', (0.0, 1.0)))
    panels.append((
        create_prediction_overlay(rgb_float, pred_masked, alpha=alpha),
        f"Наложение предсказания\n{horizon}", None, None,
    ))

    if gt_traps is not None:
        gt_float = (np.clip(gt_traps.astype(np.float32) / 255.0, 0, 1)) * on_map
        panels.append((gt_float, f"Ловушки (GT)\n{horizon}", 'gray', (0.0, 1.0)))
        panels.append((
            create_error_map(gt_float, pred_masked, valid_mask),
            f"Карта ошибок |GT - Pred|\n{horizon}", 'RdYlBu_r', (0, 1),
        ))

    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (img, title, cmap, vlim) in zip(axes, panels):
        if cmap == 'RdYlBu_r':
            im = ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        elif cmap is not None:
            ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Визуализация: {out_path}")
