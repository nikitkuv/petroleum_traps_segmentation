import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.cps_utils import read_cps_grid, cps_to_rgb, cps_to_binary_mask


HORIZON_NAME = "H150_TWT"

STRUCTURAL_CPS_PATH = f"./data/raw/CPS3_faults/x_structuralNOisoline_{HORIZON_NAME}"
FAULTS_CPS_PATH = f"./data/raw/CPS3_faults/x_faults_{HORIZON_NAME}"


def check_faults_resize(rgb_cps_path: str, faults_cps_path: str):
    # Проверка путей
    for path in [rgb_cps_path, faults_cps_path]:
        if not Path(path).exists():
            print(f"Ошибка: Файл не найден - {path}")
            return

    print("Загрузка CPS гридов...")
    
    # 1. Загружаем структурную карту (определяет целевой размер)
    structural_grid, _ = read_cps_grid(rgb_cps_path)
    rgb_img = cps_to_rgb(structural_grid)
    target_shape = rgb_img.shape[:2] # (H, W)
    
    # 2. Загружаем оригинальную карту разломов
    faults_grid, _ = read_cps_grid(faults_cps_path)
    faults_mask_orig = cps_to_binary_mask(faults_grid)
    
    print(f"RGB Target Shape: {target_shape}")
    print(f"Faults Original Shape: {faults_mask_orig.shape}")
    
    # Если размеры и так совпадают, ресайз не нужен
    if faults_mask_orig.shape == target_shape:
        print("Grid sizes are the same, no resizing needed")
        faults_mask_resized = faults_mask_orig
    else:
        # 3. Ресайзим маску разломов до размеров структурной карты (INTER_NEAREST - чтобы не было размытия)
        print(f"Resizing: {faults_mask_orig.shape} -> {target_shape}")
        faults_mask_resized = cv2.resize(
            faults_mask_orig, 
            (target_shape[1], target_shape[0]), # cv2.resize принимает (W, H)
            interpolation=cv2.INTER_NEAREST
        )

    # 4. Создаем оверлей: RGB + Ресайзнутая маска разломов (черные линии)
    rgb_overlay = rgb_img.astype(np.float32) / 255.0
    fault_pixels = faults_mask_resized > 0.5
    rgb_overlay[fault_pixels] = [0.0, 0.0, 0.0] # Красим пиксели разломов в черный

    # 5. Визуализация
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Колонка 1: RGB
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"RGB Structural\nSize: {target_shape}", fontsize=12)
    axes[0].axis('off')
    
    # Колонка 2: Маска разломов (Ресайзнутая)
    axes[1].imshow(faults_mask_resized, cmap='gray', vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Faults Mask (Resized)\nSize: {faults_mask_resized.shape[:2]}", fontsize=12)
    axes[1].axis('off')
    
    # Колонка 3: Маска разломов (Оригинальная)
    # Используем aspect='auto', чтобы оригинальная вытянутая карта не ломала пропорции графика
    axes[2].imshow(faults_mask_orig, cmap='gray', vmin=0.0, vmax=1.0, aspect='auto')
    axes[2].set_title(f"Faults Mask (Original)\nSize: {faults_mask_orig.shape[:2]}", fontsize=12)
    axes[2].axis('off')
    
    # Колонка 4: Оверлей (RGB + Resized Faults)
    axes[3].imshow(rgb_overlay)
    axes[3].set_title(f"RGB + Resized Faults Overlay\n(Black = Faults)", fontsize=12)
    axes[3].axis('off')
    
    plt.suptitle(f"Faults Resize Quality Check: {HORIZON_NAME}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_faults_resize(STRUCTURAL_CPS_PATH, FAULTS_CPS_PATH)
