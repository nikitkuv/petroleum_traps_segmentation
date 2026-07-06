"""
Скрипт для визуализации RGB CPS грида и границы карты.

Скрипт:
1. Проверяет, что имя файла содержит '_structuralNOisoline_'
2. Загружает CPS грид
3. Конвертирует в RGB изображение (purple_jet colormap)
4. Создает изображение границы карты
5. Сохраняет результат в data/test_cps/ с размерами в имени файла
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь для импорта
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2

from utils.cps_utils import read_cps_grid, cps_to_rgb


# ============================================================
# Параметры
# ============================================================

# Путь к CPS гриду (должен содержать '_structuralNOisoline_' в имени)
CPS_PATH = "./data/cps/x_structuralNOisoline_H150_toptop1"

# Директория для сохранения результата
OUTPUT_DIR = "./data/test_cps"

# Отступ между изображениями в пикселях
GAP = 20


# ============================================================
# Функции
# ============================================================


def create_map_boundary(grid: np.ndarray) -> np.ndarray:
    """
    Создает изображение границы карты.
    Граница - это контур между валидной областью (не NaN) и фоном (NaN).

    Args:
        grid: 2D numpy array с NaN для невалидных областей

    Returns:
        boundary_img: (H, W) uint8 array (255 = граница, 0 = остальное)
    """
    valid_mask = (~np.isnan(grid)).astype(np.uint8)

    # Находим контуры с помощью cv2
    contours, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры на черном фоне
    boundary_img = np.zeros(grid.shape, dtype=np.uint8)
    cv2.drawContours(boundary_img, contours, -1, 255, thickness=2)

    return boundary_img


def visualize_cps_rgb_and_boundary(
    cps_path: str,
    output_dir: str = "./data/test_cps",
    gap: int = 20,
) -> str:
    """
    Загружает CPS грид, создает RGB изображение и границу карты,
    сохраняет их рядом с отступом.

    Args:
        cps_path: Путь к CPS гриду (должен содержать '_structuralNOisoline_')
        output_dir: Директория для сохранения результата
        gap: Отступ между изображениями в пикселях

    Returns:
        Путь к сохраненному изображению или None при ошибке
    """
    path = Path(cps_path)

    # Проверка существования файла
    if not path.exists():
        print(f"Ошибка: Файл не найден - {cps_path}")
        return None

    # Проверка имени файла
    if "_structuralNOisoline_" not in path.name:
        print(f"Ошибка: Имя файла не содержит '_structuralNOisoline_' - {path.name}")
        return None

    # Загрузка CPS грида
    print(f"Загрузка CPS грида: {cps_path}")
    grid, meta = read_cps_grid(cps_path)

    grid_h, grid_w = grid.shape
    print(f"Размер грида: {grid_w}x{grid_h} (nx={meta['nx']}, ny={meta['ny']})")

    # Конвертация в RGB
    rgb_img = cps_to_rgb(grid, cmap_name='purple_jet')
    img_h, img_w = rgb_img.shape[:2]
    print(f"Размер изображения: {img_w}x{img_h}")

    # Создание границы карты
    boundary_img = create_map_boundary(grid)

    # Конвертация границы в 3-канальное изображение для объединения
    boundary_rgb = cv2.cvtColor(boundary_img, cv2.COLOR_GRAY2RGB)

    # Создание объединенного изображения с отступом
    # Белый отступ между изображениями
    gap_strip = np.ones((img_h, gap, 3), dtype=np.uint8) * 255

    # Объединяем: RGB | отступ | граница
    combined = np.hstack([rgb_img, gap_strip, boundary_rgb])

    # Извлекаем имя горизонта из имени файла
    # Формат: x_structuralNOisoline_{horizon_name}
    horizon_name = path.name.replace("x_structuralNOisoline_", "")

    # Формируем имя выходного файла с размерами
    output_filename = f"{horizon_name}_grid-{grid_w}x{grid_h}_img-{img_w}x{img_h}.png"
    output_path = Path(output_dir) / output_filename

    # Создаем директорию если не существует
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Сохраняем
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"Сохранено: {output_path}")

    return str(output_path)


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":
    result = visualize_cps_rgb_and_boundary(
        cps_path=CPS_PATH,
        output_dir=OUTPUT_DIR,
        gap=GAP,
    )

    if result is None:
        sys.exit(1)
