import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
import re

sys.path.append(str(Path(__file__).parent.parent))

from utils.cps_utils import read_cps_grid, cps_to_rgb, cps_to_grayscale
from settings import settings


def extract_horizon_name(filename: str) -> str:
    """
    Извлекает название горизонта из имени файла.

    Примеры:
        x_structuralNOisoline_Ach3-2-1_toptop1 -> Ach3-2-1_toptop1
        y_traps_U2_3_kolltop1 -> U2_3_kolltop1
    """
    # Паттерн: {prefix}_{horizon_name}
    pattern = r'^(x_structuralNOisoline_|y_traps_)(.+)$'
    match = re.match(pattern, filename)
    if match:
        return match.group(2)
    return None


def find_cps_files(cps_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Находит все CPS файлы и группирует их по горизонтам.

    Returns:
        Dict[horizon_name] -> {
            'structural': path_to_x_structuralNOisoline,
            'traps': path_to_y_traps
        }
    """
    horizons = {}

    for filename in os.listdir(cps_dir):
        if filename.startswith('x_structuralNOisoline_'):
            horizon_name = extract_horizon_name(filename)
            if horizon_name:
                if horizon_name not in horizons:
                    horizons[horizon_name] = {}
                horizons[horizon_name]['structural'] = os.path.join(cps_dir, filename)

        elif filename.startswith('y_traps_'):
            horizon_name = extract_horizon_name(filename)
            if horizon_name:
                if horizon_name not in horizons:
                    horizons[horizon_name] = {}
                horizons[horizon_name]['traps'] = os.path.join(cps_dir, filename)

    return horizons


def save_large_images(horizons: Dict[str, Dict[str, str]], output_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Конвертирует CPS файлы в PNG и сохраняет большие изображения.

    Returns:
        Dict[horizon_name] -> {
            'rgb': RGB image array,
            'grayscale': grayscale image array,
            'traps': traps image array
        }
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    images_data = {}

    for horizon_name, files in horizons.items():
        print(f"\nProcessing horizon: {horizon_name}")
        images_data[horizon_name] = {}

        # Загружаем structuralNOisoline
        if 'structural' in files:
            print(f"  Loading structural: {files['structural']}")
            grid, meta = read_cps_grid(files['structural'])

            # Конвертируем в RGB
            rgb_img = cps_to_rgb(grid, cmap_name='purple_jet')
            # Поворачиваем на 180 градусов
            rgb_img = np.rot90(rgb_img, k=2)
            rgb_path = os.path.join(output_dir, f'x_structuralNOisoline_{horizon_name}.png')
            save_png(rgb_img, rgb_path)
            print(f"  Saved RGB: {rgb_path} (shape={rgb_img.shape})")
            images_data[horizon_name]['rgb'] = rgb_img

            # Конвертируем в grayscale (для depth_norm)
            gray_img = cps_to_grayscale(grid, invert=True)
            # Поворачиваем на 180 градусов
            gray_img = np.rot90(gray_img, k=2)
            gray_path = os.path.join(output_dir, f'x_structuralBlackWhite_{horizon_name}.png')
            save_png(gray_img, gray_path)
            print(f"  Saved Grayscale: {gray_path} (shape={gray_img.shape})")
            images_data[horizon_name]['grayscale'] = gray_img

        # Загружаем traps
        if 'traps' in files:
            print(f"  Loading traps: {files['traps']}")
            grid, meta = read_cps_grid(files['traps'])

            # Для traps используем бинаризацию
            traps_img = cps_to_binary_mask(grid)
            traps_img = (traps_img * 255).astype(np.uint8)
            # Поворачиваем на 180 градусов
            traps_img = np.rot90(traps_img, k=2)
            traps_path = os.path.join(output_dir, f'y_traps_{horizon_name}.png')
            save_png(traps_img, traps_path)
            print(f"  Saved Traps: {traps_path} (shape={traps_img.shape})")
            images_data[horizon_name]['traps'] = traps_img

    return images_data


def save_png(img_array: np.ndarray, path: str):
    """Сохраняет numpy array как PNG изображение."""
    from PIL import Image

    if len(img_array.shape) == 2:
        # Grayscale
        img = Image.fromarray(img_array, mode='L')
    else:
        # RGB
        img = Image.fromarray(img_array, mode='RGB')

    img.save(path)


def cps_to_binary_mask(grid: np.ndarray, threshold: float = 128) -> np.ndarray:
    """Конвертирует CPS грид в бинарную маску."""
    valid_mask = ~np.isnan(grid)

    if valid_mask.sum() > 0:
        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        grid_norm = (grid - vmin) / (vmax - vmin + 1e-8)
        grid_norm = np.clip(grid_norm, 0, 1) * 255
    else:
        grid_norm = np.zeros_like(grid)

    mask = (grid_norm < threshold).astype(np.float32)
    mask[~valid_mask] = 0

    return mask


def split_into_tiles(images_data: Dict[str, Dict[str, np.ndarray]],
                     output_dir: str,
                     tile_width: int = None,
                     tile_height: int = None,
                     overlap_ratio: float = None) -> List[str]:
    """
    Разбивает большие изображения на тайлы с перекрытием.

    Для каждого горизонта создаются тайлы с одинаковыми координатами
    для всех трех типов изображений (rgb, grayscale, traps).

    Args:
        images_data: Данные изображений по горизонтам
        output_dir: Директория для сохранения тайлов
        tile_width: Ширина тайла (по умолчанию settings.TARGET_WIDTH)
        tile_height: Высота тайла (по умолчанию settings.TARGET_HEIGHT)
        overlap_ratio: Процент перекрытия (по умолчанию settings.TILE_OVERLAP_RATIO)

    Returns:
        Список сохраненных файлов
    """
    tile_width = tile_width or settings.TARGET_WIDTH
    tile_height = tile_height or settings.TARGET_HEIGHT
    overlap_ratio = overlap_ratio if overlap_ratio is not None else settings.TILE_OVERLAP_RATIO

    # Вычисляем stride (шаг) с учетом перекрытия
    stride_h = int(tile_height * (1 - overlap_ratio))
    stride_w = int(tile_width * (1 - overlap_ratio))

    # Гарантируем минимальный шаг хотя бы 1 пиксель
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved_files = []

    for horizon_name, images in images_data.items():
        print(f"\nSplitting horizon {horizon_name} into tiles ({tile_width}x{tile_height}, overlap={overlap_ratio*100:.0f}%)...")

        # Получаем размеры изображений (все должны быть одинаковыми)
        rgb_img = images.get('rgb')
        grayscale_img = images.get('grayscale')
        traps_img = images.get('traps')

        # Используем rgb для определения размеров
        if rgb_img is not None:
            h, w = rgb_img.shape[:2]
        elif grayscale_img is not None:
            h, w = grayscale_img.shape[:2]
        elif traps_img is not None:
            h, w = traps_img.shape[:2]
        else:
            print(f"  No images found for horizon {horizon_name}, skipping...")
            continue

        print(f"  Image size: {h}x{w}")
        print(f"  Stride: {stride_h}x{stride_w}")

        # Если изображение меньше или равно размеру тайла, сохраняем его как один тайл с паддингом
        if h <= tile_height and w <= tile_width:
            tile_index = 1
            tile_prefix = f"{tile_index:03d}_"

            # Сохраняем rgb тайл
            if rgb_img is not None:
                tile_rgb = pad_image(rgb_img, tile_height, tile_width)
                rgb_tile_path = os.path.join(output_dir, f'{tile_prefix}x_structuralNOisoline_{horizon_name}.png')
                save_png(tile_rgb, rgb_tile_path)
                saved_files.append(rgb_tile_path)

            # Сохраняем grayscale тайл
            if grayscale_img is not None:
                tile_gray = pad_image(grayscale_img, tile_height, tile_width)
                gray_tile_path = os.path.join(output_dir, f'{tile_prefix}x_structuralBlackWhite_{horizon_name}.png')
                save_png(tile_gray, gray_tile_path)
                saved_files.append(gray_tile_path)

            # Сохраняем traps тайл
            if traps_img is not None:
                tile_traps = pad_image(traps_img, tile_height, tile_width)
                traps_tile_path = os.path.join(output_dir, f'{tile_prefix}y_traps_{horizon_name}.png')
                save_png(tile_traps, traps_tile_path)
                saved_files.append(traps_tile_path)

            print(f"  Image is smaller than tile size, saved as 1 tile with padding")
            continue

        # Вычисляем количество тайлов с учетом stride
        n_tiles_h = max(1, (h - tile_height) // stride_h + 1)
        n_tiles_w = max(1, (w - tile_width) // stride_w + 1)

        # Корректируем если последний тайл выходит за границы
        # Добавляем дополнительный тайл если нужно
        if (n_tiles_h - 1) * stride_h + tile_height < h:
            n_tiles_h += 1
        if (n_tiles_w - 1) * stride_w + tile_width < w:
            n_tiles_w += 1

        print(f"  Number of tiles: {n_tiles_h} x {n_tiles_w} = {n_tiles_h * n_tiles_w}")

        tile_index = 0

        for row in range(n_tiles_h):
            for col in range(n_tiles_w):
                tile_index += 1

                # Вычисляем координаты с учетом stride
                y_start = row * stride_h
                x_start = col * stride_w

                # Вычисляем конечные координаты
                y_end = min(y_start + tile_height, h)
                x_end = min(x_start + tile_width, w)

                # Корректируем начальные координаты если мы у края
                if y_end == h:
                    y_start = max(0, y_end - tile_height)
                if x_end == w:
                    x_start = max(0, x_end - tile_width)

                tile_prefix = f"{tile_index:03d}_"

                # Сохраняем rgb тайл
                if rgb_img is not None:
                    tile_rgb = rgb_img[y_start:y_end, x_start:x_end]
                    # Паддинг если нужно (для краевых тайлов)
                    tile_rgb = pad_image(tile_rgb, tile_height, tile_width)
                    rgb_tile_path = os.path.join(output_dir, f'{tile_prefix}x_structuralNOisoline_{horizon_name}.png')
                    save_png(tile_rgb, rgb_tile_path)
                    saved_files.append(rgb_tile_path)

                # Сохраняем grayscale тайл
                if grayscale_img is not None:
                    tile_gray = grayscale_img[y_start:y_end, x_start:x_end]
                    tile_gray = pad_image(tile_gray, tile_height, tile_width)
                    gray_tile_path = os.path.join(output_dir, f'{tile_prefix}x_structuralBlackWhite_{horizon_name}.png')
                    save_png(tile_gray, gray_tile_path)
                    saved_files.append(gray_tile_path)

                # Сохраняем traps тайл
                if traps_img is not None:
                    tile_traps = traps_img[y_start:y_end, x_start:x_end]
                    tile_traps = pad_image(tile_traps, tile_height, tile_width)
                    traps_tile_path = os.path.join(output_dir, f'{tile_prefix}y_traps_{horizon_name}.png')
                    save_png(tile_traps, traps_tile_path)
                    saved_files.append(traps_tile_path)

        print(f"  Created {tile_index} tile sets for horizon {horizon_name}")

    return saved_files


def pad_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Паддит изображение до целевого размера.

    Args:
        img: Исходное изображение (H, W) или (H, W, C)
        target_h: Целевая высота
        target_w: Целевая ширина

    Returns:
        Паддитое изображение
    """
    current_h, current_w = img.shape[:2]

    if current_h >= target_h and current_w >= target_w:
        # Обрезаем если больше
        return img[:target_h, :target_w]

    # Создаем паддированное изображение
    if len(img.shape) == 2:
        padded = np.zeros((target_h, target_w), dtype=img.dtype)
    else:
        padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)

    # Копируем исходное изображение в верхний левый угол
    h_copy = min(current_h, target_h)
    w_copy = min(current_w, target_w)
    padded[:h_copy, :w_copy] = img[:h_copy, :w_copy]

    return padded


def main():
    """Основная функция."""
    cps_dir = './data/cps/'
    full_images_dir = './data/images_cps_full/'
    tiles_dir = './data/images_cps/'

    print("=" * 60)
    print("CPS to PNG Converter and Tile Splitter")
    print("=" * 60)
    print(f"CPS directory: {cps_dir}")
    print(f"Full images output: {full_images_dir}")
    print(f"Tiles output: {tiles_dir}")
    print(f"Tile size: {settings.TARGET_WIDTH}x{settings.TARGET_HEIGHT}")

    # Шаг 1: Найти CPS файлы
    print("\n" + "=" * 60)
    print("Step 1: Finding CPS files...")
    horizons = find_cps_files(cps_dir)
    print(f"Found {len(horizons)} horizons:")
    for name in sorted(horizons.keys()):
        files = horizons[name]
        print(f"  {name}: structural={'structural' in files}, traps={'traps' in files}")

    # Шаг 2: Конвертировать в PNG и сохранить большие изображения
    print("\n" + "=" * 60)
    print("Step 2: Converting CPS to PNG and saving large images...")
    images_data = save_large_images(horizons, full_images_dir)

    # Шаг 3: Разбить на тайлы
    print("\n" + "=" * 60)
    print("Step 3: Splitting large images into tiles...")
    saved_files = split_into_tiles(images_data, tiles_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Saved {len(saved_files)} tile files to {tiles_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
