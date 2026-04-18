import numpy as np
import warnings
from typing import Tuple, Dict, List
import os
import re
from pathlib import Path
import cv2

from settings import settings
from utils.images_utils import pad_image


def read_cps_grid(file_path: str, vertical_flip: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Загружает CPS грид из файла.
    
    Args:
        file_path: Путь к CPS файлу
        vertical_flip: Флип по вертикали
    
    Returns:
        grid: 2D numpy array с значениями
        metadata: dict с xmin, xmax, ymin, ymax, nx, ny, null_value
    """
    vertical_flip = vertical_flip if vertical_flip is not None else settings.CPS_VERTICAL_FLIP
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Инициализация параметров грида
    nx = ny = None
    xmin = xmax = ymin = ymax = None
    null_value = settings.CPS_NULL_VALUE
    data_start = None
    
    # Парсинг заголовков файла
    for i, line in enumerate(lines):
        if line.startswith("FSASCI"):
            parts = line.split()
            if len(parts) >= 2:
                null_value = float(parts[-1])
        
        elif line.startswith("FSNROW"):
            parts = line.split()
            if len(parts) >= 3:
                ny, nx = int(parts[1]), int(parts[2])
        
        elif line.startswith("FSLIMI"):
            parts = line.split()
            if len(parts) >= 5:
                xmin, xmax = float(parts[1]), float(parts[2])
                ymin, ymax = float(parts[3]), float(parts[4])
        
        elif line.startswith("->"):  # start of data
            data_start = i + 1
            break
    
    if None in (nx, ny, xmin, xmax, ymin, ymax):
        raise ValueError(f"Failed to parse CPS header: {file_path}")
    
    # Читаем значения
    values = []
    for line in lines[data_start:]:
        values.extend([float(x) for x in line.split()])
    
    values = np.array(values, dtype=np.float32)
    
    # Проверка количества ячеек
    expected = nx * ny
    if len(values) != expected:
        warnings.warn(f"CPS {file_path}: expected {expected}, got {len(values)}. Truncating.")
        values = values[:expected]
    
    # Решейп по Фортрану
    grid = values.reshape((ny, nx), order='F')
    
    # Вертикальный флип
    if vertical_flip:
        grid = np.flipud(grid)

    # Поворот на 180 градусов
    grid = np.rot90(grid, k=2)
    
    # Заменяем пустые значения
    grid[np.isclose(grid, null_value)] = np.nan
    
    metadata = {
        'nx': nx,
        'ny': ny,
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'null_value': null_value,
        'file_path': file_path
    }
    
    return grid, metadata


def cps_to_rgb(grid: np.ndarray, cmap_name: str = 'purple_jet') -> np.ndarray:
    """
    Конвертирует CPS грид в RGB изображение с цветовой палитрой.
    
    Args:
        grid: 2D numpy array
        cmap_name: Название colormap ('purple_jet', 'jet', 'seismic', etc.)
    
    Returns:
        rgb: (H, W, 3) uint8 array
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    # Создаём палитру
    if cmap_name == 'purple_jet':
        colors = [
            '#4B0082',  # Индиго/фиолетовый (мин)
            '#0000FF',  # Синий
            '#00FFFF',  # Голубой
            '#00FF00',  # Зелёный
            '#FFFF00',  # Жёлтый
            '#FF8000',  # Оранжевый
            '#FF0000'   # Красный (макс)
        ]
        cmap = LinearSegmentedColormap.from_list('purple_jet', colors, N=256)
    else:
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(cmap_name)
    
    # Нормализуем грид (игнорируя NaN)
    valid_mask = ~np.isnan(grid)
    if valid_mask.sum() == 0:
        # Все NaN — возвращаем чёрное изображение
        return np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    
    # Нормализация к [0, 1]
    grid_norm = (grid - vmin) / (vmax - vmin + 1e-8)
    grid_norm = np.clip(grid_norm, 0, 1)
    
    # Применяем colormap
    rgb_float = cmap(grid_norm)[:, :, :3]  # Убираем alpha канал
    
    # Конвертируем в uint8
    rgb = (rgb_float * 255).astype(np.uint8)
    
    # Маскируем NaN (чёрный цвет)
    rgb[~valid_mask] = 0
    
    return rgb


def cps_to_grayscale(grid: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Конвертирует CPS грид в черно-белое изображение.
    
    Args:
        grid: 2D numpy array
        invert: Если True — инвертируем (черный = макс, белый = мин)
    
    Returns:
        gray: (H, W) uint8 array
    """
    valid_mask = ~np.isnan(grid)
    
    if valid_mask.sum() == 0:
        return np.zeros(grid.shape, dtype=np.uint8)
    
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    
    # Нормализация к [0, 255]
    grid_norm = (grid - vmin) / (vmax - vmin + 1e-8)
    grid_norm = np.clip(grid_norm, 0, 1)
    
    if invert:
        grid_norm = 1.0 - grid_norm
    
    gray = (grid_norm * 255).astype(np.uint8)
    gray[~valid_mask] = 0  # NaN = чёрный
    
    return gray


def cps_to_isolines(grid: np.ndarray, step: float = 5.0) -> np.ndarray:
    """
    Генерирует изображение с изолиниями из CPS грида с помощью OpenCV.
    
    Args:
        grid: 2D numpy array (с NaN в качестве пустот)
        step: Шаг изолиний в метрах (по умолчанию 5)
    
    Returns:
        isolines: (H, W) uint8 array (черный фон=0, белые линии=255, вне карты=0)
    """
    valid_mask = ~np.isnan(grid)
    if valid_mask.sum() == 0:
        return np.zeros(grid.shape, dtype=np.uint8) # Черный фон
    
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    
    # Вычисляем уровни изолиний (например: -3000, -2995, -2990...)
    start_bound = np.floor(vmin / step) * step
    end_bound = np.ceil(vmax / step) * step
    levels = np.arange(start_bound, end_bound + step, step)
    
    # Создаем черное полотно (фон = 0)
    isolines_img = np.zeros(grid.shape, dtype=np.uint8)
    
    # Временно заменяем NaN на значение ниже минимума, 
    # чтобы cv2.findContours не упал, но контуры там не рисовались
    grid_filled = np.copy(grid)
    grid_filled[~valid_mask] = vmin - 1000 
    
    # Проходим по каждому уровню
    for level in levels:
        # Создаем бинарную маску: 1 там, где глубина >= level, 0 там, где меньше
        binary_mask = (grid_filled >= level).astype(np.uint8)
        
        # Убираем из маски области, где были NaN (пустоты/разломы)
        binary_mask[~valid_mask] = 0
        
        # Находим контуры этой бинарной маски
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Рисуем найденные контуры белым цветом (255) толщиной 1 пиксель
        cv2.drawContours(isolines_img, contours, -1, 255, thickness=1)
        
    return isolines_img


def cps_to_binary_mask(grid: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Конвертирует CPS грид в бинарную маску (для fault/trap).

    Для traps: карта (ловушки) = значения >= threshold, фон = NaN
    Сохраняем фон как 0 (черный), чтобы при загрузке он оставался черным.

    Args:
        grid: 2D numpy array
        threshold: Порог бинаризации (по умолчанию 128)

    Returns:
        mask: (H, W) float32 array (0 или 1)
    """
    if threshold is None:
        threshold = settings.BINARY_THRESHOLD

    valid_mask = ~np.isnan(grid)

    # Нормализуем к [0, 255]
    if valid_mask.sum() > 0:
        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        grid_norm = (grid - vmin) / (vmax - vmin + 1e-8)
        grid_norm = np.clip(grid_norm, 0, 1) * 255
    else:
        grid_norm = np.zeros_like(grid)

    # Бинаризация: ловушки (значения >= threshold) = 1, остальное = 0
    mask = (grid_norm >= threshold).astype(np.float32)
    # Фон (NaN) остается 0 (черный)
    mask[~valid_mask] = 0

    return mask


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


def save_large_images(horizons: Dict[str, Dict[str, str]], output_dir: str, isoline_step: float = 5.0) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Конвертирует CPS файлы в PNG и сохраняет большие изображения.

    Returns:
        Dict[horizon_name] -> {
            'rgb': RGB image array,
            'grayscale': grayscale image array,
            'isolines': isolines image array,
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
            gray_img = cps_to_grayscale(grid, invert=False)
            # Поворачиваем на 180 градусов
            gray_img = np.rot90(gray_img, k=2)
            gray_path = os.path.join(output_dir, f'x_structuralBlackWhite_{horizon_name}.png')
            save_png(gray_img, gray_path)
            print(f"  Saved Grayscale: {gray_path} (shape={gray_img.shape})")
            images_data[horizon_name]['grayscale'] = gray_img

            # Конвертируем в изолинии
            isolines_img = cps_to_isolines(grid, step=isoline_step)
            # Поворачиваем на 180 градусов
            isolines_img = np.rot90(isolines_img, k=2)
            isolines_path = os.path.join(output_dir, f'x_isolines_{horizon_name}.png')
            save_png(isolines_img, isolines_path)
            print(f"  Saved Isolines: {isolines_path} (shape={isolines_img.shape})")
            images_data[horizon_name]['isolines'] = isolines_img

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


def split_into_tiles(images_data: Dict[str, Dict[str, np.ndarray]],
                     output_dir: str,
                     tile_width: int = None,
                     tile_height: int = None,
                     overlap_ratio: float = None,
                     min_traps_pixels: int = None,
) -> List[str]:
    """
    Разбивает большие изображения на тайлы с перекрытием.

    Для каждого горизонта создаются тайлы с одинаковыми координатами
    для всех типов изображений (rgb, grayscale, isolines, traps).

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
    min_traps_pixels = min_traps_pixels if min_traps_pixels is not None else settings.MIN_NUM_PIXS_OF_TRAPS_IN_TILES

    # Вычисляем stride (шаг) с учетом перекрытия
    stride_h = int(tile_height * (1 - overlap_ratio))
    stride_w = int(tile_width * (1 - overlap_ratio))

    # Гарантируем минимальный шаг хотя бы 1 пиксель
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved_files = []
    skipped_tiles = 0 # Счетчик пропущенных пустых тайлов

    for horizon_name, images in images_data.items():
        print(f"\nSplitting horizon {horizon_name} into tiles ({tile_width}x{tile_height}, overlap={overlap_ratio*100:.0f}%)...")

        # Получаем размеры изображений (все должны быть одинаковыми)
        rgb_img = images.get('rgb')
        grayscale_img = images.get('grayscale')
        isolines_img = images.get('isolines')
        traps_img = images.get('traps')

        # Используем rgb для определения размеров
        if rgb_img is not None:
            h, w = rgb_img.shape[:2]
        elif grayscale_img is not None:
            h, w = grayscale_img.shape[:2]
        elif isolines_img is not None:
            h, w = isolines_img.shape[:2]
        elif traps_img is not None:
            h, w = traps_img.shape[:2]
        else:
            print(f"  No images found for horizon {horizon_name}, skipping...")
            continue

        print(f"  Image size: {h}x{w}")
        print(f"  Stride: {stride_h}x{stride_w}")

        if h <= tile_height and w <= tile_width:
            # Проверяем, есть ли достаточное количество ловушек на ВСЕЙ карте
            if traps_img is not None and np.sum(traps_img > 128) < min_traps_pixels:
                print(f"  Skipping horizon {horizon_name}: not enough trap pixels ({np.sum(traps_img > 128)} < {min_traps_pixels})")
                skipped_tiles += 1
                continue # Пропускаем ВЕСЬ горизонт (все его каналы)

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

            # Сохраняем isolines тайл
            if isolines_img is not None:
                tile_iso = pad_image(isolines_img, tile_height, tile_width)
                iso_tile_path = os.path.join(output_dir, f'{tile_prefix}x_isolines_{horizon_name}.png')
                save_png(tile_iso, iso_tile_path)
                saved_files.append(iso_tile_path)

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

        print(f"  Number of potential tiles: {n_tiles_h} x {n_tiles_w} = {n_tiles_h * n_tiles_w}")

        saved_tile_count = 0

        for row in range(n_tiles_h):
            for col in range(n_tiles_w):
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

                # ВАЖНО: Сначала проверяем ловушки, и только потом решаем сохранять ли тайл
                if traps_img is not None:
                    tile_traps_raw = traps_img[y_start:y_end, x_start:x_end]
                    
                    # Если ловушек в тайле меньше порога - пропускаем ВЕСЬ тайл (все каналы)
                    if np.sum(tile_traps_raw > 128) < min_traps_pixels:
                        skipped_tiles += 1
                        continue

                # Генерируем префикс только если тайл прошел проверку
                saved_tile_count += 1
                tile_prefix = f"{saved_tile_count:03d}_"

                # Сохраняем rgb тайл
                if rgb_img is not None:
                    tile_rgb = rgb_img[y_start:y_end, x_start:x_end]
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

                # Сохраняем isolines тайл
                if isolines_img is not None:
                    tile_iso = isolines_img[y_start:y_end, x_start:x_end]
                    tile_iso = pad_image(tile_iso, tile_height, tile_width)
                    iso_tile_path = os.path.join(output_dir, f'{tile_prefix}x_isolines_{horizon_name}.png')
                    save_png(tile_iso, iso_tile_path)
                    saved_files.append(iso_tile_path)

                # Сохраняем traps тайл
                if traps_img is not None:
                    tile_traps = pad_image(tile_traps_raw, tile_height, tile_width) # Используем уже нарезанный кусок
                    traps_tile_path = os.path.join(output_dir, f'{tile_prefix}y_traps_{horizon_name}.png')
                    save_png(tile_traps, traps_tile_path)
                    saved_files.append(traps_tile_path)

        print(f"  Created {saved_tile_count} tile sets for horizon {horizon_name} (skipped {n_tiles_h * n_tiles_w - saved_tile_count} empty tiles)")

    print(f"\nTotal skipped empty tiles across all horizons: {skipped_tiles}")
    return saved_files
