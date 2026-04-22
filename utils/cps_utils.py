import numpy as np
import warnings
from typing import Tuple, Dict, List
import os
import re
from pathlib import Path
import cv2
import scipy.ndimage as ndimage

from settings import settings


def pad_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Добавляет паддинг до target_h x target_w нулями."""
    curr_h, curr_w = img.shape[0], img.shape[1]
    if curr_h > target_h or curr_w > target_w:
        raise ValueError(f"Image {curr_h}x{curr_w} exceeds target {target_h}x{target_w}")
    
    pad_top = (target_h - curr_h) // 2
    pad_bottom = target_h - curr_h - pad_top
    pad_left = (target_w - curr_w) // 2
    pad_right = target_w - curr_w - pad_left
    
    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )
    return padded


def save_png(img_array: np.ndarray, path: str):
    from PIL import Image
    if len(img_array.shape) == 2:
        img = Image.fromarray(img_array, mode='L')
    else:
        img = Image.fromarray(img_array, mode='RGB')
    img.save(path)


def extract_horizon_name(filename: str) -> str:
    pattern = r'^(x_structuralNOisoline_|y_traps_)(.+)$'
    match = re.match(pattern, filename)
    if match:
        return match.group(2)
    return None


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


def cps_to_grayscale(
    grid: np.ndarray, 
    invert: bool = False, 
    global_vmin: float = None, 
    global_vmax: float = None
) -> np.ndarray:
    """
    Конвертирует CPS грид в черно-белое изображение.
    
    Args:
        grid: 2D numpy array
        invert: Если True — инвертируем (черный = макс, белый = мин)
        global_vmin: Глобальный минимум (если нужен, например, для тайлов)
        global_vmax: Глобальный максимум (если нужен, например, для тайлов)
    
    Returns:
        gray: (H, W) uint8 array
    """
    valid_mask = ~np.isnan(grid)
    
    if valid_mask.sum() == 0:
        return np.zeros(grid.shape, dtype=np.uint8)
    
    # Используем глобальные min/max, если они переданы, иначе вычисляем локальные
    vmin = global_vmin if global_vmin is not None else np.nanmin(grid)
    vmax = global_vmax if global_vmax is not None else np.nanmax(grid)
    
    # Нормализация к [0, 255]
    grid_norm = (grid - vmin) / (vmax - vmin + 1e-8)
    # Клиппинг обязателен при глобальной нормализации, так как значения тайла могут выходить за пределы global_vmin/global_vmax из-за NaN
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


def cps_to_closed_mask(grid: np.ndarray, step: float = 5.0, min_area: int = settings.CLOSED_ISO_MIN_AREA) -> np.ndarray:
    """
    Генерирует маску замкнутых контуров (ловушек) напрямую из CPS грида.
    
    В отличие от поиска по картинке изолиний, этот метод математически точен:
    он не страдает от артефактов пикселизации (толстых линий на крутых склонах) 
    и сразу выдает сплошные белые пятна без внутренних "царапин" изолиний.
    
    Args:
        grid: 2D numpy array (с NaN в качестве пустот)
        step: Шаг изолиний в метрах (по умолчанию 5, должен совпадать с шагом генерации изолиний)
        min_area: Минимальная площадь замкнутого контура в пикселях (меньше считаются шумом)
    
    Returns:
        mask: (H, W) float32 array (1.0 - замкнутая область/ловушка, 0.0 - остальное)
    """
    valid_mask = ~np.isnan(grid)
    if valid_mask.sum() == 0:
        return np.zeros(grid.shape, dtype=np.float32)
    
    # Находим границу валидной области (1 пиксель по краю карты)
    eroded_mask = ndimage.binary_erosion(valid_mask)
    boundary_mask = valid_mask & ~eroded_mask
    
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    
    # Вычисляем уровни изолиний (точно так же, как в cps_to_isolines)
    start_bound = np.floor(vmin / step) * step
    end_bound = np.ceil(vmax / step) * step
    levels = np.arange(start_bound, end_bound + step, step)
    
    # Итоговая маска замкнутых областей
    closed_mask = np.zeros(grid.shape, dtype=bool)
    
    # Временно заменяем NaN для корректной работы условия >= level
    grid_filled = np.copy(grid)
    grid_filled[~valid_mask] = vmin - 1000 
    
    for level in levels:
        # Бинарная маска: 1 там, где поверхность выше или равна уровню
        binary_mask = (grid_filled >= level) & valid_mask
        
        # Находим связные компоненты (4-связность, чтобы диагонали не считались за проход)
        labeled_array, num_features = ndimage.label(binary_mask)
        
        # Проверяем каждый компонент
        for i in range(1, num_features + 1):
            component_mask = (labeled_array == i)
            
            # Если компонент слишком мелкий (точечный шум), пропускаем его
            if component_mask.sum() < min_area:
                continue
            
            # Если компонент касается границы карты -> он разомкнут, пропускаем
            if np.any(component_mask & boundary_mask):
                continue
            
            # Если не касается -> замкнут, добавляем к итоговой маске
            closed_mask |= component_mask

    # Убираем черные точки (микро-впадины или NaN) внутри замкнутых белых областей
    closed_mask = ndimage.binary_fill_holes(closed_mask)

    return closed_mask.astype(np.float32)


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


def save_large_images(horizons: Dict[str, Dict[str, str]], output_dir: str, isoline_step: float = 5.0) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Конвертирует CPS файлы в PNG и сохраняет большие изображения.

    Returns:
        Dict[horizon_name] -> {
            'rgb': RGB image array,
            'grayscale': grayscale image array,
            'isolines': isolines image array,
            'closed_isolines': closed_isolines image array,
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
            rgb_path = os.path.join(output_dir, f'x_structuralNOisoline_{horizon_name}.png')
            save_png(rgb_img, rgb_path)
            print(f"  Saved RGB: {rgb_path} (shape={rgb_img.shape})")
            images_data[horizon_name]['rgb'] = rgb_img

            # Конвертируем в grayscale (для depth_norm)
            gray_img = cps_to_grayscale(grid, invert=False)
            gray_path = os.path.join(output_dir, f'x_structuralBlackWhite_{horizon_name}.png')
            save_png(gray_img, gray_path)
            print(f"  Saved Grayscale: {gray_path} (shape={gray_img.shape})")
            images_data[horizon_name]['grayscale'] = gray_img

            # Конвертируем в изолинии
            isolines_img = cps_to_isolines(grid, step=isoline_step)
            isolines_path = os.path.join(output_dir, f'x_isolines_{horizon_name}.png')
            save_png(isolines_img, isolines_path)
            print(f"  Saved Isolines: {isolines_path} (shape={isolines_img.shape})")
            images_data[horizon_name]['isolines'] = isolines_img

            # Конвертируем в маску замкнутых изолиний
            closed_mask = cps_to_closed_mask(grid, step=isoline_step)
            closed_img = (closed_mask * 255).astype(np.uint8) # В uint8 для PNG
            closed_path = os.path.join(output_dir, f'x_closedIsolines_{horizon_name}.png')
            save_png(closed_img, closed_path)
            print(f"  Saved Closed Isolines: {closed_path} (shape={closed_img.shape})")
            images_data[horizon_name]['closed_isolines'] = closed_img

        # Загружаем traps
        if 'traps' in files:
            print(f"  Loading traps: {files['traps']}")
            grid, meta = read_cps_grid(files['traps'])

            # Для traps используем бинаризацию
            traps_img = cps_to_binary_mask(grid)
            traps_img = (traps_img * 255).astype(np.uint8)
            traps_path = os.path.join(output_dir, f'y_traps_{horizon_name}.png')
            save_png(traps_img, traps_path)
            print(f"  Saved Traps: {traps_path} (shape={traps_img.shape})")
            images_data[horizon_name]['traps'] = traps_img

    return images_data


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


def split_cps_grids_into_tiles(
    horizons: Dict[str, Dict[str, str]],
    output_dir: str,
    tile_width: int = None,
    tile_height: int = None,
    overlap_ratio: float = None,
    min_traps_pixels: int = None,
    isoline_step: float = 5.0
) -> List[str]:
    """
    Загружает CPS гриды, нарезает их на тайлы и конвертирует в PNG.
    RGB использует ЛОКАЛЬНУЮ палитру, Depth_norm использует ГЛОБАЛЬНУЮ палитру всего грида.
    """
    tile_width = tile_width or settings.TARGET_WIDTH
    tile_height = tile_height or settings.TARGET_HEIGHT
    overlap_ratio = overlap_ratio if overlap_ratio is not None else settings.TILE_OVERLAP_RATIO
    min_traps_pixels = min_traps_pixels if min_traps_pixels is not None else settings.MIN_NUM_PIXS_OF_TRAPS_IN_TILES

    stride_h = max(1, int(tile_height * (1 - overlap_ratio)))
    stride_w = max(1, int(tile_width * (1 - overlap_ratio)))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_files = []
    skipped_tiles = 0

    for horizon_name, files in horizons.items():
        print(f"\nProcessing horizon: {horizon_name}")

        struct_grid = None
        global_vmin, global_vmax = None, None  # Инициализация глобальных min/max
        
        if 'structural' in files:
            struct_grid, _ = read_cps_grid(files['structural'])
            # Вычисляем глобальные min и max для всего горизонта один раз
            valid_mask = ~np.isnan(struct_grid)
            if valid_mask.sum() > 0:
                global_vmin = np.nanmin(struct_grid)
                global_vmax = np.nanmax(struct_grid)
                print(f"  Global depth range: [{global_vmin:.1f}, {global_vmax:.1f}]")

        traps_grid = None
        if 'traps' in files:
            traps_grid, _ = read_cps_grid(files['traps'])

        if struct_grid is None and traps_grid is None:
            continue

        # Определяем размер по структурной карте
        h, w = struct_grid.shape if struct_grid is not None else traps_grid.shape
        print(f"  Grid size: {h}x{w}")

        # Заблаговременно конвертируем ловушки, чтобы быстро нарезать и считать пиксели
        traps_mask_full = None
        if traps_grid is not None:
            traps_mask_full = (cps_to_binary_mask(traps_grid) * 255).astype(np.uint8)

        n_tiles_h = max(1, (h - tile_height) // stride_h + 1)
        n_tiles_w = max(1, (w - tile_width) // stride_w + 1)
        if (n_tiles_h - 1) * stride_h + tile_height < h: n_tiles_h += 1
        if (n_tiles_w - 1) * stride_w + tile_width < w: n_tiles_w += 1

        print(f"  Potential tiles: {n_tiles_h} x {n_tiles_w} = {n_tiles_h * n_tiles_w}")
        saved_tile_count = 0

        for row in range(n_tiles_h):
            for col in range(n_tiles_w):
                y_start = row * stride_h
                x_start = col * stride_w
                y_end = min(y_start + tile_height, h)
                x_end = min(x_start + tile_width, w)

                if y_end == h: y_start = max(0, y_end - tile_height)
                if x_end == w: x_start = max(0, x_end - tile_width)

                # Проверяем ловушки
                if traps_mask_full is not None:
                    tile_traps_raw = traps_mask_full[y_start:y_end, x_start:x_end]
                    if np.sum(tile_traps_raw > 128) < min_traps_pixels:
                        skipped_tiles += 1
                        continue
                else:
                    tile_traps_raw = None

                saved_tile_count += 1
                tile_prefix = f"{saved_tile_count:03d}_"

                # Нарезка и конвертация структурного грида
                if struct_grid is not None:
                    tile_struct = struct_grid[y_start:y_end, x_start:x_end]

                    # RGB с ЛОКАЛЬНОЙ нормализацией
                    tile_rgb = cps_to_rgb(tile_struct, cmap_name='purple_jet')
                    
                    # Depth_norm с ГЛОБАЛЬНОЙ нормализацией грида
                    tile_gray = cps_to_grayscale(
                        tile_struct, 
                        global_vmin=global_vmin, 
                        global_vmax=global_vmax
                    )
                    
                    tile_isolines = cps_to_isolines(tile_struct, step=isoline_step)
                    tile_closed_mask = cps_to_closed_mask(tile_struct, step=isoline_step)
                    tile_closed = (tile_closed_mask * 255).astype(np.uint8)

                    # Сохраняем и добавляем пути в список
                    rgb_path = os.path.join(output_dir, f'{tile_prefix}x_structuralNOisoline_{horizon_name}.png')
                    save_png(pad_image(tile_rgb, tile_height, tile_width), rgb_path)
                    saved_files.append(rgb_path)

                    gray_path = os.path.join(output_dir, f'{tile_prefix}x_structuralBlackWhite_{horizon_name}.png')
                    save_png(pad_image(tile_gray, tile_height, tile_width), gray_path)
                    saved_files.append(gray_path)

                    iso_path = os.path.join(output_dir, f'{tile_prefix}x_isolines_{horizon_name}.png')
                    save_png(pad_image(tile_isolines, tile_height, tile_width), iso_path)
                    saved_files.append(iso_path)

                    closed_path = os.path.join(output_dir, f'{tile_prefix}x_closedIsolines_{horizon_name}.png')
                    save_png(pad_image(tile_closed, tile_height, tile_width), closed_path)
                    saved_files.append(closed_path)

                if tile_traps_raw is not None:
                    traps_path = os.path.join(output_dir, f'{tile_prefix}y_traps_{horizon_name}.png')
                    save_png(pad_image(tile_traps_raw, tile_height, tile_width), traps_path)
                    saved_files.append(traps_path)

        print(f"  Saved {saved_tile_count} tiles (skipped {skipped_tiles} empty)")

    print(f"\nTotal skipped tiles: {skipped_tiles}")
    return saved_files


def clean_cps_filenames(cps_dir: str, suffixes_to_remove: list = None):
    """
    Удаляет указанные суффиксы из имен файлов в директории cps_dir.
    
    Args:
        cps_dir: Путь к директории с CPS файлами
        suffixes_to_remove: Список суффиксов для удаления (например, ['.cps3', '-UNIQ1'])
    """
    if suffixes_to_remove is None:
        suffixes_to_remove = [".cps3", "-UNIQ1"]

    if not os.path.exists(cps_dir):
        print(f"Directory not found: {cps_dir}. Skipping filename cleaning.")
        return

    print(f"Cleaning filenames in {cps_dir}...")
    print(f"Suffixes to remove: {suffixes_to_remove}")
    
    renamed_count = 0

    for filename in os.listdir(cps_dir):
        old_path = os.path.join(cps_dir, filename)

        if not os.path.isfile(old_path):
            continue

        new_name = filename

        # Удаляем все суффиксы из имени
        for suffix in suffixes_to_remove:
            if new_name.endswith(suffix):
                new_name = new_name[:-len(suffix)]

        # Если имя изменилось, переименовываем
        if new_name != filename:
            new_path = os.path.join(cps_dir, new_name)
            
            # Защита от перезаписи существующих файлов
            if os.path.exists(new_path):
                print(f"  WARNING: Cannot rename '{filename}' -> '{new_name}'. File already exists!")
                continue
                
            print(f"  Renaming: {filename} -> {new_name}")
            os.rename(old_path, new_path)
            renamed_count += 1

    print(f"Filename cleaning done. Renamed {renamed_count} files.\n")
