import numpy as np
import warnings
from typing import Tuple, Dict, List
import os
import re
from pathlib import Path
import cv2
import scipy.ndimage as ndimage

from settings import settings
from utils.images_utils import pad_image


def read_cps_grid(file_path: str, vertical_flip: bool = False) -> Tuple[np.ndarray, dict]:
    vertical_flip = vertical_flip if vertical_flip is not None else settings.CPS_VERTICAL_FLIP
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    nx = ny = None
    xmin = xmax = ymin = ymax = None
    null_value = settings.CPS_NULL_VALUE
    data_start = None
    
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
        
        elif line.startswith("->"):
            data_start = i + 1
            break
    
    if None in (nx, ny, xmin, xmax, ymin, ymax):
        raise ValueError(f"Failed to parse CPS header: {file_path}")
    
    values = []
    for line in lines[data_start:]:
        values.extend([float(x) for x in line.split()])
    
    values = np.array(values, dtype=np.float32)
    
    expected = nx * ny
    if len(values) != expected:
        warnings.warn(f"CPS {file_path}: expected {expected}, got {len(values)}. Truncating.")
        values = values[:expected]
    
    grid = values.reshape((ny, nx), order='F')
    
    if vertical_flip:
        grid = np.flipud(grid)
    
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


def write_cps_grid(grid: np.ndarray, meta: dict, file_path: str,
                   null_value: float = None, values_per_line: int = 5,
                   value_format: str = '%.7g', label: str = 'Predicted traps mask') -> None:
    """
    Обратная операция к read_cps_grid: пишет 2D-массив (ny, nx) в CPS-3 ASCII-грид.

    Геометрия (nx, ny, bbox, null) берётся из meta — поэтому выходной грид идеально
    накладывается на исходный (структурный) грид. Данные пишутся в Fortran-порядке
    (по столбцам), NaN помечается как null-значение — ровно так, как ожидает read_cps_grid.

    Заголовок воспроизводит структуру реальных файлов: FSASCI (null), FSATTR,
    FSLIMI (xmin xmax ymin ymax zmin zmax), FSNROW ny nx, FSXINC dx dy, ->маркер.

    Args:
        grid: 2D массив формы (ny, nx). NaN = «нет данных» (станет null в файле).
        meta: метаданные референсного грида (как из read_cps_grid): nx, ny, xmin,
            xmax, ymin, ymax, null_value.
        file_path: куда сохранить CPS-файл.
        null_value: null-значение (default: meta['null_value'], иначе settings.CPS_NULL_VALUE).
        values_per_line: чисел в строке блока данных (как в исходных файлах — 5).
        value_format: формат вещественных чисел (default '%.7g').
        label: текст после маркера '->' (не влияет на чтение, только для человека).
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"write_cps_grid: ожидается 2D массив, получено shape={grid.shape}")

    ny_meta, nx_meta = int(meta['ny']), int(meta['nx'])
    ny, nx = grid.shape
    if (ny, nx) != (ny_meta, nx_meta):
        raise ValueError(
            f"write_cps_grid: форма грида {(ny, nx)} не совпадает с meta {(ny_meta, nx_meta)}"
        )

    if null_value is None:
        null_value = meta.get('null_value', settings.CPS_NULL_VALUE)

    xmin, xmax = float(meta['xmin']), float(meta['xmax'])
    ymin, ymax = float(meta['ymin']), float(meta['ymax'])

    # z-диапазон — только для человекочитаемости (читалку интересуют первые 4 числа FSLIMI)
    valid = ~np.isnan(grid)
    if valid.any():
        zmin, zmax = float(np.nanmin(grid)), float(np.nanmax(grid))
    else:
        zmin = zmax = 0.0

    # Размер ячейки (node-centered): dx = (xmax - xmin) / (nx - 1)
    xinc = (xmax - xmin) / (nx - 1) if nx > 1 else 0.0
    yinc = (ymax - ymin) / (ny - 1) if ny > 1 else 0.0

    lines = []
    lines.append(f"FSASCI 0 1 COMPUTED 0 {null_value:.6E}")
    lines.append("FSATTR 0 0")
    lines.append(f"FSLIMI {xmin:.6f} {xmax:.6f} {ymin:.6f} {ymax:.6f} {zmin:.6f} {zmax:.6f}")
    lines.append(f"FSNROW {ny} {nx}")
    lines.append(f"FSXINC {xinc:.6f} {yinc:.6f}")
    lines.append(f"->MSMODL: {label}")

    # Данные в Fortran-порядке, NaN -> null
    flat = grid.astype(np.float64).flatten(order='F')
    flat = np.where(np.isnan(flat), null_value, flat)

    for i in range(0, flat.size, values_per_line):
        chunk = flat[i:i + values_per_line]
        lines.append(" ".join(value_format % float(v) for v in chunk))

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def resample_grid_to_reference(src_grid: np.ndarray, src_meta: dict, ref_meta: dict) -> np.ndarray:
    """
    Переносит исходный грид на геометрию референсного грида (структурной карты)
    по мировым координатам.

    Главная цель: привести faults/traps к той же пиксельной сетке, что и structural,
    чтобы все карты одного горизонта (rgb/depth/isolines/faults/traps/mask) легли
    пиксель-в-пиксель и корректно накладывались друг на друга перед обучением.

    Гриды CPS node-centered:
        col j -> X = xmin + j * (xmax - xmin) / (nx - 1)
        row i -> Y = ymax - i * (ymax - ymin) / (ny - 1)   (row 0 = верх = ymax)

    Args:
        src_grid: 2D массив исходного грида (с NaN для невалидных ячеек)
        src_meta: метаданные исходного грида (nx, ny, xmin, xmax, ymin, ymax)
        ref_meta: метаданные референсного грида (структурная карта)

    Returns:
        2D массив формы (ref_meta['ny'], ref_meta['nx']), выровненный по координатам.
        Точки за пределами охвата src_grid помечаются NaN.

    Примечание: используется INTER_NEAREST. При даунсэмплинге бинарной маски
    (например, traps 25м -> 50м) тонкие элементы могут прореживаться; это компромисс
    в пользу точного геометрического выравнивания узел-в-узел.
    """
    ny_r, nx_r = ref_meta['ny'], ref_meta['nx']

    # Мировые координаты узлов референса
    X = ref_meta['xmin'] + np.arange(nx_r) * (ref_meta['xmax'] - ref_meta['xmin']) / (nx_r - 1)
    Y = ref_meta['ymax'] - np.arange(ny_r) * (ref_meta['ymax'] - ref_meta['ymin']) / (ny_r - 1)
    YY, XX = np.meshgrid(Y, X, indexing='ij')

    # Те же точки в индексах исходного грида
    fj = ((XX - src_meta['xmin']) / (src_meta['xmax'] - src_meta['xmin']) * (src_meta['nx'] - 1)).astype(np.float32)
    fi = ((src_meta['ymax'] - YY) / (src_meta['ymax'] - src_meta['ymin']) * (src_meta['ny'] - 1)).astype(np.float32)

    # Ремапим и сами значения, и маску валидности (чтобы корректно отметить NaN)
    valid = (~np.isnan(src_grid)).astype(np.float32)
    g = np.nan_to_num(src_grid, nan=0.0).astype(np.float32)

    resampled = cv2.remap(g, fj, fi, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid_resampled = cv2.remap(valid, fj, fi, cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Всё, что вышло за охват src (в т.ч. out-of-bounds референса) -> NaN
    resampled[valid_resampled < 0.5] = np.nan

    return resampled


def cps_to_rgb(grid: np.ndarray, cmap_name: str = 'purple_jet') -> np.ndarray:
    from matplotlib.colors import LinearSegmentedColormap
    
    if cmap_name == 'purple_jet':
        colors = [
            '#4B0082', '#0000FF', '#00FFFF', '#00FF00', 
            '#FFFF00', '#FF8000', '#FF0000'
        ]
        cmap = LinearSegmentedColormap.from_list('purple_jet', colors, N=256)
    else:
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(cmap_name)
    
    valid_mask = ~np.isnan(grid)
    if valid_mask.sum() == 0:
        return np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    grid_norm = (grid - vmin) / (vmax - vmin + 1e-8)
    grid_norm = np.clip(grid_norm, 0, 1)
    
    rgb_float = cmap(grid_norm)[:, :, :3]
    rgb = (rgb_float * 255).astype(np.uint8)
    rgb[~valid_mask] = 0
    
    return rgb


def cps_to_depth_norm_float(grid: np.ndarray) -> np.ndarray:
    """
    Конвертирует CPS грид глубин в нормализованный массив float32 [0, 1].
    NaN (края карты) заполняются 0.0.
    Разломы будут обнулены позже в save_large_images.
    """
    valid_mask = ~np.isnan(grid)
    if valid_mask.sum() == 0:
        return np.zeros(grid.shape, dtype=np.float32)
    
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    grid_norm = (grid - vmin) / (vmax - vmin + 1e-8)
    grid_norm = np.clip(grid_norm, 0, 1)
    
    # Заменяем NaN (вне карты) на 0.0
    grid_norm = np.nan_to_num(grid_norm, nan=0.0)
    
    return grid_norm.astype(np.float32)


def cps_to_isolines(grid: np.ndarray, step: float = 5.0) -> np.ndarray:
    valid_mask = ~np.isnan(grid)
    if valid_mask.sum() == 0:
        return np.zeros(grid.shape, dtype=np.uint8)
    
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    
    start_bound = np.floor(vmin / step) * step
    end_bound = np.ceil(vmax / step) * step
    levels = np.arange(start_bound, end_bound + step, step)
    
    isolines_img = np.zeros(grid.shape, dtype=np.uint8)
    
    grid_filled = np.copy(grid)
    grid_filled[~valid_mask] = vmin - 1000 
    
    for level in levels:
        binary_mask = (grid_filled >= level).astype(np.uint8)
        binary_mask[~valid_mask] = 0
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(isolines_img, contours, -1, 255, thickness=1)
        
    return isolines_img


def cps_to_binary_mask(grid: np.ndarray, threshold: float = None) -> np.ndarray:
    if threshold is None:
        threshold = settings.BINARY_THRESHOLD

    valid_mask = ~np.isnan(grid)

    if valid_mask.sum() > 0:
        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        grid_norm = (grid - vmin) / (vmax - vmin + 1e-8)
        grid_norm = np.clip(grid_norm, 0, 1) * 255
    else:
        grid_norm = np.zeros_like(grid)

    grid_norm = np.nan_to_num(grid_norm, nan=0.0)

    mask = (grid_norm >= threshold).astype(np.float32)
    mask[~valid_mask] = 0

    return mask


def save_png(img_array: np.ndarray, path: str):
    from PIL import Image

    if len(img_array.shape) == 2:
        img = Image.fromarray(img_array, mode='L')
    else:
        img = Image.fromarray(img_array, mode='RGB')

    img.save(path)


def save_large_images(horizons: Dict[str, Dict[str, str]], output_dir: str, isoline_step: float = 5.0) -> Dict[str, Dict[str, np.ndarray]]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    images_data = {}

    for horizon_name, files in horizons.items():
        print(f"\nProcessing horizon: {horizon_name}")
        images_data[horizon_name] = {}

        reference_shape = None
        meta = None  # метаданные структурной карты — референс геометрии для faults/traps

        # 1. Структурная карта — референс: её cell size + bbox наследуют faults и traps,
        #    чтобы все типы карт горизонта легли на одну пиксельную сетку.
        structural_grid = None
        if 'structural' in files:
            print(f"  Loading structural: {files['structural']}")
            structural_grid, meta = read_cps_grid(files['structural'])

        # 2. Разломы: перегоняем на сетку структурной карты по мировым координатам
        faults_img_original = None
        if 'faults' in files:
            print(f"  Loading faults: {files['faults']}")
            faults_grid, faults_meta = read_cps_grid(files['faults'])
            if structural_grid is not None:
                faults_grid = resample_grid_to_reference(faults_grid, faults_meta, meta)
                print(f"  Resampled faults to structural grid geometry")
            faults_img_original = (cps_to_binary_mask(faults_grid) * 255).astype(np.uint8)

        # 3. Из структурной карты строим rgb/depth/isolines и вырезаем разломы
        if structural_grid is not None:
            rgb_img = cps_to_rgb(structural_grid, cmap_name='purple_jet')
            depth_float_img = cps_to_depth_norm_float(structural_grid)
            isolines_img = cps_to_isolines(structural_grid, step=isoline_step)

            reference_shape = rgb_img.shape[:2]
            print(f"  Reference shape set to: {reference_shape}")

            if faults_img_original is not None:
                # faults уже на сетке structural, resize остаётся как страховочный no-op
                if faults_img_original.shape != reference_shape:
                    faults_resized_for_cut = cv2.resize(
                        faults_img_original,
                        (reference_shape[1], reference_shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    faults_resized_for_cut = faults_img_original

                fault_pixels = faults_resized_for_cut > 128

                rgb_img[fault_pixels] = 0
                print(f"  Cut interpolated data from RGB at faults")

                depth_float_img[fault_pixels] = 0.0
                print(f"  Cut interpolated data from Depth at faults")

                isolines_img[fault_pixels] = 0
                print(f"  Cut isolines at faults")

            rgb_path = os.path.join(output_dir, f'x_structuralNOisoline_{horizon_name}.png')
            save_png(rgb_img, rgb_path)
            images_data[horizon_name]['rgb'] = rgb_img

            depth_path = os.path.join(output_dir, f'x_structuralBlackWhite_{horizon_name}.npy')
            np.save(depth_path, depth_float_img)
            images_data[horizon_name]['grayscale'] = depth_float_img

            isolines_path = os.path.join(output_dir, f'x_isolines_{horizon_name}.png')
            save_png(isolines_img, isolines_path)
            images_data[horizon_name]['isolines'] = isolines_img

        # 4. Сохраняем маску разломов (уже выровнена на сетку structural)
        if faults_img_original is not None:
            faults_img_final = faults_img_original
            if reference_shape is not None and faults_img_original.shape != reference_shape:
                print(f"  Resizing faults image from {faults_img_original.shape} to {reference_shape} for final save")
                faults_img_final = cv2.resize(
                    faults_img_original,
                    (reference_shape[1], reference_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            faults_path = os.path.join(output_dir, f'x_faults_{horizon_name}.png')
            save_png(faults_img_final, faults_path)
            print(f"  Saved Faults: {faults_path} (shape={faults_img_final.shape})")
            images_data[horizon_name]['faults'] = faults_img_final

        # 5. Ловушки: перегоняем на сетку структурной карты по мировым координатам
        if 'traps' in files:
            print(f"  Loading traps: {files['traps']}")
            traps_grid, traps_meta = read_cps_grid(files['traps'])
            if structural_grid is not None:
                traps_grid = resample_grid_to_reference(traps_grid, traps_meta, meta)
                print(f"  Resampled traps to structural grid geometry")
            traps_img = (cps_to_binary_mask(traps_grid) * 255).astype(np.uint8)

            if reference_shape is not None and traps_img.shape != reference_shape:
                print(f"  Resizing traps image from {traps_img.shape} to {reference_shape} for final save")
                traps_img = cv2.resize(
                    traps_img,
                    (reference_shape[1], reference_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            traps_path = os.path.join(output_dir, f'y_traps_{horizon_name}.png')
            save_png(traps_img, traps_path)
            print(f"  Saved Traps: {traps_path} (shape={traps_img.shape})")
            images_data[horizon_name]['traps'] = traps_img

    return images_data


def extract_horizon_name(filename: str) -> str:
    pattern = r'^(x_structuralNOisoline_|y_traps_|x_faults_)(.+)$'
    match = re.match(pattern, filename)
    if match:
        return match.group(2)
    return None


def find_cps_files(cps_dir: str) -> Dict[str, Dict[str, str]]:
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
                
        elif filename.startswith('x_faults_'):
            horizon_name = extract_horizon_name(filename)
            if horizon_name:
                if horizon_name not in horizons:
                    horizons[horizon_name] = {}
                horizons[horizon_name]['faults'] = os.path.join(cps_dir, filename)

    return horizons


def split_into_tiles(images_data: Dict[str, Dict[str, np.ndarray]],
                     output_dir: str,
                     tile_width: int = None,
                     tile_height: int = None,
                     overlap_ratio: float = None,
                     min_traps_pixels: int = None,
) -> List[str]:
    tile_width = tile_width or settings.TARGET_WIDTH
    tile_height = tile_height or settings.TARGET_HEIGHT
    overlap_ratio = overlap_ratio if overlap_ratio is not None else settings.TILE_OVERLAP_RATIO
    min_traps_pixels = min_traps_pixels if min_traps_pixels is not None else settings.MIN_NUM_PIXS_OF_TRAPS_IN_TILES

    stride_h = int(tile_height * (1 - overlap_ratio))
    stride_w = int(tile_width * (1 - overlap_ratio))

    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved_files = []
    skipped_tiles = 0

    for horizon_name, images in images_data.items():
        print(f"\nSplitting horizon {horizon_name} into tiles ({tile_width}x{tile_height}, overlap={overlap_ratio*100:.0f}%)...")

        rgb_img = images.get('rgb')
        grayscale_img = images.get('grayscale')
        isolines_img = images.get('isolines')
        faults_img = images.get('faults')
        traps_img = images.get('traps')

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
            if traps_img is not None and np.sum(traps_img > 128) < min_traps_pixels:
                print(f"  Skipping horizon {horizon_name}: not enough trap pixels ({np.sum(traps_img > 128)} < {min_traps_pixels})")
                skipped_tiles += 1
                continue

            tile_index = 1
            tile_prefix = f"{tile_index:03d}_"

            if rgb_img is not None:
                tile_rgb = pad_image(rgb_img, tile_height, tile_width)
                rgb_tile_path = os.path.join(output_dir, f'{tile_prefix}x_structuralNOisoline_{horizon_name}.png')
                save_png(tile_rgb, rgb_tile_path)
                saved_files.append(rgb_tile_path)

            if grayscale_img is not None:
                tile_gray = pad_image(grayscale_img, tile_height, tile_width)
                gray_tile_path = os.path.join(output_dir, f'{tile_prefix}x_structuralBlackWhite_{horizon_name}.npy')
                np.save(gray_tile_path, tile_gray)
                saved_files.append(gray_tile_path)

            if isolines_img is not None:
                tile_iso = pad_image(isolines_img, tile_height, tile_width)
                iso_tile_path = os.path.join(output_dir, f'{tile_prefix}x_isolines_{horizon_name}.png')
                save_png(tile_iso, iso_tile_path)
                saved_files.append(iso_tile_path)

            if faults_img is not None:
                tile_faults = pad_image(faults_img, tile_height, tile_width)
                faults_tile_path = os.path.join(output_dir, f'{tile_prefix}x_faults_{horizon_name}.png')
                save_png(tile_faults, faults_tile_path)
                saved_files.append(faults_tile_path)

            if traps_img is not None:
                tile_traps = pad_image(traps_img, tile_height, tile_width)
                traps_tile_path = os.path.join(output_dir, f'{tile_prefix}y_traps_{horizon_name}.png')
                save_png(tile_traps, traps_tile_path)
                saved_files.append(traps_tile_path)

            print(f"  Image is smaller than tile size, saved as 1 tile with padding")
            continue

        n_tiles_h = max(1, (h - tile_height) // stride_h + 1)
        n_tiles_w = max(1, (w - tile_width) // stride_w + 1)

        if (n_tiles_h - 1) * stride_h + tile_height < h:
            n_tiles_h += 1
        if (n_tiles_w - 1) * stride_w + tile_width < w:
            n_tiles_w += 1

        print(f"  Number of potential tiles: {n_tiles_h} x {n_tiles_w} = {n_tiles_h * n_tiles_w}")

        saved_tile_count = 0

        for row in range(n_tiles_h):
            for col in range(n_tiles_w):
                y_start = row * stride_h
                x_start = col * stride_w

                y_end = min(y_start + tile_height, h)
                x_end = min(x_start + tile_width, w)

                if y_end == h:
                    y_start = max(0, y_end - tile_height)
                if x_end == w:
                    x_start = max(0, x_end - tile_width)

                if traps_img is not None:
                    tile_traps_raw = traps_img[y_start:y_end, x_start:x_end]
                    if np.sum(tile_traps_raw > 128) < min_traps_pixels:
                        skipped_tiles += 1
                        continue

                saved_tile_count += 1
                tile_prefix = f"{saved_tile_count:03d}_"

                if rgb_img is not None:
                    tile_rgb = rgb_img[y_start:y_end, x_start:x_end]
                    tile_rgb = pad_image(tile_rgb, tile_height, tile_width)
                    rgb_tile_path = os.path.join(output_dir, f'{tile_prefix}x_structuralNOisoline_{horizon_name}.png')
                    save_png(tile_rgb, rgb_tile_path)
                    saved_files.append(rgb_tile_path)

                if grayscale_img is not None:
                    tile_gray = grayscale_img[y_start:y_end, x_start:x_end]
                    tile_gray = pad_image(tile_gray, tile_height, tile_width)
                    gray_tile_path = os.path.join(output_dir, f'{tile_prefix}x_structuralBlackWhite_{horizon_name}.npy')
                    np.save(gray_tile_path, tile_gray)
                    saved_files.append(gray_tile_path)

                if isolines_img is not None:
                    tile_iso = isolines_img[y_start:y_end, x_start:x_end]
                    tile_iso = pad_image(tile_iso, tile_height, tile_width)
                    iso_tile_path = os.path.join(output_dir, f'{tile_prefix}x_isolines_{horizon_name}.png')
                    save_png(tile_iso, iso_tile_path)
                    saved_files.append(iso_tile_path)

                if faults_img is not None:
                    tile_faults = faults_img[y_start:y_end, x_start:x_end]
                    tile_faults = pad_image(tile_faults, tile_height, tile_width)
                    faults_tile_path = os.path.join(output_dir, f'{tile_prefix}x_faults_{horizon_name}.png')
                    save_png(tile_faults, faults_tile_path)
                    saved_files.append(faults_tile_path)

                if traps_img is not None:
                    tile_traps = pad_image(tile_traps_raw, tile_height, tile_width)
                    traps_tile_path = os.path.join(output_dir, f'{tile_prefix}y_traps_{horizon_name}.png')
                    save_png(tile_traps, traps_tile_path)
                    saved_files.append(traps_tile_path)

        print(f"  Created {saved_tile_count} tile sets for horizon {horizon_name} (skipped {n_tiles_h * n_tiles_w - saved_tile_count} empty tiles)")

    print(f"\nTotal skipped empty tiles across all horizons: {skipped_tiles}")
    return saved_files


def load_existing_images(horizons: Dict[str, Dict[str, str]], full_images_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    from PIL import Image

    images_data = {}

    for horizon_name, files in horizons.items():
        print(f"  Loading existing images for horizon: {horizon_name}")
        images_data[horizon_name] = {}

        rgb_path = os.path.join(full_images_dir, f'x_structuralNOisoline_{horizon_name}.png')
        if os.path.exists(rgb_path):
            img = np.array(Image.open(rgb_path))
            images_data[horizon_name]['rgb'] = img
            print(f"    Loaded RGB: {rgb_path} (shape={img.shape})")

        # Ищем .npy для глубины
        gray_path = os.path.join(full_images_dir, f'x_structuralBlackWhite_{horizon_name}.npy')
        if os.path.exists(gray_path):
            img = np.load(gray_path)
            images_data[horizon_name]['grayscale'] = img
            print(f"    Loaded Grayscale (NPY): {gray_path} (shape={img.shape})")

        isolines_path = os.path.join(full_images_dir, f'x_isolines_{horizon_name}.png')
        if os.path.exists(isolines_path):
            img = np.array(Image.open(isolines_path))
            images_data[horizon_name]['isolines'] = img
            print(f"    Loaded Isolines: {isolines_path} (shape={img.shape})")

        faults_path = os.path.join(full_images_dir, f'x_faults_{horizon_name}.png')
        if os.path.exists(faults_path):
            img = np.array(Image.open(faults_path))
            images_data[horizon_name]['faults'] = img
            print(f"    Loaded Faults: {faults_path} (shape={img.shape})")

        traps_path = os.path.join(full_images_dir, f'y_traps_{horizon_name}.png')
        if os.path.exists(traps_path):
            img = np.array(Image.open(traps_path))
            images_data[horizon_name]['traps'] = img
            print(f"    Loaded Traps: {traps_path} (shape={img.shape})")

    return images_data


def clean_cps_filenames(cps_dir: str, suffixes_to_remove: list = None):
    if suffixes_to_remove is None:
        suffixes_to_remove = [".cps3", ".cps", ".grd", "-UNIQ1", "-UNIQ"]

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
        changed = True
        while changed:
            changed = False
            for suffix in suffixes_to_remove:
                if new_name.endswith(suffix):
                    new_name = new_name[:-len(suffix)]
                    changed = True 

        if new_name != filename:
            new_path = os.path.join(cps_dir, new_name)
            
            if os.path.exists(new_path):
                print(f"  WARNING: Cannot rename '{filename}' -> '{new_name}'. File already exists!")
                continue
                
            print(f"  Renaming: {filename} -> {new_name}")
            os.rename(old_path, new_path)
            renamed_count += 1

    print(f"Filename cleaning done. Renamed {renamed_count} files.\n")


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
