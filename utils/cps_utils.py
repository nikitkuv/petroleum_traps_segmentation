import numpy as np
import warnings
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from settings import settings


def read_cps_grid(file_path: str, vertical_flip: bool = None) -> Tuple[np.ndarray, dict]:
    """
    Загружает CPS грид из файла.
    
    Args:
        file_path: Путь к CPS файлу
        vertical_flip: Флип по вертикали (по умолчанию из settings)
    
    Returns:
        grid: 2D numpy array с значениями
        metadata: dict с xmin, xmax, ymin, ymax, nx, ny, null_value
    """
    vertical_flip = vertical_flip if vertical_flip is not None else settings.CPS_VERTICAL_FLIP
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # --- Initialize ---
    nx = ny = None
    xmin = xmax = ymin = ymax = None
    null_value = settings.CPS_NULL_VALUE
    data_start = None
    
    # --- Parse header ---
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
    
    # --- Read grid values ---
    values = []
    for line in lines[data_start:]:
        values.extend([float(x) for x in line.split()])
    
    values = np.array(values, dtype=np.float32)
    
    # --- Safety check ---
    expected = nx * ny
    if len(values) != expected:
        warnings.warn(f"CPS {file_path}: expected {expected}, got {len(values)}. Truncating.")
        values = values[:expected]
    
    # --- Reshape (Fortran order) ---
    grid = values.reshape((ny, nx), order='F')
    
    # --- Flip vertically ---
    if vertical_flip:
        grid = np.flipud(grid)
    
    # --- Replace null values ---
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


def cps_to_binary_mask(grid: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Конвертирует CPS грид в бинарную маску (для fault/trap).
    
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
    
    # Бинаризация
    mask = (grid_norm < threshold).astype(np.float32)
    mask[~valid_mask] = 0  # NaN = 0
    
    return mask