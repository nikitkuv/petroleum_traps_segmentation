import cv2
import numpy as np

from settings import settings


def load_image(path: str) -> np.ndarray:
    """Загружает изображение в RGB."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_grayscale_image(path: str) -> np.ndarray:
    """Загружает изображение в grayscale (1 канал)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def create_binary_mask(img: np.ndarray, invert: bool = False) -> np.ndarray:
    """Создает бинарную маску 0/1 (float32)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    mask = (gray < settings.BINARY_THRESHOLD).astype(np.float32)
    if invert:
        mask = 1.0 - mask
    return mask


def create_map_mask(structural_img: np.ndarray) -> np.ndarray:
    """Создает маску карты (1 внутри карты, 0 снаружи)."""
    if len(structural_img.shape) == 3:
        is_not_background = np.any(structural_img < 250, axis=2)
    else:
        is_not_background = structural_img < 250
    return is_not_background.astype(np.float32)


def pad_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Добавляет паддинг до target_h x target_w."""
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