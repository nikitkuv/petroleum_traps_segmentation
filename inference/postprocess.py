"""
Постобработка склеенной карты вероятностей в бинарную маску ловушек.

Шаги (все опционаемы через настройки/параметры):
  - бинаризация по порогу;
  - маскирование по границе карты (вне карты -> фон);
  - морфологическая очистка: remove_small_objects (убрать шум/спорадические блобы),
    binary_fill_holes (закрыть дыры внутри замкнутых ловушек).
"""
import numpy as np
import scipy.ndimage as ndimage


def postprocess_prediction(
    prob: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float = 0.5,
    min_trap_area_px: int = 0,
    fill_holes: bool = False,
) -> np.ndarray:
    """
    Карта вероятностей -> бинарная маска ловушек (float32, 1.0/0.0).

    Args:
        prob: склеенная карта вероятностей (h, w).
        valid_mask: булева/флоатная маска валидной области карты (1 = на карте).
        threshold: порог бинаризации.
        min_trap_area_px: убрать связные компоненты ловушек меньше этой площади
            (в пикселях). 0 = не убирать.
        fill_holes: заполнить дыры внутри ловушек.

    Returns:
        float32 маска (h, w): 1.0 — ловушка, 0.0 — фон (в т.ч. вне карты).
    """
    on_map = valid_mask.astype(bool)

    binary = (prob >= threshold).astype(bool)
    binary &= on_map

    if min_trap_area_px and min_trap_area_px > 0:
        binary = _remove_small_objects(binary, int(min_trap_area_px))

    if fill_holes:
        binary = ndimage.binary_fill_holes(binary)
        # fill_holes не должен раздувать ловушки вне карты, но подстрахуемся
        binary &= on_map

    return binary.astype(np.float32)


def _remove_small_objects(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Убирает связные компоненты True площадью < min_size (аналог
    skimage.morphology.remove_small_objects / scipy.ndimage.remove_small_objects,
    но работает на любой версии scipy через ndimage.label + bincount)."""
    labeled, n = ndimage.label(binary)
    if n == 0:
        return binary.copy()
    sizes = np.bincount(labeled.ravel())
    too_small = sizes < min_size
    too_small[0] = False  # фон (label 0) не трогаем
    return binary & ~too_small[labeled]


def build_output_grid(binary_mask: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Бинарная маска -> грид для записи в CPS:
      на карте & ловушка -> 1.0,
      на карте & нет ловушки -> 0.0,
      вне карты -> NaN (станет null в CPS-файле, совпадает с границей карты).

    Грид идеально накладывается на исходную структурную карту (та же геометрия).
    """
    grid = np.full(binary_mask.shape, np.nan, dtype=np.float32)
    on_map = valid_mask.astype(bool)
    grid[on_map] = binary_mask[on_map].astype(np.float32)
    return grid


def build_probability_grid(prob: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Карта вероятностей -> грид для записи в CPS (опционально, settings.INFERENCE_SAVE_PROBABILITY):
      на карте -> вероятность [0, 1], вне карты -> NaN.
    Удобен для подбора порога в downstream-проекте.
    """
    grid = np.full(prob.shape, np.nan, dtype=np.float32)
    on_map = valid_mask.astype(bool)
    grid[on_map] = prob[on_map].astype(np.float32)
    return grid
