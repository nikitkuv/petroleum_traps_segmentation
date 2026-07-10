"""
Тайлинг и обратная склейка для инференса.

Тот же детерминированный layout окон, что и при подготовке данных
(utils/cps_utils.split_into_tiles / data_validation.validate_tiling.compute_windows),
но:
  - никаких пропусков по ловушкам (на инференсе таргета может не быть, а покрытие
    карты должно быть полным);
  - для каждого тайла запоминается его положение в полной карте (placements),
    чтобы потом склеить предсказания в единую маску;
  - всегда сохраняется y_traps-тайл (реальный, если есть GT, иначе нули) —
    GeologyTrapsDataset требует файл ловушек на каждый семпл.
"""
import os
from typing import Dict, List, Tuple

import numpy as np

from settings import settings
from utils.images_utils import pad_image, create_map_mask
from utils.cps_utils import save_png


def compute_inference_windows(
    h: int,
    w: int,
    tile_h: int = None,
    tile_w: int = None,
    overlap: float = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Сетка окон (y0, y1, x0, x1) с прижимом краёв — точная копия логики окон из
    split_into_tiles. Покрытие полной карты без дыр (см. data_validation.validate_tiling).
    """
    tile_h = tile_h or settings.TARGET_HEIGHT
    tile_w = tile_w or settings.TARGET_WIDTH
    overlap = overlap if overlap is not None else settings.TILE_OVERLAP_RATIO

    stride_h = max(1, int(tile_h * (1 - overlap)))
    stride_w = max(1, int(tile_w * (1 - overlap)))

    # Карта меньше тайла целиком — одно окно во всю карту (далее добивается паддингом)
    if h <= tile_h and w <= tile_w:
        return [(0, h, 0, w)]

    nh = max(1, (h - tile_h) // stride_h + 1)
    nw = max(1, (w - tile_w) // stride_w + 1)
    if (nh - 1) * stride_h + tile_h < h:
        nh += 1
    if (nw - 1) * stride_w + tile_w < w:
        nw += 1

    windows = []
    for r in range(nh):
        for c in range(nw):
            y0 = r * stride_h
            x0 = c * stride_w
            y1 = min(y0 + tile_h, h)
            x1 = min(x0 + tile_w, w)
            # Прижим последнего окна к краю, чтобы не терять полосу пикселей
            if y1 == h:
                y0 = max(0, y1 - tile_h)
            if x1 == w:
                x0 = max(0, x1 - tile_w)
            windows.append((y0, y1, x0, x1))
    return windows


def unpad_like_pad_image(tile_arr: np.ndarray, ch: int, cw: int,
                         tile_h: int, tile_w: int) -> np.ndarray:
    """
    Точное обращение utils.images_utils.pad_image: вырезает контент размера (ch, cw)
    из падденного тайла (tile_h, tile_w). pad_image центрирует контент, поэтому
    отступ сверху/слева = (tile - content) // 2.
    """
    pt = (tile_h - ch) // 2
    pl = (tile_w - cw) // 2
    return tile_arr[pt:pt + ch, pl:pl + cw]


def tile_for_inference(
    images_data: Dict[str, Dict[str, np.ndarray]],
    horizon: str,
    out_dir: str,
    tile_h: int = None,
    tile_w: int = None,
    overlap: float = None,
    use_faults: bool = None,
) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Режет полную карту горизонта на тайлы для инференса и сохраняет их в номенклатуре
    обучения ({NNN}_x_{type}_{horizon}.{png|npy}, {NNN}_y_traps_{horizon}.png).

    Окна целиком вне карты (create_map_mask == 0) пропускаются — это чистый паддинг
    без сигнала. Окна, захватывающие хотя бы часть карты, никогда не пропускаются,
    поэтому покрытие карты остаётся без дыр.

    Args:
        images_data: вывод save_large_images: images_data[horizon] = {rgb, grayscale,
            isolines, faults(опц.), traps(опц.)}.
        horizon: имя горизонта.
        out_dir: куда складывать тайлы.
        use_faults: сохранять ли тайлы разломов (только если есть изображение faults).

    Returns:
        placements: {sample_key '{NNN}_{horizon}': {y0,y1,x0,x1,ch,cw}}.
        saved_basenames: список имён сохранённых файлов (для датасета).
    """
    tile_h = tile_h or settings.TARGET_HEIGHT
    tile_w = tile_w or settings.TARGET_WIDTH
    overlap = overlap if overlap is not None else settings.TILE_OVERLAP_RATIO
    use_faults = use_faults if use_faults is not None else settings.USE_FAULTS

    images = images_data[horizon]
    rgb = images.get('rgb')
    gray = images.get('grayscale')
    iso = images.get('isolines')
    faults = images.get('faults')
    traps = images.get('traps')  # может быть None — нет GT

    # Размер полной карты по первому доступному каналу
    h = w = None
    for ref in (rgb, gray, iso, traps, faults):
        if ref is not None:
            h, w = ref.shape[:2]
            break
    if h is None:
        print(f"  {horizon}: нет изображений, пропуск")
        return {}, []

    os.makedirs(out_dir, exist_ok=True)
    windows = compute_inference_windows(h, w, tile_h, tile_w, overlap)

    placements: Dict[str, Dict] = {}
    saved_basenames: List[str] = []
    n_saved = 0
    n_skipped_empty = 0

    for (y0, y1, x0, x1) in windows:
        ch, cw = y1 - y0, x1 - x0

        # Пропуск окон целиком вне карты (по rgb; rgb уже обнулён на разломах,
        # но для решения «есть ли карта» это безопасный критерий)
        if rgb is not None and create_map_mask(rgb[y0:y1, x0:x1]).sum() == 0:
            n_skipped_empty += 1
            continue

        n_saved += 1
        prefix = f"{n_saved:03d}_"
        sample_key = f"{n_saved:03d}_{horizon}"
        placements[sample_key] = {'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1, 'ch': ch, 'cw': cw}

        if rgb is not None:
            name = f"{prefix}x_structuralNOisoline_{horizon}.png"
            save_png(pad_image(rgb[y0:y1, x0:x1], tile_h, tile_w), os.path.join(out_dir, name))
            saved_basenames.append(name)

        if gray is not None:
            name = f"{prefix}x_structuralBlackWhite_{horizon}.npy"
            np.save(os.path.join(out_dir, name), pad_image(gray[y0:y1, x0:x1], tile_h, tile_w))
            saved_basenames.append(name)

        if iso is not None:
            name = f"{prefix}x_isolines_{horizon}.png"
            save_png(pad_image(iso[y0:y1, x0:x1], tile_h, tile_w), os.path.join(out_dir, name))
            saved_basenames.append(name)

        if faults is not None and use_faults:
            name = f"{prefix}x_faults_{horizon}.png"
            save_png(pad_image(faults[y0:y1, x0:x1], tile_h, tile_w), os.path.join(out_dir, name))
            saved_basenames.append(name)

        # Ловушки: реальный кроп, если есть GT, иначе нули (датасет требует traps-файл).
        if traps is not None:
            trap_tile = pad_image(traps[y0:y1, x0:x1], tile_h, tile_w)
        else:
            trap_tile = np.zeros((tile_h, tile_w), dtype=np.uint8)
        name = f"{prefix}y_traps_{horizon}.png"
        save_png(trap_tile, os.path.join(out_dir, name))
        saved_basenames.append(name)

    print(f"  {horizon}: {n_saved} тайлов (пропущено полностью-пустых: {n_skipped_empty})")
    return placements, saved_basenames


def stitch_predictions(
    prob_by_key: Dict[str, np.ndarray],
    placements: Dict[str, Dict],
    h: int,
    w: int,
    tile_h: int = None,
    tile_w: int = None,
    dtype=np.float32,
) -> np.ndarray:
    """
    Склеивает поканальные вероятности тайлов в полную карту (h, w).

    Перекрытия усредняются: в каждой точке накапливается сумма вероятностей по всем
    накрывшим её тайлам и их количество, итог = сумма/счётчик. Это сглаживает швы на
    границах тайлов и снижает краевые артефакты.

    Args:
        prob_by_key: {sample_key: вероятности тайла (tile_h, tile_w)}.
        placements: вывод tile_for_inference.
        h, w: размер полной карты.

    Returns:
        Усреднённая карта вероятностей (h, w).
    """
    tile_h = tile_h or settings.TARGET_HEIGHT
    tile_w = tile_w or settings.TARGET_WIDTH

    accum = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    for sample_key, placement in placements.items():
        if sample_key not in prob_by_key:
            continue
        ch, cw = placement['ch'], placement['cw']
        content = unpad_like_pad_image(prob_by_key[sample_key], ch, cw, tile_h, tile_w)
        y0, y1, x0, x1 = placement['y0'], placement['y1'], placement['x0'], placement['x1']
        accum[y0:y1, x0:x1] += content
        count[y0:y1, x0:x1] += 1.0

    avg = accum / np.maximum(count, 1e-12)
    return avg.astype(dtype)
