import os
import re
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import settings
from utils.cps_utils import (
    find_cps_files,
    save_large_images,
    split_into_tiles,
    clean_cps_filenames,
)
from utils.dataset_utils import extract_base_horizon
from data.dataset import GeologyTrapsDataset
from models.unetplusplus import load_unetplusplus, load_model_checkpoint
from metrics.metrics import MetricsCalculator
from visualization.visualize import visualize_test_results


def evaluate_on_raw_cps(
    checkpoint_path: str,
    cps_dir: str = None,
    horizon_prefixes: Optional[List[str]] = None,
    batch_size: int = None,
    threshold: float = None,
    use_faults: bool = None,
    min_traps_pixels: Optional[int] = None,
    keep_temp: bool = False,
    save_viz: bool = True,
    viz_alpha: float = 0.4,
    save_metrics: bool = True,
) -> Dict:
    """
    Загружает CPS-гриды, оставляет только горизонты из horizon_prefixes (по умолчанию —
    тестовая выборка из settings.TEST_HORIZONS), нарезает тайлы, прогоняет через модель,
    считает метрики по каждому тайлу, микро-агрегирует их по горизонтам и в целом,
    сохраняет визуализации и JSON с метриками в logs/test_visualizations/{дата_время_старта}.

    Тайлы режутся тем же детерминированным алгоритмом, что и при подготовке данных
    (data/convert_cps_to_tiles.py), поэтому для тестовых горизонтов воспроизводятся
    ровно те же тайлы, что лежат в data/images_cps/.

    Args:
        checkpoint_path: Путь к чекпоинту (.pth)
        cps_dir: Папка с CPS-файлами (default settings.CPS_SOURCE_DIR — содержит все горизонты)
        horizon_prefixes: Список префиксов базовых горизонтов для отбора (default TEST_HORIZONS).
            Отбор делается по extract_base_horizon(...).startswith(prefix) — ровно та же логика,
            что и в split_data_by_groups. Передайте [], чтобы обработать ВСЕ горизонты.
        batch_size: Размер батча (default из settings)
        threshold: Порог бинаризации (default из settings)
        use_faults: Использовать разломы (default из settings)
        min_traps_pixels: Мин. пикселей ловушек в тайле; None = settings.MIN_NUM_PIXS_OF_TRAPS_IN_TILES
            (воспроизводит тайлы из data/images_cps/)
        keep_temp: Сохранить промежуточные файлы тайлов
        save_viz: Сохранять визуализации предсказаний по тайлам
        viz_alpha: Прозрачность наложения предсказания
        save_metrics: Сохранять JSON с метриками
    """
    device = settings.DEVICE
    cps_dir = cps_dir or settings.CPS_SOURCE_DIR
    if horizon_prefixes is None:
        horizon_prefixes = settings.TEST_HORIZONS
    use_faults = use_faults if use_faults is not None else settings.USE_FAULTS
    batch_size = batch_size or settings.BATCH_SIZE
    threshold = threshold or settings.TEST_THRESHOLD

    if not os.path.isdir(cps_dir):
        raise ValueError(f"Папка не найдена: {cps_dir}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    # Дата+время начала тестирования — имя папки с результатами этого прогона
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(settings.LOGS_TEST_VIZ_DIR, run_timestamp)
    if save_viz or save_metrics:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 90)
    print("CPS INFERENCE PIPELINE (TEST HORIZONS)")
    print("=" * 90)
    print(f"  CPS папка:       {cps_dir}")
    print(f"  Чекпоинт:        {checkpoint_path}")
    print(f"  Горизонты:       {horizon_prefixes} (отбор по префиксу базового горизонта)")
    print(f"  Устройство:      {device}")
    print(f"  Разломы:         {use_faults}")
    print(f"  Порог:           {threshold}")
    print(f"  Выходная папка:  {output_dir}")
    print("=" * 90)

    # --- CPS → изображения → тайлы ---
    print("\n[1/7] Очистка имён файлов...")
    clean_cps_filenames(cps_dir)

    print("\n[2/7] Поиск CPS файлов...")
    horizons = find_cps_files(cps_dir)
    print(f"Найдено горизонтов (всего): {len(horizons)}")

    horizons = _filter_horizons_by_prefix(horizons, horizon_prefixes)
    print(f"После отбора по горизонтам: {len(horizons)}")
    for name in sorted(horizons.keys()):
        files = horizons[name]
        print(f"  {name}: {', '.join(sorted(files.keys()))}")
    if not horizons:
        raise ValueError(f"Нет CPS-горизонтов, соответствующих {horizon_prefixes} в {cps_dir}")

    temp_dir = tempfile.mkdtemp(prefix="cps_eval_")
    try:
        full_dir = os.path.join(temp_dir, "full_images")
        tiles_dir = os.path.join(temp_dir, "tiles")

        print("\n[3/7] Конвертация CPS -> изображения...")
        images_data = save_large_images(horizons, full_dir)

        print("\n[4/7] Нарезка на тайлы...")
        saved_files = split_into_tiles(images_data, tiles_dir, min_traps_pixels=min_traps_pixels)
        print(f"Создано файлов тайлов: {len(saved_files)}")
        if not saved_files:
            raise ValueError("Тайлы не созданы. Проверьте данные.")

        # --- Датасет + модель ---
        print("\n[5/7] Датасет и модель...")
        tile_basenames = [
            os.path.basename(f)
            for f in saved_files
            if f.endswith('.png') or f.endswith('.npy')
        ]

        dataset = GeologyTrapsDataset(
            file_list=tile_basenames,
            data_dir=tiles_dir,
            augment=False,
            use_faults=use_faults,
            # Оценка идёт в model.eval(): BatchNorm использует замороженные статистики,
            # поэтому NoData-фильтр (защита для обучения) здесь не нужен — иначе
            # тайлы с большим фоном отбрасываются и выборка необоснованно урезается.
            max_nodata_ratio=1.0,
        )
        if len(dataset) == 0:
            raise ValueError("Датасет пуст после фильтрации NoData.")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        model = _load_model(checkpoint_path, device)
        print(f"Модель загружена: {settings.IN_CHANNELS} каналов, {len(dataset)} тайлов")

        # --- Инференс ---
        print("\n[6/7] Вычисление метрик по тайлам и горизонтам...")
        per_tile, aggregated, per_horizon = _evaluate_tiles(model, loader, device, threshold)

        if save_viz:
            print("\n[6.5/7] Сохранение визуализаций по тайлам...")
            _save_visualizations(model, loader, device, output_dir, viz_alpha, per_tile)

    finally:
        if keep_temp:
            print(f"\nПромежуточные файлы тайлов: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if save_metrics:
        _save_metrics(
            output_dir, checkpoint_path, cps_dir, threshold,
            run_timestamp, horizon_prefixes, per_tile, aggregated, per_horizon,
        )

    _print_results(per_tile, aggregated, per_horizon)

    print("\n" + "=" * 90)
    print(f"Визуализации и метрики сохранены в: {output_dir}")
    print("=" * 90)

    return {
        'checkpoint_path': checkpoint_path,
        'cps_dir': cps_dir,
        'horizon_prefixes': horizon_prefixes,
        'run_timestamp': run_timestamp,
        'threshold': threshold,
        'n_tiles': len(per_tile),
        'per_tile_metrics': per_tile,
        'per_horizon_metrics': per_horizon,
        'aggregated_metrics': aggregated,
    }


def _filter_horizons_by_prefix(
    horizons: Dict[str, Dict[str, str]],
    prefixes: List[str],
) -> Dict[str, Dict[str, str]]:
    """
    Оставляет только горизонты, базовое имя которых начинается с одного из префиксов.
    Логика отбора совпадает с data.dataloaders.split_data_by_groups, чтобы тестовая
    выборка здесь и при обучении была одной и той же.
    """
    if not prefixes:
        return horizons
    kept = {}
    for name, files in horizons.items():
        base = extract_base_horizon(name)
        if any(base.startswith(p) for p in prefixes):
            kept[name] = files
    return kept


def _load_model(checkpoint_path: str, device: str):
    """Загружает модель из чекпоинта."""
    in_channels = settings.IN_CHANNELS
    model = load_unetplusplus(
        in_channels=in_channels,
        classes=1,
        encoder_name=settings.ENCODER_NAME,
        encoder_weights=None,
        device=device,
    )
    model = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


def _extract_sample_info(loader, idx: int):
    """
    Возвращает (имя_тайла, базовый_горизонт) для тайла по его индексу в датасете.
    Имя тайла: '{номер}_{полное_имя_горизонта}' (напр. '001_D_70_bottop1'),
    базовый горизонт — изолинии/суффикса (напр. 'D_70').
    """
    sample_paths = loader.dataset.samples[idx]
    first_path = list(sample_paths.values())[0]
    filename = Path(first_path).stem
    match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
    if match:
        number = match.group(1)
        name = match.group(2)
        return f"{number}_{name}", extract_base_horizon(name)
    return filename, extract_base_horizon(filename)


def _evaluate_tiles(model, loader, device: str, threshold: float):
    """
    Прогоняет все тайлы через модель и считает:
      - per-tile метрики,
      - микро-агрегированные метрики по каждому горизонту (все тайлы горизонта
        склеиваются в один массив — точнее, чем усреднение per-tile метрик),
      - микро-агрегированные метрики по всей тестовой выборке.

    Тензоры хранятся только в горизонтальных бакетах; глобальная агрегация
    считается из тех же бакетов, чтобы не дублировать память.
    """
    metrics_calc = MetricsCalculator(threshold=threshold)
    per_tile_results: List[Dict] = []
    groups: Dict[str, Dict[str, list]] = {}  # base_horizon -> {preds/targets/masks}

    with torch.no_grad():
        pbar = tqdm(loader, desc='Инференс')
        for batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask_map = batch.get('mask_map')
            if mask_map is not None:
                mask_map = mask_map.to(device)

            predictions = model(x)

            bs = x.shape[0]
            for i in range(bs):
                pred_i = predictions[i:i + 1]
                target_i = y[i:i + 1]
                mask_i = mask_map[i:i + 1] if mask_map is not None else None

                metrics = metrics_calc.compute_all(pred_i, target_i, mask_i)

                sample_name, base_horizon = _extract_sample_info(loader, batch['sample_idx'][i].item())
                per_tile_results.append({
                    'sample_name': sample_name,
                    'horizon': base_horizon,
                    **metrics,
                })

                group = groups.setdefault(base_horizon, {'preds': [], 'targets': [], 'masks': []})
                group['preds'].append(pred_i.cpu())
                group['targets'].append(target_i.cpu())
                if mask_i is not None:
                    group['masks'].append(mask_i.cpu())

                pbar.set_postfix(sample=sample_name[:25], dice=f"{metrics['dice']:.3f}")

    if not groups:
        empty = {'iou': 0.0, 'dice': 0.0, 'recall': 0.0, 'precision': 0.0,
                 'f1': 0.0, 'fp_area': 0.0, 'fn_area': 0.0}
        return per_tile_results, empty, {}

    # Микро-агрегация по каждому горизонту
    per_horizon: Dict[str, Dict] = {}
    cat_preds, cat_targets, cat_masks = [], [], []
    has_masks = False
    for h in sorted(groups.keys()):
        group = groups[h]
        h_preds = torch.cat(group['preds'], dim=0)
        h_targets = torch.cat(group['targets'], dim=0)
        h_masks = torch.cat(group['masks'], dim=0) if group['masks'] else None

        per_horizon[h] = metrics_calc.compute_all(h_preds, h_targets, h_masks)
        per_horizon[h]['n_tiles'] = len(group['preds'])

        cat_preds.append(h_preds)
        cat_targets.append(h_targets)
        if h_masks is not None:
            cat_masks.append(h_masks)
            has_masks = True

    # Глобальная микро-агрегация — из тех же бакетов
    aggregated = metrics_calc.compute_all(
        torch.cat(cat_preds, dim=0),
        torch.cat(cat_targets, dim=0),
        torch.cat(cat_masks, dim=0) if has_masks else None,
    )

    return per_tile_results, aggregated, per_horizon


def _save_visualizations(model, loader, device: str, viz_dir: str, alpha: float, per_tile_results: List[Dict]):
    """Сохраняет визуализацию предсказания для каждого тестового тайла."""
    metrics_by_sample = {r['sample_name']: r for r in per_tile_results}

    with torch.no_grad():
        pbar = tqdm(loader, desc='Визуализации')
        for batch in pbar:
            x = batch['x'].to(device)
            predictions = model(x)

            real_sample_indices = batch['sample_idx'].tolist()

            visualize_test_results(
                batch=batch,
                predictions=predictions.cpu(),
                sample_indices=real_sample_indices,
                dataset=loader.dataset,
                save_path=viz_dir,
                alpha=alpha,
                metrics_by_sample=metrics_by_sample,
            )


def _save_metrics(
    output_dir: str,
    checkpoint_path: str,
    cps_dir: str,
    threshold: float,
    run_timestamp: str,
    horizon_prefixes: List[str],
    per_tile: List[Dict],
    aggregated: Dict,
    per_horizon: Dict,
) -> None:
    """Сохраняет все метрики в JSON в папку прогона."""
    results = {
        'checkpoint_path': checkpoint_path,
        'cps_dir': cps_dir,
        'threshold': threshold,
        'run_timestamp': run_timestamp,
        'horizon_prefixes': horizon_prefixes,
        'n_tiles': len(per_tile),
        'aggregated_metrics': aggregated,
        'per_horizon_metrics': per_horizon,
        'per_tile_metrics': per_tile,
    }
    path = os.path.join(output_dir, 'test_metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Метрики сохранены: {path}")


def _print_results(per_tile: List[Dict], aggregated: Dict, per_horizon: Dict):
    """Выводит таблицы с метриками в консоль: по горизонтам, агрегированные и по тайлам."""
    if not per_tile:
        print("\nНет результатов.")
        return

    metric_keys = ['dice', 'iou', 'recall', 'precision', 'f1', 'fp_area', 'fn_area']

    print("\n" + "=" * 100)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 100)

    print(f"\nАгрегированные метрики по всем тайлам ({len(per_tile)}):")
    for key in metric_keys:
        print(f"  {key:<12} {aggregated[key]:.4f}")

    print("\nМетрики по горизонтам (микро-агрегация тайлов горизонта):")
    header_h = f"{'Горизонт':<20} {'Тайлов':>7} {'Dice':>8} {'IoU':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'FP%':>8} {'FN%':>8}"
    print(header_h)
    print("-" * 100)
    for h, m in sorted(per_horizon.items()):
        print(
            f"{h:<20} {m['n_tiles']:>7} "
            f"{m['dice']:>8.4f} {m['iou']:>8.4f} {m['recall']:>8.4f} "
            f"{m['precision']:>8.4f} {m['f1']:>8.4f} {m['fp_area']:>8.4f} {m['fn_area']:>8.4f}"
        )

    header_t = f"{'Тайл':<40} {'Горизонт':<14} {'Dice':>8} {'IoU':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'FP%':>8} {'FN%':>8}"
    print(f"\n{header_t}")
    print("-" * 100)
    for r in sorted(per_tile, key=lambda x: x['dice'], reverse=True):
        print(
            f"{r['sample_name']:<40} {r['horizon']:<14} "
            f"{r['dice']:>8.4f} {r['iou']:>8.4f} {r['recall']:>8.4f} "
            f"{r['precision']:>8.4f} {r['f1']:>8.4f} {r['fp_area']:>8.4f} {r['fn_area']:>8.4f}"
        )
    print("=" * 100)
