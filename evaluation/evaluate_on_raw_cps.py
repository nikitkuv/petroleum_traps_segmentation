import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List

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
from data.dataset import GeologyTrapsDataset
from models.unetplusplus import load_unetplusplus, load_model_checkpoint
from metrics.metrics import MetricsCalculator


def evaluate_on_raw_cps(
    cps_dir: str,
    checkpoint_path: str,
    batch_size: int = None,
    threshold: float = None,
    use_faults: bool = None,
    min_traps_pixels: int = 0,
    keep_temp: bool = False,
) -> Dict:
    """
    Загружает CPS файлы, нарезает тайлы, прогоняет через модель,
    возвращает метрики по каждому тайлу + агрегированные.

    Args:
        cps_dir: Папка с CPS файлами (x_structuralNOisoline_*, y_traps_*, x_faults_*)
        checkpoint_path: Путь к чекпоинту (.pth)
        batch_size: Размер батча (default из settings)
        threshold: Порог бинаризации (default из settings)
        use_faults: Использовать разломы (default из settings)
        min_traps_pixels: Мин. пикселей ловушек в тайле, 0 = все тайлы
        keep_temp: Сохранить промежуточные файлы
    """
    device = settings.DEVICE
    use_faults = use_faults if use_faults is not None else settings.USE_FAULTS
    batch_size = batch_size or settings.BATCH_SIZE
    threshold = threshold or settings.TEST_THRESHOLD

    if not os.path.isdir(cps_dir):
        raise ValueError(f"Папка не найдена: {cps_dir}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    print("=" * 90)
    print("CPS INFERENCE PIPELINE")
    print("=" * 90)
    print(f"  CPS папка:       {cps_dir}")
    print(f"  Чекпоинт:        {checkpoint_path}")
    print(f"  Устройство:      {device}")
    print(f"  Разломы:         {use_faults}")
    print(f"  Порог:           {threshold}")
    print(f"  Min traps px:    {min_traps_pixels}")
    print(f"  Tile size:       {settings.TARGET_WIDTH}x{settings.TARGET_HEIGHT}")
    print("=" * 90)

    # --- CPS → изображения → тайлы ---
    print("\n[1/6] Очистка имён файлов...")
    clean_cps_filenames(cps_dir)

    print("\n[2/6] Поиск CPS файлов...")
    horizons = find_cps_files(cps_dir)
    print(f"Найдено горизонтов: {len(horizons)}")
    for name, files in sorted(horizons.items()):
        print(f"  {name}: {', '.join(sorted(files.keys()))}")
    if not horizons:
        raise ValueError(f"CPS файлы не найдены в {cps_dir}")

    temp_dir = tempfile.mkdtemp(prefix="cps_eval_")
    try:
        full_dir = os.path.join(temp_dir, "full_images")
        tiles_dir = os.path.join(temp_dir, "tiles")

        print("\n[3/6] Конвертация CPS -> изображения...")
        images_data = save_large_images(horizons, full_dir)

        print("\n[4/6] Нарезка на тайлы...")
        saved_files = split_into_tiles(
            images_data, tiles_dir,
            min_traps_pixels=min_traps_pixels,
        )
        print(f"Создано файлов тайлов: {len(saved_files)}")
        if not saved_files:
            raise ValueError("Тайлы не созданы. Проверьте данные.")

        # --- Датасет + модель ---
        print("\n[5/6] Датасет и модель...")
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
        print("\n[6/6] Вычисление метрик...")
        per_tile, aggregated = _evaluate_tiles(model, loader, device, threshold)

    finally:
        if keep_temp:
            print(f"\nПромежуточные файлы: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    _print_results(per_tile, aggregated)

    return {
        'checkpoint_path': checkpoint_path,
        'cps_dir': cps_dir,
        'threshold': threshold,
        'n_tiles': len(per_tile),
        'per_tile_metrics': per_tile,
        'aggregated_metrics': aggregated,
    }


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


def _evaluate_tiles(model, loader, device: str, threshold: float):
    """
    Прогоняет все тайлы через модель, считает per-tile и aggregated метрики.

    Агрегация — на всём массиве данных разом (как в evaluate_on_test_data.py),
    точнее чем простое усреднение per-tile метрик.
    """
    metrics_calc = MetricsCalculator(threshold=threshold)
    per_tile_results: List[Dict] = []
    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        pbar = tqdm(loader, desc='Инференс')
        for batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask_map = batch.get('mask_map')
            if mask_map is not None:
                mask_map = mask_map.to(device)

            predictions = model(x)

            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())
            if mask_map is not None:
                all_masks.append(mask_map.cpu())

            bs = x.shape[0]
            for i in range(bs):
                pred_i = predictions[i:i+1]
                target_i = y[i:i+1]
                mask_i = mask_map[i:i+1] if mask_map is not None else None

                metrics = metrics_calc.compute_all(pred_i, target_i, mask_i)

                sample_name = _extract_sample_name(loader, batch['sample_idx'][i].item())
                per_tile_results.append({
                    'sample_name': sample_name,
                    **metrics,
                })

                pbar.set_postfix(sample=sample_name[:25], dice=f"{metrics['dice']:.3f}")

    # Агрегированные метрики на всём массиве (точнее среднего по тайлам)
    all_preds = torch.cat(all_predictions, dim=0)
    all_tgts = torch.cat(all_targets, dim=0)
    all_msks = torch.cat(all_masks, dim=0) if all_masks else None
    aggregated = metrics_calc.compute_all(all_preds, all_tgts, all_msks)

    return per_tile_results, aggregated


def _extract_sample_name(loader, idx: int) -> str:
    """Извлекает имя тайла из датасета (номер_горизонт)."""
    sample_paths = loader.dataset.samples[idx]
    first_path = list(sample_paths.values())[0]
    filename = Path(first_path).stem

    match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
    return f"{match.group(1)}_{match.group(2)}" if match else filename


def _print_results(per_tile: List[Dict], aggregated: Dict):
    """Выводит таблицу с метриками в консоль."""
    if not per_tile:
        print("\nНет результатов.")
        return

    header = f"{'Тайл':<40} {'Dice':>8} {'IoU':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'FP%':>8} {'FN%':>8}"

    print("\n" + "=" * 100)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 100)

    print(f"\nАгрегированные метрики ({len(per_tile)} тайлов):")
    for key in ['dice', 'iou', 'recall', 'precision', 'f1', 'fp_area', 'fn_area']:
        print(f"  {key:<12} {aggregated[key]:.4f}")

    print(f"\n{header}")
    print("-" * 100)

    for r in sorted(per_tile, key=lambda x: x['dice'], reverse=True):
        print(
            f"{r['sample_name']:<40} "
            f"{r['dice']:>8.4f} {r['iou']:>8.4f} "
            f"{r['recall']:>8.4f} {r['precision']:>8.4f} "
            f"{r['f1']:>8.4f} {r['fp_area']:>8.4f} {r['fn_area']:>8.4f}"
        )

    print("-" * 100)
    print(
        f"{'АГРЕГИРОВАНО':<40} "
        f"{aggregated['dice']:>8.4f} {aggregated['iou']:>8.4f} "
        f"{aggregated['recall']:>8.4f} {aggregated['precision']:>8.4f} "
        f"{aggregated['f1']:>8.4f} {aggregated['fp_area']:>8.4f} {aggregated['fn_area']:>8.4f}"
    )
    print("=" * 100)
