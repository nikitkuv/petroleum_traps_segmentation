"""
Инференс обученной U-Net++ на новых CPS-гридах.

Вход: папка с CPS-гридами в номенклатуре обучения (x_structuralNOisoline_*, [x_faults_*],
[y_traps_*]). Для каждого горизонта:
  1. полная подготовка каналов (та же, что при обучении — через save_large_images
     и GeologyTrapsDataset);
  2. нарезка на тайлы с полным покрытием карты + инференс + склейка тайлов в единую
     маску ловушек;
  3. запись cps-грида предсказаний (идеально накладывается на исходную карту);
  4. визуализация на полную карту;
  5. метрики по карте (если есть GT-ловушки).

Входная папка не модифицируется: clean_cps_filenames работает над копией во временной папке.
"""
import os
import json
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import settings
from utils.cps_utils import (
    find_cps_files,
    save_large_images,
    clean_cps_filenames,
    read_cps_grid,
    write_cps_grid,
)
from utils.dataset_utils import extract_base_horizon
from data.dataset import GeologyTrapsDataset
from models.unetplusplus import load_unetplusplus, load_model_checkpoint
from metrics.metrics import MetricsCalculator

from inference.tiling import tile_for_inference, stitch_predictions
from inference.postprocess import (
    postprocess_prediction,
    build_output_grid,
    build_probability_grid,
)
from inference.visualize_inference import visualize_full_map


def run_inference(
    checkpoint_path: str,
    cps_dir: str,
    horizon_prefixes: Optional[List[str]] = None,
    batch_size: int = None,
    threshold: float = None,
    use_faults: bool = None,
    min_trap_area_px: int = None,
    fill_holes: bool = None,
    tta: bool = None,
    save_probability: bool = None,
    viz_alpha: float = 0.4,
    keep_temp: bool = False,
) -> Dict:
    """
    Прогоняет модель по всем горизонтам из cps_dir и сохраняет артефакты инференса.

    Args:
        checkpoint_path: путь к чекпоинту (.pth).
        cps_dir: папка с входными CPS-гридами (номенклатура обучения).
        horizon_prefixes: оставить только горизонты, базовое имя которых начинается с
            одного из префиксов (та же логика, что в split_data_by_groups).
            None или [] — обработать ВСЕ горизонты.
        batch_size: размер батча (default settings.BATCH_SIZE).
        threshold: порог бинаризации (default settings.INFERENCE_THRESHOLD).
        use_faults: использовать канал разломов (default settings.USE_FAULTS).
        min_trap_area_px: морфологическая очистка — мин. площадь ловушки (default settings).
        fill_holes: заполнять дыры в ловушках (default settings).
        tta: test-time augmentation горизонтальным отражением (default settings).
        save_probability: дополнительно сохранить cps вероятностей (default settings).
        viz_alpha: прозрачность наложения предсказания.
        keep_temp: сохранить промежуточные файлы (тайлы/полные карты).

    Returns:
        Словарь с итогами прогона (он же пишется в inference_results.json).
    """
    device = settings.DEVICE
    if horizon_prefixes is None:
        horizon_prefixes = []
    use_faults = use_faults if use_faults is not None else settings.USE_FAULTS
    batch_size = batch_size or settings.BATCH_SIZE
    threshold = threshold if threshold is not None else settings.INFERENCE_THRESHOLD
    min_trap_area_px = min_trap_area_px if min_trap_area_px is not None else settings.INFERENCE_MIN_TRAP_AREA_PX
    fill_holes = fill_holes if fill_holes is not None else settings.INFERENCE_FILL_HOLES
    tta = tta if tta is not None else settings.INFERENCE_TTA
    save_probability = save_probability if save_probability is not None else settings.INFERENCE_SAVE_PROBABILITY

    if not os.path.isdir(cps_dir):
        raise ValueError(f"Папка не найдена: {cps_dir}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    viz_dir = os.path.join(settings.LOGS_INFER_VIZ_DIR, run_timestamp)
    cps_out_root = os.path.join(settings.INFERENCE_OUTPUT_DIR, run_timestamp)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(cps_out_root, exist_ok=True)

    print("=" * 90)
    print("INFERENCE PIPELINE (CPS -> маски ловушек)")
    print("=" * 90)
    print(f"  Входная папка:   {cps_dir}  (не модифицируется)")
    print(f"  Чекпоинт:        {checkpoint_path}")
    print(f"  Горизонты:       {horizon_prefixes if horizon_prefixes else 'ВСЕ'}")
    print(f"  Устройство:      {device}")
    print(f"  Разломы:         {use_faults}")
    print(f"  Порог:           {threshold}")
    print(f"  Постобработка:   min_area={min_trap_area_px}, fill_holes={fill_holes}")
    print(f"  TTA:             {tta}    Сохранять вероятности: {save_probability}")
    print(f"  Каналы:          {settings.IN_CHANNELS}")
    print(f"  CPS-артефакты:   {cps_out_root}")
    print(f"  Визуализации:    {viz_dir}")
    print("=" * 90)

    # --- Модель ---
    model = _load_model(checkpoint_path, device)

    # --- Копия входной папки во temp (не трогаем оригинал) -> чистка имён ---
    temp_root = tempfile.mkdtemp(prefix="cps_infer_")
    try:
        temp_input = os.path.join(temp_root, "input")
        _copy_tree(cps_dir, temp_input)

        print("\n[1/4] Очистка имён файлов (над копией)...")
        clean_cps_filenames(temp_input)

        print("\n[2/4] Поиск CPS файлов...")
        horizons = find_cps_files(temp_input)
        horizons = _filter_horizons_by_prefix(horizons, horizon_prefixes)
        print(f"Горизонтов к обработке: {len(horizons)}")
        if not horizons:
            raise ValueError(f"В {cps_dir} нет горизонтов, соответствующих фильтру {horizon_prefixes}")
        for name in sorted(horizons.keys()):
            files = horizons[name]
            print(f"  {name}: structural={bool(files.get('structural'))}, "
                  f"faults={bool(files.get('faults'))}, traps={bool(files.get('traps'))}")

        full_dir = os.path.join(temp_root, "full_images")
        tiles_dir = os.path.join(temp_root, "tiles")

        # --- Постепенная обработка горизонтов ---
        print("\n[3/4] Инференс по горизонтам...")
        per_horizon_results: Dict[str, Dict] = {}

        for horizon in tqdm(sorted(horizons.keys()), desc='Горизонты'):
            result = _infer_horizon(
                horizon=horizon,
                files=horizons[horizon],
                model=model,
                device=device,
                full_dir=full_dir,
                tiles_dir=tiles_dir,
                use_faults=use_faults,
                batch_size=batch_size,
                threshold=threshold,
                min_trap_area_px=min_trap_area_px,
                fill_holes=fill_holes,
                tta=tta,
                save_probability=save_probability,
                viz_alpha=viz_alpha,
                viz_dir=viz_dir,
                cps_out_root=cps_out_root,
            )
            per_horizon_results[horizon] = result

    finally:
        if keep_temp:
            print(f"\nПромежуточные файлы: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    # --- Итоги ---
    print("\n[4/4] Сохранение метрик и сводки...")
    aggregated = _macro_aggregate(per_horizon_results)

    summary = {
        'checkpoint_path': checkpoint_path,
        'cps_dir': cps_dir,
        'run_timestamp': run_timestamp,
        'threshold': threshold,
        'use_faults': use_faults,
        'tta': tta,
        'postprocess': {'min_trap_area_px': min_trap_area_px, 'fill_holes': fill_holes},
        'save_probability': save_probability,
        'viz_dir': viz_dir,
        'cps_out_root': cps_out_root,
        'n_horizons': len(per_horizon_results),
        'per_horizon': per_horizon_results,
        'aggregated_metrics_macro': aggregated,
    }
    json_path = os.path.join(viz_dir, 'inference_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Метрики сохранены: {json_path}")

    _print_results(per_horizon_results, aggregated)

    print("\n" + "=" * 90)
    print(f"CPS-артефакты:  {cps_out_root}")
    print(f"Визуализации:   {viz_dir}")
    print("=" * 90)

    return summary


# --------------------------------------------------------------------------------------
# Внутренние хелперы
# --------------------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: str):
    """Загружает модель из чекпоинта (та же схема, что в evaluate_on_raw_cps)."""
    model = load_unetplusplus(
        in_channels=settings.IN_CHANNELS,
        classes=1,
        encoder_name=settings.ENCODER_NAME,
        encoder_weights=None,
        device=device,
    )
    model = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


def _filter_horizons_by_prefix(horizons: Dict[str, Dict], prefixes: List[str]) -> Dict[str, Dict]:
    """Отбор горизонтов по префиксу базового имени (как split_data_by_groups)."""
    if not prefixes:
        return horizons
    kept = {}
    for name, files in horizons.items():
        base = extract_base_horizon(name)
        if any(base.startswith(p) for p in prefixes):
            kept[name] = files
    return kept


def _copy_tree(src: str, dst: str) -> None:
    """Копирует папку src в dst (создавая dst). Используется, чтобы не трогать оригинал."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst)


def _predict_batch(model, x: torch.Tensor, tta: bool) -> torch.Tensor:
    """Вероятности ловушек (B,1,H,W). С опциональным flip-TTA."""
    prob = torch.sigmoid(model(x))
    if tta:
        prob_flip = torch.sigmoid(model(torch.flip(x, dims=[3])))
        prob = (prob + torch.flip(prob_flip, dims=[3])) / 2.0
    return prob


def _infer_horizon(
    horizon: str,
    files: Dict[str, str],
    model,
    device: str,
    full_dir: str,
    tiles_dir: str,
    use_faults: bool,
    batch_size: int,
    threshold: float,
    min_trap_area_px: int,
    fill_holes: bool,
    tta: bool,
    save_probability: bool,
    viz_alpha: float,
    viz_dir: str,
    cps_out_root: str,
) -> Dict:
    """Полная обработка одного горизонта: каналы -> тайлы -> модель -> склейка -> артефакты."""
    has_gt = 'traps' in files

    # 1. Структурный грид — референс геометрии: meta для записи CPS + valid_mask карты
    struct_grid, struct_meta = read_cps_grid(files['structural'])
    valid_mask = ~np.isnan(struct_grid)
    ny, nx = struct_grid.shape

    # 2. Полные карты каналов (rgb/depth/isolines/faults/traps) в геометрии структурной карты
    images_data = save_large_images({horizon: files}, full_dir)
    images = images_data[horizon]
    if images.get('rgb') is None:
        print(f"\n  [{horizon}] нет структурной карты, пропуск")
        return {'has_gt': has_gt, 'skipped': True, 'reason': 'no structural image'}

    # 3. Тайлы с полным покрытием + размещения
    placements, basenames = tile_for_inference(
        images_data, horizon, tiles_dir, use_faults=use_faults,
    )
    if not placements:
        print(f"\n  [{horizon}] нет валидных тайлов, пропуск")
        return {'has_gt': has_gt, 'skipped': True, 'reason': 'no valid tiles'}

    # 4. Датасет (та же предобработка, что при обучении/валидации) + инференс
    dataset = GeologyTrapsDataset(
        file_list=basenames,
        data_dir=tiles_dir,
        augment=False,
        use_faults=use_faults,
        max_nodata_ratio=1.0,  # eval-режим: BN использует замороженные статистики
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )

    prob_by_key: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  тайлы {horizon}', leave=False):
            x = batch['x'].to(device)
            prob = _predict_batch(model, x, tta)  # (B,1,H,W)
            real_indices = batch['sample_idx'].tolist()
            for i, idx in enumerate(real_indices):
                sample_key = dataset.samples[idx].get('_sample_key')
                prob_by_key[sample_key] = prob[i, 0].detach().cpu().numpy()

    # 5. Склейка тайлов в полную карту вероятностей
    full_prob = stitch_predictions(prob_by_key, placements, ny, nx)

    # 6. Постобработка -> бинарная маска ловушек
    binary_mask = postprocess_prediction(
        full_prob, valid_mask,
        threshold=threshold, min_trap_area_px=min_trap_area_px, fill_holes=fill_holes,
    )

    # 7. CPS-артефакт (накладывается на исходную карту 1:1)
    cps_path = os.path.join(cps_out_root, f"{horizon}.cps")
    write_cps_grid(
        build_output_grid(binary_mask, valid_mask), struct_meta, cps_path,
        label=f"Predicted traps mask ({horizon})",
    )
    prob_path = None
    if save_probability:
        prob_path = os.path.join(cps_out_root, f"{horizon}_prob.cps")
        write_cps_grid(
            build_probability_grid(full_prob, valid_mask), struct_meta, prob_path,
            label=f"Predicted traps probability ({horizon})",
        )

    # 8. Визуализация на полную карту
    gt_traps = images.get('traps') if has_gt else None
    viz_path = os.path.join(viz_dir, f"{horizon}.png")

    # 9. Метрики по карте (если есть GT)
    metrics = None
    if has_gt and gt_traps is not None:
        metrics = _compute_fullmap_metrics(binary_mask, gt_traps, valid_mask, threshold)

    visualize_full_map(
        rgb_img=images['rgb'],
        depth_img=images['grayscale'],
        isolines_img=images['isolines'],
        pred_mask=binary_mask,
        valid_mask=valid_mask,
        horizon=horizon,
        out_path=viz_path,
        gt_traps=gt_traps,
        metrics=metrics,
        alpha=viz_alpha,
    )

    print(f"\n  [{horizon}] тайлов: {len(placements)} | GT: {has_gt} | "
          f"ловушек в маске: {int(binary_mask.sum())} пикс.")
    if metrics is not None:
        print(f"           Dice={metrics['dice']:.3f} IoU={metrics['iou']:.3f} "
              f"Recall={metrics['recall']:.3f} Prec={metrics['precision']:.3f}")

    return {
        'has_gt': has_gt,
        'skipped': False,
        'n_tiles': len(placements),
        'n_trap_pixels': int(binary_mask.sum()),
        'metrics': metrics,
        'prediction_cps': cps_path,
        'probability_cps': prob_path,
        'visualization': viz_path,
        'grid_shape': [int(ny), int(nx)],
    }


def _compute_fullmap_metrics(binary_mask: np.ndarray, gt_traps: np.ndarray,
                             valid_mask: np.ndarray, threshold: float) -> Dict[str, float]:
    """Метрики по полной карте: предсказанная маска vs GT, в границах карты."""
    gt_full = (gt_traps > 128).astype(np.float32)
    pred_t = torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt_full).unsqueeze(0).unsqueeze(0)
    mask_t = torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    # binary_mask уже 0/1; compute_all применяет sigmoid -> порог 0.5 оставляет 0/1 как есть
    return MetricsCalculator(threshold=threshold).compute_all(pred_t, gt_t, mask_t)


def _macro_aggregate(per_horizon_results: Dict[str, Dict]) -> Dict:
    """Макро-усреднение метрик по горизонтам с GT (просто осреднение per-horizon значений)."""
    keys = ['iou', 'dice', 'recall', 'precision', 'f1', 'fp_area', 'fn_area']
    sums = {k: 0.0 for k in keys}
    n = 0
    for res in per_horizon_results.values():
        m = res.get('metrics')
        if m:
            n += 1
            for k in keys:
                sums[k] += m.get(k, 0.0)
    if n == 0:
        return {'n_horizons_with_gt': 0}
    return {'n_horizons_with_gt': n, **{k: sums[k] / n for k in keys}}


def _print_results(per_horizon_results: Dict[str, Dict], aggregated: Dict) -> None:
    """Таблица метрик по горизонтам + сводная строка."""
    metric_keys = ['dice', 'iou', 'recall', 'precision', 'f1', 'fp_area', 'fn_area']

    print("\n" + "=" * 110)
    print("РЕЗУЛЬТАТЫ ИНФЕРЕНСА")
    print("=" * 110)
    header = (f"{'Горизонт':<26} {'GT':>4} {'Тайлов':>7} {'Ловушек':>9} "
              f"{'Dice':>8} {'IoU':>8} {'Recall':>8} {'Prec':>8} {'F1':>8}")
    print(header)
    print("-" * 110)
    for horizon, res in sorted(per_horizon_results.items()):
        if res.get('skipped'):
            print(f"{horizon:<26} {'-':>4} {'-':>7} {'-':>9}   пропущен: {res.get('reason')}")
            continue
        m = res.get('metrics') or {}
        gt = 'да' if res.get('has_gt') else 'нет'
        def cell(k):
            return f"{m[k]:>8.3f}" if k in m else f"{'-':>8}"
        print(f"{horizon:<26} {gt:>4} {res.get('n_tiles', 0):>7} "
              f"{res.get('n_trap_pixels', 0):>9} "
              f"{cell('dice')} {cell('iou')} {cell('recall')} {cell('precision')} {cell('f1')}")
    print("-" * 110)
    if aggregated.get('n_horizons_with_gt', 0) > 0:
        print(f"{'МАКРО-СРЕДНЕЕ (GT)':<26} {aggregated['n_horizons_with_gt']:>4} "
              f"{'':>7} {'':>9} "
              f"{aggregated.get('dice', 0):>8.3f} {aggregated.get('iou', 0):>8.3f} "
              f"{aggregated.get('recall', 0):>8.3f} {aggregated.get('precision', 0):>8.3f} "
              f"{aggregated.get('f1', 0):>8.3f}")
    else:
        print("Макро-усреднение недоступно: нет горизонтов с GT-ловушками.")
    print("=" * 110)
