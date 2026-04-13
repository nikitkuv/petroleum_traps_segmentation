import os
import json
from pathlib import Path
from typing import Dict
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from settings import settings
from data.dataloaders import get_file_list, split_data_by_groups, create_dataloaders
from models.unetplusplus import load_unetplusplus, load_model_checkpoint
from metrics.metrics import MetricsCalculator
from visualization.visualize import visualize_test_results
from data.dataset import GeologyTrapsDataset
    

def evaluate_all_test_samples(
    checkpoint_path: str,
    use_faults: bool = None,
    data_source: str = None,
    batch_size: int = None,
    threshold: float = None,
    save_viz_dir: str = None,
    save_metrics_path: str = None,
    seed: int = None,
    custom_test_files: list = None
) -> Dict:
    """
    Загружает модель и оценивает её на всех тестовых семплах.

    Args:
        checkpoint_path: Путь к чекпоинту модели
        data_dir: Путь к данным (для data_source='png')
        cps_tiles_dir: Путь к CPS tiles данным (для data_source='cps_tiles')
        use_faults: Использовать ли разломы (должно совпадать с обучением)
        data_source: Источник данных ('png' или 'cps_tiles')
        batch_size: Размер батча
        threshold: Порог бинаризации
        save_viz_dir: Директория для сохранения визуализаций
        save_metrics_path: Путь для сохранения JSON с метриками
        seed: Random seed для воспроизведения разбиения (должен совпадать с обучением)

    Returns:
        Словарь с результатами: per_sample_metrics, aggregated_metrics, sample_names
    """
    device = settings.DEVICE

    # Настройки по умолчанию
    use_faults = use_faults if use_faults is not None else settings.USE_FAULTS
    data_source = data_source or settings.DATA_SOURCE
    batch_size = batch_size or settings.BATCH_SIZE
    threshold = threshold or settings.TEST_THRESHOLD
    seed = seed or settings.SEED

    data_dir = settings.DATA_DIR
    cps_tiles_dir = settings.CPS_TILES_DIR

    if save_viz_dir is None:
        save_viz_dir = os.path.join(settings.LOGS_DIR, 'test_all_samples_viz')
    if save_metrics_path is None:
        save_metrics_path = os.path.join(settings.LOGS_DIR, 'test_all_samples_metrics.json')

    # Создаем директорию для визуализаций
    os.makedirs(save_viz_dir, exist_ok=True)

    print("=" * 80)
    print("EVALUATING MODEL ON ALL TEST SAMPLES")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data source: {data_source}")
    print(f"TARGET_HEIGHT: {settings.TARGET_HEIGHT}")
    print(f"TARGET_WIDTH: {settings.TARGET_WIDTH}")
    print(f"Data dir: {data_dir}")
    print(f"CPS tiles dir: {cps_tiles_dir}")
    print(f"Use faults: {use_faults}")
    print(f"Device: {device}")
    print(f"Threshold: {threshold}")
    print(f"Seed for split: {seed}")
    print("=" * 80)

    print("\n[STEP 1] Loading data and reproducing test split...")
    dir_to_load_data_from = data_dir if data_source == "png" else cps_tiles_dir
    print(f"Directory to load data from: {dir_to_load_data_from}")
    all_files = get_file_list(dir_to_load_data_from, data_source=data_source)

    if len(all_files) == 0:
        raise ValueError("No data files found!")

    # Воспроизводим разбиение с тем же seed что и при обучении
    if not custom_test_files:
        print("Using data split")
        _, _, test_files = split_data_by_groups(
            file_list=all_files,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed
        )
    else:
        print("Using custom test_files")
        test_files = custom_test_files
    

    print(f"Test files: {len(test_files)} files")

    # Создаем dataloader только для теста (напрямую, без create_dataloaders)
    test_dataset = GeologyTrapsDataset(
        file_list=test_files,
        data_dir=data_dir,
        cps_tiles_dir=cps_tiles_dir,
        augment=False,
        use_faults=use_faults,
        data_source=data_source
    )

    print(f"Test dataset initialized with {len(test_dataset)} samples")

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty! Check that test files have all required components (rgb, depth_norm, traps).")

    pin_memory_flag = torch.cuda.is_available()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory_flag,
        drop_last=False
    )

    print("\n[STEP 2] Loading model from checkpoint...")
    in_channels = 5 if use_faults else 4
    model = load_unetplusplus(
        in_channels=in_channels,
        classes=1,
        encoder_name='resnet34',
        encoder_weights=None,  # Не загружаем веса энкодера
        device=device
    )
    model = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    print(f"Model loaded with {in_channels} input channels")

    print("\n[STEP 3] Evaluating on each test sample...")

    metrics_calc = MetricsCalculator(threshold=threshold)

    per_sample_results = []
    all_predictions = []
    all_targets = []
    all_masks = []
    sample_names = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Processing samples')
        for batch in pbar:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask_map = batch.get('mask_map', None)
            if mask_map is not None:
                mask_map = mask_map.to(device)

            predictions = model(x)

            # Сохраняем для агрегированных метрик
            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())
            if mask_map is not None:
                all_masks.append(mask_map.cpu())

            # Вычисляем метрики для каждого семпла в батче
            batch_size_current = x.shape[0]
            for i in range(batch_size_current):
                pred_i = predictions[i:i+1]
                target_i = y[i:i+1]
                mask_i = mask_map[i:i+1] if mask_map is not None else None

                metrics = metrics_calc.compute_all(pred_i, target_i, mask_i)

                # Получаем имя семпла
                sample_paths = test_loader.dataset.samples[batch['sample_idx'][i].item()]
                first_path = list(sample_paths.values())[0]
                filename = Path(first_path).stem
                import re
                match = re.match(r'^(\d+)_[xy]_[^_]+_(.+)$', filename)
                if match:
                    number = match.group(1)
                    name = match.group(2)
                    sample_name = f"{number}_{name}"
                else:
                    sample_name = f"sample_{batch['sample_idx'][i].item()}"

                sample_names.append(sample_name)

                result = {
                    'sample_name': sample_name,
                    'dataset_idx': batch['sample_idx'][i].item(),
                    **metrics
                }
                per_sample_results.append(result)

                pbar.set_postfix({
                    'sample': sample_name,
                    'dice': f"{metrics['dice']:.3f}",
                    'iou': f"{metrics['iou']:.3f}"
                })

    print("\n[STEP 4] Computing aggregated metrics...")

    all_preds = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0) if all_masks else None

    aggregated_metrics = metrics_calc.compute_all(all_preds, all_targets, all_masks)
    aggregated_metrics['loss'] = None

    print("\n[STEP 5] Saving visualizations for each test sample...")

    test_loader.dataset.augment = False

    metrics_by_sample = {r['sample_name']: r for r in per_sample_results}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Saving visualizations')
        for batch in pbar:
            x = batch['x'].to(device)
            predictions = model(x)

            # Получаем реальные индексы семплов из батча
            real_sample_indices = batch['sample_idx'].tolist()

            # Визуализируем каждый семпл в батче, передавая реальные индексы
            visualize_test_results(
                batch=batch,
                predictions=predictions.cpu(),
                sample_indices=real_sample_indices,
                dataset=test_loader.dataset,
                save_path=save_viz_dir,
                alpha=0.4,
                metrics_by_sample=metrics_by_sample
            )

    print("\n[STEP 6] Saving metrics to JSON...")

    results = {
        'checkpoint_path': checkpoint_path,
        'threshold': threshold,
        'seed': seed,
        'n_test_samples': len(per_sample_results),
        'per_sample_metrics': per_sample_results,
        'aggregated_metrics': {
            'dice': aggregated_metrics['dice'],
            'iou': aggregated_metrics['iou'],
            'recall': aggregated_metrics['recall'],
            'precision': aggregated_metrics['precision'],
            'f1': aggregated_metrics['f1'],
            'fp_area': aggregated_metrics['fp_area'],
            'fn_area': aggregated_metrics['fn_area']
        },
        'sample_names': sample_names
    }

    with open(f"{save_metrics_path}/test_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nTotal test samples: {len(per_sample_results)}")

    print("\n--- Aggregated Metrics ---")
    print(f"Dice:      {aggregated_metrics['dice']:.4f}")
    print(f"IoU:       {aggregated_metrics['iou']:.4f}")
    print(f"Recall:    {aggregated_metrics['recall']:.4f}")
    print(f"Precision: {aggregated_metrics['precision']:.4f}")
    print(f"F1:        {aggregated_metrics['f1']:.4f}")
    print(f"FP Area:   {aggregated_metrics['fp_area']:.4f}")
    print(f"FN Area:   {aggregated_metrics['fn_area']:.4f}")

    print("\n--- Per-Sample Metrics ---")
    print(f"{'Sample Name':<30} {'Dice':>8} {'IoU':>8} {'Recall':>8} {'Prec':>8} {'F1':>8}")
    print("-" * 78)

    # Сортируем по Dice для удобства
    sorted_results = sorted(per_sample_results, key=lambda x: x['dice'], reverse=True)
    for r in sorted_results:
        print(f"{r['sample_name']:<30} {r['dice']:>8.4f} {r['iou']:>8.4f} {r['recall']:>8.4f} {r['precision']:>8.4f} {r['f1']:>8.4f}")

    print("\n" + "=" * 80)
    print(f"Visualizations saved to: {save_viz_dir}")
    print(f"Metrics saved to: {save_metrics_path}")
    print("=" * 80)

    return results
