import os
import json
from pathlib import Path
from typing import Dict
import torch
from tqdm import tqdm

from settings import settings
from data.dataloaders import get_file_list, split_data_by_groups, create_dataloaders
from models.unetplusplus import load_unetplusplus, load_model_checkpoint
from metrics.metrics import MetricsCalculator
from visualization.visualize import visualize_test_results


def evaluate_all_test_samples(
    checkpoint_path: str,
    data_dir: str = None,
    use_faults: bool = None,
    data_source: str = None,
    batch_size: int = None,
    threshold: float = None,
    save_viz_dir: str = None,
    save_metrics_path: str = None,
    seed: int = None
) -> Dict:
    """
    Загружает модель и оценивает её на всех тестовых семплах.

    Args:
        checkpoint_path: Путь к чекпоинту модели
        data_dir: Путь к данным
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
    data_dir = data_dir or settings.DATA_DIR
    use_faults = use_faults if use_faults is not None else settings.USE_FAULTS
    data_source = data_source or settings.DATA_SOURCE
    batch_size = batch_size or settings.BATCH_SIZE
    threshold = threshold or settings.TEST_THRESHOLD
    seed = seed or settings.SEED

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
    print(f"Data dir: {data_dir}")
    print(f"Use faults: {use_faults}")
    print(f"Device: {device}")
    print(f"Threshold: {threshold}")
    print(f"Seed for split: {seed}")
    print("=" * 80)

    print("\n[STEP 1] Loading data and reproducing test split...")
    all_files = get_file_list(data_dir, data_source=data_source)

    if len(all_files) == 0:
        raise ValueError("No data files found!")

    # Воспроизводим разбиение с тем же seed что и при обучении
    train_files, val_files, test_files = split_data_by_groups(
        file_list=all_files,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=seed
    )

    print(f"Test files: {len(test_files)} files")

    # Создаем dataloader только для теста
    _, _, test_loader = create_dataloaders(
        train_files=[],
        val_files=[],
        test_files=test_files,
        data_dir=data_dir,
        batch_size=batch_size,
        use_faults=use_faults,
        data_source=data_source,
        augment_train=False
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

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Saving visualizations')
        for batch in pbar:
            x = batch['x'].to(device)
            predictions = model(x)

            # Передаем все индексы семплов из текущего батча
            sample_indices = list(range(x.shape[0]))

            # Визуализируем каждый семпл в батче
            # Функция сама сохранит изображения в save_viz_dir
            visualize_test_results(
                batch=batch,
                predictions=predictions.cpu(),
                sample_indices=sample_indices,
                dataset=test_loader.dataset,
                save_path=save_viz_dir,
                alpha=0.4
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

    with open(save_metrics_path, 'w', encoding='utf-8') as f:
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained model on all test samples')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--use-faults', action='store_true', default=None, help='Use faults as input')
    parser.add_argument('--data-source', type=str, default=None, choices=['png', 'cps_tiles'], help='Data source')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--threshold', type=float, default=None, help='Binary threshold')
    parser.add_argument('--save-viz-dir', type=str, default=None, help='Directory to save visualizations')
    parser.add_argument('--save-metrics', type=str, default=None, help='Path to save metrics JSON')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for data split')

    args = parser.parse_args()

    # Определяем use_faults из названия чекпоинта если не указано
    use_faults = args.use_faults
    if use_faults is None:
        checkpoint_name = os.path.basename(args.checkpoint).lower()
        use_faults = 'faults' in checkpoint_name and 'no_faults' not in checkpoint_name
        print(f"Auto-detected use_faults={use_faults} from checkpoint name")

    results = evaluate_all_test_samples(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        use_faults=use_faults,
        data_source=args.data_source,
        batch_size=args.batch_size,
        threshold=args.threshold,
        save_viz_dir=args.save_viz_dir,
        save_metrics_path=args.save_metrics,
        seed=args.seed
    )