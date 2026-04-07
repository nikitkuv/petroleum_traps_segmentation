import os
import argparse
import torch
import json
from pathlib import Path
from typing import Dict
import sys

sys.path.append(str(Path(__file__).parent.parent))


def load_anomaly_info(anomaly_dir: str) -> Dict:
    """Загрузить информацию об аномалиях из JSON."""
    list_path = os.path.join(anomaly_dir, 'anomalies_list.json')
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Не найден файл anomalies_list.json в {anomaly_dir}")
    
    with open(list_path, 'r') as f:
        return json.load(f)


def analyze_batch_grad_norms(
    model_path: str,
    batch_path: str,
    criterion,
    device: str = 'cuda'
) -> Dict:
    """
    Проанализировать вклад каждого семпла в батче в норму градиента.
    
    Returns:
        dict с пер-семпл статистикой
    """
    from models.unetplusplus import UNetPlusPlus
    from settings import settings
    
    # Загружаем модель
    checkpoint = torch.load(model_path, map_location=device)
    model = UNetPlusPlus(
        encoder_name='resnet34',
        in_channels=settings.in_channels,
        classes=1
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Загружаем батч
    batch_data = torch.load(batch_path, map_location=device)
    
    # Вычисляем пер-семпл градиентные нормы
    from training.gradient_tracker import compute_per_sample_grad_norms
    
    per_sample_norms = compute_per_sample_grad_norms(
        model=model,
        batch_data=batch_data,
        criterion=criterion,
        device=device
    )
    
    # Находим семплы с максимальными градиентами
    top_indices = torch.argsort(per_sample_norms, descending=True)
    
    results = {
        'batch_file': batch_path,
        'per_sample_grad_norms': per_sample_norms.tolist(),
        'top_samples': [
            {
                'index': int(idx),
                'grad_norm': float(per_sample_norms[idx])
            }
            for idx in top_indices[:5]
        ],
        'statistics': {
            'mean': float(per_sample_norms.mean()),
            'std': float(per_sample_norms.std()),
            'max': float(per_sample_norms.max()),
            'min': float(per_sample_norms.min())
        }
    }
    
    return results


def print_anomaly_summary(anomaly_info: Dict):
    """Вывести сводку по аномалиям."""
    print("=" * 70)
    print("GRADIENT ANOMALY SUMMARY")
    print("=" * 70)
    print(f"Total anomalies detected: {anomaly_info['total_anomalies']}")
    print(f"\nTracker configuration:")
    config = anomaly_info['tracker_config']
    print(f" - Absolute threshold: {config['abs_threshold']}")
    print(f" - Std multiplier: {config['std_multiplier']}")
    print(f" - Min samples for std: {config['min_samples_for_std']}")
    
    print(f"\nFinal statistics:")
    stats = anomaly_info['final_stats']
    print(f" - Running mean: {stats['running_mean']:.2f}")
    print(f" - Running std: {stats['running_std']:.2f}")
    print(f" - Total batches processed: {stats['count']}")
    
    print(f"\nAnomaly rate: {anomaly_info['total_anomalies'] / max(stats['count'], 1) * 100:.2f}%")
    
    if anomaly_info['anomalies']:
        print(f"\nFirst 10 anomalies:")
        print("-" * 70)
        print(f"{'Epoch':<8} {'Batch':<10} {'Grad Norm':<12} {'Running Mean':<14} {'Running Std':<12}")
        print("-" * 70)
        
        for i, anomaly in enumerate(anomaly_info['anomalies'][:10]):
            print(f"{anomaly['epoch']:<8} {anomaly['batch_idx']:<10} {anomaly['grad_norm']:<12.2f} "
                  f"{anomaly['running_mean']:<14.2f} {anomaly['running_std']:<12.2f}")


def main():
    parser = argparse.ArgumentParser(description='Анализ аномальных градиентов')
    parser.add_argument('--anomaly_dir', type=str, default='./gradient_anomalies/',
                        help='Директория с сохраненными аномалиями')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth',
                        help='Путь к чекпоинту модели для анализа')
    parser.add_argument('--analyze_all', action='store_true',
                        help='Анализировать все сохраненные батчи')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Проанализировать топ N батчей по grad_norm')
    
    args = parser.parse_args()
    
    # Загружаем информацию об аномалиях
    anomaly_info = load_anomaly_info(args.anomaly_dir)
    print_anomaly_summary(anomaly_info)
    
    # Находим файлы батчей
    anomaly_dir = Path(args.anomaly_dir)
    batch_files = sorted(anomaly_dir.glob('*_batch.pt'))
    
    print(f"\nFound {len(batch_files)} saved anomaly batches")
    
    if not batch_files:
        print("No batch files found for detailed analysis.")
        return
    
    # Сортируем батчи по grad_norm (из имени файла)
    def extract_grad_norm(path):
        try:
            # Формат: epochX_batchY_gradZ.ZZ_batch.pt
            grad_part = path.stem.split('_grad')[1].split('_batch')[0]
            return float(grad_part)
        except:
            return 0.0
    
    batch_files_sorted = sorted(batch_files, key=extract_grad_norm, reverse=True)
    
    # Анализируем топ батчи
    n_to_analyze = min(args.top_n, len(batch_files_sorted)) if not args.analyze_all else len(batch_files_sorted)
    
    if n_to_analyze > 0:
        print(f"\nAnalyzing top {n_to_analyze} batches with highest grad norms...")
        print("=" * 70)
        
        # Импортируем criterion
        from losses.losses import CombinedLoss
        criterion = CombinedLoss()
        
        results = []
        for i, batch_file in enumerate(batch_files_sorted[:n_to_analyze]):
            print(f"\n[{i+1}/{n_to_analyze}] Analyzing {batch_file.name}...")
            
            try:
                result = analyze_batch_grad_norms(
                    model_path=args.model_path,
                    batch_path=str(batch_file),
                    criterion=criterion
                )
                results.append(result)
                
                print(f" Per-sample grad norms: {result['per_sample_grad_norms']}")
                print(f" Top sample: index={result['top_samples'][0]['index']}, "
                      f"grad_norm={result['top_samples'][0]['grad_norm']:.2f}")
                print(f" Statistics: mean={result['statistics']['mean']:.2f}, "
                      f"std={result['statistics']['std']:.2f}, "
                      f"max={result['statistics']['max']:.2f}")
                
            except Exception as e:
                print(f" Error analyzing batch: {e}")
        
        # Сохраняем результаты анализа
        if results:
            output_path = os.path.join(args.anomaly_dir, 'detailed_analysis.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed analysis saved to: {output_path}")


if __name__ == '__main__':
    main()
