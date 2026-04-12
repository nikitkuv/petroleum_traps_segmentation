import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_utils import collect_samples


def check_samples(samples: dict) -> list:
    """
    Проверяет наличие всех необходимых карт для каждого семпла.

    Args:
        samples: dict от collect_samples()

    Returns:
        список семплов, у которых отсутствуют необходимые карты
    """
    incomplete_samples = []
    required_maps = ['rgb', 'depth_norm', 'traps']

    for sample_key, maps in samples.items():
        missing_maps = [m for m in required_maps if maps.get(m) is None]

        if missing_maps:
            incomplete_samples.append({
                'sample_key': sample_key,
                'missing': missing_maps,
                'available': {k: v for k, v in maps.items() if v is not None}
            })

    return incomplete_samples


def main():
    # Путь к папке с данными
    script_dir = Path(__file__).parent.parent  # /workspace
    data_dir = script_dir / 'data' / 'images'

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return

    print(f"Checking directory: {data_dir}")
    print(f"Mode: USE_FAULTS=False (checking for rgb, depth_norm, traps)")
    print("=" * 80)

    # Собираем семплы
    file_list = [f for f in os.listdir(str(data_dir)) if f.endswith('.png')]
    samples = collect_samples(file_list)
    total_samples = len(samples)

    print(f"Total unique samples found: {total_samples}")
    print()

    # Проверяем наличие всех карт
    incomplete = check_samples(samples)

    if not incomplete:
        print("✓ All samples have all required maps (rgb, depth_norm, traps)")
        return

    # Выводим проблемные семплы
    print(f"⚠ Found {len(incomplete)} incomplete samples:")
    print("=" * 80)

    for item in sorted(incomplete, key=lambda x: str(x['sample_key'])):
        sample_key = item['sample_key']
        missing = ', '.join(item['missing'])
        available = ', '.join(item['available'].keys()) if item['available'] else 'none'

        print(f"\nSample: {sample_key}")
        print(f"  Missing: {missing}")
        print(f"  Available: {available}")

        # Показываем имена файлов для доступных карт
        for map_type, filename in item['available'].items():
            print(f"    - {map_type}: {filename}")


if __name__ == '__main__':
    main()
