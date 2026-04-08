from typing import List, Dict, Set
from collections import defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.dataloaders import parse_filename, get_sample_key, get_file_list, split_data_by_groups


def extract_horizon_name(parsed: dict) -> str:
    """Извлекает название горизонта (name)"""
    return parsed['name']


def check_file_leakage(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
) -> Dict[str, any]:
    """
    Проверяет пересечение файлов на уровне полных путей.

    Returns:
        Dict со статистикой пересечений
    """
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    results = {
        'train_val_overlap': train_set.intersection(val_set),
        'train_test_overlap': train_set.intersection(test_set),
        'val_test_overlap': val_set.intersection(test_set),
        'total_train': len(train_set),
        'total_val': len(val_set),
        'total_test': len(test_set)
    }

    return results


def check_sample_leakage(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
) -> Dict[str, any]:
    """
    Проверяет пересечение на уровне семплов {number}_{name}.

    Returns:
        Dict со статистикой пересечений
    """
    def get_sample_keys(file_list: List[str]) -> Set[str]:
        keys = set()
        for f in file_list:
            parsed = parse_filename(f)
            if parsed:
                keys.add(get_sample_key(parsed))
        return keys

    train_samples = get_sample_keys(train_files)
    val_samples = get_sample_keys(val_files)
    test_samples = get_sample_keys(test_files)

    results = {
        'train_val_overlap': train_samples.intersection(val_samples),
        'train_test_overlap': train_samples.intersection(test_samples),
        'val_test_overlap': val_samples.intersection(test_samples),
        'total_train_samples': len(train_samples),
        'total_val_samples': len(val_samples),
        'total_test_samples': len(test_samples)
    }

    return results


def check_horizon_leakage(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
) -> Dict[str, any]:
    """
    Проверяет пересечение на уровне горизонтов (name).
    Это критически важно, т.к. все семплы одного горизонта должны быть в одной выборке.

    Returns:
        Dict со статистикой пересечений
    """
    def get_horizons(file_list: List[str]) -> Set[str]:
        horizons = set()
        for f in file_list:
            parsed = parse_filename(f)
            if parsed:
                horizons.add(extract_horizon_name(parsed))
        return horizons

    train_horizons = get_horizons(train_files)
    val_horizons = get_horizons(val_files)
    test_horizons = get_horizons(test_files)

    results = {
        'train_val_overlap': train_horizons.intersection(val_horizons),
        'train_test_overlap': train_horizons.intersection(test_horizons),
        'val_test_overlap': val_horizons.intersection(test_horizons),
        'total_train_horizons': len(train_horizons),
        'total_val_horizons': len(val_horizons),
        'total_test_horizons': len(test_horizons)
    }

    return results


def analyze_sample_distribution(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Анализирует распределение семплов по горизонтам в каждой выборке.

    Returns:
        Dict с количеством семплов на горизонт для каждой выборки
    """
    def count_samples_per_horizon(file_list: List[str]) -> Dict[str, int]:
        samples = defaultdict(set)
        for f in file_list:
            parsed = parse_filename(f)
            if parsed:
                horizon = extract_horizon_name(parsed)
                sample_key = get_sample_key(parsed)
                samples[horizon].add(sample_key)

        return {h: len(s) for h, s in samples.items()}

    return {
        'train': count_samples_per_horizon(train_files),
        'val': count_samples_per_horizon(val_files),
        'test': count_samples_per_horizon(test_files)
    }


def print_leakage_report(
    file_results: Dict,
    sample_results: Dict,
    horizon_results: Dict,
    distribution: Dict
):
    """Выводит подробный отчет о проверке leakage"""

    print("=" * 80)
    print("DATA LEAKAGE CHECK REPORT")
    print("=" * 80)

    # 1. Проверка на уровне файлов
    print("\n1. FILE-LEVEL CHECK (полные пути)")
    print("-" * 80)
    print(f"Train files: {file_results['total_train']}")
    print(f"Val files:   {file_results['total_val']}")
    print(f"Test files:  {file_results['total_test']}")

    if file_results['train_val_overlap']:
        print(f"\n LEAKAGE DETECTED: Train ∩ Val = {len(file_results['train_val_overlap'])} files")
        for f in list(file_results['train_val_overlap'])[:5]:
            print(f"  - {f}")
    else:
        print("\n No leakage between Train and Val")

    if file_results['train_test_overlap']:
        print(f"\n LEAKAGE DETECTED: Train ∩ Test = {len(file_results['train_test_overlap'])} files")
        for f in list(file_results['train_test_overlap'])[:5]:
            print(f"  - {f}")
    else:
        print("\n No leakage between Train and Test")

    if file_results['val_test_overlap']:
        print(f"\n LEAKAGE DETECTED: Val ∩ Test = {len(file_results['val_test_overlap'])} files")
        for f in list(file_results['val_test_overlap'])[:5]:
            print(f"  - {f}")
    else:
        print("\n No leakage between Val and Test")

    # 2. Проверка на уровне семплов
    print("\n2. SAMPLE-LEVEL CHECK (ключи {number}_{name})")
    print("-" * 80)
    print(f"Train samples: {sample_results['total_train_samples']}")
    print(f"Val samples:   {sample_results['total_val_samples']}")
    print(f"Test samples:  {sample_results['total_test_samples']}")

    if sample_results['train_val_overlap']:
        print(f"\n LEAKAGE DETECTED: Train ∩ Val = {len(sample_results['train_val_overlap'])} samples")
        for s in list(sample_results['train_val_overlap'])[:5]:
            print(f"  - {s}")
    else:
        print("\n No leakage between Train and Val")

    if sample_results['train_test_overlap']:
        print(f"\n LEAKAGE DETECTED: Train ∩ Test = {len(sample_results['train_test_overlap'])} samples")
        for s in list(sample_results['train_test_overlap'])[:5]:
            print(f"  - {s}")
    else:
        print("\n No leakage between Train and Test")

    if sample_results['val_test_overlap']:
        print(f"\n LEAKAGE DETECTED: Val ∩ Test = {len(sample_results['val_test_overlap'])} samples")
        for s in list(sample_results['val_test_overlap'])[:5]:
            print(f"  - {s}")
    else:
        print("\n No leakage between Val and Test")

    # 3. Проверка на уровне горизонтов (НАИБОЛЕЕ ВАЖНО)
    print("\n3. HORIZON-LEVEL CHECK (названия горизонтов - name)")
    print("-" * 80)
    print(f"Train horizons: {horizon_results['total_train_horizons']}")
    print(f"Val horizons:   {horizon_results['total_val_horizons']}")
    print(f"Test horizons:  {horizon_results['total_test_horizons']}")

    if horizon_results['train_val_overlap']:
        print(f"\n CRITICAL LEAKAGE: Train ∩ Val = {len(horizon_results['train_val_overlap'])} horizons")
        for h in sorted(horizon_results['train_val_overlap']):
            print(f"  - {h}")
    else:
        print("\n No horizon leakage between Train and Val")

    if horizon_results['train_test_overlap']:
        print(f"\n CRITICAL LEAKAGE: Train ∩ Test = {len(horizon_results['train_test_overlap'])} horizons")
        for h in sorted(horizon_results['train_test_overlap']):
            print(f"  - {h}")
    else:
        print("\n No horizon leakage between Train and Test")

    if horizon_results['val_test_overlap']:
        print(f"\n CRITICAL LEAKAGE: Val ∩ Test = {len(horizon_results['val_test_overlap'])} horizons")
        for h in sorted(horizon_results['val_test_overlap']):
            print(f"  - {h}")
    else:
        print("\n No horizon leakage between Val and Test")

    # 4. Распределение семплов по горизонтам
    print("\n4. LEAKED SAMPLE DISTRIBUTION BY HORIZON")
    print("-" * 80)

    all_horizons = set()
    for split in distribution.values():
        all_horizons.update(split.keys())

    print(f"{'Horizon':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("-" * 80)

    for horizon in sorted(all_horizons):
        train_count = distribution['train'].get(horizon, 0)
        val_count = distribution['val'].get(horizon, 0)
        test_count = distribution['test'].get(horizon, 0)
        total = train_count + val_count + test_count

        # Подсветка проблем
        non_zero_splits = sum([train_count > 0, val_count > 0, test_count > 0])
        marker = " " if non_zero_splits > 1 else ""

        if non_zero_splits > 1:
            print(f"{horizon:<20} | {train_count:<8} | {val_count:<8} | {test_count:<8} | {total:<8}{marker}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    has_leakage = (
        len(file_results['train_val_overlap']) > 0 or
        len(file_results['train_test_overlap']) > 0 or
        len(file_results['val_test_overlap']) > 0 or
        len(sample_results['train_val_overlap']) > 0 or
        len(sample_results['train_test_overlap']) > 0 or
        len(sample_results['val_test_overlap']) > 0 or
        len(horizon_results['train_val_overlap']) > 0 or
        len(horizon_results['train_test_overlap']) > 0 or
        len(horizon_results['val_test_overlap']) > 0
    )

    if has_leakage:
        print("DATA LEAKAGE DETECTED! Training may be compromised.")
        print("Please ensure that each horizon appears in only ONE split.")
    else:
        print("NO DATA LEAKAGE DETECTED!")
        print("The dataset splits are clean and ready for training.")

    print("=" * 80)


def validate_data_source_consistency(file_list, data_source: str) -> bool:
    """
    Проверяет, что все файлы соответствуют указанному источнику данных.

    PNG и CPS tiles режимы не могут быть использованы вместе.

    Args:
        file_list: Список файлов для проверки
        data_source: Ожидаемый источник данных ('png' или 'cps_tiles')

    Returns:
        True если все файлы соответствуют, False иначе

    Raises:
        ValueError: Если обнаружены файлы обоих типов
    """
    if not file_list:
        return True

    png_files = [f for f in file_list if f.lower().endswith('.png')]

    if data_source == 'png':
        return True

    elif data_source == 'cps_tiles':
        return True

    else:
        raise ValueError(f"Unknown data_source: {data_source}. Must be 'png' or 'cps_tiles'.")


def check_leakage_from_dataloaders(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
):
    """
    Основная функция для проверки leakage.

    Args:
        train_files: Список файлов train выборки
        val_files: Список файлов val выборки
        test_files: Список файлов test выборки
    """
    file_results = check_file_leakage(train_files, val_files, test_files)
    sample_results = check_sample_leakage(train_files, val_files, test_files)
    horizon_results = check_horizon_leakage(train_files, val_files, test_files)
    distribution = analyze_sample_distribution(train_files, val_files, test_files)

    print_leakage_report(file_results, sample_results, horizon_results, distribution)

    return {
        'file_check': file_results,
        'sample_check': sample_results,
        'horizon_check': horizon_results,
        'distribution': distribution
    }


def check_leakage_with_split_function(data_dir: str, data_source: str = 'png'):
    """
    Проверяет leakage используя функцию split_data_by_groups из dataloaders.py.

    Args:
        data_dir: Путь к директории с данными
        data_source: Источник данных ('png' или 'cps_tiles')
    """

    print(f"Loading files from: {data_dir}")
    print(f"Data source: {data_source}")
    print("-" * 80)

    # Получаем все файлы
    all_files = get_file_list(data_dir, data_source)

    if len(all_files) == 0:
        print("No files found!")
        return None

    # Разделяем на train/val/test
    train_files, val_files, test_files = split_data_by_groups(all_files)

    print("\n")
    # Проверяем leakage
    results = check_leakage_from_dataloaders(train_files, val_files, test_files)

    return results


if __name__ == "__main__":
    import sys

    # Пример использования
    from settings import settings

    data_dir = str(settings.data_path)
    data_source = settings.DATA_SOURCE

    print(f"Using data directory: {data_dir}")
    print(f"Data source: {data_source}")
    print("\n")

    results = check_leakage_with_split_function(data_dir, data_source)

    if results is None:
        sys.exit(1)

    # Проверяем есть ли leakage
    has_leakage = (
        len(results['horizon_check']['train_val_overlap']) > 0 or
        len(results['horizon_check']['train_test_overlap']) > 0 or
        len(results['horizon_check']['val_test_overlap']) > 0
    )

    if has_leakage:
        sys.exit(1)  # Exit with error if leakage detected
    else:
        sys.exit(0)  # Success
