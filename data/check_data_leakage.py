from typing import List, Dict, Set
from collections import defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.dataloaders import get_file_list, split_data_by_groups
from utils.dataset_utils import parse_filename, get_sample_key
from settings import settings


def extract_horizon_name(parsed: dict) -> str:
    """Извлекает название горизонта (name)"""
    return parsed['name']


def check_file_leakage(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
) -> Dict[str, any]:
    results = {
        'train_val_overlap': set(train_files).intersection(val_files),
        'train_test_overlap': set(train_files).intersection(test_files),
        'val_test_overlap': set(val_files).intersection(test_files),
        'total_train': len(train_files),
        'total_val': len(val_files),
        'total_test': len(test_files)
    }
    return results


def check_sample_leakage(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
) -> Dict[str, any]:
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

    return {
        'train_val_overlap': train_samples.intersection(val_samples),
        'train_test_overlap': train_samples.intersection(test_samples),
        'val_test_overlap': val_samples.intersection(test_samples),
        'total_train_samples': len(train_samples),
        'total_val_samples': len(val_samples),
        'total_test_samples': len(test_samples)
    }


def check_horizon_leakage(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
) -> Dict[str, any]:
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

    return {
        'train_val_overlap': train_horizons.intersection(val_horizons),
        'train_test_overlap': train_horizons.intersection(test_horizons),
        'val_test_overlap': val_horizons.intersection(test_horizons),
        'total_train_horizons': len(train_horizons),
        'total_val_horizons': len(val_horizons),
        'total_test_horizons': len(test_horizons)
    }


def check_horizon_lists_overlap(
    train_horizons: List[str],
    val_horizons: List[str],
    test_horizons: List[str]
) -> Dict[str, any]:
    """
    Проверяет, что списки горизонтов из settings не пересекаются.
    Одна подстрока может совпасть с другой (например, 'H' и 'H150').
    """
    train_set = set(train_horizons)
    val_set = set(val_horizons)
    test_set = set(test_horizons)

    results = {
        'train_val_overlap': train_set.intersection(val_set),
        'train_test_overlap': train_set.intersection(test_set),
        'val_test_overlap': val_set.intersection(test_set),
    }
    return results


def analyze_sample_distribution(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
) -> Dict[str, Dict[str, int]]:
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
    print("=" * 80)
    print("DATA LEAKAGE CHECK REPORT")
    print("=" * 80)

    print("\n1. FILE-LEVEL CHECK (полные пути)")
    print("-" * 80)
    print(f"Train files: {file_results['total_train']}")
    print(f"Val files:   {file_results['total_val']}")
    print(f"Test files:  {file_results['total_test']}")

    if file_results['train_val_overlap']:
        print(f"\n LEAKAGE DETECTED: Train ∩ Val = {len(file_results['train_val_overlap'])} files")
    else:
        print("\n No leakage between Train and Val")

    if file_results['train_test_overlap']:
        print(f"\n LEAKAGE DETECTED: Train ∩ Test = {len(file_results['train_test_overlap'])} files")
    else:
        print("\n No leakage between Train and Test")

    if file_results['val_test_overlap']:
        print(f"\n LEAKAGE DETECTED: Val ∩ Test = {len(file_results['val_test_overlap'])} files")
    else:
        print("\n No leakage between Val and Test")

    print("\n2. SAMPLE-LEVEL CHECK (ключи {number}_{name})")
    print("-" * 80)
    print(f"Train samples: {sample_results['total_train_samples']}")
    print(f"Val samples:   {sample_results['total_val_samples']}")
    print(f"Test samples:  {sample_results['total_test_samples']}")

    if sample_results['train_val_overlap']:
        print(f"\n LEAKAGE DETECTED: Train ∩ Val = {len(sample_results['train_val_overlap'])} samples")
    else:
        print("\n No leakage between Train and Val")

    if sample_results['train_test_overlap']:
        print(f"\n LEAKAGE DETECTED: Train ∩ Test = {len(sample_results['train_test_overlap'])} samples")
    else:
        print("\n No leakage between Train and Test")

    if sample_results['val_test_overlap']:
        print(f"\n LEAKAGE DETECTED: Val ∩ Test = {len(sample_results['val_test_overlap'])} samples")
    else:
        print("\n No leakage between Val and Test")

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

        non_zero_splits = sum([train_count > 0, val_count > 0, test_count > 0])

        if non_zero_splits > 1:
            print(f"{horizon:<20} | {train_count:<8} | {val_count:<8} | {test_count:<8} | {total:<8}")

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


def check_leakage_from_dataloaders(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str]
):
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


def check_leakage_with_split_function(data_dir: str):
    print(f"Loading files from: {data_dir}")
    print("-" * 80)

    # Проверяем пересечение списков горизонтов в settings
    lists_overlap = check_horizon_lists_overlap(
        settings.TRAIN_HORIZONS, settings.VAL_HORIZONS, settings.TEST_HORIZONS
    )
    has_list_overlap = any(lists_overlap.values())
    if has_list_overlap:
        print("WARNING: Horizon lists in settings have overlaps:")
        for pair, overlap in lists_overlap.items():
            if overlap:
                print(f"  {pair}: {overlap}")
        print()

    print(f"TRAIN_HORIZONS: {settings.TRAIN_HORIZONS}")
    print(f"VAL_HORIZONS:   {settings.VAL_HORIZONS}")
    print(f"TEST_HORIZONS:  {settings.TEST_HORIZONS}")
    print("-" * 80)

    all_files = get_file_list(data_dir)

    if len(all_files) == 0:
        print("No files found!")
        return None

    train_files, val_files, test_files = split_data_by_groups(all_files)

    print("\n")
    results = check_leakage_from_dataloaders(train_files, val_files, test_files)

    return results


if __name__ == "__main__":
    import sys
    from settings import settings

    data_dir = settings.CPS_TILES_DIR
    print(f"Using data directory: {data_dir}")
    print("\n")

    results = check_leakage_with_split_function(data_dir)

    if results is None:
        sys.exit(1)

    has_leakage = (
        len(results['horizon_check']['train_val_overlap']) > 0 or
        len(results['horizon_check']['train_test_overlap']) > 0 or
        len(results['horizon_check']['val_test_overlap']) > 0
    )

    if has_leakage:
        sys.exit(1)
    else:
        sys.exit(0)
