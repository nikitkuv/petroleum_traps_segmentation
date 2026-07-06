import os
from typing import List, Tuple
from torch.utils.data import DataLoader
import torch
import json

from data.dataset import GeologyTrapsDataset
from utils.dataset_utils import parse_filename, collect_samples, extract_base_horizon
from settings import settings


def save_list(paths, filepath=settings.CUSTOM_TEST_FILES_DIR):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    files = [os.path.basename(p) for p in paths if p.endswith('.png') or p.endswith('.npy')]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(files, f, indent=2, ensure_ascii=False)


def load_list(filepath=settings.CUSTOM_TEST_FILES_DIR):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_file_list(data_dir: str) -> List[str]:
    """
    Получает список всех файлов данных из указанной директории.
    
    Формат: {number}_{x|y}_{type}_{name}.png
    
    Args:
        data_dir: Путь к директории с данными
    
    Returns:
        Список путей к файлам
    """
    valid_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.npy'):
                parsed = parse_filename(file)
                if parsed:
                    valid_files.append(os.path.join(root, file))
    
    print(f"Found {len(valid_files)} files matching format")
    return valid_files


def split_data_by_groups(
    file_list: List[str],
    train_horizons: List[str] = None,
    val_horizons: List[str] = None,
    test_horizons: List[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Разделяет данные на train/val/test по спискам горизонтов.

    Сначала из полного имени тайла извлекается базовый горизонт (тайл-суффикс
    вида 'bottop1'/'toptop5' отбрасывается через extract_base_horizon). Затем
    базовый горизонт сопоставляется со списками ПРЕФИКСОМ (startswith), а не
    подстрокой — это исключает случайные совпадения в середине имени (например,
    однобуквенный 'U' не цепляет горизонты, где 'U' встречается внутри).

    Файл попадает в выборку, если его базовый горизонт начинается с одной из
    строк соответствующего списка. Горизонт должен попасть ровно в одну
    выборку — иначе поднимается ValueError (защита от утечки и опечаток).

    Args:
        file_list: Список всех файлов
        train_horizons: Список префиксов горизонтов для train
        val_horizons: Список префиксов горизонтов для val
        test_horizons: Список префиксов горизонтов для test

    Returns:
        Кортеж (train_files, val_files, test_files)
    """
    train_horizons = train_horizons or settings.TRAIN_HORIZONS
    val_horizons = val_horizons or settings.VAL_HORIZONS
    test_horizons = test_horizons or settings.TEST_HORIZONS

    samples = collect_samples(file_list)

    if len(samples) == 0:
        raise ValueError(
            "No valid samples found! Files do not match the expected format:\n"
            "  {number}_{x|y}_{type}_{name}.png\n"
            "Examples:\n"
            "  001_x_structuralNOisoline_H150.png\n"
            "  001_y_traps_H150.png\n"
            "  001_x_structuralBlackWhite_H150.npy\n"
            "  001_x_isolines_H150.png\n"
            "  001_x_faults_H150.png\n"
            f"\nChecked {len(file_list)} files."
        )

    def get_split_for_name(name: str) -> str:
        """Определяет выборку по префиксу базового горизонта."""
        base = extract_base_horizon(name)
        matches = []
        if any(base.startswith(h) for h in train_horizons):
            matches.append('train')
        if any(base.startswith(h) for h in val_horizons):
            matches.append('val')
        if any(base.startswith(h) for h in test_horizons):
            matches.append('test')

        if len(matches) == 0:
            raise ValueError(
                f"Горизонт '{name}' (base='{base}') не попал ни в одну выборку. "
                f"Проверьте TRAIN_HORIZONS, VAL_HORIZONS, TEST_HORIZONS в settings."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Горизонт '{name}' (base='{base}') попал в несколько выборок: {matches}. "
                f"Исправьте пересечения в TRAIN_HORIZONS, VAL_HORIZONS, TEST_HORIZONS."
            )
        return matches[0]

    split_files = {'train': [], 'val': [], 'test': []}
    split_names = {'train': set(), 'val': set(), 'test': set()}

    for key, files in samples.items():
        if not files:
            continue
        first_file = list(files.values())[0]
        parsed = parse_filename(first_file)
        if not parsed:
            continue

        name = parsed['name']
        split = get_split_for_name(name)
        split_names[split].add(extract_base_horizon(name))

        for file_path in files.values():
            split_files[split].append(file_path)

    n_groups = len(split_names['train']) + len(split_names['val']) + len(split_names['test'])
    print(f"Found {n_groups} unique map groups (by name)")
    print(f"Total samples: {len(samples)}")
    print(f"Split: train={len(split_files['train'])} files (horizons: {sorted(split_names['train'])}), "
          f"val={len(split_files['val'])} files (horizons: {sorted(split_names['val'])}), "
          f"test={len(split_files['test'])} files (horizons: {sorted(split_names['test'])})")

    return split_files['train'], split_files['val'], split_files['test']


def create_dataloaders(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    data_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    use_faults: bool = False,
    use_rgb: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создает DataLoader для train/val/test выборок.
    
    Args:
        train_files: Файлы обучающей выборки
        val_files: Файлы валидационной выборки
        test_files: Файлы тестовой выборки
        data_dir: Путь к данным (CPS tiles)
        batch_size: Размер батча
        num_workers: Количество рабочих процессов
        use_faults: Использовать ли разломы
        use_rgb: Использовать ли RGB каналы
    
    Returns:
        Кортеж (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or settings.BATCH_SIZE
    num_workers = num_workers or settings.NUM_WORKERS
    data_dir = data_dir or settings.CPS_TILES_DIR
    
    train_dataset = GeologyTrapsDataset(
        file_list=train_files,
        data_dir=data_dir,
        augment=settings.AUGMENT_TRAIN,
        use_faults=use_faults,
        use_rgb=use_rgb,
    )
    
    val_dataset = GeologyTrapsDataset(
        file_list=val_files,
        data_dir=data_dir,
        augment=False,
        use_faults=use_faults,
        use_rgb=use_rgb,
    )
    
    test_dataset = GeologyTrapsDataset(
        file_list=test_files,
        data_dir=data_dir,
        augment=False,
        use_faults=use_faults,
        use_rgb=use_rgb,
    )

    pin_memory_flag = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
