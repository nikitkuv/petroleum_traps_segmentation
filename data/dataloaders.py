import os
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import random
import torch

from data.dataset import GeologyTrapsDataset
from utils.dataset_utils import parse_filename, collect_samples
from settings import settings


def get_file_list(data_dir: str, data_source: str = 'png') -> List[str]:
    """
    Получает список всех файлов данных из указанной директории.
    
    Args:
        data_dir: Путь к директории с данными
        data_source: Источник данных ('png' или 'cps_tiles')
    
    Returns:
        Список путей к файлам
    """
    if data_source == 'png':
        # Ищем все PNG файлы в директории и поддиректориях
        png_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png'):
                    # Проверяем, что файл соответствует формату
                    parsed = parse_filename(file)
                    if parsed:
                        png_files.append(os.path.join(root, file))
        
        print(f"Found {len(png_files)} PNG files matching format {{number}}_{{x|y}}_{{type}}_{{name}}.png")
        return png_files
    
    elif data_source == 'cps_tiles':
        # Ищем все PNG файлы в директории images_cps/
        # Формат: {number}_{x|y}_{type}_{name}.png
        png_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png'):
                    # Проверяем, что файл соответствует формату
                    parsed = parse_filename(file)
                    if parsed:
                        png_files.append(os.path.join(root, file))
        
        print(f"Found {len(png_files)} CPS tile PNG files matching format {{number}}_{{x|y}}_{{type}}_{{name}}.png")
        return png_files
    
    else:
        raise ValueError(f"Unknown data_source: {data_source}")


def split_data_by_groups(
    file_list: List[str], 
    train_ratio: float = None, 
    val_ratio: float = None,
    seed: int = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Разделяет данные на train/val/test с учетом группировки по горизонтам.
    Все семплы из одного горизонта (name) попадают в одну выборку.
    
    Формат названий: {number}_{x|y}_{type}_{name}.png
    Группировка производится по {name}.
    
    Args:
        file_list: Список всех файлов
        train_ratio: Доля обучающей выборки
        val_ratio: Доля валидационной выборки
        test_ratio: Доля тестовой выборки
        seed: Random seed
    
    Returns:
        Кортеж (train_files, val_files, test_files)
    """
    train_ratio = train_ratio or settings.TRAIN_RATIO
    val_ratio = val_ratio or settings.VAL_RATIO
    seed = seed or settings.SEED

    random.seed(seed)
    
    # Сначала группируем файлы по семплам (number + name)
    samples = collect_samples(file_list)
    
    if len(samples) == 0:
        raise ValueError(
            "No valid samples found! Files do not match the expected format:\n"
            "  {number}_{x|y}_{type}_{name}.png\n"
            "Examples:\n"
            "  001_x_structuralNOisoline_H150.png\n"
            "  001_y_traps_H150.png\n"
            "  001_x_structuralBlackWhite_H150.png\n"
            f"\nChecked {len(file_list)} files."
        )

    # Затем группируем семплы по горизонтам (name)
    groups = {}  # name -> list of sample_keys
    for key, files in samples.items():
        if not files:
            continue
        first_file = list(files.values())[0]
        parsed = parse_filename(first_file)
        if not parsed:
            continue
        
        name = parsed['name']
        if name not in groups:
            groups[name] = []
        groups[name].append(key)
    
    print(f"Found {len(groups)} unique map groups (by name)")
    print(f"Total samples: {len(samples)}")
    
    # Разделяем группы горизонтов
    unique_names = list(groups.keys())
    random.shuffle(unique_names)
    
    n_total = len(unique_names)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))

    # Гарантируем что останется хотя бы 1 группа для test
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = max(1, n_total - n_train - 1)
    
    train_names = unique_names[:n_train]
    val_names = unique_names[n_train : n_train + n_val]
    test_names = unique_names[n_train + n_val:]

    assert set(train_names).isdisjoint(val_names), "Leakage: train and val share groups"
    assert set(train_names).isdisjoint(test_names), "Leakage: train and test share groups"
    assert set(val_names).isdisjoint(test_names), "Leakage: val and test share groups"
    
    # Собираем файлы по группам
    def build_split(names_list):
        split_files = []
        for name in names_list:
            for sample_key in groups[name]:
                # Добавляем все файлы семпла
                for file_path in samples[sample_key].values():
                    split_files.append(file_path)
        return split_files
    
    train_files = build_split(train_names)
    val_files = build_split(val_names)
    test_files = build_split(test_names)
    
    print(f"Split: train={len(train_files)} files ({len(train_names)} maps), "
          f"val={len(val_files)} files ({len(val_names)} maps), "
          f"test={len(test_files)} files ({len(test_names)} maps)")
    
    return train_files, val_files, test_files


def create_dataloaders(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    data_dir: str = None,
    cps_tiles_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    use_faults: bool = False,
    data_source: str = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создает DataLoader для train/val/test выборок.
    
    Args:
        train_files: Файлы обучающей выборки
        val_files: Файлы валидационной выборки
        test_files: Файлы тестовой выборки
        data_dir: Путь к данным
        cps_tiles_dir: Путь к CPS tiles данным (PNG файлы из images_cps/)
        batch_size: Размер батча
        num_workers: Количество рабочих процессов
        use_faults: Использовать ли разломы
        data_source: Источник данных ('png' или 'cps_tiles')
    
    Returns:
        Кортеж (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or settings.BATCH_SIZE
    num_workers = num_workers or settings.NUM_WORKERS
    data_dir = data_dir or settings.DATA_DIR
    cps_tiles_dir = cps_tiles_dir or settings.CPS_TILES_DIR
    data_source = data_source or settings.DATA_SOURCE
    
    # Создаем датасеты
    train_dataset = GeologyTrapsDataset(
        file_list=train_files,
        data_dir=data_dir,
        cps_tiles_dir=cps_tiles_dir,
        augment=settings.AUGMENT_TRAIN,
        use_faults=use_faults,
        data_source=data_source
    )
    
    val_dataset = GeologyTrapsDataset(
        file_list=val_files,
        data_dir=data_dir,
        cps_tiles_dir=cps_tiles_dir,
        augment=False,
        use_faults=use_faults,
        data_source=data_source
    )
    
    test_dataset = GeologyTrapsDataset(
        file_list=test_files,
        data_dir=data_dir,
        cps_tiles_dir=cps_tiles_dir,
        augment=False,
        use_faults=use_faults,
        data_source=data_source
    )

    pin_memory_flag = torch.cuda.is_available()
    
    # Создаем dataloaders
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
