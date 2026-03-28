"""
Модуль для загрузки данных и создания DataLoader.
Формат файлов: {number}_{x|y}_{type}_{name}.png
Примеры:
    001_x_structuralNOisoline_H150.png → rgb
    001_x_structuralBlackWhite_H150.png → depth_norm
    001_x_faults_H150.png → faults
    001_y_traps_H150.png → traps
Группировка производится по {name} (например, H150)
Семпл определяется как {number}_{name} - все файлы с одинаковым номером и именем месторождения
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import random

from dataset import GeologyTrapsDataset
from settings import settings


def get_file_list(data_dir: str, data_source: str = 'png') -> List[str]:
    """
    Получает список всех файлов данных из указанной директории.
    
    Args:
        data_dir: Путь к директории с данными
        data_source: Источник данных ('png' или 'cps')
    
    Returns:
        Список путей к файлам
    """
    if data_source == 'png':
        # Ищем все PNG файлы в директории и поддиректориях
        png_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.png'):
                    # Проверяем, что файл соответствует формату
                    parsed = parse_filename(file)
                    if parsed:
                        png_files.append(os.path.join(root, file))
        
        print(f"Found {len(png_files)} PNG files matching format {{number}}_{{x|y}}_{{type}}_{{name}}.png")
        return png_files
    
    elif data_source == 'cps':
        # Для CPS данных (если понадобится в будущем)
        cps_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.cps'):
                    cps_files.append(os.path.join(root, file))
        
        print(f"Found {len(cps_files)} CPS files")
        return cps_files
    
    else:
        raise ValueError(f"Unknown data_source: {data_source}")


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Парсит имя файла формата: {number}_{x|y}_{type}_{name}.png
    
    Примеры:
        001_x_structuralNOisoline_H150.png -> {'number': '001', 'role': 'x', 'type': 'structuralNOisoline', 'name': 'H150'}
        001_y_traps_H150.png             -> {'number': '001', 'role': 'y', 'type': 'traps', 'name': 'H150'}
    
    Returns:
        Словарь с компонентами или None, если формат не совпадает.
    """
    # Удаляем расширение
    name_no_ext = Path(filename).stem
    
    # Регулярное выражение: число_роль_тип_имя
    pattern = r'^(\d+)_(x|y)_([^_]+)_(.+)$'
    
    match = re.match(pattern, name_no_ext)
    
    if match:
        return {
            'number': match.group(1),
            'role': match.group(2),
            'type': match.group(3),
            'name': match.group(4)
        }
    return None


def get_sample_key(parsed: Dict[str, str]) -> str:
    """Формирует уникальный ключ семпла: {number}_{name}"""
    return f"{parsed['number']}_{parsed['name']}"


def collect_samples(file_list: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Группирует файлы по семплам.
    
    Семпл определяется парой (number, name).
    Ожидаемые файлы для полного семпла:
        - {number}_x_structuralNOisoline_{name}.png (rgb)
        - {number}_x_structuralBlackWhite_{name}.png (depth)
        - {number}_x_faults_{name}.png (faults)
        - {number}_y_traps_{name}.png (traps)
        
    Returns:
        Dict[key_sempla] -> { 'rgb': filename, 'depth': filename, 'faults': filename, 'traps': filename }
    """
    samples = {}
    
    for filename in file_list:
        parsed = parse_filename(filename)
        if not parsed:
            continue
            
        key = get_sample_key(parsed)
        
        if key not in samples:
            samples[key] = {}
            
        # Маппинг типа файла из имени в внутреннее имя канала
        file_type = parsed['type']
        role = parsed['role']
        
        if role == 'x':
            if file_type == 'structuralNOisoline':
                samples[key]['rgb'] = filename
            elif file_type == 'structuralBlackWhite':
                samples[key]['depth'] = filename
            elif file_type == 'faults':
                samples[key]['faults'] = filename
        elif role == 'y':
            if file_type == 'traps':
                samples[key]['traps'] = filename
                
    return samples


def split_data_by_groups(
    file_list: List[str], 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Разделяет данные на train/val/test с учетом группировки по месторождениям.
    Все семплы из одного месторождения (name) попадают в одну выборку.
    
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
    random.seed(seed)
    
    # Сначала группируем файлы по семплам (number + name)
    samples = collect_samples(file_list)
    
    # Затем группируем семплы по месторождениям (name)
    groups = {}  # name -> list of sample_keys
    for key, files in samples.items():
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
    
    # Разделяем группы месторождений
    unique_names = list(groups.keys())
    random.shuffle(unique_names)
    
    n_total = len(unique_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_names = unique_names[:n_train]
    val_names = unique_names[n_train : n_train + n_val]
    test_names = unique_names[n_train + n_val:]
    
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
    cps_dir: str = None,
    batch_size: int = None,
    num_workers: int = None,
    use_faults: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создает DataLoader для train/val/test выборок.
    
    Args:
        train_files: Файлы обучающей выборки
        val_files: Файлы валидационной выборки
        test_files: Файлы тестовой выборки
        data_dir: Путь к данным
        batch_size: Размер батча
        num_workers: Количество рабочих процессов
        use_faults: Использовать ли разломы
    
    Returns:
        Кортеж (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or settings.BATCH_SIZE
    num_workers = num_workers or settings.NUM_WORKERS
    data_dir = data_dir or settings.DATA_DIR
    cps_dir = cps_dir or settings.CPS_DIR
    
    # Создаем датасеты
    train_dataset = GeologyTrapsDataset(
        file_list=train_files,
        data_dir=data_dir,
        cps_dir=cps_dir,
        augment=True,
        use_faults=use_faults,
    )
    
    val_dataset = GeologyTrapsDataset(
        file_list=val_files,
        data_dir=data_dir,
        cps_dir=cps_dir,
        augment=False,
        use_faults=use_faults,
    )
    
    test_dataset = GeologyTrapsDataset(
        file_list=test_files,
        data_dir=data_dir,
        cps_dir=cps_dir,
        augment=False,
        use_faults=use_faults,
    )
    
    # Создаем dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
