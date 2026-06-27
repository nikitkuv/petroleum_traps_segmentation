import sys
import os
import re
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from data.dataloaders import get_file_list, split_data_by_groups
from utils.dataset_utils import parse_filename
from settings import settings


def get_base_horizon_name(full_name: str) -> str:
    """
    Удаляет последние 1-3 цифры (и предшествующий разделитель _, -) из имени горизонта.
    
    Примеры:
        Ach3-2-1_toptop29 -> Ach3-2-1_toptop
        H150_1            -> H150
        Ach3-2-1          -> Ach3-2 (если 1 - это индекс, который нужно убрать)
    """
    # Паттерн ищет опциональный разделитель (- или _) и от 1 до 3 цифр в самом конце строки.
    # Используем (100|[1-9]\d?), чтобы ограничить числа диапазоном 1-100 согласно условию.
    # Если вы хотите удалять цифры, только если перед ними был разделитель, 
    # используйте паттерн: r'[-_](100|[1-9]\d?)$'
    pattern = r'[-_]?(100|[1-9]\d?)$'
    
    base_name = re.sub(pattern, '', full_name)
    return base_name


def count_base_names(file_list):
    """Считает уникальные базовые названия горизонтов в списке файлов."""
    counter = Counter()
    
    for filepath in file_list:
        filename = os.path.basename(filepath)
        parsed = parse_filename(filename)
        
        if parsed:
            # Считаем только по файлам ловушек (y_traps), 
            # чтобы не учитывать один и тот же семпл 5 раз (rgb, depth, isolines, faults, traps)
            if parsed['role'] == 'y' and parsed['type'] == 'traps':
                base_name = get_base_horizon_name(parsed['name'])
                counter[base_name] += 1
                
    return counter


def main():
    data_dir = settings.CPS_TILES_DIR
    print(f"Analyzing data from: {data_dir}\n")
    
    # 1. Получаем список файлов и разбиваем на выборки
    all_files = get_file_list(data_dir)
    
    if len(all_files) == 0:
        print("No files found!")
        return
        
    train_files, val_files, test_files = split_data_by_groups(
        file_list=all_files,
    )
    
    # 2. Считаем статистику
    train_counts = count_base_names(train_files)
    val_counts = count_base_names(val_files)
    test_counts = count_base_names(test_files)
    
    # Собираем все уникальные названия горизонтов
    all_base_names = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
    
    # 3. Выводим красивую таблицу
    print("\n" + "=" * 70)
    print("HORIZON DISTRIBUTION ACROSS TRAIN / VAL / TEST SETS")
    print("(Counting based on number of unique sample tiles per base horizon)")
    print("=" * 70)
    
    header = f"{'Base Horizon Name':<35} | {'Train':>6} | {'Val':>6} | {'Test':>6} | {'Total':>6}"
    print(header)
    print("-" * 70)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for name in all_base_names:
        t = train_counts.get(name, 0)
        v = val_counts.get(name, 0)
        te = test_counts.get(name, 0)
        total = t + v + te
        
        total_train += t
        total_val += v
        total_test += te
        
        print(f"{name:<35} | {t:>6} | {v:>6} | {te:>6} | {total:>6}")
        
    print("-" * 70)
    grand_total = total_train + total_val + total_test
    print(f"{'TOTAL SAMPLES':<35} | {total_train:>6} | {total_val:>6} | {total_test:>6} | {grand_total:>6}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    