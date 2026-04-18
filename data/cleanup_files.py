import os
import re
from pathlib import Path
from PIL import Image


DATA_DIR = "./data/images_cps_full"
TARGET_WIDTH = 579
TARGET_HEIGHT = 782


def parse_filename(filename: str) -> str | None:
    """Извлекает name из формата {x|y}_{type}_{name}.png"""
    stem = Path(filename).stem
    match = re.match(r'^(x|y)_([^_]+)_(.+)$', stem)
    return match.group(3) if match else None


def get_image_size(filepath: str) -> tuple[int, int] | None:
    """Возвращает (width, height) или None при ошибке"""
    try:
        with Image.open(filepath) as img:
            return img.size
    except Exception:
        return None


def main():
    print("="*60)
    print("ОЧИСТКА ФАЙЛОВ ПО РАЗМЕРУ И ИМЕНИ")
    print(f" Директория : {DATA_DIR}")
    print(f" Целевой размер: {TARGET_WIDTH}×{TARGET_HEIGHT}")
    print("="*60)

    if not os.path.isdir(DATA_DIR):
        print(f" Директория не найдена: {DATA_DIR}")
        return

    # 1. Сканирование
    print("\n Сканирование файлов...")
    all_files = {}  # path -> (w, h, name)
    skipped = 0

    for filepath in Path(DATA_DIR).rglob("*.png"):
        size = get_image_size(str(filepath))
        if size is None:
            skipped += 1
            continue
        w, h = size
        name = parse_filename(filepath.name)
        if name:
            all_files[str(filepath)] = (w, h, name)

    print(f" Обработано: {len(all_files)} файлов (пропущено: {skipped})")

    # 2. Поиск имён по целевому размеру
    names_to_delete = set()
    target_count = 0
    for fp, (w, h, name) in all_files.items():
        if w == TARGET_WIDTH and h == TARGET_HEIGHT:
            names_to_delete.add(name)
            target_count += 1

    if not names_to_delete:
        print(f"\n Файлов размером {TARGET_WIDTH}×{TARGET_HEIGHT} не найдено.")
        return

    print(f" Найдено {target_count} файлов с целевым размером ({len(names_to_delete)} уникальных имён)")

    # 3. Сбор всех файлов на удаление
    files_to_delete = [fp for fp, (_, _, name) in all_files.items() if name in names_to_delete]
    print(f" Всего под удаление попадёт: {len(files_to_delete)} файлов")

    # 4. Превью
    print("\n Примеры файлов (первые 10):")
    for f in files_to_delete[:10]:
        print(f"   • {Path(f).name}")
    if len(files_to_delete) > 10:
        print(f"   ... и ещё {len(files_to_delete) - 10}")

    # 5. Подтверждение
    resp = input("\n Удалить эти файлы? (y/n): ").strip().lower()
    if resp not in ("y", "yes", "д", "да"):
        print(" Отменено.")
        return

    # 6. Удаление
    print("\n Удаление...")
    success = fail = 0
    for fp in files_to_delete:
        try:
            os.remove(fp)
            success += 1
        except Exception as e:
            print(f"    Ошибка: {Path(fp).name} → {e}")
            fail += 1

    print(f"\n Готово! Удалено: {success} | Ошибок: {fail}")
    print("="*60)


if __name__ == "__main__":
    main()
