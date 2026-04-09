import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.cps_utils import (
    find_cps_files,
    save_large_images,
    split_into_tiles
)
from settings import settings


def main():
    """Основная функция."""
    cps_dir = './data/cps/'
    full_images_dir = './data/images_cps_full/'
    tiles_dir = './data/images_cps/'

    print("=" * 60)
    print("CPS to PNG Converter and Tile Splitter")
    print("=" * 60)
    print(f"CPS directory: {cps_dir}")
    print(f"Full images output: {full_images_dir}")
    print(f"Tiles output: {tiles_dir}")
    print(f"Tile size: {settings.TARGET_WIDTH}x{settings.TARGET_HEIGHT}")

    # Шаг 1: Найти CPS файлы
    print("\n" + "=" * 60)
    print("Step 1: Finding CPS files...")
    horizons = find_cps_files(cps_dir)
    print(f"Found {len(horizons)} horizons:")
    for name in sorted(horizons.keys()):
        files = horizons[name]
        print(f"  {name}: structural={'structural' in files}, traps={'traps' in files}")

    # Шаг 2: Конвертировать в PNG и сохранить большие изображения
    print("\n" + "=" * 60)
    print("Step 2: Converting CPS to PNG and saving large images...")
    images_data = save_large_images(horizons, full_images_dir)

    # Шаг 3: Разбить на тайлы
    print("\n" + "=" * 60)
    print("Step 3: Splitting large images into tiles...")
    saved_files = split_into_tiles(images_data, tiles_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Saved {len(saved_files)} tile files to {tiles_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
