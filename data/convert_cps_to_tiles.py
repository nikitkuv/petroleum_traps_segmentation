import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.cps_utils import (
    find_cps_files,
    save_large_images,
    split_into_tiles
)
from settings import settings


CREATE_TILES: bool = True


def main():
    cps_dir = settings.CPS_SOURCE_DIR
    full_images_dir = settings.CPS_FULL_DIR
    tiles_dir = settings.CPS_TILES_DIR

    print("=" * 60)
    print("CPS to PNG Converter and Tile Splitter")
    print("=" * 60)
    print(f"Data source: {settings.DATA_SOURCE}")

    # Проверка источника данных
    if settings.DATA_SOURCE != "cps_tiles":
        print(f"ERROR: DATA_SOURCE is set to '{settings.DATA_SOURCE}', but this script requires 'cps_tiles'")
        print("Please set DATA_SOURCE='cps_tiles' in settings before running this script.")
        sys.exit(1)
    
    print(f"CPS directory: {cps_dir}")
    print(f"Full images output: {full_images_dir}")
    print(f"Tiles output: {tiles_dir}")
    print(f"Tile size: {settings.TARGET_WIDTH}x{settings.TARGET_HEIGHT}")

    # Собираем cps файлы
    print("\n" + "=" * 60)
    print("Step 1: Finding CPS files...")
    horizons = find_cps_files(cps_dir)
    print(f"Found {len(horizons)} horizons:")
    for name in sorted(horizons.keys()):
        files = horizons[name]
        print(f"  {name}: structural={'structural' in files}, traps={'traps' in files}")

    # Конвертируем в PNG и сохранить полные карты
    print("\n" + "=" * 60)
    print("Step 2: Converting CPS to PNG and saving large images...")
    images_data = save_large_images(horizons, full_images_dir)

    if CREATE_TILES:
        # Разбиваем на тайлы
        print("\n" + "=" * 60)
        print("Step 3: Splitting large images into tiles...")
        saved_files = split_into_tiles(images_data, tiles_dir)

        print("\n" + "=" * 60)
        print("Done!")
        print(f"Saved {len(saved_files)} tile files to {tiles_dir}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Step 3: Skip tiles")


if __name__ == '__main__':
    main()
