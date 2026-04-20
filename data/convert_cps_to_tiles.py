import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.cps_utils import find_cps_files, split_cps_grids_into_tiles, clean_cps_filenames
from settings import settings


def main():
    cps_dir = settings.CPS_SOURCE_DIR
    tiles_dir = settings.CPS_TILES_DIR

    print("=" * 60)
    print("CPS to PNG Tile Splitter (Local Palette Mode)")
    print("=" * 60)
    
    if settings.DATA_SOURCE != "cps_tiles":
        print(f"ERROR: DATA_SOURCE is set to '{settings.DATA_SOURCE}', but this script requires 'cps_tiles'")
        sys.exit(1)

    print(f"CPS directory: {cps_dir}")
    print(f"Tiles output: {tiles_dir}")
    print(f"Tile size: {settings.TARGET_WIDTH}x{settings.TARGET_HEIGHT}")

    # Шаг 0: Очистка имен файлов (удаление мусорных суффиксов)
    print("\nStep 0: Cleaning CPS filenames...")
    clean_cps_filenames(cps_dir, suffixes_to_remove=[".cps3", "-UNIQ1"])

    # Шаг 1: Поиск CPS файлов
    print("Step 1: Finding CPS files...")
    horizons = find_cps_files(cps_dir)
    print(f"Found {len(horizons)} horizons")

    # Шаг 2: Непосредственная нарезка гридов и конвертация с локальными палитрами
    print("\nStep 2: Splitting grids into tiles and converting...")
    saved_files = split_cps_grids_into_tiles(horizons, tiles_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Saved {len(saved_files)} tile files to {tiles_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
    