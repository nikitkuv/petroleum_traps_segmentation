import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.cps_utils import (
    find_cps_files,
    save_large_images,
    split_into_tiles,
    load_existing_images,
    clean_cps_filenames
)
from settings import settings


# False + False = полный скрипт
ONLY_SAVE = False
ONLY_SPLIT = False


def main():
    if ONLY_SAVE and ONLY_SPLIT:
        print("ERROR: ONLY_SAVE and ONLY_SPLIT cannot both be True")
        sys.exit(1)

    cps_dir = settings.CPS_SOURCE_DIR
    full_images_dir = settings.CPS_FULL_DIR
    tiles_dir = settings.CPS_TILES_DIR

    print("=" * 60)
    print("CPS to PNG Converter and Tile Splitter")
    print("=" * 60)

    print(f"CPS directory: {cps_dir}")
    print(f"Full images output: {full_images_dir}")
    print(f"Tiles output: {tiles_dir}")
    print(f"Tile size: {settings.TARGET_WIDTH}x{settings.TARGET_HEIGHT}")

    if ONLY_SAVE:
        mode = "save_only"
    elif ONLY_SPLIT:
        mode = "split_only"
    else:
        mode = "both"

    print(f"\nMode: {mode}")
    print("=" * 60)

    print("\nStep 0: Cleaning CPS filenames...")
    clean_cps_filenames(cps_dir, suffixes_to_remove=[".cps3", "-UNIQ1"])

    horizons = None
    if mode in ("save_only", "both"):
        print("\n" + "=" * 60)
        print("Step 1: Finding CPS files...")
        horizons = find_cps_files(cps_dir)
        print(f"Found {len(horizons)} horizons:")
        for name in sorted(horizons.keys()):
            files = horizons[name]
            print(f"  {name}: structural={'structural' in files}, traps={'traps' in files}")

    images_data = None
    if mode in ("save_only", "both"):
        print("\n" + "=" * 60)
        print("Step 2: Converting CPS to PNG and saving large images...")
        images_data = save_large_images(horizons, full_images_dir)

    saved_files = []
    if mode in ("split_only", "both"):
        print("\n" + "=" * 60)
        print("Step 3: Splitting large images into tiles...")

        if horizons is None:
            print("Finding CPS files for tile splitting...")
            horizons = find_cps_files(cps_dir)
            print(f"Found {len(horizons)} horizons")

        if mode == "split_only":
            images_data = load_existing_images(horizons, full_images_dir)

        saved_files = split_into_tiles(images_data, tiles_dir)

    print("\n" + "=" * 60)
    print("Done!")
    if mode in ("split_only", "both"):
        print(f"Saved {len(saved_files)} tile files to {tiles_dir}")
    elif mode == "save_only":
        print(f"Large images saved to {full_images_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
    