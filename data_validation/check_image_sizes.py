import sys
import cv2
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from settings import settings


def check_image_sizes(folder_path: str):
    """
    Analyze image sizes in a given folder and print statistics.

    Args:
        folder_path: Path to the folder containing images
    """
    data_path = Path(folder_path)

    if not data_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    files = sorted(list(data_path.glob("*.png")))

    if len(files) == 0:
        print(f"No PNG images found in '{folder_path}'.")
        return

    print("=" * 70)
    print("📏 IMAGE SIZE ANALYSIS")
    print("=" * 70)
    print(f"Folder: {data_path.absolute()}")
    print(f"Total files: {len(files)}\n")

    # Группируем по размерам
    size_groups = defaultdict(list)

    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            size_groups[(h, w)].append(f.name)

    # Выводим статистику
    print("SIZE DISTRIBUTION:")
    print("-" * 70)

    for (h, w), file_list in sorted(size_groups.items(), key=lambda x: -len(x[1])):
        pct = len(file_list) / len(files) * 100
        print(f"{h}×{w}: {len(file_list)} files ({pct:.1f}%)")

        # Показываем первые 5 имён
        for fname in file_list[:5]:
            print(f"     - {fname}")
        if len(file_list) > 5:
            print(f"     ... and {len(file_list) - 5} more")
        print()

    # Сохранение отчёта
    report_path = data_path / 'image_size_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("IMAGE SIZE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Folder: {data_path.absolute()}\n")
        f.write(f"Total files: {len(files)}\n\n")

        for (h, w), file_list in sorted(size_groups.items(), key=lambda x: -len(x[1])):
            f.write(f"{h}×{w}: {len(file_list)} files\n")
            for fname in file_list:
                f.write(f"  {fname}\n")

    print(f"Report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    check_image_sizes(settings.CPS_FULL_DIR)
