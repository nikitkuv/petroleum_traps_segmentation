import sys
import cv2
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from settings import settings


def check_all_image_sizes():
    data_path = Path(settings.DATA_DIR)
    files = sorted(list(data_path.glob("*.png")))
    
    print("=" * 70)
    print("📏 IMAGE SIZE ANALYSIS")
    print("=" * 70)
    print(f"Total files: {len(files)}\n")
    
    # Группируем по размерам
    size_groups = defaultdict(list)
    
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape
            size_groups[(h, w)].append(f.name)
    
    # Выводим статистику
    print("📊 SIZE DISTRIBUTION:")
    print("-" * 70)
    
    for (h, w), file_list in sorted(size_groups.items(), key=lambda x: -len(x[1])):
        pct = len(file_list) / len(files) * 100
        status = "✅" if h <= settings.TARGET_HEIGHT and w <= settings.TARGET_WIDTH else "⚠️"
        print(f"{status} {h}×{w}: {len(file_list)} files ({pct:.1f}%)")
        
        # Показываем первые 5 имён
        for fname in file_list[:5]:
            print(f"      - {fname}")
        if len(file_list) > 5:
            print(f"      ... and {len(file_list) - 5} more")
        print()
    
    # Сохранение отчёта
    report_path = settings.logs_path / 'image_size_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("IMAGE SIZE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total files: {len(files)}\n\n")
        
        for (h, w), file_list in sorted(size_groups.items(), key=lambda x: -len(x[1])):
            f.write(f"{h}×{w}: {len(file_list)} files\n")
            for fname in file_list:
                f.write(f"  {fname}\n")
    
    print(f"📄 Report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    check_all_image_sizes()