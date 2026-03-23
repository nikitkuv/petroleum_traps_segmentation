import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def move_to_images_folder(
    source_root: str = "data/выгрузка",
    target_dir = "data/images"
):

    os.makedirs(target_dir, exist_ok=True)

    for root, _, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(".png"):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)

                if os.path.exists(target_path):
                    name, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_path):
                        new_name = f"{name}_{counter}{ext}"
                        target_path = os.path.join(target_dir, new_name)
                        counter += 1

                shutil.move(source_path, target_path)
                print(f"Перемещён: {source_path} -> {target_path}")

    print("Готово!")


if __name__ == "__main__":
    move_to_images_folder()