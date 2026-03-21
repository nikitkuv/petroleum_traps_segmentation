import cv2
import numpy as np
from pathlib import Path
from settings import settings


def generate_empty_fault_mask(template_path: str, output_path: str):

    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    height, width = template.shape[:2]
    empty_mask = np.full((height, width), 255, dtype=np.uint8)

    cv2.imwrite(output_path, empty_mask)
    print(f"Created empty fault mask: {output_path} ({width}x{height})")


def process_dataset(data_dir: str):
    
    data_path = Path(data_dir)
    
    rgb_files = list(data_path.glob("*_x_structuralNOisoline_*.png"))
    
    print(f"Found {len(rgb_files)} RGB maps")
    
    created_count = 0
    for rgb_file in rgb_files:
        parts = rgb_file.name.split('_')
        number = parts[0]
        name = "_".join(parts[3:])
        fault_mask_name = f"{number}_x_faults_{name}"
        fault_mask_path = data_path / fault_mask_name
        
        if not fault_mask_path.exists():
            generate_empty_fault_mask(
                template_path=str(rgb_file),
                output_path=str(fault_mask_path)
            )
            created_count += 1
        else:
            print(f"Fault mask already exists: {fault_mask_name}")
    
    print(f"Completed! Created {created_count} empty fault masks")


# if __name__ == "__main__":
#     process_dataset(str(settings.data_path))