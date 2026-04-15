import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from visualization.visualize import overlay_isolines_on_rgb


rgb, isolines, overlay = overlay_isolines_on_rgb(
    rgb_path='./data/images_cps_full/x_structuralNOisoline_Ach3-2-1_toptop1.png',
    isolines_path='./data/images_cps_full/x_isolines_Ach3-2-1_toptop1.png',
    alpha=0.5,
    save_path='./data/test_cps_visualization/overlay_Ach3-2-1_toptop1.png',
    show=False
)
