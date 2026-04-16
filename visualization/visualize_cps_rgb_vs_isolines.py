import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from visualization.visualize import overlay_isolines_on_rgb


RGB_PATH: str = './data/images_cps/003_x_structuralNOisoline_U9_10_kolltop3.png'
ISOLINES_PATH: str = './data/images_cps/003_x_isolines_U9_10_kolltop3.png'
SAVE_PATH: str = './data/test_cps_visualization/overlay_003_U9_10_kolltop3.png'


rgb, isolines, overlay = overlay_isolines_on_rgb(
    rgb_path=RGB_PATH, 
    isolines_path=ISOLINES_PATH,
    alpha=0.5,
    save_path=SAVE_PATH,
    show=False
)
