import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from visualization.visualize import overlay_isolines_on_rgb_from_cps


RGB_PATH: str = './data/cps/x_structuralNOisoline_U9_10_kolltop3'


overlay_isolines_on_rgb_from_cps(
    cps_path=RGB_PATH,
    isoline_step=5.0,
    alpha=0.6
)
