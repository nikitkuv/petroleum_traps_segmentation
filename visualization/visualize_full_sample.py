import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from visualization.visualize import visualize_full_cps_analysis


STRUCTURAL_CPS_PATH = "./data/cps/x_structuralNOisoline_Ach5_toptop77"
TRAPS_CPS_PATH = "./data/cps/y_traps_Ach5_toptop77"

# STRUCTURAL_CPS_PATH = "./data/raw/CP3_ver2/x_structuralNOisoline_B_I_2top21"
# TRAPS_CPS_PATH = "./data/raw/CP3_ver2/y_traps_B_I_2top21"


visualize_full_cps_analysis(
    rgb_cps_path=STRUCTURAL_CPS_PATH, 
    traps_cps_path=TRAPS_CPS_PATH, 
    isoline_step=5,
    overlay_alpha=0.4
)
