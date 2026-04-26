import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from visualization.visualize import visualize_full_cps_analysis


HORIZON_NAME = "H150_TWT"

# Флаг использования разломов
USE_FAULTS = True

STRUCTURAL_CPS_PATH = f"./data/raw/CPS3_faults/x_structuralNOisoline_{HORIZON_NAME}"
TRAPS_CPS_PATH = f"./data/raw/CPS3_faults/y_traps_{HORIZON_NAME}"
FAULTS_CPS_PATH = f"./data/raw/CPS3_faults/x_faults_{HORIZON_NAME}"

# STRUCTURAL_CPS_PATH = f"./data/raw/CPS3_faults/x_structuralNOisoline_{HORIZON_NAME}"
# TRAPS_CPS_PATH = f"./data/raw/CPS3_faults/y_traps_{HORIZON_NAME}"


visualize_full_cps_analysis(
    rgb_cps_path=STRUCTURAL_CPS_PATH, 
    traps_cps_path=TRAPS_CPS_PATH, 
    faults_cps_path=FAULTS_CPS_PATH if USE_FAULTS else None,
    use_faults=USE_FAULTS,
    isoline_step=5,
    overlay_alpha=0.4
)
