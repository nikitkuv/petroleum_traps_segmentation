import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from visualization.visualize import visualize_full_cps_analysis


DATA_FOLDER = "raw"
SUBDATA_FOLDER = "test_cps"
HORIZON_NAME = "Top_D_50"

USE_FAULTS = True

STRUCTURAL_CPS_PATH = f"./data/{DATA_FOLDER}/{SUBDATA_FOLDER}/x_structuralNOisoline_{HORIZON_NAME}"
TRAPS_CPS_PATH = f"./data/{DATA_FOLDER}/{SUBDATA_FOLDER}/y_traps_{HORIZON_NAME}"
FAULTS_CPS_PATH = f"./data/{DATA_FOLDER}/{SUBDATA_FOLDER}/x_faults_{HORIZON_NAME}"

# STRUCTURAL_CPS_PATH = f"./data/cps/x_structuralNOisoline_H150_toptop1"
# TRAPS_CPS_PATH = f"./data/cps/y_traps_H150_toptop1"
# FAULTS_CPS_PATH = f"./data/cps//x_faults_H150_toptop1"


visualize_full_cps_analysis(
    rgb_cps_path=STRUCTURAL_CPS_PATH, 
    traps_cps_path=TRAPS_CPS_PATH, 
    faults_cps_path=FAULTS_CPS_PATH if USE_FAULTS else None,
    use_faults=USE_FAULTS,
    isoline_step=5,
    overlay_alpha=0.4
)
