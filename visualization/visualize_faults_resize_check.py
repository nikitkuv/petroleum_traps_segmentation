import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from visualization.visualize import check_faults_resize_and_cut


HORIZON_NAME = "H150_TWT"

STRUCTURAL_CPS_PATH = f"./data/raw/CPS3_faults/x_structuralNOisoline_{HORIZON_NAME}"
FAULTS_CPS_PATH = f"./data/raw/CPS3_faults/x_faults_{HORIZON_NAME}"


check_faults_resize_and_cut(STRUCTURAL_CPS_PATH, FAULTS_CPS_PATH, HORIZON_NAME)
