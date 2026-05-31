import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.evaluate_on_raw_cps import evaluate_on_raw_cps


CPS_DIR = "data/test_cps/"
CHECKPOINT_PATH = "checkpoints/v4_cps_tiles_640x448_faults_e-50_bs-4_lr-0.0003(0.1)_wd-0.005_rlp.pth"

USE_FAULTS = True
BATCH_SIZE = None
THRESHOLD = None
MIN_TRAPS_PIXELS = 0
KEEP_TEMP = False


evaluate_on_raw_cps(
    cps_dir=CPS_DIR,
    checkpoint_path=CHECKPOINT_PATH,
    batch_size=BATCH_SIZE,
    threshold=THRESHOLD,
    use_faults=USE_FAULTS,
    min_traps_pixels=MIN_TRAPS_PIXELS,
    keep_temp=KEEP_TEMP,
)
