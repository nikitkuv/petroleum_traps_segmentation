from inference.predict import run_inference
from settings import settings


CHECKPOINT_PATH = "checkpoints/v4_cps_tiles_640x448_faults_e-50_bs-4_lr-0.0003(0.1)_wd-0.005_rlp.pth"

# Папка с ВХОДНЫМИ CPS-гридами в номенклатуре обучения. Замените на путь к своим новым гридам.
#   x_structuralNOisoline_<horizon>   — структурная карта глубин (обязательно)
#   x_faults_<horizon>                — разломы (опционально; при отсутствии канал = нули)
#   y_traps_<horizon>                 — ловушки GT (опционально; если есть — считаются метрики)
INPUT_CPS_DIR = settings.CPS_SOURCE_DIR

settings.create_dirs()

# Инференс по всем горизонтам папки. horizon_prefixes=[] => обрабатываются ВСЕ горизонты.
# Параметры постобработки/качества можно задать явно, иначе берутся из settings (см. docs/INFERENCE.md).
results = run_inference(
    checkpoint_path=CHECKPOINT_PATH,
    cps_dir=INPUT_CPS_DIR,
    horizon_prefixes=[],          # [] = все горизонты; напр. ["D_70"] — только D_70
    use_faults=settings.USE_FAULTS,
    threshold=settings.INFERENCE_THRESHOLD,
    min_trap_area_px=settings.INFERENCE_MIN_TRAP_AREA_PX,
    fill_holes=settings.INFERENCE_FILL_HOLES,
    tta=settings.INFERENCE_TTA,
    save_probability=settings.INFERENCE_SAVE_PROBABILITY,
    save_viz=True,                         # False = только CPS-гриды, без PNG визуализаций
    compute_metrics=True,                  # False = не считать метрики (y_traps GT не нужен)
)
