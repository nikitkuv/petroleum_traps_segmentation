from evaluation.evaluate_on_raw_cps import evaluate_on_raw_cps
from settings import settings


CHECKPOINT_PATH = "checkpoints/v4_cps_tiles_640x448_faults_e-50_bs-4_lr-0.0003(0.1)_wd-0.005_rlp.pth"

# Оценка модели на тестовых горизонтах.
#
# CPS-гриды берутся напрямую из data/cps/ (settings.CPS_SOURCE_DIR) и фильтруются по
# settings.TEST_HORIZONS — тем же детерминированным разбиением по горизонтам, что и при
# обучении. Затем они режутся на тайлы (воспроизведение data/images_cps/), прогоняются
# через модель. Метрики считаются по тайлам и микро-агрегируются по каждому горизонту.
#
# Визуализации предсказаний по тайлам и test_metrics.json сохраняются в папку
# logs/test_visualizations/{дата_время_начала_теста}.
#
# CPS-гриды читаем из источника, а не из data/images_cps/: тайлы пересоздаются заново,
# поэтому всегда синхронны с текущими settings (TEST_HORIZONS, размеры тайла и т.д.).
results = evaluate_on_raw_cps(
    checkpoint_path=CHECKPOINT_PATH,
    cps_dir=settings.CPS_SOURCE_DIR,
    horizon_prefixes=settings.TEST_HORIZONS,
    use_faults=settings.USE_FAULTS,
    batch_size=settings.BATCH_SIZE,
    threshold=settings.TEST_THRESHOLD,
    min_traps_pixels=None,   # None = settings.MIN_NUM_PIXS_OF_TRAPS_IN_TILES (воспроизводит data/images_cps/)
    keep_temp=False,
)
