from evaluation.evaluate_on_test_data import evaluate_all_test_samples
from settings import settings


checkpoint_path = "checkpoints/v4_cps_tiles_640x448_faults_e-50_bs-4_lr-0.0003(0.1)_wd-0.005_rlp.pth"

# Воспроизводим детерминированный test-сплит по группам (горизонтам) из settings
# (custom_test_files=None внутри evaluate_all_test_samples вызывает split_data_by_groups).
# custom_test_files.json намеренно НЕ используем: он сохраняется во время обучения
# (pipeline.py) и рассинхронизируется при любом изменении TEST_HORIZONS — в текущем
# файле оказались train/val горизонты (Ach, H150, Yellow, CS1) и пропали 2 тестовых
# (TOP_D, BSMNT).
results = evaluate_all_test_samples(
    checkpoint_path=checkpoint_path,
    use_faults=settings.USE_FAULTS,
    batch_size=settings.BATCH_SIZE,
    threshold=settings.TEST_THRESHOLD,
    save_viz_dir=settings.LOGS_TEST_VIZ_DIR,
    save_metrics_path=settings.LOGS_TEST_VIZ_DIR,
    seed=settings.SEED,
    custom_test_files=None,
)
