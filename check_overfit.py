from pipeline import run_full_pipeline
from settings import settings


test_metrics = run_full_pipeline(
    data_dir=settings.DATA_DIR,
    use_faults=False,  # Без разломов
    data_source='png',
    overfit_check_mode=True,  # Режим проверки overfit
    n_epochs=100,
    batch_size=4,
    learning_rate=1e-4
)