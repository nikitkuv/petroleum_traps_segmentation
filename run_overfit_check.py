from training.pipeline import run_full_pipeline
from settings import settings


test_metrics = run_full_pipeline(
    data_dir=settings.DATA_DIR,
    use_faults=settings.USE_FAULTS,
    data_source=settings.DATA_SOURCE,
    overfit_check_mode=True,
    n_epochs=100,
    batch_size=settings.BATCH_SIZE,
    learning_rate=settings.LEARNING_RATE
)
