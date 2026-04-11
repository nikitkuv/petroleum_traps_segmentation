from evaluation.evaluate_on_test_data import evaluate_all_test_samples
from settings import settings


checkpoint_path = "checkpoints/no_faults_epochs-50_lr-0.0001_bs-4.pth"

results = evaluate_all_test_samples(
    checkpoint_path=checkpoint_path,
    use_faults=settings.USE_FAULTS,
    data_source=settings.DATA_SOURCE,
    batch_size=settings.BATCH_SIZE,
    threshold=settings.TEST_THRESHOLD,
    save_viz_dir=settings.LOGS_TEST_VIZ_DIR,
    save_metrics_path=settings.LOGS_TEST_VIZ_DIR,
    seed=settings.SEED
)
