from evaluation.evaluate_on_test_data import evaluate_all_test_samples
from settings import settings
from data.dataloaders import load_list


checkpoint_path = "checkpoints/no_faults_epochs-50_lr-0.0001_bs-4_v2.pth"

custom_test_files = load_list()

results = evaluate_all_test_samples(
    checkpoint_path=checkpoint_path,
    use_faults=settings.USE_FAULTS,
    data_source=settings.DATA_SOURCE,
    batch_size=settings.BATCH_SIZE,
    threshold=settings.TEST_THRESHOLD,
    save_viz_dir=settings.LOGS_TEST_VIZ_DIR,
    save_metrics_path=settings.LOGS_TEST_VIZ_DIR,
    seed=settings.SEED,
    custom_test_files=custom_test_files
)
