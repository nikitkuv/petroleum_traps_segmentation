from training.pipeline import run_full_pipeline
from settings import settings


model_name = f'{settings.DATA_SOURCE}_{settings.TARGET_HEIGHT}x{settings.TARGET_WIDTH}_{"faults" if settings.USE_FAULTS else "no_faults"}_e-{settings.NUM_EPOCHS}_bs-{settings.BATCH_SIZE}_lr-{settings.LEARNING_RATE}({settings.ENCODER_LR_MULTIPLIER})_wd-{settings.WEIGHT_DECAY}'

test_metrics = run_full_pipeline(
    data_dir=settings.DATA_DIR,
    use_faults=settings.USE_FAULTS,
    data_source=settings.DATA_SOURCE,
    overfit_check_mode=False,
    wandb_project='geology-traps-segmentation',
    wandb_run_name=model_name,
    n_epochs=settings.NUM_EPOCHS,
    batch_size=settings.BATCH_SIZE,
    learning_rate=settings.LEARNING_RATE,
    early_stopping_patience=settings.ES_PATIANCE
)
