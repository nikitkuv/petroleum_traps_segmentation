from training.pipeline import run_full_pipeline
from settings import settings


model_name = 'v2_' \
f'cps_tiles_' \
f'{settings.TARGET_HEIGHT}x{settings.TARGET_WIDTH}_' \
f'{"faults" if settings.USE_FAULTS else "no_faults"}_' \
f'e-{settings.NUM_EPOCHS}_' \
f'bs-{settings.BATCH_SIZE}_' \
f'lr-{settings.LEARNING_RATE}({settings.ENCODER_LR_MULTIPLIER})_' \
f'wd-{settings.WEIGHT_DECAY}_' \
f'{"cos" if settings.SCHEDULER_NAME == "cosine_annealing" else "rlp"}'

test_metrics = run_full_pipeline(
    data_dir=settings.CPS_TILES_DIR,
    use_faults=settings.USE_FAULTS,
    overfit_check_mode=False,
    wandb_project='geology-traps-segmentation',
    wandb_run_name=model_name,
    n_epochs=settings.NUM_EPOCHS,
    batch_size=settings.BATCH_SIZE,
    learning_rate=settings.LEARNING_RATE,
    early_stopping_patience=settings.ES_PATIANCE
)
