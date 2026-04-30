from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
from pathlib import Path
import torch


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )

    # Работаем с разломами или нет
    USE_FAULTS: bool = False

    @computed_field
    @property
    def IN_CHANNELS(self) -> int:
        channels = 6  # RGB (3) + Depth (1) + Isolines (1) + MapMask (1)
        if self.USE_FAULTS:
            channels += 1  # + Faults (1)
        return channels
    
    # Пути
    CPS_SOURCE_DIR: str = './data/cps/'
    CPS_TILES_DIR: str = './data/images_cps/'
    CPS_FULL_DIR: str = './data/images_cps_full/'
    CHECKPOINT_DIR: str = './checkpoints/'
    LOGS_DIR: str = './logs/'
    LOGS_TRAIN_VIZ_DIR: str = './logs/visualizations/'
    LOGS_VAL_VIZ_DIR: str = './logs/val_visualizations/'
    LOGS_TEST_VIZ_DIR: str = './logs/test_visualizations/'
    LOGS_OVERFIT_CHECK_DIR: str = './logs/overfit_check/'
    GRAD_ANOMALIES_DIR: str = './gradient_anomalies/'
    CUSTOM_TEST_FILES_DIR: str = './logs/custom_test_files.json'
    CUSTOM_VAL_FILES_DIR: str = './logs/custom_val_files.json'

    # Размеры изображений
    TARGET_HEIGHT: int = 640
    TARGET_WIDTH: int = 448

    # Фильтрация тайлов
    # Если доля невалидных пикселей (края карты + разломы) в тайле больше этого значения, тайл исключается из обучения. 0.4 = 40%.
    MAX_NODATA_RATIO: float = 0.4  
    
    # Порог бинаризации масок
    BINARY_THRESHOLD: int = 128

    # Аугментации
    AUGMENT_PROB: float = 0.5

    # CPS настройки
    TILE_OVERLAP_RATIO: float = 0.25
    MIN_NUM_PIXS_OF_TRAPS_IN_TILES: int = 100
    CPS_NULL_VALUE: float = -999.0
    CPS_VERTICAL_FLIP: bool = False

    # Разделение данных
    SEED: int = 24
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15    
    
    # Обучение
    OVERFIT_SIZE: int = 2
    AUGMENT_TRAIN: bool = False
    ENCODER_NAME: str = "resnet34"
    SCHEDULER_NAME: str = "reduce_lr_plateau"
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    LEARNING_RATE: float = 3e-4
    ENCODER_LR_MULTIPLIER: float = 0.1
    WEIGHT_DECAY: float = 5e-3
    DECODER_DROPOUT: float = 0.2
    NUM_EPOCHS: int = 50
    ES_PATIANCE: int = 7
    GRADIENT_ACC_STEPS: int = 1

    # Трекинг градиентов
    LOG_GRADIENTS: bool = True
    TRACK_GRADIENT_ANOMALIES: bool = True
    GRAD_ALPHA: float = 0.95
    ABS_THRESHOLD: float = 10
    STD_MULTIPLIER: float = 3
    MIN_SAMPLES_FOR_STD: int = 20

    # Loss
    BCE_WEIGHT_RATIO: float = 0.5
    DICE_WEIGHT_RATIO: float = 0.5

    # Метрики
    TEST_THRESHOLD: float = 0.5
    
    # Устройство
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def data_path(self) -> Path:
        return Path(self.CPS_TILES_DIR)
    
    @property
    def checkpoint_path(self) -> Path:
        return Path(self.CHECKPOINT_DIR)
    
    @property
    def logs_path(self) -> Path:
        return Path(self.LOGS_DIR)
    
    def create_dirs(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)


# Глобальный экземпляр настроек
settings = Settings()
