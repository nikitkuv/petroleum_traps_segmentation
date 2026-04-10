from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal
import torch


class Settings(BaseSettings):

    # Источник данных
    DATA_SOURCE: Literal['png', 'cps_tiles'] = 'cps_tiles'

    # Работаем с разломами или нет
    USE_FAULTS: bool = False
    
    # Пути
    DATA_DIR: str = './data/images/'
    CPS_TILES_DIR: str = './data/images_cps/'
    CPS_FULL_DIR: str = './data/images_cps_full/'
    CHECKPOINT_DIR: str = './checkpoints/'
    LOGS_DIR: str = './logs/'
    LOGS_TRAIN_VIZ_DIR: str = './logs/visualizations/'
    LOGS_VAL_VIZ_DIR: str = './logs/val_visualizations/'
    LOGS_TEST_VIZ_DIR: str = './logs/test_visualizations/'
    LOGS_OVERFIT_CHECK_DIR: str = './logs/overfit_check/'
    GRAD_ANOMALIES_DIR: str = './gradient_anomalies/'

    # Размеры изображений
    if DATA_SOURCE == "cps_tiles":
        TARGET_HEIGHT: int = 864
        TARGET_WIDTH: int = 448
    else:
        TARGET_HEIGHT: int = 1248
        TARGET_WIDTH: int = 512
    
    # Порог бинаризации масок
    BINARY_THRESHOLD: int = 128

    # Аугментации
    AUGMENT_PROB: float = 0.5

    # CPS настройки
    TILE_OVERLAP_RATIO: float = 0.25
    CPS_NULL_VALUE: float = -999.0
    CPS_VERTICAL_FLIP: bool = False

    # Разделение данных
    SEED: int = 24
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1    
    
    # Обучение
    OVERFIT_SIZE: int = 2
    AUGMENT_TRAIN: bool = False
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 2
    LEARNING_RATE: float = 1e-4
    ENCODER_LR_MULTIPLIER: float = 0.1
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 50
    ES_PATIANCE: int = 15
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
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
    
    @property
    def data_path(self) -> Path:
        return Path(self.DATA_DIR)

    @property
    def cps_tiles_path(self) -> Path:
        return Path(self.CPS_TILES_DIR)
    
    @property
    def cps_full_path(self) -> Path:
        return Path(self.CPS_FULL_DIR)
    
    @property
    def checkpoint_path(self) -> Path:
        return Path(self.CHECKPOINT_DIR)
    
    @property
    def logs_path(self) -> Path:
        return Path(self.LOGS_DIR)
    
    @property
    def logs_train_viz_path(self) -> Path:
        return Path(self.LOGS_TRAIN_VIZ_DIR)

    @property
    def logs_val_viz_path(self) -> Path:
        return Path(self.LOGS_VAL_VIZ_DIR)
    
    @property
    def logs_test_viz_path(self) -> Path:
        return Path(self.LOGS_TEST_VIZ_DIR)
    
    @property
    def logs_overfit_check_path(self) -> Path:
        return Path(self.LOGS_OVERFIT_CHECK_DIR)
    
    @property
    def grad_anomalies_path(self) -> Path:
        return Path(self.GRAD_ANOMALIES_DIR)

    @property
    def is_cps_tiles(self) -> bool:
        return self.DATA_SOURCE.lower() == 'cps_tiles'
    
    @property
    def in_channels(self) -> int:
        return 5 if self.USE_FAULTS else 4
    
    def create_dirs(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        self.logs_train_viz_path.mkdir(parents=True, exist_ok=True)
        self.logs_val_viz_path.mkdir(parents=True, exist_ok=True)
        self.logs_test_viz_path.mkdir(parents=True, exist_ok=True)
        self.logs_overfit_check_path.mkdir(parents=True, exist_ok=True)
        self.grad_anomalies_path.mkdir(parents=True, exist_ok=True)


# Глобальный экземпляр настроек
settings = Settings()
