from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Literal
import torch


class Settings(BaseSettings):

    # Источник данных
    DATA_SOURCE: Literal['png', 'cps'] = 'png'

    # Работаем с разломами или нет
    USE_FAULTS: bool = False
    
    # Пути
    DATA_DIR: str = './data/images/'
    CPS_DIR: str = './data/cps/'
    CHECKPOINT_DIR: str = './checkpoints/'
    LOGS_DIR: str = './logs/'

    # Размеры изображений
    TARGET_HEIGHT: int = 1248
    TARGET_WIDTH: int = 512
    
    # Порог бинаризации масок
    BINARY_THRESHOLD: int = 128

    # Аугментации
    AUGMENT_PROB: float = 0.5

    # CPS настройки
    CPS_NULL_VALUE: float = -99999.0  
    CPS_VERTICAL_FLIP: bool = True    
    
    # Обучение
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    
    # Устройство
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
    
    @property
    def data_path(self) -> Path:
        return Path(self.DATA_DIR)

    @property
    def cps_path(self) -> Path:
        return Path(self.CPS_DIR)
    
    @property
    def checkpoint_path(self) -> Path:
        return Path(self.CHECKPOINT_DIR)
    
    @property
    def logs_path(self) -> Path:
        return Path(self.LOGS_DIR)

    @property
    def is_cps(self) -> bool:
        return self.DATA_SOURCE.lower() == 'cps'
    
    @property
    def in_channels(self) -> int:
        return 5 if self.USE_FAULTS else 4
    
    def create_dirs(self):
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)


# Глобальный экземпляр настроек
settings = Settings()