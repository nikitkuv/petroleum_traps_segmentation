from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    
    # Размеры изображений
    TARGET_HEIGHT: int = 1248
    TARGET_WIDTH: int = 512
    
    # Порог бинаризации масок
    BINARY_THRESHOLD: int = 128
    
    # Пути
    DATA_DIR: str = './data/images/'
    CHECKPOINT_DIR: str = './checkpoints/'
    LOGS_DIR: str = './logs/'
    
    # Обучение
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 4
    LEARNING_RATE: float = 1e-4
    NUM_EPOCHS: int = 100
    
    # Аугментации
    AUGMENT_PROB: float = 0.5
    
    # Устройство
    DEVICE: str = 'cuda'
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
    
    @property
    def data_path(self) -> Path:
        return Path(self.DATA_DIR)
    
    @property
    def checkpoint_path(self) -> Path:
        return Path(self.CHECKPOINT_DIR)
    
    @property
    def logs_path(self) -> Path:
        return Path(self.LOGS_DIR)
    
    def create_dirs(self):
        """Создаёт необходимые директории."""
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)


# Глобальный экземпляр настроек
settings = Settings()