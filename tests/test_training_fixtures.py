import pytest
import torch
import os
import tempfile
import shutil


@pytest.fixture
def device():
    """Фикстура для устройства (CPU для тестов)."""
    return 'cpu'


@pytest.fixture
def sample_batch():
    """Создает тестовый батч данных."""
    batch_size = 2
    in_channels = 6
    height, width = 256, 256
    
    return {
        'x': torch.randn(batch_size, in_channels, height, width),
        'y': torch.randint(0, 2, (batch_size, 1, height, width)).float(),
        'mask_map': torch.ones(batch_size, 1, height, width),
        'sample_idx': torch.tensor([0, 1])
    }


@pytest.fixture
def sample_batch_with_faults():
    """Создает тестовый батч с faults (7 каналов)."""
    batch_size = 2
    in_channels = 7
    height, width = 256, 256
    
    return {
        'x': torch.randn(batch_size, in_channels, height, width),
        'y': torch.randint(0, 2, (batch_size, 1, height, width)).float(),
        'mask_map': torch.ones(batch_size, 1, height, width),
        'sample_idx': torch.tensor([0, 1])
    }


@pytest.fixture
def sample_batch_no_mask():
    """Создает тестовый батч без mask_map."""
    batch_size = 2
    in_channels = 6
    height, width = 256, 256
    
    return {
        'x': torch.randn(batch_size, in_channels, height, width),
        'y': torch.randint(0, 2, (batch_size, 1, height, width)).float(),
        'sample_idx': torch.tensor([0, 1])
    }


@pytest.fixture
def temp_dir():
    """Создает временную директорию для тестов."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def temp_checkpoint_path(temp_dir):
    """Путь для временных чекпоинтов."""
    return os.path.join(temp_dir, 'checkpoints')


@pytest.fixture
def small_model_params():
    """Параметры для создания маленькой модели в тестах."""
    return {
        'in_channels': 6,
        'classes': 1,
        'encoder_name': 'resnet18',
        'encoder_weights': None,
        'decoder_channels': (64, 32, 16, 8, 4),
        'decoder_dropout': 0.1
    }


@pytest.fixture
def mock_wandb_run():
    """Mock для wandb run."""
    class MockWandbRun:
        def __init__(self):
            self.name = "test_run"
            self.logs = []
        
        def log(self, data, commit=True):
            self.logs.append((data, commit))
        
        def finish(self):
            pass
        
        def save(self, path):
            pass
    
    return MockWandbRun()


@pytest.fixture
def gradient_norms_sequence():
    """Последовательность норм градиентов для тестирования трекера."""
    # Нормальные значения
    normal_grads = [1.0, 1.2, 0.9, 1.1, 1.0, 1.3, 0.8, 1.1]
    # Аномальное значение
    anomaly_grad = 10.0
    return normal_grads, anomaly_grad
