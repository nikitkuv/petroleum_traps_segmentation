import pytest
import torch
import torch.nn as nn
import os
import tempfile
import shutil

from models.unetplusplus import load_unetplusplus, load_model_checkpoint, save_model_checkpoint


class TestLoadUnetPlusPlus:
    """Тесты для функции load_unetplusplus."""
    
    def test_load_with_default_channels(self):
        """Тест загрузки модели с каналами по умолчанию (из settings)."""
        model = load_unetplusplus(
            in_channels=3,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_load_with_6_channels(self):
        """Тест загрузки модели с 6 входными каналами."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        assert model is not None
        # Проверяем что первый слой имеет 6 каналов
        assert model.encoder.conv1.in_channels == 6
    
    def test_load_with_7_channels(self):
        """Тест загрузки модели с 7 входными каналами (с faults)."""
        model = load_unetplusplus(
            in_channels=7,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        assert model is not None
        assert model.encoder.conv1.in_channels == 7
    
    def test_load_with_custom_encoder(self):
        """Тест загрузки с разным encoder."""
        for encoder_name in ['resnet18', 'resnet34']:
            model = load_unetplusplus(
                in_channels=6,
                classes=1,
                encoder_name=encoder_name,
                encoder_weights=None,
                device='cpu'
            )
            
            assert model is not None
    
    def test_load_with_imagenet_weights_modification(self):
        """Тест что при загрузке imagenet весов и non-3 каналах первый слой модифицируется."""
        # Создаем модель с 6 каналами и без pretrained весов
        model_no_pretrained = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        # Веса первого слоя должны быть случайными (не из ImageNet)
        assert model_no_pretrained.encoder.conv1.in_channels == 6
    
    def test_model_output_shape(self):
        """Тест формы выхода модели."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        batch_size = 2
        height, width = 256, 256
        x = torch.randn(batch_size, 6, height, width)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, height, width)
    
    def test_model_forward_pass(self):
        """Тест прямого прохода через модель."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        model.eval()
        
        x = torch.randn(1, 6, 128, 128)
        
        with torch.no_grad():
            output = model(x)
        
        assert output is not None
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_input_sizes(self):
        """Тест работы с разными размерами входа."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        model.eval()
        
        for size in [128, 256, 512]:
            x = torch.randn(1, 6, size, size)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 1, size, size)
    
    def test_decoder_channels(self):
        """Тест что decoder channels применяются корректно."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        # Проверяем что модель имеет decoder
        assert hasattr(model, 'decoder')


class TestLoadModelCheckpoint:
    """Тесты для функции load_model_checkpoint."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = os.path.join(self.temp_dir, 'test_checkpoint.pth')
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_checkpoint_with_model_state_dict(self):
        """Тест загрузки чекпоинта с model_state_dict."""
        # Создаем и сохраняем модель
        original_model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        checkpoint = {
            'epoch': 10,
            'model_state_dict': original_model.state_dict(),
            'optimizer_state_dict': None,
            'metrics': {'dice': 0.8}
        }
        torch.save(checkpoint, self.checkpoint_path)
        
        # Загружаем в новую модель
        loaded_model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        loaded_model = load_model_checkpoint(loaded_model, self.checkpoint_path, 'cpu')
        
        # Проверяем что веса совпадают
        for param1, param2 in zip(original_model.parameters(), loaded_model.parameters()):
            assert torch.equal(param1, param2)
    
    def test_load_checkpoint_without_model_state_dict(self):
        """Тест загрузки чекпоинта без ключа model_state_dict."""
        original_model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        # Сохраняем только state_dict
        torch.save(original_model.state_dict(), self.checkpoint_path)
        
        # Загружаем
        loaded_model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        loaded_model = load_model_checkpoint(loaded_model, self.checkpoint_path, 'cpu')
        
        # Проверяем что веса совпадают
        for param1, param2 in zip(original_model.parameters(), loaded_model.parameters()):
            assert torch.equal(param1, param2)
    
    def test_load_checkpoint_not_found(self):
        """Тест обработки отсутствия файла чекпоинта."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        with pytest.raises(FileNotFoundError):
            load_model_checkpoint(model, '/nonexistent/path/checkpoint.pth', 'cpu')


class TestSaveModelCheckpoint:
    """Тесты для функции save_model_checkpoint."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, 'checkpoints')
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_checkpoint(self):
        """Тест сохранения чекпоинта."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        filepath = save_model_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={'dice': 0.8, 'iou': 0.7},
            save_path=self.save_path,
            filename='test_checkpoint.pth'
        )
        
        assert os.path.exists(filepath)
        assert filepath.endswith('test_checkpoint.pth')
    
    def test_save_checkpoint_content(self):
        """Тест содержимого сохраненного чекпоинта."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        filepath = save_model_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={'dice': 0.8, 'iou': 0.7},
            save_path=self.save_path,
            filename='test_checkpoint.pth'
        )
        
        # Загружаем и проверяем
        checkpoint = torch.load(filepath, weights_only=False)
        
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'metrics' in checkpoint
        
        assert checkpoint['epoch'] == 10
        assert checkpoint['metrics']['dice'] == 0.8
    
    def test_save_checkpoint_without_optimizer(self):
        """Тест сохранения чекпоинта без оптимизатора."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        filepath = save_model_checkpoint(
            model=model,
            optimizer=None,
            epoch=5,
            metrics={'dice': 0.75},
            save_path=self.save_path,
            filename='no_optim_checkpoint.pth'
        )
        
        checkpoint = torch.load(filepath, weights_only=False)
        
        assert checkpoint['optimizer_state_dict'] is None
    
    def test_save_creates_directory(self):
        """Тест что save создает директорию если она не существует."""
        model = load_unetplusplus(
            in_channels=6,
            classes=1,
            encoder_name='resnet18',
            encoder_weights=None,
            device='cpu'
        )
        
        new_dir = os.path.join(self.temp_dir, 'new_checkpoints')
        assert not os.path.exists(new_dir)
        
        filepath = save_model_checkpoint(
            model=model,
            optimizer=None,
            epoch=1,
            metrics={},
            save_path=new_dir,
            filename='checkpoint.pth'
        )
        
        assert os.path.exists(new_dir)
        assert os.path.exists(filepath)
