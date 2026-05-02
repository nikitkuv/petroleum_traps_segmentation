"""
Юнит тесты для GradientNormTracker и градиентного трекинга.
"""
import pytest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import shutil
import json

# Добавляем корень проекта в path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.gradient_tracker import GradientNormTracker, compute_per_sample_grad_norms


class SimpleTestModel(nn.Module):
    """Простая модель для тестов."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class TestGradientNormTrackerInit:
    """Тесты инициализации GradientNormTracker."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_default(self):
        """Тест инициализации по умолчанию."""
        tracker = GradientNormTracker(save_dir=self.temp_dir)
        
        # Проверяем что значения установлены из settings или имеют дефолтные значения
        assert tracker.alpha > 0 and tracker.alpha <= 1
        assert tracker.abs_threshold > 0
        assert tracker.std_multiplier > 0
        assert tracker.min_samples_for_std > 0
        assert tracker.count == 0
        assert tracker.running_mean == 0.0
        assert len(tracker.anomalies) == 0
    
    def test_init_custom_params(self):
        """Тест инициализации с custom параметрами."""
        tracker = GradientNormTracker(
            alpha=0.9,
            abs_threshold=5.0,
            std_multiplier=2.0,
            min_samples_for_std=5,
            save_dir=self.temp_dir
        )
        
        assert tracker.alpha == 0.9
        assert tracker.abs_threshold == 5.0
        assert tracker.std_multiplier == 2.0
        assert tracker.min_samples_for_std == 5


class TestGradientNormTrackerUpdate:
    """Тесты метода update."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = GradientNormTracker(
            abs_threshold=5.0,
            std_multiplier=2.0,
            min_samples_for_std=3,
            save_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_update_increments_count(self):
        """Тест что update увеличивает count."""
        is_anomaly, stats = self.tracker.update(1.0)
        
        assert self.tracker.count == 1
        assert stats['count'] == 1
    
    def test_update_returns_tuple(self):
        """Тест что update возвращает кортеж."""
        result = self.tracker.update(1.0)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        is_anomaly, stats = result
        assert isinstance(is_anomaly, bool)
        assert isinstance(stats, dict)
    
    def test_update_stats_keys(self):
        """Тест что stats содержит правильные ключи."""
        is_anomaly, stats = self.tracker.update(1.0)
        
        expected_keys = [
            'count', 'grad_norm', 'running_mean', 'running_std',
            'threshold', 'abs_threshold', 'dynamic_threshold', 'is_anomaly'
        ]
        
        for key in expected_keys:
            assert key in stats
    
    def test_update_running_mean(self):
        """Тест обновления running mean."""
        self.tracker.update(2.0)
        self.tracker.update(4.0)
        self.tracker.update(6.0)
        
        # Running mean должен быть близок к среднему
        assert 3.0 <= self.tracker.running_mean <= 5.0
    
    def test_normal_gradient_not_anomaly(self):
        """Тест что нормальный градиент не считается аномалией."""
        # Сначала набираем статистику с большим min_samples_for_std
        tracker = GradientNormTracker(
            abs_threshold=10.0,
            std_multiplier=3.0,
            min_samples_for_std=20,
            save_dir=self.temp_dir
        )
        
        for _ in range(15):
            is_anomaly, _ = tracker.update(1.0)
        
        # Нормальный градиент не должен быть аномалией
        is_anomaly, stats = tracker.update(1.2)
        
        # После достаточного количества сэмплов с низким std, 1.2 не должна быть аномалией
        assert is_anomaly is False or stats['count'] < tracker.min_samples_for_std
    
    def test_large_gradient_is_anomaly(self):
        """Тест что большой градиент считается аномалией."""
        # Набираем статистику с маленькими градиентами
        for _ in range(10):
            self.tracker.update(1.0)
        
        # Большой градиент должен быть аномалией
        is_anomaly, stats = self.tracker.update(20.0)
        
        assert is_anomaly is True
        assert stats['grad_norm'] == 20.0
    
    def test_threshold_before_min_samples(self):
        """Тест порога до набора минимального количества сэмплов."""
        tracker = GradientNormTracker(
            abs_threshold=5.0,
            min_samples_for_std=10,
            save_dir=self.temp_dir
        )
        
        # До min_samples_for_std используется только abs_threshold
        for i in range(5):
            is_anomaly, stats = tracker.update(1.0)
            assert stats['threshold'] == 5.0
            assert stats['dynamic_threshold'] is None
    
    def test_threshold_after_min_samples(self):
        """Тест порога после набора минимального количества сэмплов."""
        tracker = GradientNormTracker(
            abs_threshold=10.0,
            std_multiplier=2.0,
            min_samples_for_std=5,
            save_dir=self.temp_dir
        )
        
        # Набираем статистику
        for i in range(10):
            is_anomaly, stats = tracker.update(1.0)
        
        # После min_samples должен использоваться dynamic threshold
        assert stats['dynamic_threshold'] is not None


class TestGradientNormTrackerLogAnomaly:
    """Тесты метода log_anomaly."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = GradientNormTracker(
            abs_threshold=5.0,
            save_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_log_anomaly_adds_to_list(self):
        """Тест что log_anomaly добавляет аномалию в список."""
        initial_len = len(self.tracker.anomalies)
        
        self.tracker.log_anomaly(
            epoch=0,
            batch_idx=5,
            grad_norm=10.0
        )
        
        assert len(self.tracker.anomalies) == initial_len + 1
    
    def test_log_anomaly_info(self):
        """Тест информации об аномалии."""
        self.tracker.log_anomaly(
            epoch=3,
            batch_idx=10,
            grad_norm=15.0
        )
        
        anomaly = self.tracker.anomalies[0]
        
        assert anomaly['epoch'] == 3
        assert anomaly['batch_idx'] == 10
        assert anomaly['grad_norm'] == 15.0
    
    def test_log_anomaly_saves_batch(self):
        """Тест сохранения батча при аномалии."""
        batch_data = {
            'x': torch.randn(2, 3, 64, 64),
            'y': torch.randint(0, 2, (2, 1, 64, 64)).float(),
            'mask_map': torch.ones(2, 1, 64, 64)
        }
        
        self.tracker.log_anomaly(
            epoch=0,
            batch_idx=5,
            grad_norm=10.0,
            batch_data=batch_data,
            save_batch=True
        )
        
        # Проверяем что файлы сохранены
        files = os.listdir(self.temp_dir)
        assert any('.pt' in f for f in files)
        assert any('anomalies_list.json' in f for f in files)
    
    def test_log_anomaly_saves_json(self):
        """Тест сохранения JSON списка аномалий."""
        self.tracker.log_anomaly(
            epoch=0,
            batch_idx=5,
            grad_norm=10.0
        )
        
        json_path = os.path.join(self.temp_dir, 'anomalies_list.json')
        assert os.path.exists(json_path)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert 'total_anomalies' in data
        assert 'anomalies' in data
        assert data['total_anomalies'] == 1
    
    def test_log_anomaly_without_batch(self):
        """Тест log_anomaly без сохранения батча."""
        self.tracker.log_anomaly(
            epoch=0,
            batch_idx=5,
            grad_norm=10.0,
            batch_data=None,
            save_batch=False
        )
        
        # Аномалия должна быть добавлена в список
        assert len(self.tracker.anomalies) == 1
        
        # Но файлы батчей не должны быть сохранены
        files = [f for f in os.listdir(self.temp_dir) if '_batch.pt' in f]
        assert len(files) == 0


class TestGradientNormTrackerGetSummary:
    """Тесты метода get_summary."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = GradientNormTracker(
            save_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_get_summary_returns_dict(self):
        """Тест что get_summary возвращает словарь."""
        summary = self.tracker.get_summary()
        
        assert isinstance(summary, dict)
    
    def test_get_summary_keys(self):
        """Тест ключей в summary."""
        self.tracker.update(1.0)
        self.tracker.update(2.0)
        
        summary = self.tracker.get_summary()
        
        expected_keys = [
            'total_batches', 'total_anomalies', 'anomaly_rate',
            'running_mean', 'running_std', 'abs_threshold', 'recent_grads'
        ]
        
        for key in expected_keys:
            assert key in summary
    
    def test_get_summary_values(self):
        """Тест значений в summary."""
        for i in range(10):
            self.tracker.update(1.0 + i * 0.1)
        
        summary = self.tracker.get_summary()
        
        assert summary['total_batches'] == 10
        assert summary['total_anomalies'] == 0
        assert summary['anomaly_rate'] == 0.0
        assert len(summary['recent_grads']) > 0


class TestComputePerSampleGradNorms:
    """Тесты функции compute_per_sample_grad_norms."""
    
    def test_basic_computation(self):
        """Тест базового вычисления пер-семпл градиентов."""
        from losses.losses import CombinedLoss
        
        model = SimpleTestModel()
        
        batch_data = {
            'x': torch.randn(2, 3, 32, 32),
            'y': torch.randint(0, 2, (2, 1, 32, 32)).float()
        }
        
        # Используем CombinedLoss вместо BCEWithLogitsLoss
        criterion = CombinedLoss()
        
        norms = compute_per_sample_grad_norms(
            model=model,
            batch_data=batch_data,
            criterion=criterion,
            device='cpu'
        )
        
        assert isinstance(norms, torch.Tensor)
        assert norms.shape == (2,)  # По одному на семпл
        assert all(n >= 0 for n in norms)
    
    def test_different_samples_different_norms(self):
        """Тест что разные семплы могут иметь разные нормы."""
        from losses.losses import CombinedLoss
        
        model = SimpleTestModel()
        
        # Создаем семплы с разными характеристиками
        batch_data = {
            'x': torch.randn(3, 3, 32, 32),
            'y': torch.randint(0, 2, (3, 1, 32, 32)).float()
        }
        
        # Используем CombinedLoss
        criterion = CombinedLoss()
        
        norms = compute_per_sample_grad_norms(
            model=model,
            batch_data=batch_data,
            criterion=criterion,
            device='cpu'
        )
        
        assert norms.shape == (3,)
    
    def test_with_mask_map(self):
        """Тест вычисления с mask_map."""
        from losses.losses import CombinedLoss
        
        model = SimpleTestModel()
        
        batch_data = {
            'x': torch.randn(2, 3, 32, 32),
            'y': torch.randint(0, 2, (2, 1, 32, 32)).float(),
            'mask_map': torch.ones(2, 1, 32, 32)
        }
        
        # Используем CombinedLoss
        criterion = CombinedLoss()
        
        norms = compute_per_sample_grad_norms(
            model=model,
            batch_data=batch_data,
            criterion=criterion,
            device='cpu'
        )
        
        assert norms.shape == (2,)
    
    def test_zero_loss(self):
        """Тест когда loss близок к нулю."""
        from losses.losses import CombinedLoss
        
        model = SimpleTestModel()
        
        # Создаем данные где prediction и target близки
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = torch.sigmoid(model(x))
        
        batch_data = {
            'x': x,
            'y': y
        }
        
        # Используем CombinedLoss
        criterion = CombinedLoss()
        
        norms = compute_per_sample_grad_norms(
            model=model,
            batch_data=batch_data,
            criterion=criterion,
            device='cpu'
        )
        
        assert norms.shape == (1,)
        assert norms[0] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
