import torch
import torch.nn as nn
import os
import tempfile
import shutil

from losses.losses import CombinedLoss
from metrics.metrics import MetricsCalculator
from optimizers.optimizers import create_optimizer_and_scheduler, get_gradient_stats
from training.gradient_tracker import GradientNormTracker


class SimpleUNet(nn.Module):
    """Простая U-Net-like модель для интеграционных тестов."""
    
    def __init__(self, in_channels=6, classes=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, classes, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class TestTrainingPipelineIntegration:
    """Интеграционные тесты для training pipeline."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = 'cpu'
        self.model = SimpleUNet(in_channels=6, classes=1).to(self.device)
        self.criterion = CombinedLoss()
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(self.model)
    
    def teardown_method(self):
        """Очистка после каждого теста."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_training_step(self):
        """Тест одного шага обучения."""
        self.model.train()
        
        # Создаем батч данных
        x = torch.randn(2, 6, 64, 64).to(self.device)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float().to(self.device)
        mask_map = torch.ones(2, 1, 64, 64).to(self.device)
        
        # Forward pass
        predictions = self.model(x)
        
        # Loss
        loss, metrics = self.criterion(predictions, y, mask_map)
        
        # Backward pass
        loss.backward()
        
        # Проверяем что градиенты вычислены
        for param in self.model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Проверяем что loss конечный
        assert torch.isfinite(loss)
        assert loss.item() > 0
    
    def test_validation_step(self):
        """Тест одного шага валидации."""
        self.model.eval()
        
        x = torch.randn(2, 6, 64, 64).to(self.device)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float().to(self.device)
        mask_map = torch.ones(2, 1, 64, 64).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x)
            loss, metrics = self.criterion(predictions, y, mask_map)
            
            metrics_calc = MetricsCalculator()
            eval_metrics = metrics_calc.compute_all(predictions, y, mask_map)
        
        # Проверяем метрики
        assert 'dice' in eval_metrics
        assert 'iou' in eval_metrics
        assert 0 <= eval_metrics['dice'] <= 1
        assert 0 <= eval_metrics['iou'] <= 1
    
    def test_gradient_tracking_integration(self):
        """Тест интеграции gradient tracking в обучение."""
        grad_tracker = GradientNormTracker(
            abs_threshold=5.0,
            save_dir=self.temp_dir
        )
        
        self.model.train()
        
        x = torch.randn(2, 6, 64, 64).to(self.device)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float().to(self.device)
        
        predictions = self.model(x)
        loss, _ = self.criterion(predictions, y)
        
        loss.backward()
        
        # Вычисляем norm градиента
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Трекаем аномалии
        is_anomaly, stats = grad_tracker.update(grad_norm.item())
        
        # Проверяем что статистика обновлена
        assert grad_tracker.count == 1
        assert 'grad_norm' in stats
        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def test_multiple_training_steps(self):
        """Тест нескольких шагов обучения."""
        self.model.train()
        
        losses = []
        for step in range(5):
            x = torch.randn(2, 6, 64, 64).to(self.device)
            y = torch.randint(0, 2, (2, 1, 64, 64)).float().to(self.device)
            
            predictions = self.model(x)
            loss, _ = self.criterion(predictions, y)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            losses.append(loss.item())
        
        # Все losses должны быть конечными
        for loss in losses:
            assert torch.isfinite(torch.tensor(loss))
    
    def test_scheduler_step(self):
        """Тест шага планировщика learning rate."""
        initial_lr = self.optimizer.param_groups[1]['lr']
        
        # Симулируем несколько эпох
        for epoch in range(3):
            x = torch.randn(2, 6, 64, 64).to(self.device)
            y = torch.randint(0, 2, (2, 1, 64, 64)).float().to(self.device)
            
            predictions = self.model(x)
            loss, _ = self.criterion(predictions, y)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step(loss)
        
        # LR должен измениться (или остаться тем же для ReduceLROnPlateau)
        current_lr = self.optimizer.param_groups[1]['lr']
        assert current_lr > 0


class TestEvaluationPipelineIntegration:
    """Интеграционные тесты для evaluation pipeline."""
    
    def setup_method(self):
        """Настройка перед каждым тестом."""
        self.device = 'cpu'
        self.model = SimpleUNet(in_channels=6, classes=1).to(self.device)
        self.criterion = CombinedLoss()
        self.metrics_calc = MetricsCalculator()
    
    def test_batch_evaluation(self):
        """Тест оценки на батче данных."""
        self.model.eval()
        
        batch_size = 4
        x = torch.randn(batch_size, 6, 64, 64).to(self.device)
        y = torch.randint(0, 2, (batch_size, 1, 64, 64)).float().to(self.device)
        mask_map = torch.ones(batch_size, 1, 64, 64).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x)
            loss, loss_metrics = self.criterion(predictions, y, mask_map)
            eval_metrics = self.metrics_calc.compute_all(predictions, y, mask_map)
        
        # Проверяем метрики
        assert 'loss' in {**loss_metrics, 'loss': loss.item()}
        assert 'dice' in eval_metrics
        assert 'iou' in eval_metrics
        assert eval_metrics['dice'] >= 0
        assert eval_metrics['iou'] >= 0
    
    def test_aggregate_metrics(self):
        """Тест агрегации метрик по нескольким батчам."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_masks = []
        
        # Симулируем несколько батчей
        for _ in range(3):
            x = torch.randn(2, 6, 64, 64).to(self.device)
            y = torch.randint(0, 2, (2, 1, 64, 64)).float().to(self.device)
            
            with torch.no_grad():
                predictions = self.model(x)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())
            all_masks.append(torch.ones_like(y).cpu())
        
        # Агрегируем
        all_preds = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Вычисляем метрики на всех данных
        final_metrics = self.metrics_calc.compute_all(all_preds, all_targets, all_masks)
        
        assert isinstance(final_metrics, dict)
        assert len(final_metrics) > 0
    
    def test_threshold_variation(self):
        """Тест оценки с разными порогами."""
        self.model.eval()
        
        x = torch.randn(2, 6, 64, 64).to(self.device)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float().to(self.device)
        
        with torch.no_grad():
            predictions = self.model(x)
        
        # Разные пороги
        metrics_low = MetricsCalculator(threshold=0.3).compute_all(predictions, y)
        metrics_high = MetricsCalculator(threshold=0.7).compute_all(predictions, y)
        
        # Метрики должны быть валидными для обоих порогов
        assert 0 <= metrics_low['dice'] <= 1
        assert 0 <= metrics_high['dice'] <= 1
    
    def test_masked_evaluation(self):
        """Тест оценки с масками."""
        self.model.eval()
        
        x = torch.randn(2, 6, 64, 64).to(self.device)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float().to(self.device)
        
        # Создаем маску с игнорируемыми областями
        mask_map = torch.ones(2, 1, 64, 64).to(self.device)
        mask_map[:, :, :32, :] = 0  # Игнорируем верхнюю половину
        
        with torch.no_grad():
            predictions = self.model(x)
            loss, _ = self.criterion(predictions, y, mask_map)
            metrics = self.metrics_calc.compute_all(predictions, y, mask_map)
        
        # Метрики должны быть вычислены только по неигнорируемым областям
        assert torch.isfinite(loss)
        assert 0 <= metrics['dice'] <= 1


class TestGradientStatsIntegration:
    """Интеграционные тесты для gradient statistics."""
    
    def test_full_gradient_flow(self):
        """Тест полного потока градиентов."""
        model = SimpleUNet(in_channels=6, classes=1)
        criterion = CombinedLoss()
        
        x = torch.randn(2, 6, 64, 64)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        # Forward
        predictions = model(x)
        loss, _ = criterion(predictions, y)
        
        # Backward
        loss.backward()
        
        # Получаем статистику
        stats = get_gradient_stats(model)
        
        # Проверяем статистику
        assert stats['grad_norm_total'] > 0
        assert stats['grad_norm_encoder'] > 0
        assert stats['grad_norm_decoder'] > 0
        assert stats['grad_min'] >= 0
        assert stats['grad_max'] > 0
        
        # Очищаем градиенты
        model.zero_grad()
        
        # После zero_grad все должно быть 0
        stats_after = get_gradient_stats(model)
        assert stats_after['grad_norm_total'] == 0.0
