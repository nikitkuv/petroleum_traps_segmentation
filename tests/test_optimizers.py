import torch
import torch.nn as nn

from optimizers.optimizers import create_optimizer_and_scheduler, get_gradient_stats


class SimpleTestModel(nn.Module):
    """Простая модель для тестов."""
    
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.decoder_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.encoder_conv(x))
        x = self.decoder_conv(x)
        return x


class TestCreateOptimizerAndScheduler:
    """Тесты для функции create_optimizer_and_scheduler."""
    
    def test_create_default(self):
        """Тест создания оптимизатора и планировщика по умолчанию."""
        model = SimpleTestModel()
        
        optimizer, scheduler = create_optimizer_and_scheduler(model)
        
        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.AdamW)
        assert scheduler is not None
    
    def test_custom_learning_rate(self):
        """Тест с custom learning rate."""
        model = SimpleTestModel()
        
        optimizer, _ = create_optimizer_and_scheduler(
            model,
            learning_rate=0.01
        )
        
        # Проверяем что LR установлен корректно (encoder и decoder могут иметь разные LR)
        assert optimizer.param_groups[0]['lr'] > 0
        assert optimizer.param_groups[1]['lr'] > 0
        # Decoder LR должен быть равен заданному learning_rate
        assert optimizer.param_groups[1]['lr'] == 0.01
    
    def test_custom_weight_decay(self):
        """Тест с custom weight decay."""
        model = SimpleTestModel()
        
        optimizer, _ = create_optimizer_and_scheduler(
            model,
            weight_decay=0.05
        )
        
        assert optimizer.defaults['weight_decay'] == 0.05
    
    def test_encoder_lr_multiplier(self):
        """Тест множителя LR для энкодера."""
        model = SimpleTestModel()
        
        optimizer, _ = create_optimizer_and_scheduler(
            model,
            learning_rate=0.001,
            encoder_lr_multiplier=0.5
        )
        
        # Encoder LR должен быть умножен на multiplier
        assert optimizer.param_groups[0]['lr'] == 0.001 * 0.5
        # Decoder LR должен остаться базовым
        assert optimizer.param_groups[1]['lr'] == 0.001
    
    def test_differential_lr(self):
        """Тест что encoder и decoder имеют разные LR."""
        model = SimpleTestModel()
        
        optimizer, _ = create_optimizer_and_scheduler(
            model,
            learning_rate=0.001,
            encoder_lr_multiplier=0.1
        )
        
        encoder_lr = optimizer.param_groups[0]['lr']
        decoder_lr = optimizer.param_groups[1]['lr']
        
        assert encoder_lr != decoder_lr
        assert encoder_lr < decoder_lr
    
    def test_reduce_lr_plateau_scheduler(self):
        """Тест создания ReduceLROnPlateau планировщика."""
        model = SimpleTestModel()
        
        optimizer, scheduler = create_optimizer_and_scheduler(
            model,
            scheduler_type='reduce_lr_plateau'
        )
        
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.mode == 'min'
    
    def test_cosine_annealing_scheduler(self):
        """Тест создания CosineAnnealingWarmRestarts планировщика."""
        model = SimpleTestModel()
        
        optimizer, scheduler = create_optimizer_and_scheduler(
            model,
            scheduler_type='cosine_annealing'
        )
        
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        assert isinstance(scheduler, CosineAnnealingWarmRestarts)
    
    def test_no_scheduler(self):
        """Тест создания без планировщика."""
        model = SimpleTestModel()
        
        optimizer, scheduler = create_optimizer_and_scheduler(
            model,
            scheduler_type='unknown'
        )
        
        assert scheduler is None
    
    def test_parameter_groups(self):
        """Тест что параметры правильно разделены на группы."""
        model = SimpleTestModel()
        
        optimizer, _ = create_optimizer_and_scheduler(model)
        
        # Должно быть 2 группы параметров
        assert len(optimizer.param_groups) == 2
        
        # Проверяем что параметры распределены
        encoder_params = list(optimizer.param_groups[0]['params'])
        decoder_params = list(optimizer.param_groups[1]['params'])
        
        assert len(encoder_params) > 0
        assert len(decoder_params) > 0
    
    def test_optimizer_type(self):
        """Тест что используется AdamW."""
        model = SimpleTestModel()
        
        optimizer, _ = create_optimizer_and_scheduler(model)
        
        assert isinstance(optimizer, torch.optim.AdamW)


class TestGetGradientStats:
    """Тесты для функции get_gradient_stats."""
    
    def test_basic_stats(self):
        """Тест базовой статистики градиентов."""
        model = SimpleTestModel()
        
        # Создаем фиктивные градиенты
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        stats = get_gradient_stats(model)
        
        assert 'grad_norm_total' in stats
        assert 'grad_norm_encoder' in stats
        assert 'grad_norm_decoder' in stats
        assert 'grad_min' in stats
        assert 'grad_max' in stats
    
    def test_grad_norm_values(self):
        """Тест значений норм градиентов."""
        model = SimpleTestModel()
        
        # Создаем градиенты с известными значениями
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        
        stats = get_gradient_stats(model)
        
        # Все нормы должны быть положительными
        assert stats['grad_norm_total'] > 0
        assert stats['grad_norm_encoder'] >= 0
        assert stats['grad_norm_decoder'] >= 0
    
    def test_grad_min_max(self):
        """Тест min/max значений градиентов."""
        model = SimpleTestModel()
        
        # Создаем градиенты с разными значениями
        for param in model.parameters():
            param.grad = torch.linspace(-10, 10, param.numel()).reshape(param.shape)
        
        stats = get_gradient_stats(model)
        
        assert stats['grad_min'] >= 0  # Берется abs
        assert stats['grad_max'] > 0
    
    def test_no_gradients(self):
        """Тест когда нет градиентов."""
        model = SimpleTestModel()
        
        # Не устанавливаем градиенты
        for param in model.parameters():
            param.grad = None
        
        stats = get_gradient_stats(model)
        
        assert stats['grad_norm_total'] == 0.0
        assert stats['grad_norm_encoder'] == 0.0
        assert stats['grad_norm_decoder'] == 0.0
        assert stats['grad_min'] == 0.0
        assert stats['grad_max'] == 0.0
    
    def test_encoder_decoder_separation(self):
        """Тест разделения статистики encoder/decoder."""
        model = SimpleTestModel()
        
        # Устанавливаем разные градиенты для encoder и decoder
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.grad = torch.ones_like(param) * 2
            else:
                param.grad = torch.ones_like(param)
        
        stats = get_gradient_stats(model)
        
        # Encoder градиенты должны быть больше (т.к. мы их умножили на 2)
        assert stats['grad_norm_encoder'] > stats['grad_norm_decoder']
    
    def test_param_counts(self):
        """Тест что функция корректно обрабатывает количество параметров."""
        model = SimpleTestModel()
        
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        stats = get_gradient_stats(model)
        
        # Статистика должна быть вычислена корректно
        assert isinstance(stats['grad_norm_total'], float)
        assert isinstance(stats['grad_min'], float)
        assert isinstance(stats['grad_max'], float)
    
    def test_zero_gradients(self):
        """Тест с нулевыми градиентами."""
        model = SimpleTestModel()
        
        for param in model.parameters():
            param.grad = torch.zeros_like(param)
        
        stats = get_gradient_stats(model)
        
        assert stats['grad_norm_total'] == 0.0
        assert stats['grad_min'] == 0.0
        assert stats['grad_max'] == 0.0
