import torch

from losses.losses import MaskedBCELoss, MaskedDiceLoss, CombinedLoss


class TestMaskedBCELoss:
    """Тесты для MaskedBCELoss."""
    
    def test_init_default(self):
        """Тест инициализации по умолчанию."""
        loss_fn = MaskedBCELoss()
        assert loss_fn.reduction == 'mean'
        assert isinstance(loss_fn.bce, torch.nn.Module)
    
    def test_init_custom_reduction(self):
        """Тест инициализации с custom reduction."""
        loss_fn = MaskedBCELoss(reduction='sum')
        assert loss_fn.reduction == 'sum'
    
    def test_forward_without_mask_mean(self):
        """Тест forward без маски с mean reduction."""
        loss_fn = MaskedBCELoss(reduction='mean')
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # скаляр
        assert loss.item() >= 0
    
    def test_forward_without_mask_sum(self):
        """Тест forward без маски с sum reduction."""
        loss_fn = MaskedBCELoss(reduction='sum')
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
    
    def test_forward_with_mask(self):
        """Тест forward с маской."""
        loss_fn = MaskedBCELoss(reduction='mean')
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        mask = torch.ones(2, 1, 64, 64)
        mask[:, :, :32, :] = 0  # Игнорируем половину
        
        loss = loss_fn(predictions, targets, mask)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_forward_with_zero_mask(self):
        """Тест forward с нулевой маской (все игнорируется)."""
        loss_fn = MaskedBCELoss(reduction='mean')
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        mask = torch.zeros(2, 1, 64, 64)
        
        loss = loss_fn(predictions, targets, mask)
        
        # Должно вернуть 0 или очень маленькое значение из-за eps
        assert loss.item() >= 0
        assert torch.isfinite(loss)
    
    def test_perfect_predictions(self):
        """Тест идеальных предсказаний (должны давать низкий loss)."""
        loss_fn = MaskedBCELoss(reduction='mean')
        # Сигмоида от больших положительных чисел близка к 1
        predictions = torch.ones(2, 1, 64, 64) * 10
        targets = torch.ones(2, 1, 64, 64)
        
        loss = loss_fn(predictions, targets)
        
        # Loss должен быть близок к 0
        assert loss.item() < 0.1
    
    def test_worst_predictions(self):
        """Тест наихудших предсказаний (должны давать высокий loss)."""
        loss_fn = MaskedBCELoss(reduction='mean')
        # Предсказываем 0 когда target 1
        predictions = torch.ones(2, 1, 64, 64) * -10
        targets = torch.ones(2, 1, 64, 64)
        
        loss = loss_fn(predictions, targets)
        
        # Loss должен быть высоким
        assert loss.item() > 5


class TestMaskedDiceLoss:
    """Тесты для MaskedDiceLoss."""
    
    def test_init_default(self):
        """Тест инициализации по умолчанию."""
        loss_fn = MaskedDiceLoss()
        assert loss_fn.smooth == 1.0
        assert loss_fn.from_logits is True
    
    def test_init_custom_params(self):
        """Тест инициализации с custom параметрами."""
        loss_fn = MaskedDiceLoss(smooth=2.0, from_logits=False)
        assert loss_fn.smooth == 2.0
        assert loss_fn.from_logits is False
    
    def test_forward_without_mask(self):
        """Тест forward без маски."""
        loss_fn = MaskedDiceLoss(from_logits=True)
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert 0 <= loss.item() <= 2  # Dice loss в диапазоне [0, 2]
    
    def test_forward_with_mask(self):
        """Тест forward с маской."""
        loss_fn = MaskedDiceLoss(from_logits=True)
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        mask = torch.ones(2, 1, 64, 64)
        mask[:, :, :32, :] = 0
        
        loss = loss_fn(predictions, targets, mask)
        
        assert isinstance(loss, torch.Tensor)
        assert 0 <= loss.item() <= 2
    
    def test_perfect_dice(self):
        """Тест идеального Dice score (должен давать низкий loss)."""
        loss_fn = MaskedDiceLoss(from_logits=False)
        predictions = torch.ones(2, 1, 64, 64)
        targets = torch.ones(2, 1, 64, 64)
        
        loss = loss_fn(predictions, targets)
        
        # При идеальном совпадении Dice = 1, Dice_loss = 0
        # Но из-за smooth будет небольшое значение
        assert loss.item() < 0.5
    
    def test_no_overlap(self):
        """Тест отсутствия перекрытия (должен давать высокий loss)."""
        loss_fn = MaskedDiceLoss(from_logits=False)
        predictions = torch.zeros(2, 1, 64, 64)
        targets = torch.ones(2, 1, 64, 64)
        
        loss = loss_fn(predictions, targets)
        
        # При отсутствии перекрытия Dice близок к 0, Dice_loss близок к 1
        assert loss.item() > 0.5
    
    def test_from_logits_false(self):
        """Тест с from_logits=False (применяется sigmoid)."""
        loss_fn = MaskedDiceLoss(from_logits=False)
        # Уже вероятности
        predictions = torch.rand(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert 0 <= loss.item() <= 2


class TestCombinedLoss:
    """Тесты для CombinedLoss."""
    
    def test_init_default(self):
        """Тест инициализации по умолчанию."""
        loss_fn = CombinedLoss()
        assert loss_fn.bce_weight == 0.5
        assert loss_fn.dice_weight == 0.5
        assert loss_fn.use_map_mask is True
    
    def test_init_custom_weights(self):
        """Тест инициализации с custom весами."""
        loss_fn = CombinedLoss(bce_weight=0.7, dice_weight=0.3, use_map_mask=False)
        assert loss_fn.bce_weight == 0.7
        assert loss_fn.dice_weight == 0.3
        assert loss_fn.use_map_mask is False
    
    def test_forward_returns_tuple(self):
        """Тест что forward возвращает кортеж (loss, metrics_dict)."""
        loss_fn = CombinedLoss()
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        mask_map = torch.ones(2, 1, 64, 64)
        
        result = loss_fn(predictions, targets, mask_map)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        total_loss, metrics = result
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(metrics, dict)
    
    def test_forward_metrics_keys(self):
        """Тест что metrics содержит правильные ключи."""
        loss_fn = CombinedLoss()
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        mask_map = torch.ones(2, 1, 64, 64)
        
        _, metrics = loss_fn(predictions, targets, mask_map)
        
        assert 'total_loss' in metrics
        assert 'bce_loss' in metrics
        assert 'dice_loss' in metrics
    
    def test_forward_without_mask(self):
        """Тест forward без маски."""
        loss_fn = CombinedLoss(use_map_mask=False)
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        total_loss, metrics = loss_fn(predictions, targets, None)
        
        assert isinstance(total_loss, torch.Tensor)
        assert metrics['total_loss'] == total_loss.item()
    
    def test_loss_weights(self):
        """Тест влияния весов на итоговый loss."""
        # BCE только
        loss_fn_bce = CombinedLoss(bce_weight=1.0, dice_weight=0.0)
        # Dice только
        loss_fn_dice = CombinedLoss(bce_weight=0.0, dice_weight=1.0)
        
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        mask_map = torch.ones(2, 1, 64, 64)
        
        total_loss_bce, metrics_bce = loss_fn_bce(predictions, targets, mask_map)
        total_loss_dice, metrics_dice = loss_fn_dice(predictions, targets, mask_map)
        
        # BCE only должен равняться bce_loss
        assert abs(total_loss_bce.item() - metrics_bce['bce_loss']) < 1e-5
        # Dice only должен равняться dice_loss
        assert abs(total_loss_dice.item() - metrics_dice['dice_loss']) < 1e-5
    
    def test_combined_loss_value(self):
        """Тест значения комбинированного лосса."""
        loss_fn = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        mask_map = torch.ones(2, 1, 64, 64)
        
        total_loss, metrics = loss_fn(predictions, targets, mask_map)
        
        expected_total = 0.5 * metrics['bce_loss'] + 0.5 * metrics['dice_loss']
        assert abs(total_loss.item() - expected_total) < 1e-5
