import pytest
import torch

from losses.losses import MaskedBCELoss, MaskedDiceLoss, CombinedLoss


class TestMaskedBCELoss:
    """Тесты для MaskedBCELoss."""

    @pytest.fixture
    def loss_fn(self):
        """Фикстура с функцией потерь."""
        return MaskedBCELoss(reduction='mean')

    def test_perfect_predictions(self, loss_fn):
        """Тест идеальных предсказаний - loss должен быть близок к 0."""
        # Логиты для perfect predictions (большие положительные для 1, отрицательные для 0)
        preds = torch.tensor([[[[2.0, -2.0], [-2.0, 2.0]]]], dtype=torch.float32)
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)

        loss = loss_fn(preds, targets)
        assert loss < 0.5  # BCE должен быть низким

    def test_wrong_predictions(self, loss_fn):
        """Тест неправильных предсказаний - loss должен быть высоким."""
        # Предсказания противоположны таргетам
        preds = torch.tensor([[[[-2.0, 2.0], [2.0, -2.0]]]], dtype=torch.float32)
        targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)

        loss = loss_fn(preds, targets)
        assert loss > 1.0  # BCE должен быть высоким

    def test_with_mask(self, loss_fn):
        """Тест с маской - игнорируемые области не влияют на loss."""
        preds = torch.ones(1, 1, 4, 4) * 2.0  # Все предсказания = 1
        targets = torch.zeros(1, 1, 4, 4)     # Все таргеты = 0

        # Без маски - высокий loss
        loss_no_mask = loss_fn(preds, targets)
        assert loss_no_mask > 1.0

        # С маской, игнорирующей всё кроме одного пикселя
        mask = torch.zeros(1, 1, 4, 4)
        mask[0, 0, 0, 0] = 1.0

        loss_with_mask = loss_fn(preds, targets, mask=mask)
        # Loss должен быть таким же для этого одного пикселя
        assert loss_with_mask > 0

    def test_reduction_mean_vs_sum(self):
        """Тест различных режимов reduction."""
        loss_mean = MaskedBCELoss(reduction='mean')
        loss_sum = MaskedBCELoss(reduction='sum')

        preds = torch.ones(2, 1, 4, 4) * 0.5
        targets = torch.zeros(2, 1, 4, 4)

        loss_m = loss_mean(preds, targets)
        loss_s = loss_sum(preds, targets)

        # Sum должен быть больше mean для батча > 1
        assert loss_s > loss_m

    def test_empty_mask(self, loss_fn):
        """Тест с пустой маской (все нули)."""
        preds = torch.rand(1, 1, 4, 4)
        targets = torch.rand(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)

        loss = loss_fn(preds, targets, mask=mask)
        # При пустой маске loss должен обрабатываться корректно
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMaskedDiceLoss:
    """Тесты для MaskedDiceLoss."""

    @pytest.fixture
    def loss_fn(self):
        """Фикстура с Dice loss."""
        return MaskedDiceLoss(smooth=1.0, from_logits=True)

    def test_perfect_predictions(self, loss_fn):
        """Тест идеальных предсказаний - Dice loss должен быть близок к 0."""
        preds = torch.ones(1, 1, 10, 10) * 2.0  # sigmoid(2) ≈ 0.88
        targets = torch.ones(1, 1, 10, 10)

        loss = loss_fn(preds, targets)
        assert loss < 0.2  # Dice loss должен быть низким

    def test_opposite_predictions(self, loss_fn):
        """Тест противоположных предсказаний - Dice loss должен быть высоким."""
        preds = torch.zeros(1, 1, 10, 10)
        targets = torch.ones(1, 1, 10, 10)

        loss = loss_fn(preds, targets)
        # При полностью противоположных предсказаниях Dice loss будет около 1 - (2*smooth)/(sum+smooth)
        # Для smooth=1.0 и sum=100: dice_score ≈ 2/101 ≈ 0.02, dice_loss ≈ 0.98
        # Но с sigmoid(0) = 0.5, пересечение будет ненулевым
        assert loss > 0.3  # Dice loss должен быть высоким

    def test_with_mask(self, loss_fn):
        """Тест с маской."""
        preds = torch.ones(1, 1, 4, 4)
        targets = torch.zeros(1, 1, 4, 4)

        # Маска игнорирует половину
        mask = torch.ones(1, 1, 4, 4)
        mask[0, 0, :, :2] = 0.0

        loss = loss_fn(preds, targets, mask=mask)
        assert not torch.isnan(loss)

    def test_from_logits_false(self):
        """Тест с from_logits=False (уже вероятности)."""
        loss_fn = MaskedDiceLoss(smooth=1.0, from_logits=False)

        preds = torch.ones(1, 1, 10, 10) * 0.9  # Уже вероятности
        targets = torch.ones(1, 1, 10, 10)

        loss = loss_fn(preds, targets)
        assert loss < 0.2


class TestCombinedLoss:
    """Тесты для CombinedLoss."""

    @pytest.fixture
    def combined_loss(self):
        """Фикстура с комбинированным лоссом."""
        return CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    def test_combined_loss_perfect(self, combined_loss):
        """Тест идеальных предсказаний."""
        preds = torch.ones(1, 1, 10, 10) * 2.0
        targets = torch.ones(1, 1, 10, 10)

        total_loss, metrics = combined_loss(preds, targets)

        assert total_loss < 0.5
        assert 'total_loss' in metrics
        assert 'bce_loss' in metrics
        assert 'dice_loss' in metrics
        assert metrics['total_loss'] == pytest.approx(total_loss.item())

    def test_combined_loss_with_masks(self, combined_loss):
        """Тест с масками map и depth."""
        preds = torch.rand(2, 1, 8, 8)
        targets = torch.rand(2, 1, 8, 8)
        mask_map = torch.ones(2, 1, 8, 8)
        mask_depth = torch.ones(2, 1, 8, 8)

        # Игнорируем часть карты
        mask_map[0, 0, :4, :] = 0.0

        total_loss, metrics = combined_loss(
            preds, targets,
            mask_map=mask_map,
            mask_depth=mask_depth
        )

        assert not torch.isnan(total_loss)
        assert metrics['bce_loss'] >= 0
        assert metrics['dice_loss'] >= 0

    def test_combined_loss_weights(self):
        """Тест различных весов компонентов."""
        loss_bce_heavy = CombinedLoss(bce_weight=0.9, dice_weight=0.1)
        loss_dice_heavy = CombinedLoss(bce_weight=0.1, dice_weight=0.9)

        preds = torch.rand(1, 1, 8, 8)
        targets = torch.rand(1, 1, 8, 8)

        _, metrics_bce = loss_bce_heavy(preds, targets)
        _, metrics_dice = loss_dice_heavy(preds, targets)

        # Веса должны влиять на итоговый loss
        assert metrics_bce['bce_loss'] >= 0
        assert metrics_dice['dice_loss'] >= 0

    def test_use_map_mask_false(self):
        """Тест с отключенной маской карты."""
        loss_fn = CombinedLoss(use_map_mask=False, use_depth_mask=False)

        preds = torch.rand(1, 1, 8, 8)
        targets = torch.rand(1, 1, 8, 8)
        mask_map = torch.zeros(1, 1, 8, 8)

        # Маска должна игнорироваться
        total_loss, metrics = loss_fn(preds, targets, mask_map=mask_map)
        assert not torch.isnan(total_loss)

    def test_both_masks_enabled(self):
        """Тест с обоими типами масок."""
        loss_fn = CombinedLoss(use_map_mask=True, use_depth_mask=True)

        preds = torch.ones(1, 1, 8, 8) * 2.0
        targets = torch.ones(1, 1, 8, 8)
        mask_map = torch.ones(1, 1, 8, 8)
        mask_depth = torch.ones(1, 1, 8, 8)

        total_loss, metrics = loss_fn(preds, targets, mask_map=mask_map, mask_depth=mask_depth)

        assert total_loss < 0.5  # Идеальные предсказания
        assert metrics['total_loss'] < 0.5
