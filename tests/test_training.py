import pytest
import torch
import torch.nn as nn
import os

from models.unetplusplus import load_unetplusplus, save_model_checkpoint
from losses.losses import CombinedLoss, MaskedBCELoss, MaskedDiceLoss
from metrics.metrics import MetricsCalculator
from optimizers.optimizers import create_optimizer_and_scheduler
from settings import settings


class TestModelLoading:
    """Tests for model loading and architecture."""
    
    def test_load_unetplusplus_creates_model(self):
        """Test that model loads successfully."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_name='resnet34',
            encoder_weights=None,  # Skip imagenet for faster tests
            device='cpu'
        )
        
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_load_unetplusplus_correct_input_channels(self):
        """Test model accepts correct number of input channels."""
        model = load_unetplusplus(
            in_channels=5,  # RGB + depth + faults
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        
        # Check first conv layer has correct in_channels
        assert model.encoder.conv1.in_channels == 5
    
    def test_load_unetplusplus_output_shape(self):
        """Test model produces correct output shape."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        
        # Create dummy input (batch_size=2, channels=4, H=128, W=128)
        x = torch.randn(2, 4, 128, 128)
        
        with torch.no_grad():
            output = model(x)
        
        # Output should be (batch_size, 1, H, W)
        assert output.shape[0] == 2
        assert output.shape[1] == 1
        assert output.shape[2] == 128
        assert output.shape[3] == 128
    
    def test_load_unetplusplus_with_imagenet_weights(self):
        """Test model loads with ImageNet weights."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            device='cpu'
        )
        
        assert model is not None
        
        # Verify weights were initialized from pretrained
        # The first 3 channels should have non-zero weights from ImageNet
        first_conv_weight = model.encoder.conv1.weight
        assert first_conv_weight.shape[1] == 4  # Modified for 4 channels
        
        # Channels 0-3 should have weights (first 3 from ImageNet, 4th initialized)
        assert not torch.all(first_conv_weight[:, :3, :, :] == 0)
    
    @pytest.mark.slow
    def test_model_forward_backward_pass(self):
        """Test complete forward and backward pass."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        
        model.train()
        
        x = torch.randn(2, 4, 128, 128)
        y = torch.randn(2, 1, 128, 128)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Forward
        output = model(x)
        loss = criterion(output, y)
        
        # Backward
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_masked_bce_loss_basic(self):
        """Test basic BCE loss computation."""
        loss_fn = MaskedBCELoss(reduction='mean')
        
        predictions = torch.randn(2, 1, 10, 10)
        targets = torch.randint(0, 2, (2, 1, 10, 10)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_masked_bce_loss_with_mask(self):
        """Test BCE loss respects mask."""
        loss_fn = MaskedBCELoss(reduction='mean')
        
        predictions = torch.zeros(1, 1, 10, 10)
        targets = torch.ones(1, 1, 10, 10)
        mask = torch.zeros(1, 1, 10, 10)  # All masked out
        
        loss = loss_fn(predictions, targets, mask)
        
        # With all zeros in mask, loss should be 0 or very small
        assert loss.item() < 1e-6
    
    def test_masked_dice_loss_basic(self):
        """Test basic Dice loss computation."""
        loss_fn = MaskedDiceLoss(smooth=1.0, from_logits=False)
        
        predictions = torch.rand(2, 1, 10, 10)
        targets = torch.randint(0, 2, (2, 1, 10, 10)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert 0 <= loss.item() <= 1
        assert not torch.isnan(loss)
    
    def test_combined_loss_returns_tuple(self):
        """Test combined loss returns loss and metrics dict."""
        criterion = CombinedLoss(
            bce_weight=0.5,
            dice_weight=0.5,
            use_map_mask=True,
            use_depth_mask=False
        )
        
        predictions = torch.randn(2, 1, 10, 10)
        targets = torch.randint(0, 2, (2, 1, 10, 10)).float()
        mask_map = torch.ones(2, 1, 10, 10)
        
        total_loss, metrics = criterion(predictions, targets, mask_map=mask_map)
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert 'total_loss' in metrics
        assert 'bce_loss' in metrics
        assert 'dice_loss' in metrics
    
    def test_combined_loss_with_masks(self):
        """Test combined loss applies masks correctly."""
        criterion = CombinedLoss(
            bce_weight=0.5,
            dice_weight=0.5,
            use_map_mask=True,
            use_depth_mask=True
        )
        
        predictions = torch.randn(2, 1, 10, 10)
        targets = torch.randint(0, 2, (2, 1, 10, 10)).float()
        mask_map = torch.ones(2, 1, 10, 10)
        mask_depth = torch.ones(2, 1, 10, 10)
        
        total_loss, metrics = criterion(
            predictions, targets,
            mask_map=mask_map,
            mask_depth=mask_depth
        )
        
        assert not torch.isnan(total_loss)


class TestMetricsCalculator:
    """Tests for metrics calculation."""
    
    def test_metrics_calculator_compute_all(self):
        """Test computing all metrics at once."""
        calc = MetricsCalculator(threshold=0.5)
        
        predictions = torch.randn(2, 1, 10, 10)
        targets = torch.randint(0, 2, (2, 1, 10, 10)).float()
        
        metrics = calc.compute_all(predictions, targets)
        
        expected_keys = ['iou', 'dice', 'recall', 'precision', 'f1', 'fp_area', 'fn_area']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
    
    def test_metrics_calculator_iou_range(self):
        """Test IoU is in valid range [0, 1]."""
        calc = MetricsCalculator(threshold=0.5)
        
        predictions = torch.rand(2, 1, 10, 10)
        targets = torch.randint(0, 2, (2, 1, 10, 10)).float()
        
        metrics = calc.compute_all(predictions, targets)
        
        assert 0 <= metrics['iou'] <= 1
    
    def test_metrics_calculator_perfect_prediction(self):
        """Test metrics for perfect prediction."""
        calc = MetricsCalculator(threshold=0.5)
        
        # Perfect prediction
        predictions = torch.ones(1, 1, 10, 10) * 10  # High logit -> sigmoid ~ 1
        targets = torch.ones(1, 1, 10, 10)
        
        metrics = calc.compute_all(predictions, targets)
        
        # Should have high scores (not necessarily 1.0 due to smoothing)
        assert metrics['dice'] > 0.9
        assert metrics['iou'] > 0.8
    
    def test_metrics_calculator_with_mask(self):
        """Test metrics calculation with mask."""
        calc = MetricsCalculator(threshold=0.5)
        
        predictions = torch.randn(2, 1, 10, 10)
        targets = torch.randint(0, 2, (2, 1, 10, 10)).float()
        mask = torch.ones(2, 1, 10, 10)
        
        metrics = calc.compute_all(predictions, targets, mask=mask)
        
        assert 'iou' in metrics
        assert 'dice' in metrics


class TestOptimizerAndScheduler:
    """Tests for optimizer and scheduler creation."""
    
    def test_create_optimizer_and_scheduler_basic(self):
        """Test basic optimizer and scheduler creation."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        
        optimizer, scheduler = create_optimizer_and_scheduler(
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-4,
            scheduler_type='reduce_lr_plateau',
            encoder_lr_multiplier=0.1
        )
        
        assert optimizer is not None
        assert scheduler is not None
        
        # Check learning rates
        param_groups = optimizer.param_groups
        assert len(param_groups) == 2  # Encoder and decoder
        
        # Encoder should have lower LR
        encoder_lr = param_groups[0]['lr']
        decoder_lr = param_groups[1]['lr']
        
        assert encoder_lr < decoder_lr
    
    def test_optimizer_step_updates_parameters(self):
        """Test that optimizer step updates model parameters."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        
        optimizer, _ = create_optimizer_and_scheduler(
            model=model,
            learning_rate=1e-3
        )
        
        # Store initial weights
        initial_weight = model.encoder.conv1.weight.clone()
        
        # Forward and backward pass
        x = torch.randn(1, 4, 64, 64)
        y = torch.randn(1, 1, 64, 64)
        
        output = model(x)
        loss = nn.BCEWithLogitsLoss()(output, y)
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Weights should have changed
        updated_weight = model.encoder.conv1.weight
        assert not torch.all(initial_weight == updated_weight)


class TestTrainingReadiness:
    """Tests for training readiness checks."""
    
    def test_model_on_correct_device(self):
        """Test model is on correct device."""
        device = 'cpu'  # Use CPU for tests
        
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device=device
        )
        
        # Check model is on correct device
        assert next(model.parameters()).device.type == device
    
    def test_criterion_accepts_model_output(self):
        """Test loss function accepts model output directly."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        
        criterion = CombinedLoss()
        
        x = torch.randn(2, 4, 64, 64)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        with torch.no_grad():
            output = model(x)
        
        # CombinedLoss expects logits (raw model output)
        loss, metrics = criterion(output, y)
        
        assert not torch.isnan(loss)
    
    def test_complete_training_iteration(self):
        """Test a complete training iteration (forward, loss, backward, step)."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        model.train()
        
        criterion = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(2, 4, 64, 64)
        y = torch.randint(0, 2, (2, 1, 64, 64)).float()
        mask_map = torch.ones(2, 1, 64, 64)
        
        # Forward
        output = model(x)
        
        # Loss
        loss, metrics = criterion(output, y, mask_map=mask_map)
        
        # Backward
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        
        # Step
        optimizer.step()
        optimizer.zero_grad()
        
        # Gradients should be zeroed
        for param in model.parameters():
            assert torch.all(param.grad == 0)


class TestCheckpointSaving:
    """Tests for model checkpoint saving/loading."""
    
    def test_save_checkpoint_creates_file(self, tmp_path):
        """Test checkpoint saving creates file."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint_path = str(tmp_path / "test_checkpoint.pth")
        
        saved_path = save_model_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={'loss': 0.5},
            save_path=str(tmp_path),
            filename='test_checkpoint.pth'
        )
        
        assert os.path.exists(saved_path)
        assert saved_path == checkpoint_path
    
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoint preserves weights."""
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = str(tmp_path / "checkpoint.pth")
        save_model_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics={'dice': 0.8},
            save_path=str(tmp_path),
            filename='checkpoint.pth'
        )
        
        # Load checkpoint
        from models.unetplusplus import load_model_checkpoint
        loaded_model = load_model_checkpoint(model, checkpoint_path, device='cpu')
        
        # Verify weights are the same
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert torch.allclose(param1, param2), f"Weights differ for {name1}"


@pytest.mark.smoke
def test_training_smoke_test():
    """Smoke test for basic training components."""
    model = load_unetplusplus(
        in_channels=4,
        classes=1,
        encoder_weights=None,
        device='cpu'
    )
    
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    x = torch.randn(1, 4, 64, 64)
    y = torch.randint(0, 2, (1, 1, 64, 64)).float()
    
    model.train()
    output = model(x)
    loss, _ = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    assert loss.item() >= 0
    assert not torch.isnan(loss)
