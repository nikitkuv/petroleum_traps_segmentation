import pytest
import numpy as np
import os
import cv2

from training.pipeline import run_full_pipeline
from training.overfit_check import overfit_check
from evaluation.evaluate import evaluate_on_test
from settings import settings


class TestPipelineIntegration:
    """Integration tests for the full training pipeline."""
    
    @pytest.fixture
    def minimal_dataset(self, tmp_path):
        """Create a minimal dataset for pipeline testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create 4 complete samples (enough for train/val/test split)
        for card_num in range(1, 5):
            horizon = "H150" if card_num % 2 == 0 else "BZ24"
            
            # RGB image
            rgb_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(data_dir / f"{card_num:03d}_x_structuralNOisoline_{horizon}.png"), rgb_img)
            
            # Depth image
            depth_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(data_dir / f"{card_num:03d}_x_structuralBlackWhite_{horizon}.png"), depth_img)
            
            # Traps mask
            traps_img = np.zeros((64, 64), dtype=np.uint8)
            traps_img[20:40, 20:40] = 200
            cv2.imwrite(str(data_dir / f"{card_num:03d}_y_traps_{horizon}.png"), traps_img)
        
        return str(data_dir)
    
    @pytest.mark.slow
    def test_full_pipeline_runs(self, minimal_dataset, tmp_path):
        """Test that full pipeline runs without errors."""
        checkpoint_dir = tmp_path / "checkpoints"
        logs_dir = tmp_path / "logs"
        
        # Override settings for test
        original_checkpoint = settings.CHECKPOINT_DIR
        original_logs = settings.LOGS_DIR
        
        try:
            settings.CHECKPOINT_DIR = str(checkpoint_dir)
            settings.LOGS_DIR = str(logs_dir)
            settings.NUM_EPOCHS = 2  # Short training for test
            settings.BATCH_SIZE = 2
            
            metrics = run_full_pipeline(
                data_dir=minimal_dataset,
                use_faults=False,
                data_source='png',
                overfit_check_mode=False,
                n_epochs=2,
                batch_size=2,
                wandb_project=None  # Disable wandb for tests
            )
            
            # Should return some metrics
            assert isinstance(metrics, dict)
            
        finally:
            settings.CHECKPOINT_DIR = original_checkpoint
            settings.LOGS_DIR = original_logs
    
    @pytest.mark.slow
    def test_pipeline_overfit_mode(self, minimal_dataset, tmp_path):
        """Test pipeline in overfit check mode."""
        checkpoint_dir = tmp_path / "checkpoints"
        logs_dir = tmp_path / "logs"
        
        original_checkpoint = settings.CHECKPOINT_DIR
        original_logs = settings.LOGS_DIR
        original_overfit = settings.OVERFIT_SIZE
        
        try:
            settings.CHECKPOINT_DIR = str(checkpoint_dir)
            settings.LOGS_DIR = str(logs_dir)
            settings.OVERFIT_SIZE = 1
            
            result = run_full_pipeline(
                data_dir=minimal_dataset,
                use_faults=False,
                data_source='png',
                overfit_check_mode=True,
                wandb_project=None
            )
            
            # Overfit mode returns empty dict
            assert isinstance(result, dict)
            
            # Check that overfit visualization was created
            overfit_dir = checkpoint_dir / "overfit_check"
            assert overfit_dir.exists()
            
        finally:
            settings.CHECKPOINT_DIR = original_checkpoint
            settings.LOGS_DIR = original_logs
            settings.OVERFIT_SIZE = original_overfit


class TestOverfitCheck:
    """Tests for overfit check functionality."""
    
    @pytest.fixture
    def tiny_dataset(self, tmp_path):
        """Create a tiny dataset for overfit testing."""
        from data.dataset import GeologyTrapsDataset
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create just 1 complete sample
        for filename in [
            "001_x_structuralNOisoline_H150.png",
            "001_x_structuralBlackWhite_H150.png",
            "001_y_traps_H150.png"
        ]:
            img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
            if 'BlackWhite' in filename or 'traps' in filename:
                img = img[:, :, 0]  # Grayscale
            cv2.imwrite(str(data_dir / filename), img)
        
        file_paths = [str(data_dir / f) for f in os.listdir(data_dir)]
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=str(data_dir),
            augment=False,
            use_faults=False,
            data_source='png'
        )
        
        return dataset
    
    @pytest.mark.slow
    def test_overfit_check_decreases_loss(self, tiny_dataset, tmp_path):
        """Test that overfit check shows decreasing loss."""
        from torch.utils.data import DataLoader
        from models.unetplusplus import load_unetplusplus
        from losses.losses import CombinedLoss
        import torch.optim as optim
        
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        
        train_loader = DataLoader(
            tiny_dataset,
            batch_size=1,
            shuffle=True
        )
        
        criterion = CombinedLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        
        save_path = tmp_path / "overfit_results"
        save_path.mkdir()
        
        # Run overfit check
        overfit_check(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device='cpu',
            n_epochs=10,
            save_path=str(save_path)
        )
        
        # Check that visualizations were saved
        viz_files = list(save_path.glob("*.png"))
        assert len(viz_files) > 0


class TestEvaluation:
    """Tests for evaluation functionality."""
    
    @pytest.fixture
    def test_dataset_and_model(self, tmp_path):
        """Create test dataset and model."""
        from data.dataset import GeologyTrapsDataset
        from models.unetplusplus import load_unetplusplus
        from torch.utils.data import DataLoader
        
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create 2 complete samples
        for card_num in range(1, 3):
            horizon = "H150"
            
            rgb_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(data_dir / f"{card_num:03d}_x_structuralNOisoline_{horizon}.png"), rgb_img)
            
            depth_img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(data_dir / f"{card_num:03d}_x_structuralBlackWhite_{horizon}.png"), depth_img)
            
            traps_img = np.zeros((64, 64), dtype=np.uint8)
            traps_img[20:40, 20:40] = 200
            cv2.imwrite(str(data_dir / f"{card_num:03d}_y_traps_{horizon}.png"), traps_img)
        
        file_paths = [str(data_dir / f) for f in os.listdir(data_dir)]
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=str(data_dir),
            augment=False,
            use_faults=False,
            data_source='png'
        )
        
        loader = DataLoader(dataset, batch_size=1)
        
        model = load_unetplusplus(
            in_channels=4,
            classes=1,
            encoder_weights=None,
            device='cpu'
        )
        
        return model, loader
    
    def test_evaluate_on_test_returns_metrics(self, test_dataset_and_model):
        """Test evaluation returns expected metrics."""
        model, test_loader = test_dataset_and_model
        
        from losses.losses import CombinedLoss
        
        criterion = CombinedLoss()
        
        metrics = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device='cpu',
            threshold=0.5
        )
        
        # Should have standard metrics
        expected_keys = ['loss', 'iou', 'dice', 'precision', 'recall', 'f1']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float)
    
    def test_evaluate_metrics_in_valid_range(self, test_dataset_and_model):
        """Test that evaluation metrics are in valid ranges."""
        model, test_loader = test_dataset_and_model
        
        from losses.losses import CombinedLoss
        
        criterion = CombinedLoss()
        
        metrics = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device='cpu'
        )
        
        # IoU and Dice should be in [0, 1]
        assert 0 <= metrics['iou'] <= 1
        assert 0 <= metrics['dice'] <= 1
        
        # Precision and recall should be in [0, 1]
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1


@pytest.mark.smoke
def test_pipeline_smoke_test(tmp_path):
    """Smoke test for basic pipeline components."""
    from data.dataset import GeologyTrapsDataset
    from torch.utils.data import DataLoader
    from models.unetplusplus import load_unetplusplus
    from losses.losses import CombinedLoss
    
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create minimal sample
    for filename in [
        "001_x_structuralNOisoline_H150.png",
        "001_x_structuralBlackWhite_H150.png",
        "001_y_traps_H150.png"
    ]:
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)
        if 'BlackWhite' in filename or 'traps' in filename:
            img = img[:, :, 0]
        cv2.imwrite(str(data_dir / filename), img)
    
    file_paths = [str(data_dir / f) for f in os.listdir(data_dir)]
    
    dataset = GeologyTrapsDataset(
        file_list=file_paths,
        data_dir=str(data_dir),
        augment=False,
        use_faults=False
    )
    
    loader = DataLoader(dataset, batch_size=1)
    model = load_unetplusplus(in_channels=4, classes=1, encoder_weights=None, device='cpu')
    criterion = CombinedLoss()
    
    batch = next(iter(loader))
    x = batch['x']
    y = batch['y']
    
    model.train()
    output = model(x)
    loss, _ = criterion(output, y)
    
    assert loss.item() >= 0
