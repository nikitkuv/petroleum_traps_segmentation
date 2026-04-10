import pytest
import numpy as np
import torch
import os
import cv2
import shutil
from pathlib import Path

from data.dataset import GeologyTrapsDataset
from data.dataloaders import (
    get_file_list,
    collect_samples,
    split_data_by_groups,
    create_dataloaders
)
from settings import settings


class TestCpsTilesFileListCollection:
    """Tests for CPS tiles file list collection utilities."""
    
    @pytest.fixture
    def sample_cps_tiles_dir(self, tmp_path):
        """Create a directory with sample CPS tile PNG files."""
        data_dir = tmp_path / "images_cps"
        data_dir.mkdir()
        
        # Create complete samples for 3 cards (one for each split: train, val, test)
        # Card 001 with 3 horizons
        for filename in [
            "001_x_structuralNOisoline_Ach3-2-1_toptop1.png",
            "001_x_structuralBlackWhite_Ach3-2-1_toptop1.png",
            "001_y_traps_Ach3-2-1_toptop1.png",
            "001_x_structuralNOisoline_U2_3_kolltop1.png",
            "001_x_structuralBlackWhite_U2_3_kolltop1.png",
            "001_y_traps_U2_3_kolltop1.png",
            "001_x_structuralNOisoline_U4_42_kolltop1.png",
            "001_x_structuralBlackWhite_U4_42_kolltop1.png",
            "001_y_traps_U4_42_kolltop1.png",
        ]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(data_dir / filename), img)

        # Card 002 with 2 horizons
        for filename in [
            "002_x_structuralNOisoline_U2_3_kolltop1.png",
            "002_x_structuralBlackWhite_U2_3_kolltop1.png",
            "002_y_traps_U2_3_kolltop1.png",
            "002_x_structuralNOisoline_U4_42_kolltop1.png",
            "002_x_structuralBlackWhite_U4_42_kolltop1.png",
            "002_y_traps_U4_42_kolltop1.png",
        ]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(data_dir / filename), img)

        # Card 003 with 2 horizons
        for filename in [
            "003_x_structuralNOisoline_U2_3_kolltop1.png",
            "003_x_structuralBlackWhite_U2_3_kolltop1.png",
            "003_y_traps_U2_3_kolltop1.png",
            "003_x_structuralNOisoline_U4_42_kolltop1.png",
            "003_x_structuralBlackWhite_U4_42_kolltop1.png",
            "003_y_traps_U4_42_kolltop1.png",
        ]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(data_dir / filename), img)
        
        # Invalid file (should be skipped)
        invalid_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / "invalid_file.png"), invalid_img)
        
        return str(data_dir)
    
    def test_get_file_list_finds_cps_tiles_png_files(self, sample_cps_tiles_dir):
        """Test that get_file_list finds all valid CPS tile PNG files."""
        files = get_file_list(sample_cps_tiles_dir, data_source='cps_tiles')
        
        # Should find 21 valid files (9 + 6 + 6)
        assert len(files) == 21
        
        # All should be PNG files
        assert all(f.endswith('.png') for f in files)
    
    def test_get_file_list_skips_invalid_names_cps_tiles(self, sample_cps_tiles_dir):
        """Test that files with invalid naming are skipped for cps_tiles."""
        files = get_file_list(sample_cps_tiles_dir, data_source='cps_tiles')
        
        # invalid_file.png should not be included
        assert not any('invalid_file' in f for f in files)
    
    def test_collect_samples_groups_cps_tiles_correctly(self, sample_cps_tiles_dir):
        """Test that collect_samples groups CPS tile files by sample key."""
        files = get_file_list(sample_cps_tiles_dir, data_source='cps_tiles')
        samples = collect_samples(files)
        
        # Should have 7 samples (3 for card 001, 2 for card 002, 2 for card 003)
        assert len(samples) == 7
        
        # Check sample keys exist
        assert '001_Ach3-2-1_toptop1' in samples
        assert '001_U2_3_kolltop1' in samples
        assert '001_U4_42_kolltop1' in samples
        assert '002_U2_3_kolltop1' in samples
        assert '002_U4_42_kolltop1' in samples
        assert '003_U2_3_kolltop1' in samples
        assert '003_U4_42_kolltop1' in samples
        
        # Each sample should have rgb, depth, traps
        for key, sample_files in samples.items():
            assert 'rgb' in sample_files or any('structuralNOisoline' in v for v in sample_files.values())
            assert 'traps' in sample_files
    
    def test_split_data_by_groups_cps_tiles_returns_all_splits(self, sample_cps_tiles_dir):
        """Test that data splitting returns train/val/test sets for CPS tiles."""
        files = get_file_list(sample_cps_tiles_dir, data_source='cps_tiles')
        
        train_files, val_files, test_files = split_data_by_groups(
            files,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42
        )
        
        # All files should be distributed
        total = len(train_files) + len(val_files) + len(test_files)
        assert total == len(files)
        
        # Train should have most files
        assert len(train_files) >= len(val_files)
        assert len(train_files) >= len(test_files)


class TestCpsTilesDatasetInitialization:
    """Tests for GeologyTrapsDataset initialization with CPS tiles."""
    
    @pytest.fixture
    def sample_cps_tiles_files(self, tmp_path):
        """Create sample CPS tile files and return file list."""
        data_dir = tmp_path / "images_cps"
        data_dir.mkdir()
        
        # Create minimal complete dataset with CPS-style naming
        filenames = [
            "001_x_structuralNOisoline_Ach3-2-1_toptop1.png",
            "001_x_structuralBlackWhite_Ach3-2-1_toptop1.png",
            "001_y_traps_Ach3-2-1_toptop1.png",
        ]
        
        file_paths = []
        for filename in filenames:
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            img_path = data_dir / filename
            cv2.imwrite(str(img_path), img)
            file_paths.append(str(img_path))
        
        return file_paths, str(data_dir)
    
    def test_cps_tiles_dataset_initializes_with_valid_files(self, sample_cps_tiles_files):
        """Test CPS tiles dataset initializes successfully with valid files."""
        file_paths, data_dir = sample_cps_tiles_files
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        assert len(dataset) > 0
        assert dataset.use_faults == False
        assert dataset.data_source == 'cps_tiles'
    
    def test_cps_tiles_dataset_without_faults_excludes_fault_channel(self, sample_cps_tiles_files):
        """Test CPS tiles dataset configuration without faults."""
        file_paths, data_dir = sample_cps_tiles_files
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        # Get a sample
        sample = dataset[0]
        
        # Input should have 4 channels (RGB + depth)
        assert sample['x'].shape[0] == 4
        assert sample['use_faults'] == False
    
    def test_cps_tiles_dataset_reports_statistics_on_init(self, sample_cps_tiles_files, capsys):
        """Test that CPS tiles dataset prints statistics on initialization."""
        file_paths, data_dir = sample_cps_tiles_files
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        captured = capsys.readouterr()
        assert "Dataset initialized" in captured.out
        assert "Mode: use_faults=" in captured.out
        assert "Data source:" in captured.out
        assert "cps_tiles" in captured.out


class TestCpsTilesDatasetGetItem:
    """Tests for CPS tiles dataset __getitem__ method."""
    
    @pytest.fixture
    def complete_cps_sample(self, tmp_path):
        """Create a complete CPS tile sample with all required files."""
        data_dir = tmp_path / "images_cps"
        data_dir.mkdir()
        
        # Create RGB image (structuralNOisoline)
        rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_img[:, :, 0] = 255  # Red channel
        cv2.imwrite(str(data_dir / "001_x_structuralNOisoline_Ach3-2-1_toptop1.png"), rgb_img)
        
        # Create depth image (structuralBlackWhite)
        depth_img = np.zeros((100, 100), dtype=np.uint8)
        depth_img[:, :] = 128  # Gray
        cv2.imwrite(str(data_dir / "001_x_structuralBlackWhite_Ach3-2-1_toptop1.png"), depth_img)
        
        # Create traps mask
        traps_img = np.zeros((100, 100), dtype=np.uint8)
        traps_img[40:60, 40:60] = 255  # White square in center
        cv2.imwrite(str(data_dir / "001_y_traps_Ach3-2-1_toptop1.png"), traps_img)
        
        # Create faults mask
        faults_img = np.zeros((100, 100), dtype=np.uint8)
        faults_img[50:, :] = 255  # Bottom half white
        cv2.imwrite(str(data_dir / "001_x_faults_Ach3-2-1_toptop1.png"), faults_img)
        
        file_paths = [
            str(data_dir / "001_x_structuralNOisoline_Ach3-2-1_toptop1.png"),
            str(data_dir / "001_x_structuralBlackWhite_Ach3-2-1_toptop1.png"),
            str(data_dir / "001_y_traps_Ach3-2-1_toptop1.png"),
            str(data_dir / "001_x_faults_Ach3-2-1_toptop1.png")
        ]
        
        return file_paths, str(data_dir)
    
    def test_cps_tiles_getitem_returns_required_keys(self, complete_cps_sample):
        """Test that CPS tiles __getitem__ returns all required keys."""
        file_paths, data_dir = complete_cps_sample
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=True,
            data_source='cps_tiles'
        )
        
        sample = dataset[0]
        
        required_keys = ['x', 'y', 'mask_depth', 'mask_map', 'sample_idx', 'use_faults', 'metadata']
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
    
    def test_cps_tiles_getitem_input_channels_with_faults(self, complete_cps_sample):
        """Test CPS tiles input has correct channels when faults are used."""
        file_paths, data_dir = complete_cps_sample
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=True,
            data_source='cps_tiles'
        )
        
        sample = dataset[0]
        
        # Should have 5 channels: RGB(3) + depth(1) + faults(1)
        assert sample['x'].shape[0] == 5
    
    def test_cps_tiles_getitem_input_channels_without_faults(self, complete_cps_sample):
        """Test CPS tiles input has correct channels when faults are not used."""
        file_paths, data_dir = complete_cps_sample
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        sample = dataset[0]
        
        # Should have 4 channels: RGB(3) + depth(1)
        assert sample['x'].shape[0] == 4
    
    def test_cps_tiles_getitem_output_is_single_channel(self, complete_cps_sample):
        """Test CPS tiles output mask is single channel."""
        file_paths, data_dir = complete_cps_sample
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        sample = dataset[0]
        
        # Output should be (1, H, W)
        assert len(sample['y'].shape) == 3
        assert sample['y'].shape[0] == 1
    
    def test_cps_tiles_getitem_values_normalized(self, complete_cps_sample):
        """Test that CPS tiles input values are normalized to [0, 1]."""
        file_paths, data_dir = complete_cps_sample
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        sample = dataset[0]
        
        # RGB values should be normalized (original was 0 and 255)
        x = sample['x']
        assert x.min() >= 0.0
        assert x.max() <= 1.0
    
    def test_cps_tiles_getitem_masks_are_binary(self, complete_cps_sample):
        """Test that CPS tiles masks contain only 0 and 1 values."""
        file_paths, data_dir = complete_cps_sample
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=True,
            data_source='cps_tiles'
        )
        
        sample = dataset[0]
        
        # Check all masks are binary
        for mask_name in ['y', 'mask_depth', 'mask_map']:
            mask = sample[mask_name]
            unique_vals = set(torch.unique(mask).tolist())
            assert unique_vals.issubset({0.0, 1.0}), f"{mask_name} has non-binary values: {unique_vals}"
    
    def test_cps_tiles_getitem_applies_padding(self, complete_cps_sample):
        """Test that CPS tiles images are padded to target size."""
        file_paths, data_dir = complete_cps_sample
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            target_h=128,
            target_w=128,
            augment=False,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        sample = dataset[0]
        
        # Check spatial dimensions match target
        assert sample['x'].shape[2] == 128  # Width
        assert sample['x'].shape[1] == 128  # Height
        assert sample['y'].shape[2] == 128
        assert sample['y'].shape[1] == 128
    
    def test_cps_tiles_getitem_metadata_contains_horizon_info(self, complete_cps_sample):
        """Test that CPS tiles metadata contains horizon information."""
        file_paths, data_dir = complete_cps_sample
        
        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        sample = dataset[0]
        
        # Metadata should contain sample key with horizon info
        assert 'metadata' in sample
        assert 'sample_key' in sample['metadata']
        assert 'Ach3-2-1_toptop1' in sample['metadata']['sample_key']


class TestCpsTilesDataLoaderCreation:
    """Tests for CPS tiles DataLoader creation."""
    
    @pytest.fixture
    def sample_cps_data_for_loader(self, tmp_path):
        """Create enough CPS tile samples for DataLoader testing."""
        data_dir = tmp_path / "images_cps"
        data_dir.mkdir()
        
        # Create 6 complete CPS tile samples with different map groups (for train/val/test splits)
        # Each map group needs at least one complete sample
        sample_configs = [
            # Map group Ach3-2-1_toptop1 - will go to train
            (1, "Ach3-2-1_toptop1"),
            (2, "Ach3-2-1_toptop1"),
            # Map group U2_3_kolltop1 - will go to val
            (3, "U2_3_kolltop1"),
            (4, "U2_3_kolltop1"),
            # Map group U4_42_kolltop1 - will go to test
            (5, "U4_42_kolltop1"),
            (6, "U4_42_kolltop1"),
        ]

        for card_num, horizon in sample_configs:
            
            # RGB
            rgb_img = np.ones((100, 100, 3), dtype=np.uint8) * card_num * 50
            cv2.imwrite(str(data_dir / f"{card_num:03d}_x_structuralNOisoline_{horizon}.png"), rgb_img)
            
            # Depth
            depth_img = np.ones((100, 100), dtype=np.uint8) * card_num * 50
            cv2.imwrite(str(data_dir / f"{card_num:03d}_x_structuralBlackWhite_{horizon}.png"), depth_img)
            
            # Traps
            traps_img = np.zeros((100, 100), dtype=np.uint8)
            traps_img[40:60, 40:60] = 200
            cv2.imwrite(str(data_dir / f"{card_num:03d}_y_traps_{horizon}.png"), traps_img)
        
        return str(data_dir)
    
    def test_cps_tiles_create_dataloaders_returns_three_loaders(self, sample_cps_data_for_loader):
        """Test that create_dataloaders returns train/val/test loaders for CPS tiles."""
        files = get_file_list(sample_cps_data_for_loader, data_source='cps_tiles')
        
        train_files, val_files, test_files = split_data_by_groups(files)
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dir=sample_cps_data_for_loader,
            batch_size=2,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
    
    def test_cps_tiles_dataloader_batch_shapes(self, sample_cps_data_for_loader):
        """Test that CPS tiles DataLoader produces batches with correct shapes."""
        files = get_file_list(sample_cps_data_for_loader, data_source='cps_tiles')
        
        # Use all files for train to ensure we have enough
        train_loader, _, _ = create_dataloaders(
            train_files=files,
            val_files=[],
            test_files=[],
            data_dir=sample_cps_data_for_loader,
            batch_size=2,
            use_faults=False,
            data_source='cps_tiles'
        )
        
        batch = next(iter(train_loader))
        
        # Batch dimension should be present
        assert len(batch['x'].shape) == 4  # (B, C, H, W)
        assert batch['x'].shape[0] == 2  # Batch size
        assert batch['x'].shape[1] == 4  # Channels (RGB + depth)


@pytest.mark.integration
class TestCpsTilesWithRealData:
    """Integration tests using real CPS tiles from data/images_cps/."""
    
    def test_real_cps_tiles_directory_exists(self):
        """Test that the real CPS tiles directory exists."""
        cps_tiles_dir = Path("data/images_cps")
        assert cps_tiles_dir.exists(), "data/images_cps/ directory does not exist"
    
    def test_real_cps_tiles_files_found(self):
        """Test that real CPS tiles files are found."""
        cps_tiles_dir = "data/images_cps"
        files = get_file_list(cps_tiles_dir, data_source='cps_tiles')
        
        # Should find many files (at least 50+)
        assert len(files) > 50, f"Expected many CPS tile files, found {len(files)}"
        
        # All should be PNG files
        assert all(f.endswith('.png') for f in files)
    
    def test_real_cps_tiles_samples_collected(self):
        """Test that real CPS tiles samples are collected correctly."""
        cps_tiles_dir = "data/images_cps"
        files = get_file_list(cps_tiles_dir, data_source='cps_tiles')
        samples = collect_samples(files)
        
        # Should find multiple samples
        assert len(samples) > 10, f"Expected many CPS tile samples, found {len(samples)}"
        
        # Each sample should have required files
        for key, sample_files in samples.items():
            assert 'traps' in sample_files, f"Sample {key} missing traps file"


@pytest.mark.smoke
def test_cps_tiles_smoke_test(tmp_path):
    """Smoke test for basic CPS tiles functionality."""
    data_dir = tmp_path / "images_cps"
    data_dir.mkdir()
    
    # Create minimal CPS tile sample
    for filename in [
        "001_x_structuralNOisoline_Ach3-2-1_toptop1.png",
        "001_x_structuralBlackWhite_Ach3-2-1_toptop1.png",
        "001_y_traps_Ach3-2-1_toptop1.png"
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
        use_faults=False,
        data_source='cps_tiles'
    )
    
    assert len(dataset) > 0
    
    sample = dataset[0]
    assert 'x' in sample
    assert 'y' in sample
    assert sample['data_source'] == 'cps_tiles'
