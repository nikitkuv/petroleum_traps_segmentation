import pytest
import numpy as np
import torch
import os
import cv2

from data.dataset import GeologyTrapsDataset
from data.dataloaders import (
    get_file_list,
    split_data_by_groups,
    create_dataloaders
)
from utils.dataset_utils import collect_samples


class TestFileListCollection:
    """Tests for file list collection utilities."""

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a directory with sample data files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create complete samples for 3 cards (one for each split: train, val, test)
        # Card 001_KEK2
        for filename in [
            "001_x_structuralNOisoline_KEK2.png",
            "001_x_structuralBlackWhite_KEK2.png",
            "001_y_traps_KEK2.png"
        ]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(data_dir / filename), img)

        # Card 002_BZ24
        for filename in [
            "002_x_structuralNOisoline_BZ24.png",
            "002_x_structuralBlackWhite_BZ24.png",
            "002_y_traps_BZ24.png"
        ]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(data_dir / filename), img)

        # Card 003_XUY1
        for filename in [
            "003_x_structuralNOisoline_XUY1.png",
            "003_x_structuralBlackWhite_XUY1.png",
            "003_y_traps_XUY1.png"
        ]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(data_dir / filename), img)

        # Invalid file (should be skipped)
        invalid_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / "invalid_file.png"), invalid_img)

        return data_dir

    def test_get_file_list_finds_png_files(self, sample_data_dir):
        """Test that get_file_list finds all valid PNG files."""
        files = get_file_list(str(sample_data_dir), data_source='png')

        # Should find 6 valid files (3 per card × 2 cards)
        assert len(files) == 9

        # All should be PNG files
        assert all(f.endswith('.png') for f in files)

    def test_get_file_list_skips_invalid_names(self, sample_data_dir):
        """Test that files with invalid naming are skipped."""
        files = get_file_list(str(sample_data_dir), data_source='png')

        # invalid_file.png should not be included
        assert not any('invalid_file' in f for f in files)

    def test_collect_samples_groups_correctly(self, sample_data_dir):
        """Test that collect_samples groups files by sample key."""
        files = get_file_list(str(sample_data_dir), data_source='png')
        samples = collect_samples(files)

        # Should have 3 samples
        assert len(samples) == 3

        # Check sample keys
        assert '001_KEK2' in samples
        assert '002_BZ24' in samples
        assert '003_XUY1' in samples

        # Each sample should have rgb, depth, traps
        for key, sample_files in samples.items():
            assert 'rgb' in sample_files or any('structuralNOisoline' in v for v in sample_files.values())
            assert 'traps' in sample_files

    def test_split_data_by_groups_returns_all_splits(self, sample_data_dir):
        """Test that data splitting returns train/val/test sets."""
        files = get_file_list(str(sample_data_dir), data_source='png')

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


class TestDatasetInitialization:
    """Tests for GeologyTrapsDataset initialization."""

    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample files and return file list."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create minimal complete dataset for PNG source (no isolines/closed_isolines)
        filenames = [
            "001_x_structuralNOisoline_H150.png",
            "001_x_structuralBlackWhite_H150.png",
            "001_y_traps_H150.png",
        ]

        file_paths = []
        for filename in filenames:
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            img_path = data_dir / filename
            cv2.imwrite(str(img_path), img)
            file_paths.append(str(img_path))

        return file_paths, str(data_dir)

    def test_dataset_initializes_with_valid_files(self, sample_files):
        """Test dataset initializes successfully with valid files."""
        file_paths, data_dir = sample_files

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='png'
        )

        assert len(dataset) > 0
        assert dataset.use_faults == False
        assert dataset.data_source == 'png'

    def test_dataset_without_faults_excludes_fault_channel(self, sample_files):
        """Test dataset configuration without faults."""
        file_paths, data_dir = sample_files

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False,
            data_source='png'  # PNG source doesn't require isolines/closed_isolines
        )

        # Get a sample
        sample = dataset[0]

        # Input should have 4 channels (RGB + depth)
        assert sample['x'].shape[0] == 4
        assert sample['use_faults'] == False

    def test_dataset_reports_statistics_on_init(self, sample_files, capsys):
        """Test that dataset prints statistics on initialization."""
        file_paths, data_dir = sample_files

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False
        )

        captured = capsys.readouterr()
        assert "Dataset initialized" in captured.out
        assert "Mode: use_faults=" in captured.out
        assert "Data source:" in captured.out


class TestDatasetGetItem:
    """Tests for dataset __getitem__ method."""

    @pytest.fixture
    def complete_sample(self, tmp_path):
        """Create a complete sample with all required files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create RGB image (structuralNOisoline)
        rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_img[:, :, 0] = 255  # Red channel
        cv2.imwrite(str(data_dir / "001_x_structuralNOisoline_H150.png"), rgb_img)

        # Create depth image (structuralBlackWhite)
        depth_img = np.zeros((100, 100), dtype=np.uint8)
        depth_img[:, :] = 128  # Gray
        cv2.imwrite(str(data_dir / "001_x_structuralBlackWhite_H150.png"), depth_img)

        # Create traps mask
        traps_img = np.zeros((100, 100), dtype=np.uint8)
        traps_img[40:60, 40:60] = 255  # White square in center
        cv2.imwrite(str(data_dir / "001_y_traps_H150.png"), traps_img)

        # Create faults mask
        faults_img = np.zeros((100, 100), dtype=np.uint8)
        faults_img[50:, :] = 255  # Bottom half white
        cv2.imwrite(str(data_dir / "001_x_faults_H150.png"), faults_img)

        file_paths = [
            str(data_dir / "001_x_structuralNOisoline_H150.png"),
            str(data_dir / "001_x_structuralBlackWhite_H150.png"),
            str(data_dir / "001_y_traps_H150.png"),
            str(data_dir / "001_x_faults_H150.png")
        ]

        return file_paths, str(data_dir)

    def test_getitem_returns_required_keys(self, complete_sample):
        """Test that __getitem__ returns all required keys."""
        file_paths, data_dir = complete_sample

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=True,
            data_source='png'
        )

        sample = dataset[0]

        required_keys = ['x', 'y', 'mask_depth', 'mask_map', 'sample_idx', 'use_faults', 'metadata']
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"

    def test_getitem_input_channels_with_faults(self, complete_sample):
        """Test input has correct channels when faults are used."""
        file_paths, data_dir = complete_sample

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=True
        )

        sample = dataset[0]

        # Should have 5 channels: RGB(3) + depth(1) + faults(1)
        assert sample['x'].shape[0] == 5

    def test_getitem_input_channels_without_faults(self, complete_sample):
        """Test input has correct channels when faults are not used."""
        file_paths, data_dir = complete_sample

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False
        )

        sample = dataset[0]

        # Should have 4 channels: RGB(3) + depth(1)
        assert sample['x'].shape[0] == 4

    def test_getitem_output_is_single_channel(self, complete_sample):
        """Test output mask is single channel."""
        file_paths, data_dir = complete_sample

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False
        )

        sample = dataset[0]

        # Output should be (1, H, W)
        assert len(sample['y'].shape) == 3
        assert sample['y'].shape[0] == 1

    def test_getitem_values_normalized(self, complete_sample):
        """Test that input values are normalized to [0, 1]."""
        file_paths, data_dir = complete_sample

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=False
        )

        sample = dataset[0]

        # RGB values should be normalized (original was 0 and 255)
        x = sample['x']
        assert x.min() >= 0.0
        assert x.max() <= 1.0

    def test_getitem_masks_are_binary(self, complete_sample):
        """Test that masks contain only 0 and 1 values."""
        file_paths, data_dir = complete_sample

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            augment=False,
            use_faults=True
        )

        sample = dataset[0]

        # Check all masks are binary
        for mask_name in ['y', 'mask_depth', 'mask_map']:
            mask = sample[mask_name]
            unique_vals = set(torch.unique(mask).tolist())
            assert unique_vals.issubset({0.0, 1.0}), f"{mask_name} has non-binary values: {unique_vals}"

    def test_getitem_applies_padding(self, complete_sample):
        """Test that images are padded to target size."""
        file_paths, data_dir = complete_sample

        dataset = GeologyTrapsDataset(
            file_list=file_paths,
            data_dir=data_dir,
            target_h=128,
            target_w=128,
            augment=False,
            use_faults=False
        )

        sample = dataset[0]

        # Check spatial dimensions match target
        assert sample['x'].shape[2] == 128  # Width
        assert sample['x'].shape[1] == 128  # Height
        assert sample['y'].shape[2] == 128
        assert sample['y'].shape[1] == 128


class TestDataLoaderCreation:
    """Tests for DataLoader creation."""

    @pytest.fixture
    def sample_data_for_loader(self, tmp_path):
        """Create enough samples for DataLoader testing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create 6 complete samples with 3 different map groups (for train/val/test splits)
        # Each map group needs at least one complete sample
        sample_configs = [
            # Map group KEK2 - will go to train
            (1, "KEK2"),
            (2, "KEK2"),
            # Map group BZ24 - will go to val
            (3, "BZ24"),
            (4, "BZ24"),
            # Map group XUY1 - will go to test
            (5, "XUY1"),
            (6, "XUY1"),
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

    def test_create_dataloaders_returns_three_loaders(self, sample_data_for_loader):
        """Test that create_dataloaders returns train/val/test loaders."""
        files = get_file_list(sample_data_for_loader, data_source='png')

        train_files, val_files, test_files = split_data_by_groups(files)

        train_loader, val_loader, test_loader = create_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dir=sample_data_for_loader,
            batch_size=2,
            use_faults=False,
            data_source='png'
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_dataloader_batch_shapes(self, sample_data_for_loader):
        """Test that DataLoader produces batches with correct shapes."""
        files = get_file_list(sample_data_for_loader, data_source='png')

        # Use all files for train to ensure we have enough
        train_loader, _, _ = create_dataloaders(
            train_files=files,
            val_files=[],
            test_files=[],
            data_dir=sample_data_for_loader,
            batch_size=2,
            use_faults=False,
            data_source='png'
        )

        batch = next(iter(train_loader))

        # Batch dimension should be present
        assert len(batch['x'].shape) == 4  # (B, C, H, W)
        assert batch['x'].shape[0] == 2  # Batch size
        assert batch['x'].shape[1] == 4  # Channels (RGB + depth)

    def test_dataloader_preserves_metadata(self, sample_data_for_loader):
        """Test that DataLoader preserves sample metadata."""
        files = get_file_list(sample_data_for_loader, data_source='png')

        train_files, val_files, test_files = split_data_by_groups(files)

        _, _, test_loader = create_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dir=sample_data_for_loader,
            batch_size=1,
            use_faults=False,
            data_source='png'
        )

        batch = next(iter(test_loader))

        assert 'metadata' in batch
        assert 'sample_idx' in batch


@pytest.mark.smoke
def test_dataset_smoke_test(tmp_path):
    """Smoke test for basic dataset functionality."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create minimal sample
    for filename in [
        "001_x_structuralNOisoline_H150.png",
        "001_x_structuralBlackWhite_H150.png",
        "001_y_traps_H150.png"
    ]:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(data_dir / filename), img)

    file_paths = [str(data_dir / f) for f in os.listdir(data_dir)]

    dataset = GeologyTrapsDataset(
        file_list=file_paths,
        data_dir=str(data_dir),
        augment=False,
        use_faults=False
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert 'x' in sample
    assert 'y' in sample
