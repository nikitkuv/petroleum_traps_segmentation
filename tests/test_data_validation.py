import pytest
import numpy as np
import os

from utils.images_utils import (
    load_image,
    load_grayscale_image,
    create_binary_mask,
    create_map_mask,
    pad_image
)
from utils.cps_utils import (
    read_cps_grid,
    cps_to_rgb,
    cps_to_grayscale,
    cps_to_binary_mask
)


class TestImageLoading:
    """Tests for PNG image loading utilities."""
    
    @pytest.fixture
    def sample_rgb_image(self, tmp_path):
        """Create a sample RGB image for testing."""
        cv2 = pytest.importorskip("cv2")
        img_path = tmp_path / "test_rgb.png"
        # Create simple RGB gradient
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 100).astype(np.uint8)  # R gradient
        img[:, :, 1] = np.linspace(255, 0, 100).astype(np.uint8)  # G gradient
        img[:, :, 2] = 128  # Constant B
        cv2.imwrite(str(img_path), img)
        return img_path
    
    @pytest.fixture
    def sample_grayscale_image(self, tmp_path):
        """Create a sample grayscale image for testing."""
        cv2 = pytest.importorskip("cv2")
        img_path = tmp_path / "test_gray.png"
        img = np.linspace(0, 255, 100 * 100).reshape(100, 100).astype(np.uint8)
        cv2.imwrite(str(img_path), img)
        return img_path
    
    def test_load_image_returns_rgb(self, sample_rgb_image):
        """Test that load_image returns RGB image with correct shape."""
        img = load_image(str(sample_rgb_image))
        assert img.shape == (100, 100, 3)
        assert img.dtype == np.uint8
        # Check values are in valid range
        assert img.min() >= 0
        assert img.max() <= 255
    
    def test_load_grayscale_image_single_channel(self, sample_grayscale_image):
        """Test that load_grayscale_image returns single channel image."""
        img = load_grayscale_image(str(sample_grayscale_image))
        assert len(img.shape) == 2
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255
    
    def test_load_image_file_not_found(self):
        """Test that load_image raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")
    
    def test_load_grayscale_image_file_not_found(self):
        """Test that load_grayscale_image raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_grayscale_image("/nonexistent/path/image.png")


class TestMaskCreation:
    """Tests for mask creation utilities."""
    
    def test_create_binary_mask_threshold(self):
        """Test binary mask creation with threshold."""
        # Create image with values above and below threshold
        img = np.array([
            [0, 50, 100],
            [150, 200, 255]
        ], dtype=np.uint8)
        
        mask = create_binary_mask(img, invert=False)
        
        assert mask.shape == img.shape
        assert mask.dtype == np.float32
        # Values < 128 should be 1, >= 128 should be 0
        assert mask[0, 0] == 1.0  # 0 < 128
        assert mask[0, 1] == 1.0  # 50 < 128
        assert mask[0, 2] == 1.0  # 100 < 128
        assert mask[1, 0] == 0.0  # 150 >= 128
        assert mask[1, 1] == 0.0  # 200 >= 128
        assert mask[1, 2] == 0.0  # 255 >= 128
    
    def test_create_binary_mask_invert(self):
        """Test binary mask creation with inversion."""
        img = np.array([[100, 200]], dtype=np.uint8)
        
        mask_normal = create_binary_mask(img, invert=False)
        mask_inverted = create_binary_mask(img, invert=True)
        
        # Inverted mask should be opposite
        assert mask_normal[0, 0] == 1.0  # 100 < 128
        assert mask_inverted[0, 0] == 0.0  # inverted
        
        assert mask_normal[0, 1] == 0.0  # 200 >= 128
        assert mask_inverted[0, 1] == 1.0  # inverted
    
    def test_create_map_mask_identifies_non_background(self):
        """Test map mask correctly identifies non-background areas."""
        # Background is >= 250
        img = np.full((10, 10, 3), 255, dtype=np.uint8)  # All background
        img[5, 5] = [0, 0, 0]  # Non-background pixel
        
        mask = create_map_mask(img, data_source='images')
        
        assert mask.shape == (10, 10)
        assert mask.dtype == np.float32
        assert mask[5, 5] == 1.0  # Non-background
        assert mask[0, 0] == 0.0  # Background

    def test_create_map_mask_cps_tiles_black_background(self):
        """Test map mask correctly identifies non-background areas for 'cps_tiles' source."""
        # Background is ~0 (black background)
        img = np.full((10, 10, 3), 0, dtype=np.uint8)  # All black background
        img[5, 5] = [100, 100, 100]  # Non-background pixel (map area)

        mask = create_map_mask(img, data_source='cps_tiles')

        assert mask.shape == (10, 10)
        assert mask.dtype == np.float32
        assert mask[5, 5] == 1.0  # Non-background (map area)
        assert mask[0, 0] == 0.0  # Black background
    
    def test_create_map_mask_grayscale_input(self):
        """Test map mask works with grayscale input for default 'images' source."""
        img = np.full((10, 10), 255, dtype=np.uint8)  # All background (white)
        img[5, 5] = 0  # Non-background pixel

        mask = create_map_mask(img, data_source='images')

        assert mask[5, 5] == 1.0
        assert mask[0, 0] == 0.0

    def test_create_map_mask_grayscale_cps_tiles(self):
        """Test map mask works with grayscale input for 'cps_tiles' source."""
        img = np.full((10, 10), 0, dtype=np.uint8)  # All black background
        img[5, 5] = 100  # Non-background pixel (map area)

        mask = create_map_mask(img, data_source='cps_tiles')

        assert mask[5, 5] == 1.0
        assert mask[0, 0] == 0.0


class TestPadding:
    """Tests for image padding utilities."""
    
    def test_pad_image_adds_correct_padding(self):
        """Test that padding adds borders correctly."""
        img = np.ones((50, 50), dtype=np.float32)
        
        padded = pad_image(img, 100, 100)
        
        assert padded.shape == (100, 100)
        # Center should be 1s
        assert np.all(padded[25:75, 25:75] == 1.0)
        # Borders should be 0s
        assert np.all(padded[:25, :] == 0.0)
        assert np.all(padded[75:, :] == 0.0)
        assert np.all(padded[:, :25] == 0.0)
        assert np.all(padded[:, 75:] == 0.0)
    
    def test_pad_image_3channel(self):
        """Test padding works with 3-channel images."""
        img = np.ones((50, 50, 3), dtype=np.float32)
        
        padded = pad_image(img, 100, 100)
        
        assert padded.shape == (100, 100, 3)
        assert np.all(padded[25:75, 25:75, :] == 1.0)
    
    def test_pad_image_exceeds_target_raises_error(self):
        """Test that padding raises error if image exceeds target."""
        img = np.ones((200, 200), dtype=np.float32)
        
        with pytest.raises(ValueError, match="exceeds target"):
            pad_image(img, 100, 100)


class TestCPSUtils:
    """Tests for CPS file utilities."""
    
    @pytest.fixture
    def sample_cps_file(self, tmp_path):
        """Create a sample CPS file for testing."""
        cps_path = tmp_path / "test.cps"
        
        # Write minimal valid CPS header and data
        content = """FSASCI -99999.0
FSNROW 10 10
FSLIMI 0.0 10.0 0.0 10.0
->
"""
        # Add 10x10 grid data (10 values per line)
        for i in range(10):
            row = " ".join([str(float(i * 10 + j)) for j in range(10)])
            content += row + "\n"
        
        cps_path.write_text(content)
        return cps_path
    
    def test_read_cps_grid_parses_header(self, sample_cps_file):
        """Test CPS grid parsing extracts header information."""
        grid, meta = read_cps_grid(str(sample_cps_file))
        
        assert meta['nx'] == 10
        assert meta['ny'] == 10
        assert meta['xmin'] == 0.0
        assert meta['xmax'] == 10.0
        assert meta['null_value'] == -99999.0
    
    def test_read_cps_grid_shape(self, sample_cps_file):
        """Test CPS grid has correct shape."""
        grid, _ = read_cps_grid(str(sample_cps_file))
        
        assert grid.shape == (10, 10)
        assert grid.dtype == np.float32
    
    def test_read_cps_grid_handles_null_values(self, tmp_path):
        """Test that null values are converted to NaN."""
        cps_path = tmp_path / "test_null.cps"
        
        content = """FSASCI -99999.0
FSNROW 5 5
FSLIMI 0.0 5.0 0.0 5.0
->
"""
        # Add data with null value
        for i in range(5):
            values = []
            for j in range(5):
                if i == 2 and j == 2:
                    values.append("-99999.0")
                else:
                    values.append(str(float(i * 5 + j)))
            content += " ".join(values) + "\n"
        
        cps_path.write_text(content)
        
        grid, _ = read_cps_grid(str(cps_path))
        
        assert np.isnan(grid[2, 2])
        assert not np.isnan(grid[0, 0])
    
    def test_cps_to_rgb_produces_valid_image(self, sample_cps_file):
        """Test CPS to RGB conversion produces valid image."""
        grid, _ = read_cps_grid(str(sample_cps_file))
        
        rgb = cps_to_rgb(grid)
        
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8
        assert rgb.min() >= 0
        assert rgb.max() <= 255
    
    def test_cps_to_grayscale_produces_valid_image(self, sample_cps_file):
        """Test CPS to grayscale conversion produces valid image."""
        grid, _ = read_cps_grid(str(sample_cps_file))
        
        gray = cps_to_grayscale(grid)
        
        assert gray.shape == (10, 10)
        assert gray.dtype == np.uint8
        assert gray.min() >= 0
        assert gray.max() <= 255
    
    def test_cps_to_grayscale_invert(self, sample_cps_file):
        """Test CPS to grayscale with inversion."""
        grid, _ = read_cps_grid(str(sample_cps_file))
        
        gray_normal = cps_to_grayscale(grid, invert=False)
        gray_inverted = cps_to_grayscale(grid, invert=True)
        
        # Inverted should be opposite
        assert gray_normal[0, 0] != gray_inverted[0, 0]
    
    def test_cps_to_binary_mask_produces_float_mask(self, sample_cps_file):
        """Test CPS to binary mask produces float32 mask."""
        grid, _ = read_cps_grid(str(sample_cps_file))
        
        mask = cps_to_binary_mask(grid)
        
        assert mask.shape == grid.shape
        assert mask.dtype == np.float32
        assert set(np.unique(mask)).issubset({0.0, 1.0})
    
    def test_cps_to_rgb_handles_all_nan(self):
        """Test CPS to RGB handles all-NaN grid."""
        grid = np.full((10, 10), np.nan)
        
        rgb = cps_to_rgb(grid)
        
        assert rgb.shape == (10, 10, 3)
        assert np.all(rgb == 0)  # Should return black image


class TestFileNamingValidation:
    """Tests for file naming convention validation."""
    
    def test_parse_valid_filename(self):
        """Test parsing of valid filename."""
        from utils.dataset_utils import parse_filename
        
        result = parse_filename("001_x_structuralNOisoline_H150.png")
        
        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'x'
        assert result['type'] == 'structuralNOisoline'
        assert result['name'] == 'H150'
    
    def test_parse_traps_filename(self):
        """Test parsing of traps filename."""
        from utils.dataset_utils import parse_filename
        
        result = parse_filename("002_y_traps_BZ24.png")
        
        assert result is not None
        assert result['number'] == '002'
        assert result['role'] == 'y'
        assert result['type'] == 'traps'
        assert result['name'] == 'BZ24'
    
    def test_parse_faults_filename(self):
        """Test parsing of faults filename."""
        from utils.dataset_utils import parse_filename
        
        result = parse_filename("003_x_faults_H150.png")
        
        assert result is not None
        assert result['number'] == '003'
        assert result['role'] == 'x'
        assert result['type'] == 'faults'
        assert result['name'] == 'H150'
    
    def test_parse_invalid_filename_returns_none(self):
        """Test that invalid filenames return None."""
        from utils.dataset_utils import parse_filename
        
        invalid_names = [
            "invalid.png",
            "001_type_H150.png",  # Missing role
            "001_x_H150.png",  # Missing type
            "abc_x_structuralNOisoline_H150.png",  # Non-numeric number
        ]
        
        for name in invalid_names:
            result = parse_filename(name)
            assert result is None, f"Expected None for {name}"
    
    def test_parse_complex_horizon_name(self):
        """Test parsing filenames with complex horizon names."""
        from utils.dataset_utils import parse_filename
        
        result = parse_filename("001_x_structuralNOisoline_Ach322_top_1.png")
        
        assert result is not None
        assert result['number'] == '001'
        assert result['name'] == 'Ach322_top_1'  # Full remaining part


@pytest.mark.smoke
def test_quick_image_load_smoke(tmp_path):
    """Smoke test for basic image loading."""
    cv2 = pytest.importorskip("cv2")
    img_path = tmp_path / "smoke_test.png"
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    
    loaded = load_image(str(img_path))
    assert loaded.shape == (50, 50, 3)


class TestDatasetUtils:
    """Tests for dataset utility functions in utils/dataset_utils.py."""

    @pytest.fixture
    def sample_files_for_collect(self, tmp_path):
        """Create sample files for testing collect_samples."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create complete sample 1
        for filename in [
            "001_x_structuralNOisoline_H150.png",
            "001_x_structuralBlackWhite_H150.png",
            "001_y_traps_H150.png"
        ]:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2 = pytest.importorskip("cv2")
            cv2.imwrite(str(data_dir / filename), img)

        # Create complete sample 2 with faults
        for filename in [
            "002_x_structuralNOisoline_BZ24.png",
            "002_x_structuralBlackWhite_BZ24.png",
            "002_x_faults_BZ24.png",
            "002_y_traps_BZ24.png"
        ]:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2 = pytest.importorskip("cv2")
            cv2.imwrite(str(data_dir / filename), img)

        # Create incomplete sample (missing traps)
        for filename in [
            "003_x_structuralNOisoline_XUY1.png",
            "003_x_structuralBlackWhite_XUY1.png"
        ]:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2 = pytest.importorskip("cv2")
            cv2.imwrite(str(data_dir / filename), img)

        return [str(data_dir / f) for f in os.listdir(data_dir)]

    def test_get_sample_key_formats_correctly(self):
        """Test that get_sample_key creates correct key format."""
        from utils.dataset_utils import get_sample_key

        parsed = {
            'number': '001',
            'role': 'x',
            'type': 'structuralNOisoline',
            'name': 'H150'
        }

        key = get_sample_key(parsed)
        assert key == '001_H150'

    def test_get_sample_key_complex_name(self):
        """Test get_sample_key with complex horizon name."""
        from utils.dataset_utils import get_sample_key

        parsed = {
            'number': '002',
            'role': 'y',
            'type': 'traps',
            'name': 'Ach3-2-1_toptop1'
        }

        key = get_sample_key(parsed)
        assert key == '002_Ach3-2-1_toptop1'

    def test_collect_samples_complete_sample(self, sample_files_for_collect):
        """Test collect_samples with complete samples."""
        from utils.dataset_utils import collect_samples

        samples = collect_samples(sample_files_for_collect)

        # Should have 3 samples (including incomplete one)
        assert len(samples) == 3
        assert '001_H150' in samples
        assert '002_BZ24' in samples
        assert '003_XUY1' in samples

    def test_collect_samples_maps_channel_names(self, sample_files_for_collect):
        """Test that collect_samples correctly maps file types to channel names."""
        from utils.dataset_utils import collect_samples

        samples = collect_samples(sample_files_for_collect)

        # Check sample 1 (no faults)
        sample1 = samples['001_H150']
        assert any('structuralNOisoline' in v for v in sample1.values())  # rgb
        assert any('structuralBlackWhite' in v for v in sample1.values())  # depth_norm
        assert any('traps' in v for v in sample1.values())  # traps
        assert 'faults' not in sample1

        # Check sample 2 (with faults)
        sample2 = samples['002_BZ24']
        assert any('structuralNOisoline' in v for v in sample2.values())  # rgb
        assert any('structuralBlackWhite' in v for v in sample2.values())  # depth_norm
        assert any('traps' in v for v in sample2.values())  # traps
        assert any('faults' in v for v in sample2.values())  # faults

    def test_collect_samples_incomplete_sample(self, sample_files_for_collect):
        """Test that incomplete samples are still collected (filtering is done later)."""
        from utils.dataset_utils import collect_samples

        samples = collect_samples(sample_files_for_collect)

        # Incomplete sample should be present but missing traps
        sample3 = samples['003_XUY1']
        assert 'rgb' in sample3 or any('structuralNOisoline' in v for v in sample3.values())
        assert 'depth_norm' in sample3 or any('structuralBlackWhite' in v for v in sample3.values())
        assert 'traps' not in sample3

    def test_collect_samples_empty_list(self):
        """Test collect_samples with empty file list."""
        from utils.dataset_utils import collect_samples

        samples = collect_samples([])
        assert len(samples) == 0

    def test_collect_samples_invalid_filenames_skipped(self, tmp_path):
        """Test that files with invalid names are skipped."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create valid and invalid files
        cv2 = pytest.importorskip("cv2")
        valid_file = data_dir / "001_x_structuralNOisoline_H150.png"
        invalid_file1 = data_dir / "invalid.png"
        invalid_file2 = data_dir / "001_type_H150.png"  # Missing role

        for f in [valid_file, invalid_file1, invalid_file2]:
            cv2.imwrite(str(f), np.zeros((50, 50, 3), dtype=np.uint8))

        from utils.dataset_utils import collect_samples

        file_list = [str(f) for f in data_dir.iterdir()]
        samples = collect_samples(file_list)

        # Only valid file should be collected
        assert len(samples) == 1
        assert '001_H150' in samples

    def test_resolve_path_absolute_path_unchanged(self):
        """Test resolve_path with absolute path."""
        from utils.dataset_utils import resolve_path

        abs_path = "/absolute/path/to/file.png"
        result = resolve_path(abs_path, "/base/dir")

        assert result == abs_path

    def test_resolve_path_relative_path_joined(self):
        """Test resolve_path with relative path."""
        from utils.dataset_utils import resolve_path
        import os

        rel_path = "relative/file.png"
        base_dir = "/base/dir"
        result = resolve_path(rel_path, base_dir)

        assert result == os.path.join(base_dir, rel_path)

    def test_resolve_path_dot_slash_path_unchanged(self):
        """Test resolve_path with ./ prefix."""
        from utils.dataset_utils import resolve_path

        path = "./relative/file.png"
        result = resolve_path(path, "/base/dir")

        assert result == path

    def test_resolve_path_parent_dir_path_unchanged(self):
        """Test resolve_path with ../ prefix."""
        from utils.dataset_utils import resolve_path

        path = "../relative/file.png"
        result = resolve_path(path, "/base/dir")

        assert result == path

    def test_load_maps_into_ndarray_returns_correct_shapes(self, tmp_path):
        """Test load_maps_into_ndarray returns images with correct shapes."""
        from utils.dataset_utils import load_maps_into_ndarray

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        cv2 = pytest.importorskip("cv2")

        # Create test images
        rgb_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth_img = np.ones((50, 50), dtype=np.uint8) * 100
        traps_img = np.zeros((50, 50), dtype=np.uint8)
        traps_img[20:30, 20:30] = 255
        faults_img = np.zeros((50, 50), dtype=np.uint8)
        faults_img[30:, :] = 255

        rgb_path = str(data_dir / "rgb.png")
        depth_path = str(data_dir / "depth.png")
        traps_path = str(data_dir / "traps.png")
        faults_path = str(data_dir / "faults.png")

        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img)
        cv2.imwrite(traps_path, traps_img)
        cv2.imwrite(faults_path, faults_img)

        sample_paths = {
            'rgb': rgb_path,
            'depth_norm': depth_path,
            'traps': traps_path,
            'faults': faults_path
        }

        rgb, depth, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=True, data_source='images'
        )

        assert rgb.shape == (50, 50, 3)
        assert rgb.dtype == np.uint8
        assert depth.shape == (50, 50)
        assert depth.dtype == np.uint8
        assert traps.shape == (50, 50)
        assert traps.dtype == np.float32
        assert faults.shape == (50, 50)
        assert faults.dtype == np.float32

    def test_load_maps_into_ndarray_without_faults(self, tmp_path):
        """Test load_maps_into_ndarray without faults returns zero mask."""
        from utils.dataset_utils import load_maps_into_ndarray

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        cv2 = pytest.importorskip("cv2")

        rgb_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth_img = np.ones((50, 50), dtype=np.uint8) * 100
        traps_img = np.zeros((50, 50), dtype=np.uint8)

        rgb_path = str(data_dir / "rgb.png")
        depth_path = str(data_dir / "depth.png")
        traps_path = str(data_dir / "traps.png")

        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img)
        cv2.imwrite(traps_path, traps_img)

        sample_paths = {
            'rgb': rgb_path,
            'depth_norm': depth_path,
            'traps': traps_path
        }

        rgb, depth, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=False, data_source='images'
        )

        assert rgb.shape == (50, 50, 3)
        assert depth.shape == (50, 50)
        assert traps.shape == (50, 50)
        # Faults should be zero mask
        assert faults.shape == (50, 50)
        assert np.all(faults == 0.0)

    def test_load_maps_into_ndarray_cps_tiles_source(self, tmp_path):
        """Test load_maps_into_ndarray with cps_tiles data source."""
        from utils.dataset_utils import load_maps_into_ndarray

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        cv2 = pytest.importorskip("cv2")

        # For cps_tiles, traps/faults are white (> threshold)
        rgb_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth_img = np.ones((50, 50), dtype=np.uint8) * 100
        traps_img = np.zeros((50, 50), dtype=np.uint8)
        traps_img[20:30, 20:30] = 255  # White = trap

        rgb_path = str(data_dir / "rgb.png")
        depth_path = str(data_dir / "depth.png")
        traps_path = str(data_dir / "traps.png")

        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img)
        cv2.imwrite(traps_path, traps_img)

        sample_paths = {
            'rgb': rgb_path,
            'depth_norm': depth_path,
            'traps': traps_path
        }

        rgb, depth, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=False, data_source='cps_tiles'
        )

        assert rgb.shape == (50, 50, 3)
        # Traps should have 1s where image was white
        assert traps[25, 25] == 1.0  # Inside trap area
        assert traps[0, 0] == 0.0  # Outside trap area


@pytest.mark.smoke
def test_dataset_utils_smoke_test(tmp_path):
    """Smoke test for dataset_utils module."""
    from utils.dataset_utils import parse_filename, get_sample_key, collect_samples

    # Test parse_filename
    parsed = parse_filename("001_x_structuralNOisoline_H150.png")
    assert parsed is not None

    # Test get_sample_key
    key = get_sample_key(parsed)
    assert key == "001_H150"

    # Test collect_samples
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cv2 = pytest.importorskip("cv2")

    for filename in [
        "001_x_structuralNOisoline_H150.png",
        "001_x_structuralBlackWhite_H150.png",
        "001_y_traps_H150.png"
    ]:
        cv2.imwrite(str(data_dir / filename), np.zeros((50, 50, 3), dtype=np.uint8))

    file_list = [str(data_dir / f) for f in os.listdir(data_dir)]
    samples = collect_samples(file_list)

    assert len(samples) == 1
    assert "001_H150" in samples
    