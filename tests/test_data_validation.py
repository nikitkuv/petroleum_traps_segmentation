import pytest
import numpy as np

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
        
        mask = create_map_mask(img)
        
        assert mask.shape == (10, 10)
        assert mask.dtype == np.float32
        assert mask[5, 5] == 1.0  # Non-background
        assert mask[0, 0] == 0.0  # Background
    
    def test_create_map_mask_grayscale_input(self):
        """Test map mask works with grayscale input."""
        img = np.full((10, 10), 255, dtype=np.uint8)  # All background
        img[5, 5] = 0  # Non-background pixel
        
        mask = create_map_mask(img)
        
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
    
    def test_read_cps_grid_values(self, sample_cps_file):
        """Test CPS grid values are correctly parsed."""
        grid, _ = read_cps_grid(str(sample_cps_file))
        
        # Check some known values
        # Grid is filled in Fortran order (column-major): 
        # values are written row by row but reshaped with order='F'
        # So grid[0, 0] = 0.0, grid[1, 0] = 1.0, ..., grid[9, 0] = 9.0
        # Then grid[0, 1] = 10.0, grid[1, 1] = 11.0, etc.
        assert grid[0, 0] == 0.0
        assert grid[9, 0] == 9.0
        assert grid[0, 1] == 10.0
    
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
        from data.dataloaders import parse_filename
        
        result = parse_filename("001_x_structuralNOisoline_H150.png")
        
        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'x'
        assert result['type'] == 'structuralNOisoline'
        assert result['name'] == 'H150'
    
    def test_parse_traps_filename(self):
        """Test parsing of traps filename."""
        from data.dataloaders import parse_filename
        
        result = parse_filename("002_y_traps_BZ24.png")
        
        assert result is not None
        assert result['number'] == '002'
        assert result['role'] == 'y'
        assert result['type'] == 'traps'
        assert result['name'] == 'BZ24'
    
    def test_parse_faults_filename(self):
        """Test parsing of faults filename."""
        from data.dataloaders import parse_filename
        
        result = parse_filename("003_x_faults_H150.png")
        
        assert result is not None
        assert result['number'] == '003'
        assert result['role'] == 'x'
        assert result['type'] == 'faults'
        assert result['name'] == 'H150'
    
    def test_parse_invalid_filename_returns_none(self):
        """Test that invalid filenames return None."""
        from data.dataloaders import parse_filename
        
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
        from data.dataloaders import parse_filename
        
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
