import pytest
import numpy as np

from utils.cps_utils import (
    read_cps_grid,
    cps_to_rgb,
    cps_to_grayscale,
    cps_to_binary_mask,
    cps_to_isolines,
    cps_to_closed_mask
)


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

    @pytest.fixture
    def sample_cps_with_trap(self, tmp_path):
        """Create a CPS file with a clear trap structure (closed contour)."""
        cps_path = tmp_path / "test_trap.cps"

        # Create a dome structure: higher in the center, lower at edges
        # This should create closed isolines in the center
        content = """FSASCI -99999.0
FSNROW 20 20
FSLIMI 0.0 20.0 0.0 20.0
->
"""
        for i in range(20):
            values = []
            for j in range(20):
                # Create a dome: center is higher (less negative)
                dist_from_center = ((i - 10) ** 2 + (j - 10) ** 2) ** 0.5
                value = -1000 - dist_from_center * 10  # Deeper at edges
                values.append(str(value))
            content += " ".join(values) + "\n"

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

    def test_cps_to_isolines_produces_valid_image(self, sample_cps_file):
        """Test CPS to isolines conversion produces valid image."""
        grid, _ = read_cps_grid(str(sample_cps_file))

        isolines = cps_to_isolines(grid, step=5.0)

        assert isolines.shape == grid.shape
        assert isolines.dtype == np.uint8
        assert set(np.unique(isolines)).issubset({0, 255})

    def test_cps_to_isolines_has_lines(self, sample_cps_with_trap):
        """Test that isolines are actually generated."""
        grid, _ = read_cps_grid(str(sample_cps_with_trap))

        isolines = cps_to_isolines(grid, step=5.0)

        # Should have some white pixels (isolines)
        assert np.sum(isolines > 0) > 0

    def test_cps_to_closed_mask_produces_float_mask(self, sample_cps_with_trap):
        """Test CPS to closed mask produces float32 mask."""
        grid, _ = read_cps_grid(str(sample_cps_with_trap))

        closed_mask = cps_to_closed_mask(grid, step=5.0)

        assert closed_mask.shape == grid.shape
        assert closed_mask.dtype == np.float32
        assert set(np.unique(closed_mask)).issubset({0.0, 1.0})

    def test_cps_to_closed_mask_detects_trap(self, sample_cps_with_trap):
        """Test that closed mask detects trap structure."""
        grid, _ = read_cps_grid(str(sample_cps_with_trap))

        closed_mask = cps_to_closed_mask(grid, step=5.0)

        # Dome structure should create closed contours in the center
        # Check that center area has closed mask = 1
        center_y, center_x = 10, 10
        assert closed_mask[center_y, center_x] == 1.0

    def test_cps_to_closed_mask_no_trap_on_flat_surface(self, tmp_path):
        """Test that flat surface doesn't create closed masks."""
        cps_path = tmp_path / "test_flat.cps"

        # Create completely flat surface
        content = """FSASCI -99999.0
FSNROW 10 10
FSLIMI 0.0 10.0 0.0 10.0
->
"""
        for i in range(10):
            row = " ".join(["100.0"] * 10)
            content += row + "\n"

        cps_path.write_text(content)

        grid, _ = read_cps_grid(str(cps_path))
        closed_mask = cps_to_closed_mask(grid, step=5.0)

        # Flat surface should not have closed contours
        assert np.all(closed_mask == 0.0)

    def test_cps_to_closed_mask_handles_null_values(self, tmp_path):
        """Test that closed mask handles NaN values correctly."""
        cps_path = tmp_path / "test_null_closed.cps"

        content = """FSASCI -99999.0
FSNROW 10 10
FSLIMI 0.0 10.0 0.0 10.0
->
"""
        # Add data with null values at edges
        for i in range(10):
            values = []
            for j in range(10):
                if i < 2 or i >= 8 or j < 2 or j >= 8:
                    values.append("-99999.0")
                else:
                    values.append("100.0")
            content += " ".join(values) + "\n"

        cps_path.write_text(content)

        grid, _ = read_cps_grid(str(cps_path))
        closed_mask = cps_to_closed_mask(grid, step=5.0)

        # Area with NaN should not have closed mask
        assert closed_mask[0, 0] == 0.0
        assert closed_mask[1, 1] == 0.0

    def test_cps_to_closed_mask_min_area_filter(self, sample_cps_with_trap):
        """Test that small areas are filtered out by min_area."""
        grid, _ = read_cps_grid(str(sample_cps_with_trap))

        # With very large min_area, no closed masks should be detected
        closed_mask_large = cps_to_closed_mask(grid, step=5.0, min_area=10000)

        assert np.all(closed_mask_large == 0.0)

        # With small min_area, should detect traps
        closed_mask_small = cps_to_closed_mask(grid, step=5.0, min_area=10)

        assert np.sum(closed_mask_small > 0) > 0
