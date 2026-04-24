import pytest
import numpy as np
import os

from utils.dataset_utils import (
    parse_filename,
    get_sample_key,
    collect_samples,
    resolve_path,
    load_maps_into_ndarray
)


class TestFileNamingValidation:
    """Tests for file naming convention validation."""

    def test_parse_valid_filename(self):
        """Test parsing of valid filename."""
        result = parse_filename("001_x_structuralNOisoline_H150.png")

        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'x'
        assert result['type'] == 'structuralNOisoline'
        assert result['name'] == 'H150'

    def test_parse_traps_filename(self):
        """Test parsing of traps filename."""
        result = parse_filename("002_y_traps_BZ24.png")

        assert result is not None
        assert result['number'] == '002'
        assert result['role'] == 'y'
        assert result['type'] == 'traps'
        assert result['name'] == 'BZ24'

    def test_parse_faults_filename(self):
        """Test parsing of faults filename."""
        result = parse_filename("003_x_faults_H150.png")

        assert result is not None
        assert result['number'] == '003'
        assert result['role'] == 'x'
        assert result['type'] == 'faults'
        assert result['name'] == 'H150'

    def test_parse_invalid_filename_returns_none(self):
        """Test that invalid filenames return None."""
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
        result = parse_filename("001_x_structuralNOisoline_Ach322_top_1.png")

        assert result is not None
        assert result['number'] == '001'
        assert result['name'] == 'Ach322_top_1'  # Full remaining part


class TestDatasetUtils:
    """Tests for dataset utility functions in utils/dataset_utils.py."""

    @pytest.fixture
    def sample_files_for_collect(self, tmp_path):
        """Create sample files for testing collect_samples."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create complete sample 1 (cps_tiles format with isolines and closed_isolines)
        for filename in [
            "001_x_structuralNOisoline_H150.png",
            "001_x_structuralBlackWhite_H150.png",
            "001_x_isolines_H150.png",
            "001_x_closedIsolines_H150.png",
            "001_y_traps_H150.png"
        ]:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2 = pytest.importorskip("cv2")
            cv2.imwrite(str(data_dir / filename), img)

        # Create complete sample 2 with faults (cps_tiles format)
        for filename in [
            "002_x_structuralNOisoline_BZ24.png",
            "002_x_structuralBlackWhite_BZ24.png",
            "002_x_isolines_BZ24.png",
            "002_x_closedIsolines_BZ24.png",
            "002_x_faults_BZ24.png",
            "002_y_traps_BZ24.png"
        ]:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2 = pytest.importorskip("cv2")
            cv2.imwrite(str(data_dir / filename), img)

        # Create incomplete sample (missing traps and closed_isolines)
        for filename in [
            "003_x_structuralNOisoline_XUY1.png",
            "003_x_structuralBlackWhite_XUY1.png",
            "003_x_isolines_XUY1.png"
        ]:
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2 = pytest.importorskip("cv2")
            cv2.imwrite(str(data_dir / filename), img)

        return [str(data_dir / f) for f in os.listdir(data_dir)]

    def test_get_sample_key_formats_correctly(self):
        """Test that get_sample_key creates correct key format."""
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
        samples = collect_samples(sample_files_for_collect)

        # Should have 3 samples (including incomplete one)
        assert len(samples) == 3
        assert '001_H150' in samples
        assert '002_BZ24' in samples
        assert '003_XUY1' in samples

    def test_collect_samples_maps_channel_names(self, sample_files_for_collect):
        """Test that collect_samples correctly maps file types to channel names."""
        samples = collect_samples(sample_files_for_collect)

        # Check sample 1 (no faults, cps_tiles format)
        sample1 = samples['001_H150']
        assert any('structuralNOisoline' in v for v in sample1.values())  # rgb
        assert any('structuralBlackWhite' in v for v in sample1.values())  # depth_norm
        assert any('isolines' in v for v in sample1.values())  # isolines
        assert any('closedIsolines' in v for v in sample1.values())  # closed_isolines
        assert any('traps' in v for v in sample1.values())  # traps
        assert 'faults' not in sample1

        # Check sample 2 (with faults, cps_tiles format)
        sample2 = samples['002_BZ24']
        assert any('structuralNOisoline' in v for v in sample2.values())  # rgb
        assert any('structuralBlackWhite' in v for v in sample2.values())  # depth_norm
        assert any('isolines' in v for v in sample2.values())  # isolines
        assert any('closedIsolines' in v for v in sample2.values())  # closed_isolines
        assert any('traps' in v for v in sample2.values())  # traps
        assert any('faults' in v for v in sample2.values())  # faults

    def test_collect_samples_incomplete_sample(self, sample_files_for_collect):
        """Test that incomplete samples are still collected (filtering is done later)."""
        samples = collect_samples(sample_files_for_collect)

        # Incomplete sample should be present but missing traps and closed_isolines
        sample3 = samples['003_XUY1']
        assert 'rgb' in sample3 or any('structuralNOisoline' in v for v in sample3.values())
        assert 'depth_norm' in sample3 or any('structuralBlackWhite' in v for v in sample3.values())
        assert 'isolines' in sample3 or any('isolines' in v for v in sample3.values())
        assert 'traps' not in sample3
        assert 'closed_isolines' not in sample3

    def test_collect_samples_empty_list(self):
        """Test collect_samples with empty file list."""
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

        file_list = [str(f) for f in data_dir.iterdir()]
        samples = collect_samples(file_list)

        # Only valid file should be collected
        assert len(samples) == 1
        assert '001_H150' in samples

    def test_resolve_path_absolute_path_unchanged(self):
        """Test resolve_path with absolute path."""
        abs_path = "/absolute/path/to/file.png"
        result = resolve_path(abs_path, "/base/dir")

        assert result == abs_path

    def test_resolve_path_relative_path_joined(self):
        """Test resolve_path with relative path."""
        rel_path = "relative/file.png"
        base_dir = "/base/dir"
        result = resolve_path(rel_path, base_dir)

        assert result == os.path.join(base_dir, rel_path)

    def test_resolve_path_dot_slash_path_unchanged(self):
        """Test resolve_path with ./ prefix."""
        path = "./relative/file.png"
        result = resolve_path(path, "/base/dir")

        assert result == path

    def test_resolve_path_parent_dir_path_unchanged(self):
        """Test resolve_path with ../ prefix."""
        path = "../relative/file.png"
        result = resolve_path(path, "/base/dir")

        assert result == path

    def test_load_maps_into_ndarray_returns_correct_shapes(self, tmp_path):
        """Test load_maps_into_ndarray returns images with correct shapes."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        cv2 = pytest.importorskip("cv2")

        # Create test images for cps_tiles format
        rgb_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth_img = np.ones((50, 50), dtype=np.uint8) * 100
        isolines_img = np.zeros((50, 50), dtype=np.uint8)
        isolines_img[10:40, 10:40] = 255  # Some isolines
        closed_isolines_img = np.zeros((50, 50), dtype=np.uint8)
        closed_isolines_img[20:30, 20:30] = 255  # Closed isolines area
        traps_img = np.zeros((50, 50), dtype=np.uint8)
        traps_img[20:30, 20:30] = 255
        faults_img = np.zeros((50, 50), dtype=np.uint8)
        faults_img[30:, :] = 255

        rgb_path = str(data_dir / "rgb.png")
        depth_path = str(data_dir / "depth.png")
        isolines_path = str(data_dir / "isolines.png")
        closed_isolines_path = str(data_dir / "closed_isolines.png")
        traps_path = str(data_dir / "traps.png")
        faults_path = str(data_dir / "faults.png")

        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img)
        cv2.imwrite(isolines_path, isolines_img)
        cv2.imwrite(closed_isolines_path, closed_isolines_img)
        cv2.imwrite(traps_path, traps_img)
        cv2.imwrite(faults_path, faults_img)

        sample_paths = {
            'rgb': rgb_path,
            'depth_norm': depth_path,
            'isolines': isolines_path,
            'closed_isolines': closed_isolines_path,
            'traps': traps_path,
            'faults': faults_path
        }

        rgb, depth, isolines, closed_isolines, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=True, data_source='cps_tiles'
        )

        assert rgb.shape == (50, 50, 3)
        assert rgb.dtype == np.uint8
        assert depth.shape == (50, 50)
        assert depth.dtype == np.uint8
        assert isolines.shape == (50, 50)
        assert isolines.dtype == np.uint8
        assert closed_isolines.shape == (50, 50)
        assert closed_isolines.dtype == np.uint8
        assert traps.shape == (50, 50)
        assert traps.dtype == np.float32
        assert faults.shape == (50, 50)
        assert faults.dtype == np.float32

    def test_load_maps_into_ndarray_without_faults(self, tmp_path):
        """Test load_maps_into_ndarray without faults returns zero mask."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        cv2 = pytest.importorskip("cv2")

        rgb_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth_img = np.ones((50, 50), dtype=np.uint8) * 100
        isolines_img = np.zeros((50, 50), dtype=np.uint8)
        closed_isolines_img = np.zeros((50, 50), dtype=np.uint8)
        traps_img = np.zeros((50, 50), dtype=np.uint8)

        rgb_path = str(data_dir / "rgb.png")
        depth_path = str(data_dir / "depth.png")
        isolines_path = str(data_dir / "isolines.png")
        closed_isolines_path = str(data_dir / "closed_isolines.png")
        traps_path = str(data_dir / "traps.png")

        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img)
        cv2.imwrite(isolines_path, isolines_img)
        cv2.imwrite(closed_isolines_path, closed_isolines_img)
        cv2.imwrite(traps_path, traps_img)

        sample_paths = {
            'rgb': rgb_path,
            'depth_norm': depth_path,
            'isolines': isolines_path,
            'closed_isolines': closed_isolines_path,
            'traps': traps_path
        }

        rgb, depth, isolines, closed_isolines, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=False, data_source='cps_tiles'
        )

        assert rgb.shape == (50, 50, 3)
        assert depth.shape == (50, 50)
        assert isolines.shape == (50, 50)
        assert closed_isolines.shape == (50, 50)
        assert traps.shape == (50, 50)
        # Faults should be zero mask
        assert faults.shape == (50, 50)
        assert np.all(faults == 0.0)

    def test_load_maps_into_ndarray_cps_tiles_source(self, tmp_path):
        """Test load_maps_into_ndarray with cps_tiles data source."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        cv2 = pytest.importorskip("cv2")

        # For cps_tiles, traps/faults are white (> threshold)
        rgb_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        depth_img = np.ones((50, 50), dtype=np.uint8) * 100
        isolines_img = np.zeros((50, 50), dtype=np.uint8)
        closed_isolines_img = np.zeros((50, 50), dtype=np.uint8)
        closed_isolines_img[20:30, 20:30] = 255  # Closed isolines area
        traps_img = np.zeros((50, 50), dtype=np.uint8)
        traps_img[20:30, 20:30] = 255  # White = trap

        rgb_path = str(data_dir / "rgb.png")
        depth_path = str(data_dir / "depth.png")
        isolines_path = str(data_dir / "isolines.png")
        closed_isolines_path = str(data_dir / "closed_isolines.png")
        traps_path = str(data_dir / "traps.png")

        cv2.imwrite(rgb_path, rgb_img)
        cv2.imwrite(depth_path, depth_img)
        cv2.imwrite(isolines_path, isolines_img)
        cv2.imwrite(closed_isolines_path, closed_isolines_img)
        cv2.imwrite(traps_path, traps_img)

        sample_paths = {
            'rgb': rgb_path,
            'depth_norm': depth_path,
            'isolines': isolines_path,
            'closed_isolines': closed_isolines_path,
            'traps': traps_path
        }

        rgb, depth, isolines, closed_isolines, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=False, data_source='cps_tiles'
        )

        assert rgb.shape == (50, 50, 3)
        assert depth.shape == (50, 50)
        assert isolines.shape == (50, 50)
        assert closed_isolines.shape == (50, 50)
        # Closed isolines should have values where image was white
        assert closed_isolines[25, 25] == 255  # Inside closed area
        assert closed_isolines[0, 0] == 0  # Outside closed area
        # Traps should have 1s where image was white
        assert traps[25, 25] == 1.0  # Inside trap area
        assert traps[0, 0] == 0.0  # Outside trap area

    def test_load_maps_into_ndarray_png_source(self, tmp_path):
        """Test load_maps_into_ndarray with png data source (no isolines)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        cv2 = pytest.importorskip("cv2")

        # For png source, no isolines/closed_isolines files
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

        rgb, depth, isolines, closed_isolines, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=False, data_source='png'
        )

        assert rgb.shape == (50, 50, 3)
        assert depth.shape == (50, 50)
        # Isolines and closed_isolines should be zero masks for png source
        assert isolines.shape == (50, 50)
        assert np.all(isolines == 0)
        assert closed_isolines.shape == (50, 50)
        assert np.all(closed_isolines == 0)
        # Traps should have 1s where image was white
        assert traps[25, 25] == 1.0  # Inside trap area
        assert traps[0, 0] == 0.0  # Outside trap area


@pytest.mark.smoke
def test_dataset_utils_smoke_test(tmp_path):
    """Smoke test for dataset_utils module."""
    # Test parse_filename
    parsed = parse_filename("001_x_structuralNOisoline_H150.png")
    assert parsed is not None

    # Test get_sample_key
    key = get_sample_key(parsed)
    assert key == "001_H150"

    # Test collect_samples with cps_tiles format
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cv2 = pytest.importorskip("cv2")

    for filename in [
        "001_x_structuralNOisoline_H150.png",
        "001_x_structuralBlackWhite_H150.png",
        "001_x_isolines_H150.png",
        "001_x_closedIsolines_H150.png",
        "001_y_traps_H150.png"
    ]:
        cv2.imwrite(str(data_dir / filename), np.zeros((50, 50, 3), dtype=np.uint8))

    file_list = [str(data_dir / f) for f in os.listdir(data_dir)]
    samples = collect_samples(file_list)

    assert len(samples) == 1
    assert "001_H150" in samples
    # Check that all expected channels are present
    sample = samples["001_H150"]
    assert 'rgb' in sample
    assert 'depth_norm' in sample
    assert 'isolines' in sample
    assert 'closed_isolines' in sample
    assert 'traps' in sample


@pytest.mark.smoke
class TestClosedIsolinesChannel:
    """Tests specifically for the new closed_isolines channel."""

    def test_parse_closed_isolines_filename(self):
        """Test parsing of closedIsolines filename."""
        result = parse_filename("001_x_closedIsolines_H150.png")

        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'x'
        assert result['type'] == 'closedIsolines'
        assert result['name'] == 'H150'

    def test_collect_samples_includes_closed_isolines(self, tmp_path):
        """Test that collect_samples includes closed_isolines channel."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cv2 = pytest.importorskip("cv2")

        # Create files with closed_isolines
        for filename in [
            "001_x_structuralNOisoline_H150.png",
            "001_x_structuralBlackWhite_H150.png",
            "001_x_isolines_H150.png",
            "001_x_closedIsolines_H150.png",
            "001_y_traps_H150.png"
        ]:
            cv2.imwrite(str(data_dir / filename), np.zeros((50, 50, 3), dtype=np.uint8))

        file_list = [str(data_dir / f) for f in os.listdir(data_dir)]
        samples = collect_samples(file_list)

        assert len(samples) == 1
        sample = samples["001_H150"]
        assert 'closed_isolines' in sample
        assert 'closedIsolines' in sample['closed_isolines']
