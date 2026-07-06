"""Тесты для utils/cps_utils.py.

Фокус — resample_grid_to_reference: приведение faults/traps к геометрии
структурной карты по мировым координатам (node-centered).
"""
import numpy as np

from utils.cps_utils import resample_grid_to_reference


def _meta(nx, ny, xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0):
    """Синтетические метаданные CPS-грида (node-centered)."""
    return {
        'nx': nx, 'ny': ny,
        'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
        'null_value': -999.0, 'file_path': '',
    }


def test_resample_identity_returns_same_grid():
    """Перенос грида на собственную геометрию = тождество (включая NaN)."""
    rng = np.random.default_rng(0)
    grid = rng.random((40, 50)).astype(np.float32)
    grid[0, 0] = np.nan
    m = _meta(50, 40)

    out = resample_grid_to_reference(grid, m, m)

    assert out.shape == grid.shape
    np.testing.assert_array_equal(np.isnan(out), np.isnan(grid))
    np.testing.assert_allclose(np.nan_to_num(out), np.nan_to_num(grid), atol=1e-6)


def test_resample_output_shape_matches_reference():
    """Результат всегда имеет форму референса (ny, nx)."""
    src = np.ones((81, 81), dtype=np.float32)
    src_meta = _meta(81, 81)
    ref_meta = _meta(41, 41)

    out = resample_grid_to_reference(src, src_meta, ref_meta)

    assert out.shape == (41, 41)


def test_resample_node_alignment_2x_same_bbox():
    """При 2x разрешении с тем же bbox ref-узел (i,j) берёт src-узел (2i, 2j)."""
    nx_s, ny_s = 41, 31
    src = np.zeros((2 * ny_s - 1, 2 * nx_s - 1), dtype=np.float32)
    src[2 * 5, 2 * 7] = 1.0  # метка в src-узле (row=10, col=14)
    src_meta = _meta(2 * nx_s - 1, 2 * ny_s - 1)
    ref_meta = _meta(nx_s, ny_s)

    out = resample_grid_to_reference(src, src_meta, ref_meta)

    assert out.shape == (ny_s, nx_s)
    # метка переезжает ровно в ref-узел (5, 7)
    assert out[5, 7] == 1.0
    assert int(np.sum(out == 1.0)) == 1


def test_resample_outside_src_coverage_is_nan():
    """Точки референса вне охвата src -> NaN (корректные границы карты)."""
    src = np.ones((20, 20), dtype=np.float32)
    src_meta = _meta(20, 20, xmin=20.0, xmax=80.0, ymin=20.0, ymax=80.0)
    ref_meta = _meta(40, 40, xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)

    out = resample_grid_to_reference(src, src_meta, ref_meta)

    assert out.shape == (40, 40)
    assert np.isnan(out[0, 0])              # угол — за охватом src
    assert not np.isnan(out[20, 20])        # центр — внутри охвата src
