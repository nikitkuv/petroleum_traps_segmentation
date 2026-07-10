"""Тесты инференс-пайплайна: запись/чтение CPS, тайлинг/склейка, постобработка.

Без модели и реальных данных — всё на синтетических массивах.
"""
import numpy as np

from utils.cps_utils import write_cps_grid, read_cps_grid
from utils.images_utils import pad_image
from inference.tiling import (
    compute_inference_windows,
    unpad_like_pad_image,
    tile_for_inference,
    stitch_predictions,
)
from inference.postprocess import (
    postprocess_prediction,
    build_output_grid,
    build_probability_grid,
)


TH, TW = 640, 448


# --------------------------------- write/read CPS ---------------------------------

def _meta(ny, nx, null=1e30):
    return {
        'nx': nx, 'ny': ny,
        'xmin': 100.0, 'xmax': 200.0, 'ymin': 300.0, 'ymax': 400.0,
        'null_value': null, 'file_path': '',
    }


def test_write_read_cps_binary_roundtrip(tmp_path):
    """Бинарная маска {0,1,NaN} переживает запись->чтение без потерь и коллизий с null."""
    ny, nx = 13, 19
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 2, size=(ny, nx)).astype(np.float32)
    grid[0, 0] = np.nan
    grid[5, 8] = np.nan
    meta = _meta(ny, nx)

    path = str(tmp_path / "pred.cps")
    write_cps_grid(grid, meta, path)
    re_grid, re_meta = read_cps_grid(path)

    np.testing.assert_array_equal(np.isnan(re_grid), np.isnan(grid))
    np.testing.assert_array_equal(np.nan_to_num(re_grid), np.nan_to_num(grid))
    assert re_meta['nx'] == nx and re_meta['ny'] == ny
    assert re_meta['xmin'] == meta['xmin'] and re_meta['xmax'] == meta['xmax']
    assert re_meta['ymin'] == meta['ymin'] and re_meta['ymax'] == meta['ymax']
    assert re_meta['null_value'] == meta['null_value']


def test_write_read_cps_probability_roundtrip(tmp_path):
    """Непрерывные вероятности [0,1] + NaN переживают запись->чтение (с допустимой погрешностью)."""
    ny, nx = 7, 11
    rng = np.random.default_rng(3)
    grid = rng.random((ny, nx)).astype(np.float32)
    grid[1, 1] = np.nan
    meta = _meta(ny, nx)

    path = str(tmp_path / "prob.cps")
    write_cps_grid(grid, meta, path)
    re_grid, _ = read_cps_grid(path)

    np.testing.assert_array_equal(np.isnan(re_grid), np.isnan(grid))
    valid = ~np.isnan(grid)
    np.testing.assert_allclose(re_grid[valid], grid[valid], atol=1e-4)


# --------------------------------- окна тайлов ---------------------------------

def test_windows_cover_full_extent_no_gaps():
    """Объединение окон = вся карта, каждое окно <= размера тайла, для разных размеров."""
    for (h, w) in [(TH, TW), (TH + 1, TW + 1), (967, 2009), (2 * TH, 2 * TW), (1000, 500)]:
        wins = compute_inference_windows(h, w, TH, TW)
        cov = np.zeros((h, w), dtype=bool)
        for (y0, y1, x0, x1) in wins:
            assert 0 <= y0 < y1 <= h
            assert 0 <= x0 < x1 <= w
            assert (y1 - y0) <= TH and (x1 - x0) <= TW
            cov[y0:y1, x0:x1] = True
        assert cov.all(), f"есть дыры покрытия для {h}x{w}"


def test_windows_small_image_single():
    """Карта меньше тайла — одно окно во всю карту."""
    wins = compute_inference_windows(100, 80, TH, TW)
    assert wins == [(0, 100, 0, 80)]


# --------------------------------- unpad обратен pad ---------------------------------

def test_unpad_is_inverse_of_pad():
    """unpad_like_pad_image точно обращает pad_image для чётных/нечётных размеров контента."""
    rng = np.random.default_rng(2)
    for (ch, cw) in [(1, 1), (2, 2), (3, 3), (321, 224), (639, 447),
                     (TH, TW), (TH, 1), (1, TW)]:
        content = rng.random((ch, cw)).astype(np.float32)
        padded = pad_image(content, TH, TW)
        assert padded.shape == (TH, TW)
        recovered = unpad_like_pad_image(padded, ch, cw, TH, TW)
        np.testing.assert_array_equal(recovered, content)


# --------------------------------- склейка = обращение тайлинга ---------------------------------

def test_stitch_is_inverse_of_tiling():
    """Если каждый тайл несёт свой фрагмент полной карты, склейка восстанавливает её."""
    h, w = 1000, 700
    rng = np.random.default_rng(1)
    full_prob = rng.random((h, w)).astype(np.float32)

    wins = compute_inference_windows(h, w, TH, TW)
    placements, prob_by_key = {}, {}
    for i, (y0, y1, x0, x1) in enumerate(wins):
        key = f"{i:03d}_H"
        placements[key] = {'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1,
                           'ch': y1 - y0, 'cw': x1 - x0}
        prob_by_key[key] = pad_image(full_prob[y0:y1, x0:x1], TH, TW)

    stitched = stitch_predictions(prob_by_key, placements, h, w, TH, TW)
    np.testing.assert_allclose(stitched, full_prob, atol=1e-6)


# --------------------------------- пропуск пустых окон + покрытие ---------------------------------

def test_tile_skip_empty_keeps_map_coverage(tmp_path):
    """Окна целиком вне карты пропускаются, но карта покрывается без дыр."""
    h, w = 1000, 700
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[200:600, 150:550] = 100  # блок карты (rgb > 10)
    images_data = {'TEST': {
        'rgb': rgb,
        'grayscale': np.full((h, w), 0.5, np.float32),
        'isolines': np.zeros((h, w), np.uint8),
    }}

    placements, basenames = tile_for_inference(
        images_data, 'TEST', str(tmp_path / 'tiles'), tile_h=TH, tile_w=TW, use_faults=False,
    )

    assert len(placements) > 0
    # y_traps всегда есть (датасет требует) — хотя бы один файл ловушек сохранён
    assert any('y_traps' in b for b in basenames)

    on_map = rgb.sum(axis=2) > 10
    covered = np.zeros((h, w), dtype=bool)
    for p in placements.values():
        covered[p['y0']:p['y1'], p['x0']:p['x1']] = True
    assert covered[on_map].all(), "часть карты не покрыта тайлами"


# --------------------------------- постобработка ---------------------------------

def test_postprocess_removes_small_objects():
    prob = np.zeros((50, 50), np.float32)
    valid = np.ones((50, 50), dtype=bool)
    prob[5:25, 5:25] = 0.9    # большая ловушка
    prob[40:43, 40:43] = 0.9  # мелкий блоб 3x3=9

    mask = postprocess_prediction(prob, valid, threshold=0.5,
                                  min_trap_area_px=10, fill_holes=False)
    assert mask[10, 10] == 1.0   # большая осталась
    assert mask[41, 41] == 0.0   # мелкая убрана


def test_postprocess_fill_holes():
    prob = np.zeros((20, 20), np.float32)
    valid = np.ones((20, 20), dtype=bool)
    prob[5:15, 5:15] = 0.9
    prob[10, 10] = 0.0  # дыра внутри ловушки

    mask = postprocess_prediction(prob, valid, threshold=0.5,
                                  min_trap_area_px=0, fill_holes=True)
    assert mask[10, 10] == 1.0   # дыра заполнена


def test_postprocess_off_map_is_zero():
    prob = np.full((10, 10), 0.9, np.float32)
    valid = np.zeros((10, 10), dtype=bool)
    mask = postprocess_prediction(prob, valid, threshold=0.5)
    assert mask.sum() == 0


def test_build_output_grid_nan_off_map():
    binary = np.ones((5, 5), np.float32)
    valid = np.zeros((5, 5), dtype=bool)
    valid[2, 2] = True
    grid = build_output_grid(binary, valid)
    assert np.isnan(grid[0, 0])
    assert grid[2, 2] == 1.0


def test_build_probability_grid_nan_off_map():
    prob = np.full((4, 4), 0.7, np.float32)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    grid = build_probability_grid(prob, valid)
    assert np.isnan(grid[0, 0])
    assert grid[1, 1] == 0.7
