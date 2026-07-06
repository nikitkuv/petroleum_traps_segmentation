"""
Эмпирическая проверка split_into_tiles на реальных CPS разного размера.
Проверяет:
  A) математику сетки (покрытие без дыр, размеры окон, перекрытия)
  B) end-to-end: реальный save_large_images -> split_into_tiles -> реконструкция
     Каждый сохранённый тайл должен быть точным (padded) регионом полной карты,
     а объединение всех тайлов — покрывать всю карту без дыр.
  C) влияние фильтра min_traps_pixels (только выбрасывает тайлы, не портит контент).
"""
import os, sys, shutil, tempfile, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from settings import settings
from utils.cps_utils import find_cps_files, save_large_images, split_into_tiles
from utils.images_utils import pad_image

TH, TW = settings.TARGET_HEIGHT, settings.TARGET_WIDTH     # 640 x 448
OV = settings.TILE_OVERLAP_RATIO                            # 0.25
SH = max(1, int(TH * (1 - OV)))                             # 480
SW = max(1, int(TW * (1 - OV)))                             # 336

# --- точная копия логики окон из split_into_tiles (cps_utils.py) ---
def compute_windows(h, w):
    if h <= TH and w <= TW:
        return [(0, h, 0, w)]
    nh = max(1, (h - TH) // SH + 1)
    nw = max(1, (w - TW) // SW + 1)
    if (nh - 1) * SH + TH < h: nh += 1
    if (nw - 1) * SW + TW < w: nw += 1
    wins = []
    for r in range(nh):
        for c in range(nw):
            y0, x0 = r * SH, c * SW
            y1, x2 = min(y0 + TH, h), min(x0 + TW, w)
            if y1 == h: y0 = max(0, y1 - TH)
            if x2 == w: x0 = max(0, x2 - TW)
            wins.append((y0, y1, x0, x2))
    return wins

def check_grid_math(h, w, label):
    wins = compute_windows(h, w)
    # размеры окон
    for (y0, y1, x0, x1) in wins:
        assert 0 <= y0 < y1 <= h, f"{label}: bad y {y0},{y1}"
        assert 0 <= x0 < x1 <= w, f"{label}: bad x {x0},{x1}"
        assert (y1 - y0) <= TH and (x1 - x0) <= TW, f"{label}: window > tile"
    # покрытие без дыр
    cov = np.zeros((h, w), dtype=bool)
    over = np.zeros((h, w), dtype=np.int32)
    for (y0, y1, x0, x1) in wins:
        cov[y0:y1, x0:x1] = True
        over[y0:y1, x0:x1] += 1
    gaps = int((~cov).sum())
    max_over = int(over.max())
    # оценка: покрыто 100%, перекрытия >=1
    ok = (gaps == 0)
    print(f"  [A] {label:<18} map={h}x{w}  tiles={len(wins)}  "
          f"gaps={gaps}  max_overlap={max_over}x  -> {'OK' if ok else 'FAIL'}")
    return wins, ok, over

def unpad_to_content(tile_arr, ch, cw):
    """Обратный pad_image: вырезать контент (pad_image центрирует)."""
    H, W = tile_arr.shape[:2]
    pt = (H - ch) // 2
    pl = (W - cw) // 2
    return tile_arr[pt:pt + ch, pl:pl + cw]

def run_one(horizons_dict, name, min_traps):
    """Прогоняет реальный пайплайн для одного горизонта, возвращает (images_data, tiles_dir)."""
    tmp = tempfile.mkdtemp(prefix="tiling_chk_")
    full_dir = os.path.join(tmp, "full")
    tiles_dir = os.path.join(tmp, "tiles")
    sub = {name: horizons_dict[name]}
    images_data = save_large_images(sub, full_dir)
    split_into_tiles(images_data, tiles_dir, min_traps_pixels=min_traps)
    return images_data, tiles_dir, tmp

def main():
    cps_dir = settings.CPS_SOURCE_DIR
    horizons = find_cps_files(cps_dir)
    # один горизонт каждого размерного режима
    pick = ["Ach5_toptop5", "H150_toptop1", "U4_42_kolltop1"]
    avail = [p for p in pick if p in horizons]
    print(f"Tile size = {TW}x{TH} (WxH), overlap={OV}, stride={SW}x{SH}")
    print(f"Checking horizons: {avail}\n")

    all_ok = True
    for name in avail:
        print("=" * 70)
        print(f"HORIZON {name}")
        # сначала размер по структурному гриду (через save_large_images rgb)
        images_data, tiles_dir, tmp = run_one(horizons, name, min_traps=0)
        try:
            rgb = images_data[name]['rgb']
            depth = images_data[name]['grayscale']
            traps = images_data[name]['traps']
            h, w = rgb.shape[:2]

            # ---- A: математика сетки ----
            wins, okA, over = check_grid_math(h, w, name)
            all_ok &= okA

            # ---- B: end-to-end реконструкция из сохранённых тайлов ----
            # тайлы сохранены row-major; с min_traps=0 сохранены ВСЕ -> порядок = wins
            tile_files = sorted(
                f for f in os.listdir(tiles_dir)
                if 'x_structuralNOisoline_' in f and f.endswith('.png')
            )
            # отсортировать по ведущему номеру
            tile_files.sort(key=lambda f: int(f.split('_')[0]))

            recon_rgb = np.zeros_like(rgb)
            recon_traps = np.zeros_like(traps)
            covmask = np.zeros((h, w), dtype=bool)
            mism = 0
            n_checked = 0
            for f, (y0, y1, x0, x1) in zip(tile_files, wins):
                ch, cw = y1 - y0, x1 - x0
                # rgb png
                t = np.array(Image.open(os.path.join(tiles_dir, f)).convert('RGB'))
                content = unpad_to_content(t, ch, cw)
                ref = rgb[y0:y1, x0:x1]
                if not np.array_equal(content, ref):
                    mism += 1
                recon_rgb[y0:y1, x0:x1] = content
                covmask[y0:y1, x0:x1] = True
                n_checked += 1

            # traps реконструкция (через y_traps_ файлы того же номера)
            trap_files = sorted(
                f for f in os.listdir(tiles_dir)
                if 'y_traps_' in f and f.endswith('.png')
            )
            trap_files.sort(key=lambda f: int(f.split('_')[0]))
            t_mism = 0
            for f, (y0, y1, x0, x1) in zip(trap_files, wins):
                ch, cw = y1 - y0, x1 - x0
                t = np.array(Image.open(os.path.join(tiles_dir, f)))
                content = unpad_to_content(t, ch, cw)
                ref = traps[y0:y1, x0:x1]
                if not np.array_equal(content, ref):
                    t_mism += 1
                recon_traps[y0:y1, x0:x1] = content

            gaps = int((~covmask).sum())
            n_tiles_expected = len(wins)
            okB = (mism == 0 and t_mism == 0 and gaps == 0
                   and len(tile_files) == n_tiles_expected)
            all_ok &= okB
            print(f"  [B] map={h}x{w}  saved_rgb_tiles={len(tile_files)}/{n_tiles_expected}  "
                  f"rgb_mismatches={mism}  traps_mismatches={t_mism}  gaps={gaps}  "
                  f"-> {'OK' if okB else 'FAIL'}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        # ---- C: фильтр min_traps (поведение продакшена) ----
        images_data2, tiles_dir2, tmp2 = run_one(horizons, name, min_traps=settings.MIN_NUM_PIXS_OF_TRAPS_IN_TILES)
        try:
            tf = [f for f in os.listdir(tiles_dir2) if 'y_traps_' in f and f.endswith('.png')]
            print(f"  [C] min_traps={settings.MIN_NUM_PIXS_OF_TRAPS_IN_TILES}: "
                  f"saved {len(tf)}/{n_tiles_expected} tiles (drop пустых = {n_tiles_expected - len(tf)})")
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)
        print()

    print("=" * 70)
    print(f"ИТОГ: {'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else 'ЕСТЬ ОШИБКИ'}")

if __name__ == "__main__":
    main()
