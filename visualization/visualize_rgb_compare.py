"""
Сравнение RGB-карты Top_D_50:
  (A) Чистая загрузка из CPS (cps_to_rgb, БЕЗ вырезания разломов) — «как загрузишь ты»
  (B) Как делает код визуализации (cps_to_rgb + вырезание НАИВНО смещённых разломов) — то, что в Top_D_50.png
  (C) Подсветка: RED = код вырезал в чёрный, а там есть глубина (артефакт), синим = граница карты
  (D) Как должно быть после фикса (вырезание ПРАВИЛЬНО выровненных разломов)
Сохраняет в data/test_cps/Top_D_50_rgb_compare.png
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.cps_utils import read_cps_grid, resample_grid_to_reference, cps_to_rgb, cps_to_binary_mask

HORIZON = 'Top_D_50'
STRUCT = f'./data/raw/test_cps/x_structuralNOisoline_{HORIZON}'
FAULTS = f'./data/raw/test_cps/x_faults_{HORIZON}'
OUT = f'./data/test_cps/{HORIZON}_rgb_compare.png'


def main():
    sg, sm = read_cps_grid(STRUCT)
    fg, fm = read_cps_grid(FAULTS)
    ny, nx = sg.shape
    struct_valid = ~np.isnan(sg)

    # (A) Чистая RGB из structural (без вырезания разломов)
    rgb_clean = cps_to_rgb(sg).astype(np.float32) / 255.0

    # (B) Наивная маска разломов (как в visualize_full_cps_analysis): naive resize
    fmask_naive = cps_to_binary_mask(fg)
    if fmask_naive.shape != (ny, nx):
        fmask_naive = cv2.resize(fmask_naive, (nx, ny), interpolation=cv2.INTER_NEAREST)
    fp_naive = fmask_naive > 0.5
    rgb_viz = rgb_clean.copy()
    rgb_viz[fp_naive] = 0.0

    # (D) Правильно выровненная маска разломов (после фикса)
    fg_aligned = resample_grid_to_reference(fg, fm, sm)
    fmask_correct = cps_to_binary_mask(fg_aligned)
    fp_correct = fmask_correct > 0.5
    rgb_correct = rgb_clean.copy()
    rgb_correct[fp_correct] = 0.0

    # (C) Артефакт: код вырезал в чёрный (viz==0), а там есть глубина (clean цветная)
    clean_colored = (rgb_clean.sum(axis=2) > 0.05) & struct_valid
    viz_black = rgb_viz.sum(axis=2) < 0.01
    artifact = viz_black & clean_colored   # скрытая глубина
    fault_cut_total = fp_naive & struct_valid  # всего вырезано по валидной глубине
    artifact_offmap = (fp_naive & ~struct_valid)  # вырезано вне карты

    # граница карты
    bnd = struct_valid.astype(np.uint8)
    contours, _ = cv2.findContours(bnd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # статистика
    print('=' * 70)
    print(f'RGB-сравнение для {HORIZON} (поле {nx}x{ny})')
    print('=' * 70)
    print(f'Пикселей с глубиной (struct valid): {int(struct_valid.sum())}')
    print(f'Разломов (naive, всего в кадре):    {int(fp_naive.sum())}')
    print(f'  -> вырезано ВНЕ карты (off-map):   {int(artifact_offmap.sum())}  ({100*artifact_offmap.sum()/max(1,fp_naive.sum()):.1f}% разломов)')
    print(f'  -> вырезано ПО глубине (артефакт): {int(fault_cut_total.sum())}  ({100*fault_cut_total.sum()/max(1,fp_naive.sum()):.1f}% разломов)')
    print(f'Правильно выровненных разломов:     {int(fp_correct.sum())} (IoU naive-vs-correct = '
          f'{100*(fp_naive&fp_correct).sum()/max(1,(fp_naive|fp_correct).sum()):.1f}%)')

    # фигура
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes[0, 0].imshow(rgb_clean)
    axes[0, 0].set_title('(A) RGB ЧИСТАЯ загрузка\n(cps_to_rgb, БЕЗ вырезания разломов)')
    axes[0, 1].imshow(rgb_viz)
    axes[0, 1].set_title('(B) RGB как в коде визуализации\n(cps_to_rgb + naive-вырезание СМЕЩЁННЫХ разломов)\n= то, что в Top_D_50.png')

    diff = np.zeros((ny, nx, 3), dtype=np.float32)
    diff[..., 0] = artifact.astype(np.float32)         # RED = скрытая глубина
    diff[..., 1] = (fp_correct & struct_valid).astype(np.float32)  # GREEN = правильные разломы
    cv2.drawContours(diff, contours, -1, (0, 0, 1), 2)
    axes[1, 0].imshow(diff)
    axes[1, 0].set_title(f'(C) Артефакт: RED = код зачернил валидную глубину\n'
                         f'GREEN = правильные разломы, BLUE = граница карты\n'
                         f'скрыто глубины: {int(artifact.sum())} px')

    axes[1, 1].imshow(rgb_correct)
    axes[1, 1].set_title('(D) RGB ПОСЛЕ ФИКСА\n(вырезание ПРАВИЛЬНО выровненных разломов)')

    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle(f'{HORIZON}: сравнение загрузки RGB (чистая vs код vs фикс)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'\nСохранено: {OUT}')


if __name__ == '__main__':
    main()
