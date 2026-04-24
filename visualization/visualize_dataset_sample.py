import sys
from pathlib import Path
import torch
import random
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from settings import settings
from data.dataset import GeologyTrapsDataset
from utils.images_utils import load_grayscale_image, load_image


NUM_SAMPLES = 3
SEED = settings.SEED


def visualize_sample(
    dataset: GeologyTrapsDataset,
    idx: int
):
    sample_data = dataset[idx]
    sample_paths = dataset.samples[idx]

    # Загружаем оригиналы
    orig_rgb = load_image(sample_paths['rgb'])
    orig_depth = load_grayscale_image(sample_paths['depth_norm'])
    orig_traps = load_grayscale_image(sample_paths['traps'])
    orig_isolines = load_grayscale_image(sample_paths['isolines'])
    orig_closed_isolines = load_grayscale_image(sample_paths['closed_isolines'])

    # Извлекаем тензоры из dataset
    x_rgb = sample_data['x'][:3].permute(1, 2, 0).numpy()      # (H, W, 3)
    x_depth = sample_data['x'][3].numpy()                       # (H, W)
    x_isolines = sample_data['x'][4].numpy()                    # (H, W)
    x_closed_isolines = sample_data['x'][5].numpy()             # (H, W)

    y_traps = sample_data['y'][0].numpy()                       # (H, W)
    mask_map = sample_data['mask_map'][0].numpy()               # (H, W)

    # Формируем ID семпла
    sample_filename = Path(sample_paths['rgb']).name
    parts = sample_filename.split('_')
    sample_number = parts[0]
    sample_name = Path(sample_paths['rgb']).stem.split('_')[-1]
    sample_id = f"{sample_number}_{sample_name}"

    # Структура: 3 ряда x 4 колонки
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           height_ratios=[1, 1, 1],
                           width_ratios=[1, 1, 1, 1],
                           hspace=0.2, wspace=0.05,
                           top=0.93, bottom=0.02, left=0.02, right=0.98)

    # === РЯД 1: RGB и Depth ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_rgb)
    ax1.set_title(f'Original RGB', fontsize=12, pad=4)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(x_rgb)
    ax2.set_title(f'Processed RGB', fontsize=12, pad=4)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(orig_depth, cmap='gray')
    ax3.set_title(f'Original Depth', fontsize=12, pad=4)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(x_depth, cmap='gray')
    ax4.set_title('Processed Depth', fontsize=12, pad=4)
    ax4.axis('off')

    # === РЯД 2: Isolines и Closed Isolines ===
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(orig_isolines, cmap='gray')
    ax5.set_title('Original Isolines', fontsize=12, pad=4)
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(x_isolines, cmap='gray', vmin=0.0, vmax=1.0)
    ax6.set_title('Processed Isolines', fontsize=12, pad=4)
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(orig_closed_isolines, cmap='gray')
    ax7.set_title('Original Closed Iso', fontsize=12, pad=4)
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(x_closed_isolines, cmap='gray', vmin=0.0, vmax=1.0)
    ax8.set_title('Processed Closed Iso', fontsize=12, pad=4)
    ax8.axis('off')

    # === РЯД 3: Map Mask, Traps, Stats ===
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(mask_map, cmap='gray', vmin=0.0, vmax=1.0)
    ax9.set_title('Map Mask\n(1=Valid, 0=Pad)', fontsize=12, pad=4)
    ax9.axis('off')

    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(orig_traps, cmap='gray')
    ax10.set_title('Original Traps', fontsize=12, pad=4)
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[2, 2])
    ax11.imshow(y_traps, cmap='gray', vmin=0.0, vmax=1.0)
    ax11.set_title('Target Y (Traps)', fontsize=12, pad=4)
    ax11.axis('off')

    # Блок со статистикой
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')

    orig_h, orig_w = orig_rgb.shape[:2]
    proc_h, proc_w = x_rgb.shape[:2]
    
    map_valid_px = mask_map.sum()
    map_valid_pct = mask_map.mean() * 100
    
    traps_gt_px = y_traps.sum()
    traps_gt_pct = (y_traps.sum() / (map_valid_px + 1e-8)) * 100
    
    closed_iso_px = x_closed_isolines.sum()
    closed_iso_pct = (x_closed_isolines.sum() / (map_valid_px + 1e-8)) * 100

    info_text = (
        f"Sample ID: {sample_id}\n"
        f"Augmentations: {'ON' if dataset.augment else 'OFF'}\n\n"
        f"--- Dimensions ---\n"
        f"Original: {orig_w}x{orig_h}\n"
        f"Processed: {proc_w}x{proc_h}\n"
        f"Input Tensor: {tuple(sample_data['x'].shape)}\n\n"
        f"--- Map Coverage ---\n"
        f"Valid Map Area: {map_valid_px:.0f} px ({map_valid_pct:.1f}%)\n\n"
        f"--- Target & Features ---\n"
        f"Traps (Y): {traps_gt_px:.0f} px ({traps_gt_pct:.1f}% of map)\n"
        f"Closed Iso: {closed_iso_px:.0f} px ({closed_iso_pct:.1f}% of map)"
    )
    ax12.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=11,
              family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.8))

    plt.suptitle(f'Dataset Sample {idx}: {sample_id}', fontsize=18, fontweight='bold', y=0.98)
    plt.show()


def get_all_files() -> List[str]:
    data_dir = settings.CPS_TILES_DIR
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_files = [f.name for f in data_path.glob('*.png')]
    relevant_files = []
    for f in all_files:
        stem = Path(f).stem
        parts = stem.split('_')
        if len(parts) >= 3:
            file_type = parts[2] if len(parts) > 2 else ''
            if file_type in ['structuralNOisoline', 'structuralBlackWhite', 'isolines', 'closedIsolines', 'faults', 'traps']:
                relevant_files.append(f)

    print(f"Found {len(relevant_files)} relevant files in {data_dir}")
    return relevant_files


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("GEOLOGY TRAPS DATASET VISUALIZATION")
    print("=" * 70)
    print(f"NUM_SAMPLES:  {NUM_SAMPLES}")
    print("=" * 70)

    file_list = get_all_files()

    if len(file_list) == 0:
        print("\nNo files found! Check your directory paths.")
        return

    dataset = GeologyTrapsDataset(
        file_list,
        augment=False,
        use_faults=False,
    )

    if len(dataset) == 0:
        print("\nNo samples loaded! Check if all required files exist for samples.")
        return

    print(f"\nTotal samples in dataset: {len(dataset)}")

    num_samples = min(NUM_SAMPLES, len(dataset))
    sample_indices = random.sample(range(len(dataset)), num_samples)
    print(f"Visualizing samples: {sample_indices}")

    for idx in sample_indices:
        visualize_sample(dataset, idx)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
    