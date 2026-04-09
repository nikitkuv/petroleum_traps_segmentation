import sys
from pathlib import Path
import torch
import argparse
import random
import os
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from settings import settings
from data.dataset import GeologyTrapsDataset
from utils.images_utils import load_grayscale_image, load_image


def visualize_sample(
    dataset: GeologyTrapsDataset,
    idx: int,
    save_path: str
):
    """
    Visualize all maps for a single sample and save to PNG.

    Args:
        dataset: The dataset object
        idx: Index of the sample to visualize
        save_path: Path to save the visualization
    """
    # Получаем данные из dataset
    sample_data = dataset[idx]

    # Получаем пути к оригинальным файлам для загрузки
    sample_paths = dataset.samples[idx]

    # Загружаем оригиналы (до обработки)
    orig_rgb = load_image(sample_paths['rgb'])
    orig_depth = load_grayscale_image(sample_paths['depth_norm'])
    orig_traps = load_grayscale_image(sample_paths['traps'])
    orig_faults = load_grayscale_image(sample_paths['faults']) if 'faults' in sample_paths else None

    # Извлекаем тензоры из dataset и конвертируем в numpy
    x_rgb = sample_data['x'][:3].permute(1, 2, 0).numpy()      # (H, W, 3)
    x_depth = sample_data['x'][3].numpy()                       # (H, W)

    # Faults канал только если use_faults=True
    if sample_data['use_faults']:
        x_faults = sample_data['x'][4].numpy()                  # (H, W)
    else:
        x_faults = np.zeros_like(x_depth)

    y_traps = sample_data['y'][0].numpy()                       # (H, W)
    mask_map = sample_data['mask_map'][0].numpy()               # (H, W)

    # mask_depth только если use_faults=True
    if sample_data['mask_depth'] is not None:
        mask_depth = sample_data['mask_depth'][0].numpy()
    else:
        mask_depth = np.zeros_like(x_depth)

    # Определяем количество рядов и колонок
    n_rows = 4
    n_cols = 3
    fig = plt.figure(figsize=(20, 14))

    # Extract sample info for title
    sample_key = list(dataset.samples.keys())[idx] if hasattr(dataset.samples, 'keys') else f"sample_{idx}"
    rgb_filename = Path(sample_paths['rgb']).name
    depth_filename = Path(sample_paths['depth_norm']).name
    traps_filename = Path(sample_paths['traps']).name
    faults_filename = Path(sample_paths['faults']).name if 'faults' in sample_paths else "N/A"

    # 1. Original RGB
    ax1 = fig.add_subplot(n_rows, n_cols, 1)
    ax1.imshow(orig_rgb)
    ax1.set_title(f'Original RGB\n{rgb_filename}', fontsize=9, pad=8)
    ax1.axis('off')

    # 2. Processed RGB (из dataset)
    ax2 = fig.add_subplot(n_rows, n_cols, 2)
    ax2.imshow(x_rgb)
    ax2.set_title(f'Processed RGB\n{x_rgb.shape[0]}×{x_rgb.shape[1]}', fontsize=9, pad=8)
    ax2.axis('off')

    # 3. Original Depth
    ax3 = fig.add_subplot(n_rows, n_cols, 3)
    ax3.imshow(orig_depth, cmap='gray')
    ax3.set_title(f'Original Depth\n{depth_filename}', fontsize=9, pad=8)
    ax3.axis('off')

    # 4. Processed Depth (из dataset)
    ax4 = fig.add_subplot(n_rows, n_cols, 4)
    ax4.imshow(x_depth, cmap='gray')
    ax4.set_title('Processed Depth\nBlack=High, White=Low', fontsize=9, pad=8)
    ax4.axis('off')

    # 5. Original Faults (if available)
    ax5 = fig.add_subplot(n_rows, n_cols, 5)
    if orig_faults is not None:
        ax5.imshow(orig_faults, cmap='gray')
        ax5.set_title(f'Original Fault Mask\n{faults_filename}', fontsize=9, pad=8)
    else:
        ax5.imshow(np.zeros_like(x_depth), cmap='gray')
        ax5.set_title('Original Fault Mask\nN/A', fontsize=9, pad=8)
    ax5.axis('off')

    # 6. Processed Faults (из dataset)
    ax6 = fig.add_subplot(n_rows, n_cols, 6)
    ax6.imshow(x_faults, cmap='gray')
    ax6.set_title('Processed Fault Mask\n1=Faults (White)', fontsize=9, pad=8)
    ax6.axis('off')

    # 7. Original Traps
    ax7 = fig.add_subplot(n_rows, n_cols, 7)
    ax7.imshow(orig_traps, cmap='gray')
    ax7.set_title(f'Original Trap Mask\n{traps_filename}', fontsize=9, pad=8)
    ax7.axis('off')

    # 8. Processed Traps / Target Y (из dataset)
    ax8 = fig.add_subplot(n_rows, n_cols, 8)
    ax8.imshow(y_traps, cmap='gray')
    ax8.set_title('Target Y (Traps)\n1=Traps (White)', fontsize=9, pad=8)
    ax8.axis('off')

    # 9. Map Mask
    ax9 = fig.add_subplot(n_rows, n_cols, 9)
    ax9.imshow(mask_map, cmap='gray')
    ax9.set_title('Map Mask\n1=Map Area (White)', fontsize=9, pad=8)
    ax9.axis('off')

    # 10. Depth Mask
    ax10 = fig.add_subplot(n_rows, n_cols, 10)
    ax10.imshow(mask_depth, cmap='gray')
    ax10.set_title('Depth Mask\n1=Valid (White), 0=Faults/Pad (Black)', fontsize=9, pad=8)
    ax10.axis('off')

    # 11. Input Channels Overview (RGB+Depth+Faults)
    ax11 = fig.add_subplot(n_rows, n_cols, 11)
    # Показываем композит: RGB + контуры разломов
    composite = x_rgb.copy()
    fault_overlay = (x_faults > 0.5).astype(np.float32)
    composite[:, :, 0] = np.maximum(composite[:, :, 0], fault_overlay * 0.7)  # Красный канал для разломов
    ax11.imshow(composite)

    n_channels = 5 if sample_data['use_faults'] else 4
    ax11.set_title(f'Input Overview ({n_channels} channels)\nRGB + Depth' + (' + Faults' if sample_data['use_faults'] else ''), fontsize=9, pad=8)
    ax11.axis('off')

    # 12. Statistics Info Box
    ax12 = fig.add_subplot(n_rows, n_cols, 12)
    ax12.axis('off')

    info_text = (
        f"Sample: {sample_key}\n\n"
        f"Augmentations: {'ON' if dataset.augment else 'OFF'}\n"
        f"Data Source: {dataset.data_source.upper()}\n\n"
        f"Input Shape: {sample_data['x'].shape}\n"
        f"Target Shape: {sample_data['y'].shape}\n\n"
        f"Map Valid: {mask_map.sum():.0f} px ({mask_map.mean()*100:.1f}%)\n"
        f"Depth Valid: {mask_depth.sum():.0f} px ({mask_depth.mean()*100:.1f}%)\n"
        f"Faults: {x_faults.sum():.0f} px ({x_faults.mean()*100:.2f}%)\n"
        f"Traps (Y): {y_traps.sum():.0f} px ({y_traps.mean()*100:.2f}%)"
    )
    ax12.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1.5))

    plt.suptitle(f'Sample Visualization: {sample_key}', fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.05)

    # Save the figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def get_all_files_from_source(data_source: str) -> List[str]:
    """
    Get all relevant files from the specified data source.

    Args:
        data_source: Either 'png' or 'cps_tiles'

    Returns:
        List of filenames
    """
    if data_source == 'cps_tiles':
        data_dir = settings.CPS_TILES_DIR
    else:
        data_dir = settings.DATA_DIR

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Get all PNG files
    all_files = [f.name for f in data_path.glob('*.png')]

    # Filter to only include relevant files (x_structuralNOisoline, x_structuralBlackWhite,
    # x_faults, y_traps)
    relevant_files = []
    for f in all_files:
        stem = Path(f).stem
        parts = stem.split('_')
        if len(parts) >= 3:
            file_type = parts[2] if len(parts) > 2 else ''
            if file_type in ['structuralNOisoline', 'structuralBlackWhite', 'faults', 'traps']:
                relevant_files.append(f)

    print(f"Found {len(relevant_files)} relevant files in {data_dir}")
    return relevant_files


def main():
    parser = argparse.ArgumentParser(description='Visualize dataset samples')
    parser.add_argument('--samples', type=str, nargs='+', default=None,
                        help='Specific sample indices to visualize (e.g., --samples 0 2 4)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of random samples to visualize if --samples not specified (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--load-all', action='store_true',
                        help='Load all files from the data source instead of hardcoded list')
    parser.add_argument('--output-dir', type=str, default='./data/test_dataset_vizualisation/',
                        help='Output directory for visualizations (default: ./data/test_dataset_vizualisation/)')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    settings.create_dirs()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GEOLOGY TRAPS DATASET VISUALIZATION")
    print("=" * 70)
    print(f"DATA_SOURCE:  {settings.DATA_SOURCE}")
    print(f"USE_FAULTS:   {settings.USE_FAULTS}")
    print(f"IN_CHANNELS:  {settings.in_channels}")
    print(f"SEED:         {args.seed}")
    print(f"OUTPUT DIR:   {output_path}")
    print("=" * 70)

    # Get file list
    if args.load_all:
        file_list = get_all_files_from_source(settings.DATA_SOURCE)
    else:
        # Список файлов в зависимости от источника (hardcoded по умолчанию)
        if settings.DATA_SOURCE == 'cps_tiles':
            # CPS tiles: PNG файлы из images_cps/
            file_list = [
                '001_x_structuralNOisoline_U4_42_kolltop1.png',
                '001_x_structuralBlackWhite_U4_42_kolltop1.png',
                '001_y_traps_U4_42_kolltop1.png',
                '002_x_structuralNOisoline_U4_42_kolltop1.png',
                '002_x_structuralBlackWhite_U4_42_kolltop1.png',
                '002_y_traps_U4_42_kolltop1.png',
                '003_x_structuralNOisoline_U4_42_kolltop1.png',
                '003_x_structuralBlackWhite_U4_42_kolltop1.png',
                '003_y_traps_U4_42_kolltop1.png',
                '004_x_structuralNOisoline_U4_42_kolltop1.png',
                '004_x_structuralBlackWhite_U4_42_kolltop1.png',
                '004_y_traps_U4_42_kolltop1.png',
            ]
            data_dir = settings.CPS_TILES_DIR
        else:
            # PNG: с расширением
            file_list = [
                '001_x_structuralNOisoline_H150.png', '001_x_structuralBlackWhite_H150.png',
                '001_x_faults_H150.png', '001_y_traps_H150.png',
                '002_x_structuralNOisoline_H150.png', '002_x_structuralBlackWhite_H150.png',
                '002_x_faults_H150.png', '002_y_traps_H150.png',
                '003_x_structuralNOisoline_H150.png', '003_x_structuralBlackWhite_H150.png',
                '003_x_faults_H150.png', '003_y_traps_H150.png'
            ]
            data_dir = settings.DATA_DIR

    print(f"\nData directory: {data_dir if not args.load_all else settings.CPS_TILES_DIR if settings.DATA_SOURCE == 'cps_tiles' else settings.DATA_DIR}")
    print(f"Files to load: {len(file_list)}")

    # Create dataset
    train_dataset = GeologyTrapsDataset(
        file_list,
        data_dir=data_dir if settings.DATA_SOURCE != 'cps_tiles' and not args.load_all else None,
        cps_tiles_dir=data_dir if settings.DATA_SOURCE == 'cps_tiles' and not args.load_all else None,
        augment=False,
        data_source=settings.DATA_SOURCE
    )

    if len(train_dataset) == 0:
        print("\n No samples loaded! Check file paths and naming.")
        return

    print(f"\nTotal samples in dataset: {len(train_dataset)}")

    # Determine which samples to visualize
    if args.samples is not None:
        # Use specified sample indices
        sample_indices = [int(idx) for idx in args.samples]
        print(f"Visualizing specified samples: {sample_indices}")
    else:
        # Select random samples
        num_samples = min(args.num_samples, len(train_dataset))
        sample_indices = random.sample(range(len(train_dataset)), num_samples)
        print(f"Visualizing {num_samples} random samples (seed={args.seed}): {sample_indices}")

    # Visualize each sample
    print("\n" + "=" * 70)
    for idx in sample_indices:
        if idx >= len(train_dataset):
            print(f"Skipping index {idx} (out of range)")
            continue

        # Get sample key for filename
        sample_key = list(train_dataset.samples.keys())[idx] if hasattr(train_dataset.samples, 'keys') else f"sample_{idx}"

        # Create safe filename
        safe_name = sample_key.replace('/', '_').replace('\\', '_')
        save_filename = f"sample_{idx}_{safe_name}.png"
        save_path = output_path / save_filename

        print(f"\nVisualizing sample {idx}: {sample_key}")
        visualize_sample(train_dataset, idx, str(save_path))

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"All images saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    