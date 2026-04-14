import sys
from pathlib import Path
import torch
import argparse
import random
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    orig_isolines = load_grayscale_image(sample_paths['isolines']) if 'isolines' in sample_paths else None

    # Получаем имена файлов для заголовков
    rgb_filename = Path(sample_paths['rgb']).name
    depth_filename = Path(sample_paths['depth_norm']).name
    traps_filename = Path(sample_paths['traps']).name
    faults_filename = Path(sample_paths['faults']).name if 'faults' in sample_paths else "N/A"
    isolines_filename = Path(sample_paths['isolines']).name if 'isolines' in sample_paths else "N/A"

    # Извлекаем тензоры из dataset и конвертируем в numpy
    x_rgb = sample_data['x'][:3].permute(1, 2, 0).numpy()      # (H, W, 3)
    x_depth = sample_data['x'][3].numpy()                       # (H, W)
    x_isolines = sample_data['x'][4].numpy()                    # (H, W)

    # Faults канал только если use_faults=True
    if sample_data['use_faults']:
        x_faults = sample_data['x'][5].numpy()                  # (H, W)
    else:
        x_faults = np.zeros_like(x_depth)

    y_traps = sample_data['y'][0].numpy()                       # (H, W)
    mask_map = sample_data['mask_map'][0].numpy()               # (H, W)

    # mask_depth только если use_faults=True
    if sample_data['mask_depth'] is not None:
        mask_depth = sample_data['mask_depth'][0].numpy()
    else:
        mask_depth = np.zeros_like(x_depth)

    # Извлекаем номер и название горизонта из имени файла
    # Формат имени файла: {number}_..._{name}_...
    sample_filename = Path(sample_paths['rgb']).name
    parts = sample_filename.split('_')
    sample_number = parts[0]  # номер семпла (например, "001")
    # Название горизонта - часть после последнего подчеркивания без расширения
    sample_name = Path(sample_paths['rgb']).stem.split('_')[-1]  # например, "kolltop1" или "H150"
    sample_id = f"{sample_number}_{sample_name}"

    # Структура: 4 ряда x 4 колонки (GridSpec: 4x4)
    # Row 0: RGB orig | RGB proc | Depth orig | Depth proc
    # Row 1: Isolines orig | Isolines proc | Faults orig | Faults proc
    # Row 2: Traps orig | Traps proc (target Y) | Map mask | Depth mask
    # Row 3: Statistics (spans cols 0-3)
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(4, 4, figure=fig,
                           height_ratios=[1, 1, 1, 1],
                           width_ratios=[1, 1, 1, 1],
                           hspace=0.15, wspace=0.1,
                           top=0.94, bottom=0.05, left=0.04, right=0.97)

    # 1. Original RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_rgb)
    ax1.set_title(f'Original RGB\n{rgb_filename}', fontsize=11, pad=6)
    ax1.axis('off')

    # 2. Processed RGB
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(x_rgb)
    ax2.set_title(f'Processed RGB\n{x_rgb.shape[0]}×{x_rgb.shape[1]}', fontsize=11, pad=6)
    ax2.axis('off')

    # 3. Original Depth
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(orig_depth, cmap='gray')
    ax3.set_title(f'Original Depth\n{depth_filename}', fontsize=11, pad=6)
    ax3.axis('off')

    # 4. Processed Depth
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(x_depth, cmap='gray')
    ax4.set_title('Processed Depth\nBlack=High, White=Low', fontsize=11, pad=6)
    ax4.axis('off')

    # 5. Original Isolines
    ax5 = fig.add_subplot(gs[1, 0])
    if orig_isolines is not None:
        ax5.imshow(orig_isolines, cmap='gray')
        ax5.set_title(f'Original Isolines\n{isolines_filename}', fontsize=11, pad=6)
    else:
        ax5.imshow(np.ones_like(x_depth) * 255, cmap='gray')
        ax5.set_title('Original Isolines\nN/A', fontsize=11, pad=6)
    ax5.axis('off')

    # 6. Processed Isolines
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(x_isolines, cmap='gray')
    ax6.set_title('Processed Isolines\nWhite=Isolines, Black=Background', fontsize=11, pad=6)
    ax6.axis('off')

    # 7. Original Faults
    ax7 = fig.add_subplot(gs[1, 2])
    if orig_faults is not None:
        ax7.imshow(orig_faults, cmap='gray')
        ax7.set_title(f'Original Fault Mask\n{faults_filename}', fontsize=11, pad=6)
    else:
        ax7.imshow(np.zeros_like(x_depth), cmap='gray')
        ax7.set_title('Original Fault Mask\nN/A', fontsize=11, pad=6)
    ax7.axis('off')

    # 8. Processed Faults
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.imshow(x_faults, cmap='gray')
    ax8.set_title('Processed Fault Mask\n1=Faults (White)', fontsize=11, pad=6)
    ax8.axis('off')

    # 9. Original Traps
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.imshow(orig_traps, cmap='gray')
    ax9.set_title(f'Original Trap Mask\n{traps_filename}', fontsize=11, pad=6)
    ax9.axis('off')

    # 10. Processed Traps / Target Y
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(y_traps, cmap='gray')
    ax10.set_title('Target Y (Traps)\n1=Traps (White)', fontsize=11, pad=6)
    ax10.axis('off')

    # 11. Map Mask
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.imshow(mask_map, cmap='gray')
    ax11.set_title('Map Mask\n1=Map Area (White)', fontsize=11, pad=6)
    ax11.axis('off')

    # 12. Depth Mask
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.imshow(mask_depth, cmap='gray')
    ax12.set_title('Depth Mask\n1=Valid (White), 0=Faults/Pad (Black)', fontsize=11, pad=6)
    ax12.axis('off')

    # 13. Statistics Info Box — снизу, занимает все 4 колонки
    ax13 = fig.add_subplot(gs[3, :])
    ax13.axis('off')

    info_text = (
        f"Sample: {sample_id}\n\n"
        f"Augmentations: {'ON' if dataset.augment else 'OFF'}\n"
        f"Data Source: {dataset.data_source.upper()}\n\n"
        f"Input Shape: {sample_data['x'].shape}\n"
        f"Target Shape: {sample_data['y'].shape}\n\n"
        f"Map Valid: {mask_map.sum():.0f} px ({mask_map.mean()*100:.1f}%)\n"
        f"Depth Valid: {mask_depth.sum():.0f} px ({mask_depth.mean()*100:.1f}%)\n"
        f"Isolines: {(x_isolines < 128).sum():.0f} line px ({(x_isolines < 128).mean()*100:.2f}%)\n"
        f"Faults: {x_faults.sum():.0f} px ({x_faults.mean()*100:.2f}%)\n"
        f"Traps (Y): {y_traps.sum():.0f} px ({y_traps.mean()*100:.2f}%)"
    )
    ax13.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1.2))

    plt.suptitle(f'Sample {idx}: {sample_id}', fontsize=16, fontweight='bold', y=0.99)

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

    # Filter to only include relevant files (x_structuralNOisoline, x_structuralBlackWhite, x_faults, y_traps, x_isolines)
    relevant_files = []
    for f in all_files:
        stem = Path(f).stem
        parts = stem.split('_')
        if len(parts) >= 3:
            file_type = parts[2] if len(parts) > 2 else ''
            if file_type in ['structuralNOisoline', 'structuralBlackWhite', 'isolines', 'faults', 'traps']:
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
                '001_x_isolines_U4_42_kolltop1.png',
                '001_y_traps_U4_42_kolltop1.png',
                '002_x_structuralNOisoline_U4_42_kolltop1.png',
                '002_x_structuralBlackWhite_U4_42_kolltop1.png',
                '002_x_isolines_U4_42_kolltop1.png',
                '002_y_traps_U4_42_kolltop1.png',
                '003_x_structuralNOisoline_U4_42_kolltop1.png',
                '003_x_structuralBlackWhite_U4_42_kolltop1.png',
                '003_x_isolines_U4_42_kolltop1.png',
                '003_y_traps_U4_42_kolltop1.png',
                '004_x_structuralNOisoline_U4_42_kolltop1.png',
                '004_x_structuralBlackWhite_U4_42_kolltop1.png',
                '004_x_isolines_U4_42_kolltop1.png',
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

        # Create safe filename using sample_id format {number}_{name}
        sample_paths_dict = train_dataset.samples[sample_key] if hasattr(train_dataset.samples, 'keys') else None
        if sample_paths_dict:
            sample_filename = Path(sample_paths_dict['rgb']).name
            parts = sample_filename.split('_')
            sample_number = parts[0]
            sample_name = Path(sample_paths_dict['rgb']).stem.split('_')[-1]
            sample_id = f"{sample_number}_{sample_name}"
        else:
            sample_id = f"sample_{idx}"

        save_filename = f"sample_{sample_id}.png"
        save_path = output_path / save_filename

        print(f"\nVisualizing sample {idx}: {sample_id}")
        visualize_sample(train_dataset, idx, str(save_path))

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"All images saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    