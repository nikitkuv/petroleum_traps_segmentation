import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from settings import settings
from dataset import GeologyTrapsDataset
from utils.images_utils import load_grayscale_image, load_image


def visualize_dataset_sample(
    dataset: GeologyTrapsDataset,
    idx: int = 0,
    save_path: Optional[str] = None
):
    # Получаем данные из dataset
    sample_data = dataset[idx]
    
    # Получаем пути к оригинальным файлам для загрузки
    sample_paths = dataset.samples[idx]
    
    # Загружаем оригиналы (до обработки)
    orig_rgb = load_image(sample_paths['rgb'])
    orig_depth = load_grayscale_image(sample_paths['depth_norm'])
    orig_traps = load_grayscale_image(sample_paths['traps'])
    orig_faults = load_grayscale_image(sample_paths['faults']) if 'faults' in sample_paths else np.zeros_like(orig_depth)
    
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

    fig = plt.figure(figsize=(20, 16))
    
    # 1. Original RGB
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.imshow(orig_rgb)
    ax1.set_title(f'Original RGB\n{orig_rgb.shape[0]}×{orig_rgb.shape[1]}', fontsize=10, pad=15)
    ax1.axis('off')
    
    # 2. Processed RGB (из dataset)
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.imshow(x_rgb)
    ax2.set_title(f'Processed RGB (from Dataset)\n{x_rgb.shape[0]}×{x_rgb.shape[1]}', fontsize=10, pad=15)
    ax2.axis('off')
    
    # 3. Original Depth
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.imshow(orig_depth, cmap='gray')
    ax3.set_title('Original Depth (B&W)\nBlack=High, White=Low', fontsize=10, pad=15)
    ax3.axis('off')
    
    # 4. Processed Depth (из dataset)
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.imshow(x_depth, cmap='gray')
    ax4.set_title('Processed Depth (from Dataset)\nBlack=High, White=Low', fontsize=10, pad=15)
    ax4.axis('off')
    
    # 5. Original Faults
    ax5 = fig.add_subplot(3, 4, 5)
    ax5.imshow(orig_faults, cmap='gray')
    ax5.set_title('Original Fault Mask\nBlack=Faults', fontsize=10, pad=15)
    ax5.axis('off')
    
    # 6. Processed Faults (из dataset)
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.imshow(x_faults, cmap='gray')
    ax6.set_title('Processed Fault Mask (from Dataset)\n1=Faults (White)', fontsize=10, pad=15)
    ax6.axis('off')
    
    # 7. Original Traps
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.imshow(orig_traps, cmap='gray')
    ax7.set_title('Original Trap Mask\nBlack=Traps', fontsize=10, pad=15)
    ax7.axis('off')
    
    # 8. Processed Traps / Target Y (из dataset)
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.imshow(y_traps, cmap='gray')
    ax8.set_title('Target Y (from Dataset)\n1=Traps (White)', fontsize=10, pad=15)
    ax8.axis('off')
    
    # 9. Map Mask
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.imshow(mask_map, cmap='gray')
    ax9.set_title('Map Mask\n1=Map Area (White), 0=Padding (Black)', fontsize=10, pad=15)
    ax9.axis('off')
    
    # 10. Depth Mask
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.imshow(mask_depth, cmap='gray')
    ax10.set_title('Depth Mask\n1=Valid (White), 0=Faults/Pad (Black)', fontsize=10, pad=15)
    ax10.axis('off')
    
    # 11. Input Channels Overview (RGB+Depth+Faults)
    ax11 = fig.add_subplot(3, 4, 11)
    # Показываем композит: RGB + контуры разломов
    composite = x_rgb.copy()
    fault_overlay = (x_faults > 0.5).astype(np.float32)
    composite[:, :, 0] = np.maximum(composite[:, :, 0], fault_overlay * 0.7)  # Красный канал для разломов
    ax11.imshow(composite)
    
    # 🔧 Исправление: динамический заголовок без ...
    n_channels = 5 if sample_data['use_faults'] else 4
    ax11.set_title(f'Input Overview ({n_channels} channels)\nRGB + Depth' + (' + Faults' if sample_data['use_faults'] else ''), fontsize=10, pad=15)
    ax11.axis('off')
    
    # 12. Statistics Info Box
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    
    info_text = (
        f"Sample Statistics\n\n"
        f"Sample Index: {idx}\n"
        f"Augmentations: {'✓ ON' if dataset.augment else '✗ OFF'}\n\n"
        f"Input Shape: {sample_data['x'].shape}\n"
        f"Target Shape: {sample_data['y'].shape}\n\n"
        f"Map Valid: {mask_map.sum():.0f} px ({mask_map.mean()*100:.1f}%)\n"
        f"Depth Valid: {mask_depth.sum():.0f} px ({mask_depth.mean()*100:.1f}%)\n"
        f"Faults: {x_faults.sum():.0f} px ({x_faults.mean()*100:.2f}%)\n"
        f"Traps (Y): {y_traps.sum():.0f} px ({y_traps.mean()*100:.2f}%)"
    )
    ax12.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, pad=1.5))
    
    plt.subplots_adjust(hspace=0.35, wspace=0.15, top=0.95, bottom=0.05)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()
    

    print("\n" + "="*60)
    print(f"📋 DATASET SAMPLE REPORT (Idx={idx})")
    print("="*60)
    print(f"Augmentations: {'✓ ON' if dataset.augment else '✗ OFF'}")
    print(f"\nTensor Shapes:")
    print(f"  Input (x):      {sample_data['x'].shape}")
    print(f"  Target (y):     {sample_data['y'].shape}")
    if sample_data['mask_depth'] is not None:
        print(f"  Mask Depth:     {sample_data['mask_depth'].shape}")
    else:
        print(f"  Mask Depth:     None (not used)")
    print(f"  Mask Map:       {sample_data['mask_map'].shape}")
    print(f"\nPixel Statistics:")
    print(f"  Map Valid:      {mask_map.sum():.0f} px ({mask_map.mean()*100:.1f}%)")
    print(f"  Depth Valid:    {mask_depth.sum():.0f} px ({mask_depth.mean()*100:.1f}%)")
    print(f"  Faults:         {x_faults.sum():.0f} px ({x_faults.mean()*100:.2f}%)")
    print(f"  Traps (Y):      {y_traps.sum():.0f} px ({y_traps.mean()*100:.2f}%)")
    print(f"\nValue Ranges:")
    print(f"  RGB:            [{x_rgb.min():.3f}, {x_rgb.max():.3f}]")
    print(f"  Depth:          [{x_depth.min():.3f}, {x_depth.max():.3f}]")
    print(f"  Faults:         [{x_faults.min():.3f}, {x_faults.max():.3f}]")
    print(f"  Traps (Y):      [{y_traps.min():.3f}, {y_traps.max():.3f}]")
    print("="*60)


if __name__ == "__main__":
    settings.create_dirs()
    
    print("=" * 70)
    print("🧪 GEOLOGY TRAPS DATASET TEST")
    print("=" * 70)
    print(f"DATA_SOURCE:  {settings.DATA_SOURCE}")
    print(f"USE_FAULTS:   {settings.USE_FAULTS}")
    print(f"IN_CHANNELS:  {settings.in_channels}")
    print("=" * 70)
    
    # 🔑 Список файлов в зависимости от источника
    if settings.DATA_SOURCE == 'cps':
        # CPS: без расширения
        file_list = ['001_x_structuralNOisoline_H76', '001_y_traps_H76']
        if settings.USE_FAULTS:
            file_list.extend(['001_x_faults_H76'])
        data_dir = settings.CPS_DIR
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
    
    print(f"\n📁 Data directory: {data_dir}")
    print(f"📄 Files to load: {len(file_list)}")
    print()
    
    train_dataset = GeologyTrapsDataset(
        file_list, 
        data_dir=data_dir if settings.DATA_SOURCE == 'png' else None,
        cps_dir=data_dir if settings.DATA_SOURCE == 'cps' else None,
        augment=False
    )
    
    if len(train_dataset) > 0:
        # Визуализация
        visualize_dataset_sample(train_dataset, idx=0, 
                                save_path=str(settings.logs_path / f'viz_test_{settings.DATA_SOURCE}.png'))
        
        # Проверка DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.BATCH_SIZE,
            shuffle=True,
            num_workers=settings.NUM_WORKERS,
            pin_memory=True
        )
        
        batch = next(iter(train_loader))
        print(f"\n✅ Batch Input Shape:  {batch['x'].shape}")
        print(f"✅ Batch Target Shape: {batch['y'].shape}")
        print(f"✅ Actual channels: {batch['x'].shape[1]}")
        
        expected_channels = 5 if settings.USE_FAULTS else 4
        
        if batch['x'].shape[1] == expected_channels:
            print(f"\n✅ TEST PASSED! Ready for training! ({expected_channels} channels)")
        else:
            print(f"\n❌ TEST FAILED! Expected {expected_channels} channels, got {batch['x'].shape[1]}")
    else:
        print("\n❌ No samples loaded! Check file paths and naming.")