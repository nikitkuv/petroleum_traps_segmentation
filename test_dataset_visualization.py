from torch.utils.data import DataLoader

from settings import settings
from dataset import GeologyTrapsDataset
from visualization import visualize_dataset_sample


if __name__ == "__main__":

    settings.create_dirs()
    
    file_list = [
        '001_x_faults_H150.png', '001_x_structuralBlackWhite_H150.png', '001_x_structuralNOisoline_H150.png', '001_y_traps_H150.png',
        '002_x_faults_H150.png', '002_x_structuralBlackWhite_H150.png', '002_x_structuralNOisoline_H150.png', '002_y_traps_H150.png',
        '003_x_faults_H150.png', '003_x_structuralBlackWhite_H150.png', '003_x_structuralNOisoline_H150.png', '003_y_traps_H150.png'
    ]
    
    print("=" * 60)
    print(f"Testing GeologyTrapsDataset")
    print("=" * 60)
    print(f"📁 Settings:")
    print(f"  USE_FAULTS:     {settings.USE_FAULTS}")
    print(f"  TARGET_SIZE:    {settings.TARGET_HEIGHT}×{settings.TARGET_WIDTH}")
    print("=" * 60)
    
    train_dataset = GeologyTrapsDataset(
        file_list, 
        data_dir=settings.DATA_DIR, 
        augment=False
    )
    
    # Визуализация
    visualize_dataset_sample(train_dataset, idx=0, save_path=str(settings.logs_path / 'viz_test_sample.png'))
    
    # Проверка DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True
    )
    
    batch = next(iter(train_loader))
    print(f"Batch Input Shape:  {batch['x'].shape}")
    print(f"Batch Target Shape: {batch['y'].shape}")
    print(f"Actual channels: {batch['x'].shape[1]}")
    
    # Проверка depth_mask
    if settings.USE_FAULTS:
        print(f"mask_depth: {batch['mask_depth'].shape if batch['mask_depth'] is not None else None}")
    else:
        print(f"mask_depth: None (not used)")