from torch.utils.data import DataLoader

from settings import settings
from dataset import GeologyTrapsDataset
from visualization import visualize_dataset_sample


if __name__ == "__main__":
    # Создаём директории
    settings.create_dirs()
    
    file_list = [
        '001_x_faults_H150.png', '001_x_structuralBlackWhite_H150.png',
        '001_x_structuralNOisoline_H150.png', '001_y_traps_H150.png',
        '002_x_faults_H150.png', '002_x_structuralBlackWhite_H150.png',
        '002_x_structuralNOisoline_H150.png', '002_y_traps_H150.png',
        '003_x_faults_H150.png', '003_x_structuralBlackWhite_H150.png',
        '003_x_structuralNOisoline_H150.png', '003_y_traps_H150.png'
    ]
    
    print("="*60)
    print("Testing GeologyTrapsDataset with Visualization")
    print("="*60)
    print(f"\n📁 Settings:")
    print(f"  DATA_DIR:       {settings.DATA_DIR}")
    print(f"  TARGET_SIZE:    {settings.TARGET_HEIGHT}×{settings.TARGET_WIDTH}")
    print(f"  BATCH_SIZE:     {settings.BATCH_SIZE}")
    print(f"  NUM_WORKERS:    {settings.NUM_WORKERS}")
    print(f"  AUGMENT_PROB:   {settings.AUGMENT_PROB}")
    print(f"  DEVICE:         {settings.DEVICE}")
    print("="*60)
    
    # Создаём Dataset (с аугментациями)
    train_dataset = GeologyTrapsDataset(file_list, data_dir=settings.DATA_DIR, augment=False)
    
    # Создаём Dataset (без аугментаций для сравнения)
    val_dataset = GeologyTrapsDataset(file_list, data_dir=settings.DATA_DIR, augment=False)
    
    # 1. Визуализация с аугментациями (train)
    print("\n📊 Visualizing TRAIN sample (with augmentations)...")
    visualize_dataset_sample(train_dataset, idx=0, save_path=str(settings.logs_path / 'viz_train_sample.png'))
    
    # 2. Визуализация без аугментаций (val)
    print("\n📊 Visualizing VAL sample (without augmentations)...")
    visualize_dataset_sample(val_dataset, idx=0, save_path=str(settings.logs_path / 'viz_val_sample.png'))
    
    print("\n📊 Testing DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True
    )
    
    batch = next(iter(train_loader))
    print(f"\n✓ Batch Input Shape:  {batch['x'].shape}")
    print(f"✓ Batch Target Shape: {batch['y'].shape}")
    print(f"✓ Batch Size: {batch['x'].shape[0]}")
    
    print("\n✓ All tests completed successfully!")
    print("✓ Ready for UNet++ training!")