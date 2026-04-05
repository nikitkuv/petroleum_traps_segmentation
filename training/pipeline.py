import os
from typing import Dict
import torch
from torch.utils.data import Subset

from settings import settings
from data.dataloaders import get_file_list, split_data_by_groups, create_dataloaders
from models.unetplusplus import load_unetplusplus, load_model_checkpoint
from losses.losses import CombinedLoss
from optimizers.optimizers import create_optimizer_and_scheduler
from training.overfit_check import overfit_check
from training.train import train_with_wandb
from evaluation.evaluate import evaluate_on_test, visualize_test_predictions


def run_full_pipeline(
    data_dir: str = None,
    use_faults: bool = False,
    data_source: str = None,
    augment_train: bool = None,
    overfit_check_mode: bool = False,
    wandb_project: str = 'geology-traps-segmentation',
    wandb_run_name: str = None,
    n_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    early_stopping_patience: int = None,
    encoder_lr_multiplier: float = None
) -> Dict[str, float]:
    """
    Запускает полный пайплайн обучения и тестирования модели.
    
    Args:
        data_dir: Путь к данным
        use_faults: Использовать ли разломы
        data_source: Источник данных ('png' или 'cps')
        overfit_check_mode: Режим проверки overfit
        wandb_project: wandb
        wandb_run_name: Имя запуска
        n_epochs: Количество эпох
        batch_size: Размер батча
        learning_rate: Скорость обучения
        early_stopping_patience: Патанс для ранней остановки
        encoder_lr_multiplier: Множитель LR для энкодера
    
    Returns:
        Метрики на тестовой выборке
    """
    device = settings.DEVICE
    data_dir = data_dir or settings.DATA_DIR
    batch_size = batch_size or settings.BATCH_SIZE
    learning_rate = learning_rate or settings.LEARNING_RATE
    data_source = data_source or settings.DATA_SOURCE
    augment_train = augment_train or settings.AUGMENT_TRAIN
    n_epochs = n_epochs or settings.NUM_EPOCHS
    early_stopping_patience = early_stopping_patience or settings.ES_PATIANCE
    encoder_lr_multiplier = encoder_lr_multiplier or settings.ENCODER_LR_MULTIPLIER
    
    print("=" * 80)
    print("GEOLOGY TRAPS SEGMENTATION PIPELINE")
    print(f"Data source: {data_source}")
    print(f"Use faults: {use_faults}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # ========== 1. ЗАГРУЗКА ДАННЫХ ==========
    print("\n[STEP 1] Loading data...")
    all_files = get_file_list(data_dir, data_source=data_source)
    
    if len(all_files) == 0:
        raise ValueError("No data files found!")
    
    # ========== 2. РАЗДЕЛЕНИЕ НА ВЫБОРКИ ==========
    print("\n[STEP 2] Splitting data into train/val/test...")
    train_files, val_files, test_files = split_data_by_groups(
        file_list=all_files,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    
    # ========== 3. СОЗДАНИЕ DATALOADERS ==========
    print("\n[STEP 3] Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        data_dir=data_dir,
        batch_size=batch_size,
        use_faults=use_faults,
        data_source=data_source,
        augment_train=augment_train
    )
    
    # Если режим overfit check - берем только 1-2 карты из train
    if overfit_check_mode:
        print("\n[OVERFIT CHECK MODE] Using only first batch from train...")
        # Создаем новый dataloader с одним батчем
        overfit_dataset = train_loader.dataset
        overfit_indices = list(range(min(settings.OVERFIT_SIZE, len(overfit_dataset))))  # 2 семпла
        print(f"Selected indices for overfit: {overfit_indices}")
        overfit_subset = Subset(overfit_dataset, overfit_indices)
        train_loader = torch.utils.data.DataLoader(
            overfit_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        print(f"Size of train_loader: {len(train_loader.dataset)}")
    
    # ========== 4. ЗАГРУЗКА МОДЕЛИ ==========
    print("\n[STEP 4] Loading U-Net++ model...")
    in_channels = settings.in_channels
    model = load_unetplusplus(
        in_channels=in_channels,
        classes=1,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        device=device
    )
    print(f"Model loaded with {in_channels} input channels")
    
    # ========== 5. ЛОСС ==========
    print("\n[STEP 5] Setting up loss function...")
    criterion = CombinedLoss(
        bce_weight=settings.BCE_WEIGHT_RATIO,
        dice_weight=settings.DICE_WEIGHT_RATIO,
        use_map_mask=True,
        use_depth_mask=use_faults
    )
    
    # ========== 6. ОПТИМИЗАТОР ==========
    print("\n[STEP 6] Setting up optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        learning_rate=learning_rate,
        weight_decay=settings.WEIGHT_DECAY,
        scheduler_type='reduce_lr_plateau',
        encoder_lr_multiplier=encoder_lr_multiplier
    )
    print(f"Optimizer: AdamW, LR={learning_rate}, Encoder LR multiplier={encoder_lr_multiplier}")
    
    # ========== 7. OVERFIT CHECK ИЛИ ОБУЧЕНИЕ ==========
    if overfit_check_mode:
        print("\n[STEP 7a] Running overfit check...")
        overfit_check(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_epochs=100,
            save_path=settings.LOGS_OVERFIT_CHECK_DIR
        )
    else:
        # ========== 8. ОБУЧЕНИЕ С W&B ==========
        print("\n[STEP 8] Starting fine-tuning with W&B monitoring...")
        history = train_with_wandb(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            n_epochs=n_epochs,
            early_stopping_patience=early_stopping_patience,
            gradient_accumulation_steps=settings.GRADIENT_ACC_STEPS,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            checkpoint_path=settings.CHECKPOINT_DIR,
            log_gradients=True
        )
        
        # ========== 9. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ТЕСТИРОВАНИЕ ==========
        print("\n[STEP 9] Evaluating best model on test data...")
        best_model_path = os.path.join(settings.CHECKPOINT_DIR, f'{"faults" if settings.USE_FAULTS else "no_faults"}_epochs-{settings.NUM_EPOCHS}_lr-{settings.LEARNING_RATE}_bs-{settings.BATCH_SIZE}.pth')
        model = load_model_checkpoint(model, best_model_path, device)
        
        test_metrics = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=settings.TEST_THRESHOLD
        )
        
        # ========== 10. ВИЗУАЛИЗАЦИЯ ТЕСТОВЫХ РЕЗУЛЬТАТОВ ==========
        print("\n[STEP 10] Visualizing test results...")
        visualize_test_predictions(
            model=model,
            test_loader=test_loader,
            device=device,
            sample_indices=[0, 1, 2, 3],
            save_path=settings.LOGS_TEST_VIZ_DIR,
            alpha=0.4
        )
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Best model saved to: {best_model_path}")
        print(f"Training history: {settings.CHECKPOINT_DIR}training_history.json")
        print(f"Visualizations: ./logs/")
        print("=" * 80)
        
        return test_metrics
    
    return {}
