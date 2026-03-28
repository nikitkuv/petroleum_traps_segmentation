"""
Главный файл пайплайна обучения U-Net++ для сегментации геологических ловушек.
Вариант: PNG с RGB без изолиний + depth_norm + map_mask (без разломов)

1. Загрузка данных в датасеты и даталоадеры
2. Разделение данных на обучающую, валидационную и тестовую выборки
3. Код для загрузки модели
4. Код для нужных лоссов
5. Код для задания оптимизатора, гиперпараметров
6. Код для нужных метрик
7. Код для проверки обучения: overfit на 1–2 картах (без валидации)
8. Код для визуализации проверки обучения
9. Код fine tuning U-Net++ с мониторингом в wandb
10. Проверка дообученной модели на тестовых данных
11. Визуализация выборочных результатов тестовых данных
"""

import os
from typing import Dict, Optional
import torch

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
    data_source: str = 'png',
    overfit_check_mode: bool = False,
    wandb_project: str = 'geology-traps-segmentation',
    wandb_run_name: str = None,
    n_epochs: int = 100,
    batch_size: int = None,
    learning_rate: float = None,
    early_stopping_patience: int = 15,
    encoder_lr_multiplier: float = 0.1
) -> Dict[str, float]:
    """
    Запускает полный пайплайн обучения и тестирования модели.
    
    Args:
        data_dir: Путь к данным
        use_faults: Использовать ли разломы
        data_source: Источник данных ('png' или 'cps')
        overfit_check_mode: Режим проверки overfit
        wandb_project:项目名称 wandb
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
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        data_source=data_source
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
        data_source=data_source
    )
    
    # Если режим overfit check - берем только 1-2 карты из train
    if overfit_check_mode:
        print("\n[OVERFIT CHECK MODE] Using only first batch from train...")
        # Создаем новый dataloader с одним батчем
        from torch.utils.data import Subset
        overfit_dataset = train_loader.dataset
        overfit_indices = list(range(min(4, len(overfit_dataset))))  # 4 семпла
        overfit_subset = Subset(overfit_dataset, overfit_indices)
        train_loader = torch.utils.data.DataLoader(
            overfit_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    # ========== 4. ЗАГРУЗКА МОДЕЛИ ==========
    print("\n[STEP 4] Loading U-Net++ model...")
    in_channels = 4 if not use_faults else 5  # RGB + depth (+ faults)
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
        bce_weight=0.5,
        dice_weight=0.5,
        use_map_mask=True,
        use_depth_mask=use_faults
    )
    
    # ========== 6. ОПТИМИЗАТОР ==========
    print("\n[STEP 6] Setting up optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        learning_rate=learning_rate,
        weight_decay=1e-4,
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
            save_path='./logs/overfit_check/'
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
            gradient_accumulation_steps=1,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            checkpoint_path=settings.CHECKPOINT_DIR,
            log_gradients=True
        )
        
        # ========== 9. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ И ТЕСТИРОВАНИЕ ==========
        print("\n[STEP 9] Evaluating best model on test data...")
        best_model_path = os.path.join(settings.CHECKPOINT_DIR, 'best_model.pth')
        model = load_model_checkpoint(model, best_model_path, device)
        
        test_metrics = evaluate_on_test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            threshold=0.5
        )
        
        # ========== 10. ВИЗУАЛИЗАЦИЯ ТЕСТОВЫХ РЕЗУЛЬТАТОВ ==========
        print("\n[STEP 10] Visualizing test results...")
        visualize_test_predictions(
            model=model,
            test_loader=test_loader,
            device=device,
            sample_indices=[0, 1, 2, 3],
            save_path='./logs/test_visualizations/',
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


if __name__ == '__main__':
    # Пример запуска полного пайплайна
    test_metrics = run_full_pipeline(
        data_dir=settings.DATA_DIR,
        use_faults=False,  # Без разломов
        data_source='png',
        overfit_check_mode=False,  # Установить True для проверки overfit
        wandb_project='geology-traps-segmentation',
        wandb_run_name='unetplusplus_rgb_depth_no_faults',
        n_epochs=100,
        batch_size=4,
        learning_rate=1e-4,
        early_stopping_patience=15
    )
