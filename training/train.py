import os
import json
import time
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from metrics.metrics import MetricsCalculator
from visualization.visualize import visualize_training_results
from settings import settings
from training.gradient_tracker import GradientNormTracker


def train_with_wandb(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,  # CombinedLoss
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str = None,
    n_epochs: int = None,
    early_stopping_patience: int = settings.ES_PATIANCE,
    gradient_accumulation_steps: int = settings.GRADIENT_ACC_STEPS,
    wandb_project: str = 'geology-traps-segmentation',
    wandb_run_name: str = None,
    checkpoint_path: str = settings.CHECKPOINT_DIR,
    log_gradients: bool = settings.LOG_GRADIENTS,
    grad_anomaly_tracking: bool = settings.TRACK_GRADIENT_ANOMALIES,
    grad_abs_threshold: float = settings.ABS_THRESHOLD,
    grad_std_multiplier: float = settings.STD_MULTIPLIER,
    save_anomaly_batches: bool = True,
    max_anomalies_to_save: int = 20
) -> Dict:
    """
    Fine-tuning U-Net++ с мониторингом в wandb.
    
    Args:
        model: Модель
        train_loader: Обучающий DataLoader
        val_loader: Валидационный DataLoader
        criterion: Функция потерь
        optimizer: Оптимизатор
        scheduler: Планировщик LR
        device: Устройство
        n_epochs: Количество эпох
        early_stopping_patience: Патанс для ранней остановки
        gradient_accumulation_steps: Шаги накопления градиентов
        wandb_project: Имя проекта wandb
        wandb_run_name: Имя запуска
        checkpoint_path: Путь для сохранения чекпоинтов
        log_gradients: Логировать ли градиенты
        grad_anomaly_tracking: Трекать ли аномалии градиентов
        grad_abs_threshold: Абсолютный порог grad_norm для аномалий
        grad_std_multiplier: Множитель std для динамического порога
        save_anomaly_batches: Сохранять ли проблемные батчи
        max_anomalies_to_save: Максимум аномалий для сохранения
    
    Returns:
        История обучения
    """
    device = device or settings.DEVICE
    n_epochs = n_epochs or settings.NUM_EPOCHS
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Инициализация wandb
    if wandb_project is not None:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f'unetplusplus_{time.strftime("%Y%m%d_%H%M%S")}',
            config={
                'learning_rate': optimizer.defaults['lr'],
                'batch_size': train_loader.batch_size,
                'epochs': n_epochs,
                'optimizer': type(optimizer).__name__,
                'scheduler': type(scheduler).__name__ if scheduler else 'None',
                'criterion': 'BCE+Dice',
                'model': 'U-Net++',
                'encoder': 'ResNet34',
                'in_channels': settings.IN_CHANNELS,
                'gradient_accumulation_steps': gradient_accumulation_steps
            }
        )
    
    print("=" * 60)
    print("STARTING FINE-TUNING WITH W&B MONITORING")
    print("=" * 60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': [],
        'train_iou': [],
        'val_iou': [],
        'learning_rates': []
    }
    
    metrics_calc = MetricsCalculator()
    best_val_dice = 0.0
    patience_counter = 0
    
    # Инициализация трекера аномалий градиентов
    grad_tracker = None
    if grad_anomaly_tracking:
        grad_tracker = GradientNormTracker(
            abs_threshold=grad_abs_threshold,
            std_multiplier=grad_std_multiplier,
            save_dir=settings.GRAD_ANOMALIES_DIR
        )
        print(f"Gradient anomaly tracking enabled: abs_threshold={grad_abs_threshold}, std_multiplier={grad_std_multiplier}")
    
    global_step = 0

    for epoch in range(n_epochs):
        start_time = time.time()
        
        model.train()
        epoch_train_loss = 0.0
        epoch_train_dice = 0.0
        epoch_train_iou = 0.0
        n_train_batches = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        for batch_idx, batch in enumerate(pbar):
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            mask_map = batch.get('mask_map', None)
            if mask_map is not None:
                mask_map = mask_map.to(device)
            
            # Forward pass
            predictions = model(x)
            
            # Loss
            loss, loss_metrics = criterion(predictions, y, mask_map=mask_map)
            
            # Нормализуем loss для gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Шаг оптимизатора с накоплением градиентов
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Градиентный клиппинг
                if log_gradients and wandb.run is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_norm_value = grad_norm.item()
                    
                    # Логгируем в wandb
                    wandb.log({'grad_norm': grad_norm_value}, commit=False)
                    
                    # Проверяем на аномалии
                    if grad_tracker is not None:
                        is_anomaly, stats = grad_tracker.update(grad_norm_value)
                        
                        # Логируем расширенную статистику
                        wandb.log({
                            'grad_norm_running_mean': stats['running_mean'],
                            'grad_norm_running_std': stats['running_std'],
                            'grad_norm_threshold': stats['threshold']
                        }, commit=False)
                        
                        # Если аномалия - сохраняем батч
                        if is_anomaly and save_anomaly_batches and len(grad_tracker.anomalies) <= max_anomalies_to_save:
                            grad_tracker.log_anomaly(
                                epoch=epoch,
                                batch_idx=batch_idx,
                                grad_norm=grad_norm_value,
                                batch_data=batch,
                                model=model,
                                save_batch=True
                            )
                            
                            # Логируем в wandb
                            wandb.log({
                                'gradient_anomaly_detected': 1,
                                'gradient_anomaly_grad_norm': grad_norm_value,
                                'gradient_anomaly_batch_idx': batch_idx,
                                'gradient_anomaly_epoch': epoch
                            }, commit=False)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Metrics
            metrics = metrics_calc.compute_all(predictions, y, mask_map)
            
            epoch_train_loss += loss.item() * gradient_accumulation_steps
            epoch_train_dice += metrics['dice']
            epoch_train_iou += metrics['iou']
            n_train_batches += 1
            
            # Обновляем progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'dice': f'{metrics["dice"]:.4f}'
            })

            global_step += 1
        
        # Средние метрики за эпоху (train)
        avg_train_loss = epoch_train_loss / n_train_batches
        avg_train_dice = epoch_train_dice / n_train_batches
        avg_train_iou = epoch_train_iou / n_train_batches
        
        # Валидация
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_dice = 0.0
        epoch_val_iou = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]  ')
            for batch in pbar:
                x = batch['x'].to(device)
                y = batch['y'].to(device)
                mask_map = batch.get('mask_map', None)
                if mask_map is not None:
                    mask_map = mask_map.to(device)
                
                predictions = model(x)
                
                loss, loss_metrics = criterion(predictions, y, mask_map=mask_map)
                metrics = metrics_calc.compute_all(predictions, y, mask_map)
                
                epoch_val_loss += loss.item()
                epoch_val_dice += metrics['dice']
                epoch_val_iou += metrics['iou']
                n_val_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{metrics["dice"]:.4f}'
                })
        
        # Средние метрики за эпоху (val)
        avg_val_loss = epoch_val_loss / n_val_batches
        avg_val_dice = epoch_val_dice / n_val_batches
        avg_val_iou = epoch_val_iou / n_val_batches
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Сохраняем в историю
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_dice'].append(avg_train_dice)
        history['val_dice'].append(avg_val_dice)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)
        
        # Логируем в wandb
        if wandb.run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_dice': avg_train_dice,
                'val_dice': avg_val_dice,
                'train_iou': avg_train_iou,
                'val_iou': avg_val_iou,
                'learning_rate': current_lr,
                'epoch_time': time.time() - start_time
            })
        
        print(f"\nEpoch {epoch+1}/{n_epochs}:")
        print(f" Train: Loss={avg_train_loss:.4f}, Dice={avg_train_dice:.4f}, IoU={avg_train_iou:.4f}")
        print(f" Val:   Loss={avg_val_loss:.4f}, Dice={avg_val_dice:.4f}, IoU={avg_val_iou:.4f}")
        print(f" Time:  {time.time() - start_time:.1f}s, LR: {current_lr:.2e}")
        
        # Визуализация
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Визуализация на валидации
            with torch.no_grad():
                batch_example = next(iter(val_loader))
                x_example = batch_example['x'].to(device)
                preds_example = model(x_example)
                visualize_training_results(
                    batch_example,
                    preds_example,
                    epoch + 1,
                    dataset=val_loader.dataset,
                    save_path=settings.LOGS_VAL_VIZ_DIR,
                    n_samples=4
                )
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            patience_counter = 0
            
            # Сохраняем лучшую модель
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
                'val_iou': avg_val_iou,
                'config': {
                    'in_channels': settings.IN_CHANNELS,
                    'encoder': 'resnet34',
                    'classes': 1
                }
                
            }
            checkpoint_name = f'{"faults" if settings.USE_FAULTS else "no_faults"}_epochs-{settings.NUM_EPOCHS}_lr-{settings.LEARNING_RATE}_bs-{settings.BATCH_SIZE}.pth'
            best_model_path = os.path.join(checkpoint_path, checkpoint_name)
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model with Dice={best_val_dice:.4f}")
            
            if wandb.run is not None:
                wandb.save(best_model_path)
        else:
            patience_counter += 1
        
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step(epoch)
        
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Завершение wandb
    if wandb.run is not None:
        # Логируем финальную сводку по аномалиям
        if grad_tracker is not None:
            summary = grad_tracker.get_summary()
            wandb.log({
                'gradient_anomalies_total': summary['total_anomalies'],
                'gradient_anomaly_rate': summary['anomaly_rate'],
                'gradient_norm_final_mean': summary['running_mean'],
                'gradient_norm_final_std': summary['running_std']
            })
            print(f"\nGradient Anomaly Summary:")
            print(f" Total anomalies: {summary['total_anomalies']}")
            print(f" Anomaly rate: {summary['anomaly_rate']:.2%}")
            print(f" Final running mean: {summary['running_mean']:.2f}")
            print(f" Final running std: {summary['running_std']:.2f}")
            print(f" Saved to: {grad_tracker.save_dir}")
        
        wandb.finish()
    
    # Сохраняем историю
    history_path = os.path.join(checkpoint_path, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed. Best Val Dice: {best_val_dice:.4f}")
    print(f"History saved to {history_path}")
    
    return history
