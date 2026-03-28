"""
Модуль для создания оптимизаторов и планировщиков скорости обучения.
Поддержка differential learning rate для энкодера и декодера.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from settings import settings


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = None,
    weight_decay: float = None,
    scheduler_type: str = 'reduce_lr_plateau',
    encoder_lr_multiplier: float = 0.1
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """
    Создает оптимизатор и планировщик скорости обучения.
    
    Args:
        model: Модель
        learning_rate: Базовая скорость обучения
        weight_decay: Вес регуляризации L2
        scheduler_type: Тип планировщика ('reduce_lr_plateau' или 'cosine_annealing')
        encoder_lr_multiplier: Множитель LR для энкодера (для differential LR)
    
    Returns:
        Кортеж (optimizer, scheduler)
    """
    lr = learning_rate or settings.LEARNING_RATE
    wd = weight_decay or 1e-4
    
    # Differential Learning Rate: разные LR для энкодера и декодера
    encoder_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': lr * encoder_lr_multiplier},
        {'params': decoder_params, 'lr': lr}
    ], weight_decay=wd)
    
    # Планировщик
    if scheduler_type == 'reduce_lr_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-7
        )
    elif scheduler_type == 'cosine_annealing':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def get_gradient_stats(model: nn.Module) -> dict:
    """
    Вычисляет статистику градиентов модели.
    
    Args:
        model: Модель
    
    Returns:
        Словарь со статистикой градиентов
    """
    grad_stats = {
        'grad_norm_total': 0.0,
        'grad_norm_encoder': 0.0,
        'grad_norm_decoder': 0.0,
        'grad_min': float('inf'),
        'grad_max': float('-inf')
    }
    
    total_params = 0
    encoder_params = 0
    decoder_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats['grad_norm_total'] += grad_norm ** 2
            
            grad_stats['grad_min'] = min(grad_stats['grad_min'], param.grad.abs().min().item())
            grad_stats['grad_max'] = max(grad_stats['grad_max'], param.grad.abs().max().item())
            
            if 'encoder' in name:
                grad_stats['grad_norm_encoder'] += grad_norm ** 2
                encoder_params += param.numel()
            else:
                grad_stats['grad_norm_decoder'] += grad_norm ** 2
                decoder_params += param.numel()
            
            total_params += param.numel()
    
    grad_stats['grad_norm_total'] = grad_stats['grad_norm_total'] ** 0.5
    grad_stats['grad_norm_encoder'] = grad_stats['grad_norm_encoder'] ** 0.5
    grad_stats['grad_norm_decoder'] = grad_stats['grad_norm_decoder'] ** 0.5
    
    if total_params == 0:
        grad_stats['grad_min'] = 0.0
        grad_stats['grad_max'] = 0.0
    
    return grad_stats
