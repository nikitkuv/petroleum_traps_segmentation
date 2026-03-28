"""
Модуль для загрузки и создания моделей U-Net++.
Вариант: PNG с RGB без изолиний + depth_norm (без разломов)
"""

import os
from typing import Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from settings import settings


def load_unetplusplus(
    in_channels: int = None,
    classes: int = 1,
    encoder_name: str = 'resnet34',
    encoder_weights: str = 'imagenet',
    activation: str = None,
    device: str = None
) -> nn.Module:
    """
    Загружает предобученную модель U-Net++.
    
    Args:
        in_channels: Количество входных каналов (4 для rgb+depth, 5 для rgb+depth+faults)
        classes: Количество классов сегментации
        encoder_name: Название энкодера
        encoder_weights: Веса энкодера
        activation: Функция активации
        device: Устройство
    
    Returns:
        Модель U-Net++
    """
    in_channels = in_channels or settings.in_channels
    device = device or settings.DEVICE
    
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type=None
    )
    
    # Модифицируем первый слой энкодера если количество каналов не стандартное
    if in_channels != 3 and encoder_weights is not None:
        # Создаем новый первый слой
        old_conv1 = model.encoder.conv1
        new_conv1 = nn.Conv2d(
            in_channels,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        
        # Копируем веса для первых 3 каналов, остальные инициализируем случайно
        with torch.no_grad():
            if in_channels > 3:
                # Копируем первые 3 канала
                new_conv1.weight[:, :3, :, :] = old_conv1.weight
                # Остальные каналы инициализируем как среднее от RGB
                for i in range(3, in_channels):
                    new_conv1.weight[:, i:i+1, :, :] = old_conv1.weight.mean(dim=1, keepdim=True)
            else:
                new_conv1.weight[:, :in_channels, :, :] = old_conv1.weight[:, :in_channels, :, :]
        
        model.encoder.conv1 = new_conv1
    
    model = model.to(device)
    return model


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = None
) -> nn.Module:
    """
    Загружает веса модели из чекпоинта.
    
    Args:
        model: Модель для загрузки весов
        checkpoint_path: Путь к чекпоинту
        device: Устройство
    
    Returns:
        Модель с загруженными весами
    """
    device = device or settings.DEVICE
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def save_model_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: dict,
    save_path: str,
    filename: str = 'checkpoint.pth'
) -> str:
    """
    Сохраняет чекпоинт модели.
    
    Args:
        model: Модель
        optimizer: Оптимизатор
        epoch: Номер эпохи
        metrics: Метрики для сохранения
        save_path: Путь для сохранения
        filename: Имя файла
    
    Returns:
        Полный путь к сохраненному чекпоинту
    """
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics
    }
    
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint to {filepath}")
    return filepath
