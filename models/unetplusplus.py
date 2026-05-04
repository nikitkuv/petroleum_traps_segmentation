import os
from typing import Optional
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from settings import settings


def load_unetplusplus(
    in_channels: int = None,
    classes: int = 1,
    encoder_name: str = settings.ENCODER_NAME,
    encoder_weights: str = 'imagenet',
    activation: str = None,
    device: str = None,
    use_rgb: bool = None
) -> nn.Module:
    """
    Загружает предобученную модель U-Net++.

    Args:
        in_channels: Количество входных каналов:
            - 3 (depth+isolines+map_mask) без RGB
            - 6 (RGB+depth+isolines+map_mask) с RGB
            - +1 если есть faults
        classes: Количество классов сегментации
        encoder_name: Название энкодера
        encoder_weights: Веса энкодера
        activation: Функция активации
        device: Устройство
        use_rgb: Использовать ли RGB каналы (для корректной инициализации весов)

    Returns:
        Модель U-Net++
    """
    in_channels = in_channels or settings.IN_CHANNELS
    device = device or settings.DEVICE
    use_rgb = use_rgb if use_rgb is not None else settings.USE_RGB

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,  # Всегда 3 для загрузки imagenet весов
        classes=classes,
        activation=activation,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type='scse',
        decoder_dropout=settings.DECODER_DROPOUT
    )

    # Модифицируем первый слой энкодера если количество каналов не стандартное
    if in_channels != 3 and encoder_weights is not None:
        # Создаем новый первый слой с нужным количеством каналов
        old_conv1 = model.encoder.conv1
        new_conv1 = nn.Conv2d(
            in_channels,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )

        # Копируем веса для первых 3 каналов (RGB), остальные инициализируем случайно
        with torch.no_grad():
            if use_rgb:
                # Копируем первые 3 канала из предобученной модели (RGB)
                new_conv1.weight[:, :3, :, :] = old_conv1.weight
                # Остальные каналы (depth, isolines, map_mask, faults) инициализируем как среднее от RGB
                for i in range(3, in_channels):
                    new_conv1.weight[:, i:i+1, :, :] = old_conv1.weight.mean(dim=1, keepdim=True)
            else:
                # Нет RGB каналов - все каналы инициализируем средним от ImageNet весов
                # Это более стабильный подход чем копирование в первые каналы
                rgb_mean_weights = old_conv1.weight.mean(dim=1, keepdim=True)
                for i in range(in_channels):
                    new_conv1.weight[:, i:i+1, :, :] = rgb_mean_weights

        model.encoder.conv1 = new_conv1
    elif in_channels != 3 and encoder_weights is None:
        # Если нет предобученных весов, просто заменяем слой
        old_conv1 = model.encoder.conv1
        new_conv1 = nn.Conv2d(
            in_channels,
            old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        model.encoder.conv1 = new_conv1

    model = model.to(device)
    return model


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = None
) -> nn.Module:
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
