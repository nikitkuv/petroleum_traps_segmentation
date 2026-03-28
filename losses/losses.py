"""
Модуль с функциями потерь для сегментации геологических ловушек.
Поддержка масок для игнорирования фона и областей под разломами.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class MaskedBCELoss(nn.Module):
    """
    BCE loss с поддержкой масок для игнорирования определенных областей.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Предсказания модели (логиты)
            targets: Целевые маски
            mask: Маска для взвешивания (1 - учитывать, 0 - игнорировать)
        """
        loss = self.bce(predictions, targets)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return loss.sum()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        return loss


class MaskedDiceLoss(nn.Module):
    """
    Dice loss с поддержкой масок.
    """
    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Предсказания модели
            targets: Целевые маски
            mask: Маска для взвешивания
        """
        if self.from_logits:
            predictions = torch.sigmoid(predictions)
        
        # Применяем маску если есть
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
        
        # Вычисляем Dice
        intersection = (predictions * targets).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return dice_loss.mean()


class CombinedLoss(nn.Module):
    """
    Комбинированный лосс: BCE + Dice с масками.
    """
    def __init__(
        self, 
        bce_weight: float = 0.5, 
        dice_weight: float = 0.5,
        use_map_mask: bool = True,
        use_depth_mask: bool = False
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_map_mask = use_map_mask
        self.use_depth_mask = use_depth_mask
        
        self.bce_loss = MaskedBCELoss(reduction='mean')
        self.dice_loss = MaskedDiceLoss(smooth=1.0, from_logits=True)
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        mask_map: Optional[torch.Tensor] = None,
        mask_depth: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            predictions: Предсказания модели (логиты)
            targets: Целевые маски ловушек
            mask_map: Маска карты (игнорируем фон за пределами карты)
            mask_depth: Маска глубины (игнорируем области под разломами)
        
        Returns:
            total_loss, metrics_dict
        """
        # Объединяем маски
        if self.use_map_mask and mask_map is not None:
            combined_mask = mask_map
            if self.use_depth_mask and mask_depth is not None:
                combined_mask = combined_mask * mask_depth
        else:
            combined_mask = None
        
        # Вычисляем лоссы
        bce = self.bce_loss(predictions, targets, combined_mask)
        dice = self.dice_loss(predictions, targets, combined_mask)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        metrics = {
            'total_loss': total_loss.item(),
            'bce_loss': bce.item(),
            'dice_loss': dice.item()
        }
        
        return total_loss, metrics
