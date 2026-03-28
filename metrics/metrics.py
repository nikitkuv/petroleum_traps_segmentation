"""
Модуль для вычисления метрик сегментации.
Dice, IoU, Recall, Precision, F1, FP/FN area.
"""

from typing import Dict, Optional
import torch


class MetricsCalculator:
    """
    Калькулятор метрик для сегментации.
    """
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth
    
    def compute_all(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Вычисляет все метрики.
        
        Args:
            predictions: Предсказания модели (логиты или вероятности)
            targets: Целевые маски
            mask: Маска для фильтрации
        
        Returns:
            Словарь с метриками
        """
        # Применяем порог
        if predictions.dim() == 4:
            preds_binary = (torch.sigmoid(predictions) > self.threshold).float()
        else:
            preds_binary = (predictions > self.threshold).float()
        
        targets_binary = targets
        
        # Применяем маску если есть
        if mask is not None:
            preds_binary = preds_binary * mask
            targets_binary = targets_binary * mask
        
        # Intersection и Union
        intersection = (preds_binary * targets_binary).sum(dim=(2, 3))
        union = preds_binary.sum(dim=(2, 3)) + targets_binary.sum(dim=(2, 3))
        
        # IoU
        iou = (intersection + self.smooth) / (union - intersection + self.smooth)
        
        # Dice
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Recall (TP / (TP + FN))
        tp = intersection
        fn = targets_binary.sum(dim=(2, 3)) - intersection
        recall = (tp + self.smooth) / (tp + fn + self.smooth)
        
        # Precision (TP / (TP + FP))
        fp = preds_binary.sum(dim=(2, 3)) - intersection
        precision = (tp + self.smooth) / (tp + fp + self.smooth)
        
        # F1 Score
        f1 = (2.0 * precision * recall + self.smooth) / (precision + recall + self.smooth)
        
        # FP area и FN area
        fp_area = fp.sum() / (targets_binary.shape[0] * targets_binary.shape[2] * targets_binary.shape[3] + self.smooth)
        fn_area = fn.sum() / (targets_binary.shape[0] * targets_binary.shape[2] * targets_binary.shape[3] + self.smooth)
        
        return {
            'iou': iou.mean().item(),
            'dice': dice.mean().item(),
            'recall': recall.mean().item(),
            'precision': precision.mean().item(),
            'f1': f1.mean().item(),
            'fp_area': fp_area.item(),
            'fn_area': fn_area.item()
        }
    
    def compute_iou(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """Вычисляет только IoU."""
        metrics = self.compute_all(predictions, targets, mask)
        return metrics['iou']
    
    def compute_dice(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """Вычисляет только Dice."""
        metrics = self.compute_all(predictions, targets, mask)
        return metrics['dice']
