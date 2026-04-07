import os
import torch
import json
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np

from settings import settings


class GradientNormTracker:
    """
    Трекер для мониторинга нормы градиентов и детектирования аномалий.
    
    Использует running average с экспоненциальным сглаживанием для быстрой
    и эффективной оценки среднего значения и дисперсии.
    """
    
    def __init__(
        self,
        alpha: float = settings.GRAD_ALPHA,  # Коэффициент сглаживания (ближе к 1 = больше вес истории)
        abs_threshold: float = settings.ABS_THRESHOLD,  # Абсолютный порог для аномалий
        std_multiplier: float = settings.STD_MULTIPLIER,  # Количество стандартных отклонений для аномалий
        min_samples_for_std: int = settings.MIN_SAMPLES_FOR_STD,  # Минимум сэмплов перед использованием std-порога
        save_dir: str = settings.GRAD_ANOMALIES_DIR
    ):
        """
        Args:
            alpha: Коэффициент экспоненциального сглаживания (0-1)
            abs_threshold: Абсолютный порог grad_norm для детектирования аномалий
            std_multiplier: Множитель для std (аномалия = mean + std_multiplier * std)
            min_samples_for_std: Минимальное количество сэмплов перед использованием std-порога
            save_dir: Директория для сохранения проблемных батчей
        """
        self.alpha = alpha
        self.abs_threshold = abs_threshold
        self.std_multiplier = std_multiplier
        self.min_samples_for_std = min_samples_for_std
        self.save_dir = save_dir
        
        # Running statistics (Welford's algorithm для численной стабильности)
        self.running_mean = 0.0
        self.running_m2 = 0.0  # Для вычисления дисперсии
        self.count = 0
        
        # История аномалий
        self.anomalies: List[Dict] = []
        
        # Буфер для recent grad norms (для анализа)
        self.recent_grads = deque(maxlen=100)
        
        os.makedirs(save_dir, exist_ok=True)
    
    def update(self, grad_norm: float) -> Tuple[bool, Dict]:
        """
        Обновить статистику и проверить на аномалию.
        
        Args:
            grad_norm: Текущая норма градиента
            
        Returns:
            (is_anomaly, stats_dict)
        """
        self.count += 1
        self.recent_grads.append(grad_norm)
        
        # Welford's online algorithm для running mean и variance
        delta = grad_norm - self.running_mean
        self.running_mean += delta / self.count
        delta2 = grad_norm - self.running_mean
        self.running_m2 += delta * delta2
        
        # Вычисляем running std
        if self.count < 2:
            running_std = 0.0
        else:
            running_std = np.sqrt(self.running_m2 / (self.count - 1))
        
        # Определяем порог
        if self.count < self.min_samples_for_std:
            # Используем только абсолютный порог пока не наберем достаточно статистики
            threshold = self.abs_threshold
        else:
            # Динамический порог на основе статистики
            dynamic_threshold = self.running_mean + self.std_multiplier * running_std
            threshold = min(self.abs_threshold, dynamic_threshold)
        
        # Проверяем на аномалию
        is_anomaly = grad_norm > threshold
        
        stats = {
            'count': self.count,
            'grad_norm': grad_norm,
            'running_mean': self.running_mean,
            'running_std': running_std,
            'threshold': threshold,
            'abs_threshold': self.abs_threshold,
            'dynamic_threshold': dynamic_threshold if self.count >= self.min_samples_for_std else None,
            'is_anomaly': is_anomaly
        }
        
        return is_anomaly, stats
    
    def log_anomaly(
        self,
        epoch: int,
        batch_idx: int,
        grad_norm: float,
        batch_data: Optional[Dict] = None,
        model: Optional[torch.nn.Module] = None,
        save_batch: bool = True
    ):
        """
        Записать информацию об аномалии и опционально сохранить батч.
        
        Args:
            epoch: Номер эпохи
            batch_idx: Индекс батча
            grad_norm: Значение нормы градиента
            batch_data: Данные батча (x, y, mask_map)
            model: Модель (для сохранения состояния)
            save_batch: Сохранять ли сам батч
        """
        anomaly_info = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'grad_norm': grad_norm,
            'running_mean': self.running_mean,
            'running_std': np.sqrt(self.running_m2 / (self.count - 1)) if self.count > 1 else 0.0,
        }
        
        self.anomalies.append(anomaly_info)
        
        if save_batch and batch_data is not None:
            # Сохраняем батч
            anomaly_id = f"epoch{epoch}_batch{batch_idx}_grad{grad_norm:.2f}"
            batch_save_path = os.path.join(self.save_dir, f'{anomaly_id}_batch.pt')
            
            # Сохраняем данные батча на CPU чтобы не занимать GPU память
            batch_to_save = {
                'x': batch_data['x'].cpu(),
                'y': batch_data['y'].cpu(),
            }
            if 'mask_map' in batch_data and batch_data['mask_map'] is not None:
                batch_to_save['mask_map'] = batch_data['mask_map'].cpu()
            
            batch_to_save['metadata'] = anomaly_info
            
            torch.save(batch_to_save, batch_save_path)
            
            # Сохраняем также превью изображения для быстрого просмотра
            if 'x' in batch_data:
                self._save_batch_preview(batch_data['x'], anomaly_id)
        
        # Сохраняем общий список аномалий
        self._save_anomalies_list()
        
        print(f"\n  GRADIENT ANOMALY DETECTED!")
        print(f"  Epoch: {epoch+1}, Batch: {batch_idx}")
        print(f"  Grad Norm: {grad_norm:.2f} (threshold: {anomaly_info['running_mean'] + self.std_multiplier * np.sqrt(self.running_m2 / (self.count - 1)) if self.count > 1 else self.abs_threshold:.2f})")
        print(f"  Running Mean: {self.running_mean:.2f}, Std: {np.sqrt(self.running_m2 / (self.count - 1)) if self.count > 1 else 0.0:.2f}")
        print(f"  Saved to: {self.save_dir}")
    
    def _save_batch_preview(self, x: torch.Tensor, anomaly_id: str):
        """Сохранить превью изображений из батча для быстрого анализа."""
        try:
            import matplotlib.pyplot as plt
            
            # Берем первые 4 изображения из батча
            n_samples = min(4, x.shape[0])
            
            fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))
            if n_samples == 1:
                axes = [axes]
            
            for i in range(n_samples):
                img = x[i].cpu()
                # Если много каналов, берем среднее или первые 3
                if img.shape[0] == 1:
                    img_display = img[0].numpy()
                elif img.shape[0] >= 3:
                    img_display = img[:3].permute(1, 2, 0).numpy()
                    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
                else:
                    img_display = img.mean(dim=0).numpy()
                
                axes[i].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
                axes[i].axis('off')
                axes[i].set_title(f'Sample {i}')
            
            plt.tight_layout()
            preview_path = os.path.join(self.save_dir, f'{anomaly_id}_preview.png')
            plt.savefig(preview_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save batch preview: {e}")
    
    def _save_anomalies_list(self):
        """Сохранить список всех аномалий в JSON."""
        list_path = os.path.join(self.save_dir, 'anomalies_list.json')
        
        # Конвертируем тензоры в списки для JSON
        anomalies_serializable = []
        for a in self.anomalies:
            a_copy = a.copy()
            for k, v in a_copy.items():
                if isinstance(v, torch.Tensor):
                    a_copy[k] = v.item()
                elif isinstance(v, np.ndarray):
                    a_copy[k] = v.tolist()
            anomalies_serializable.append(a_copy)
        
        with open(list_path, 'w') as f:
            json.dump({
                'total_anomalies': len(anomalies_serializable),
                'tracker_config': {
                    'alpha': self.alpha,
                    'abs_threshold': self.abs_threshold,
                    'std_multiplier': self.std_multiplier,
                    'min_samples_for_std': self.min_samples_for_std
                },
                'final_stats': {
                    'running_mean': self.running_mean,
                    'running_std': np.sqrt(self.running_m2 / (self.count - 1)) if self.count > 1 else 0.0,
                    'count': self.count
                },
                'anomalies': anomalies_serializable
            }, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Получить сводную статистику."""
        return {
            'total_batches': self.count,
            'total_anomalies': len(self.anomalies),
            'anomaly_rate': len(self.anomalies) / max(self.count, 1),
            'running_mean': self.running_mean,
            'running_std': np.sqrt(self.running_m2 / (self.count - 1)) if self.count > 1 else 0.0,
            'abs_threshold': self.abs_threshold,
            'recent_grads': list(self.recent_grads)
        }


def compute_per_sample_grad_norms(
    model: torch.nn.Module,
    batch_data: Dict,
    criterion,
    device: str,
    loss_scale: float = 1.0
) -> torch.Tensor:
    """
    Вычислить норму градиента для каждого семпла в батче отдельно.
    
    Это дорогая операция (N forward-backward passes), поэтому используйте
    только постфактум для анализа сохраненных проблемных батчей.
    
    Args:
        model: Модель
        batch_data: Батч данных
        criterion: Функция потерь
        device: Устройство
        loss_scale: Масштаб для loss (если использовался gradient accumulation)
        
    Returns:
        Tensor пер-семпл градиентных норм [batch_size]
    """
    model.zero_grad()
    
    x = batch_data['x'].to(device)
    y = batch_data['y'].to(device)
    mask_map = batch_data.get('mask_map', None)
    if mask_map is not None:
        mask_map = mask_map.to(device)
    
    batch_size = x.shape[0]
    per_sample_grad_norms = torch.zeros(batch_size)
    
    # Градиенты для каждого семпла отдельно
    for i in range(batch_size):
        model.zero_grad()
        
        x_i = x[i:i+1]
        y_i = y[i:i+1]
        mask_map_i = mask_map[i:i+1] if mask_map is not None else None
        
        pred_i = model(x_i)
        loss_i, _ = criterion(pred_i, y_i, mask_map=mask_map_i)
        loss_i = loss_i * loss_scale
        
        loss_i.backward()
        
        # Вычисляем норму градиента
        grad_norm_i = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_i += param.grad.data.norm(2).item() ** 2
        grad_norm_i = grad_norm_i ** 0.5
        
        per_sample_grad_norms[i] = grad_norm_i
    
    return per_sample_grad_norms
