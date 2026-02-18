import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict

from settings import settings
from utils import load_image, load_grayscale_image, create_binary_mask, create_map_mask, pad_image
from augmentations import get_train_transforms, get_val_transforms


class GeologyTrapsDataset(Dataset):
    """PyTorch Dataset для сегментации структурных ловушек."""
    
    def __init__(
        self,
        file_list: List[str],
        data_dir: str = None,
        target_h: int = None,
        target_w: int = None,
        augment: bool = True
    ):
        self.file_list = file_list
        self.data_dir = data_dir or str(settings.data_path)
        self.target_h = target_h or settings.TARGET_HEIGHT
        self.target_w = target_w or settings.TARGET_WIDTH
        self.augment = augment
        
        # Выбираем трансформации
        self.transforms = get_train_transforms() if augment else get_val_transforms()
        
        # Группируем файлы по семплам
        self.samples = self._parse_files(file_list)
        print(f"✓ Dataset initialized with {len(self.samples)} samples")
        print(f"✓ Augmentations: {'ON' if augment else 'OFF'}")
        print(f"✓ Target size: {self.target_h}×{self.target_w}")
    
    def _parse_files(self, file_list: List[str]) -> List[Dict[str, str]]:
        """Группирует файлы по семплам."""
        samples = {}
        for f in file_list:
            parts = f.split('_')
            if len(parts) < 4:
                continue
            key = f"{parts[0]}_{parts[-1].replace('.png', '')}"
            subtype = parts[2]
            if key not in samples:
                samples[key] = {}
            
            if 'faults' in subtype:
                samples[key]['faults'] = f
            elif 'structuralBlackWhite' in subtype:
                samples[key]['depth_norm'] = f
            elif 'structuralNOisoline' in subtype:
                samples[key]['rgb'] = f
            elif 'traps' in subtype:
                samples[key]['traps'] = f
        
        result = []
        for key, paths in samples.items():
            if len(paths) == 4:
                result.append({k: os.path.join(self.data_dir, v) for k, v in paths.items()})
        return result
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_paths = self.samples[idx]
        
        # 1. Загрузка
        rgb_img = load_image(sample_paths['rgb'])
        faults_img = load_grayscale_image(sample_paths['faults'])
        depth_img = load_grayscale_image(sample_paths['depth_norm'])
        traps_img = load_grayscale_image(sample_paths['traps'])
        
        # 2. Маски
        fault_mask = create_binary_mask(faults_img, invert=False)
        trap_mask = create_binary_mask(traps_img, invert=False)
        map_mask = create_map_mask(rgb_img)
        depth_mask = map_mask * (1.0 - fault_mask)
        
        # 3. Нормализация
        rgb_norm = rgb_img.astype(np.float32) / 255.0
        depth_norm = depth_img.astype(np.float32) / 255.0
        
        # 4. Паддинг
        rgb_padded = pad_image(rgb_norm, self.target_h, self.target_w)
        depth_padded = pad_image(depth_norm, self.target_h, self.target_w)
        fault_mask_padded = pad_image(fault_mask, self.target_h, self.target_w)
        trap_mask_padded = pad_image(trap_mask, self.target_h, self.target_w)
        depth_mask_padded = pad_image(depth_mask, self.target_h, self.target_w)
        map_mask_padded = pad_image(map_mask, self.target_h, self.target_w)
        
        # 5. Аугментации (Albumentations)
        augmented = self.transforms(
            image=rgb_padded,
            depth=depth_padded,
            faults=fault_mask_padded,
            traps=trap_mask_padded,
            mask_depth=depth_mask_padded,
            mask_map=map_mask_padded
        )
        
        # 6. Извлекаем тензоры
        x_rgb = augmented['image']           # (3, H, W)
        x_depth = augmented['depth']         # (1, H, W) или (H, W)
        x_faults = augmented['faults']       # (H, W)
        
        y_traps = augmented['traps']         # (H, W)
        mask_depth = augmented['mask_depth'] # (H, W)
        mask_map = augmented['mask_map']     # (H, W)
        
        # 🔧 ДОБАВЛЯЕМ КАНАЛ для масок если нужно
        if x_depth.dim() == 2:
            x_depth = x_depth.unsqueeze(0)   # (H, W) → (1, H, W)
        if x_faults.dim() == 2:
            x_faults = x_faults.unsqueeze(0) # (H, W) → (1, H, W)
        if y_traps.dim() == 2:
            y_traps = y_traps.unsqueeze(0)
        if mask_depth.dim() == 2:
            mask_depth = mask_depth.unsqueeze(0)
        if mask_map.dim() == 2:
            mask_map = mask_map.unsqueeze(0)
        
        # Объединяем входы (5 каналов)
        x_in = torch.cat([x_rgb, x_depth, x_faults], dim=0)  # (5, H, W)
        
        return {
            'x': x_in,              # (5, 1248, 512)
            'y': y_traps,           # (1, 1248, 512)
            'mask_depth': mask_depth,
            'mask_map': mask_map,
            'sample_idx': idx
        }