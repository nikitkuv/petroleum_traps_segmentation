import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict

from settings import settings
from utils import load_image, load_grayscale_image, create_binary_mask, create_map_mask, pad_image
from augmentations import get_train_transforms, get_val_transforms


class GeologyTrapsDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        data_dir: str = None,
        target_h: int = None,
        target_w: int = None,
        augment: bool = True,
        use_faults: bool = None
    ):
        self.file_list = file_list
        self.data_dir = data_dir or str(settings.data_path)
        self.target_h = target_h or settings.TARGET_HEIGHT
        self.target_w = target_w or settings.TARGET_WIDTH
        self.augment = augment
        self.use_faults = use_faults if use_faults is not None else settings.USE_FAULTS
        
        # Выбираем трансформации
        self.transforms = get_train_transforms() if augment else get_val_transforms()
        
        # Группируем файлы по семплам
        self.samples = self._parse_files(file_list)
        
        # Статистика
        n_faults = sum(1 for s in self.samples if 'faults' in s)
        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"Mode: use_faults={self.use_faults}")
        print(f"Samples with fault files: {n_faults} / {len(self.samples)}")
        print(f"Augmentations: {'ON' if augment else 'OFF'}")
        print(f"Target size: {self.target_h}×{self.target_w}")
    
    def _parse_files(self, file_list: List[str]) -> List[Dict[str, str]]:
        samples = {}
        for f in file_list:
            parts = f.split('_')
            if len(parts) < 4:
                continue
            number = parts[0]
            name = "-".join(parts[3:]).replace('.png', '').replace('.irap', '').replace(' ', '-')
            key = f"{number}_{name}"
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
            if self.use_faults:
                if len(paths) == 4 and 'faults' in paths:
                    result.append({k: os.path.join(self.data_dir, v) for k, v in paths.items()})
            else:
                required_keys = ['rgb', 'depth_norm', 'traps']
                if all(k in paths for k in required_keys):
                    clean_paths = {k: os.path.join(self.data_dir, v) for k, v in paths.items() if k in required_keys}
                    if 'faults' in paths:
                        clean_paths['faults'] = os.path.join(self.data_dir, paths['faults'])
                    result.append(clean_paths)
        return result
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_paths = self.samples[idx]
        
        # 1. Загрузка (обязательные файлы)
        rgb_img = load_image(sample_paths['rgb'])
        depth_img = load_grayscale_image(sample_paths['depth_norm'])
        traps_img = load_grayscale_image(sample_paths['traps'])
        
        # 2. Загрузка fault (только если use_faults=True и файл есть)
        if self.use_faults and 'faults' in sample_paths:
            faults_img = load_grayscale_image(sample_paths['faults'])
            fault_mask = create_binary_mask(faults_img, invert=False)
        else:
            fault_mask = np.zeros_like(depth_img, dtype=np.float32)
        
        # 3. Маски
        trap_mask = create_binary_mask(traps_img, invert=False)
        map_mask = create_map_mask(rgb_img)
        
        if self.use_faults:
            depth_mask = map_mask * (1.0 - fault_mask)
        else:
            depth_mask = np.zeros_like(map_mask, dtype=np.float32)
        
        # 4. Нормализация
        rgb_norm = rgb_img.astype(np.float32) / 255.0
        depth_norm = depth_img.astype(np.float32) / 255.0
        
        # 5. Паддинг
        rgb_padded = pad_image(rgb_norm, self.target_h, self.target_w)
        depth_padded = pad_image(depth_norm, self.target_h, self.target_w)
        fault_mask_padded = pad_image(fault_mask, self.target_h, self.target_w)
        trap_mask_padded = pad_image(trap_mask, self.target_h, self.target_w)
        depth_mask_padded = pad_image(depth_mask, self.target_h, self.target_w)
        map_mask_padded = pad_image(map_mask, self.target_h, self.target_w)
        
        # 6. Аугментации (Albumentations)
        augmented = self.transforms(
            image=rgb_padded,
            depth=depth_padded,
            faults=fault_mask_padded,
            traps=trap_mask_padded,
            mask_depth=depth_mask_padded,
            mask_map=map_mask_padded
        )
        
        # 7. Извлекаем тензоры
        x_rgb = augmented['image']           # (3, H, W)
        x_depth = augmented['depth']         # (H, W) или (1, H, W)
        x_faults = augmented['faults']       # (H, W)
        
        y_traps = augmented['traps']         # (H, W)
        mask_depth = augmented['mask_depth'] # (H, W)
        mask_map = augmented['mask_map']     # (H, W)
        
        # Добавляем канал для масок если нужно
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
        
        # Объединяем входы в зависимости от режима
        if self.use_faults:
            x_in = torch.cat([x_rgb, x_depth, x_faults], dim=0)  # (5, H, W)
        else:
            x_in = torch.cat([x_rgb, x_depth], dim=0)            # (4, H, W)
        
        return {
            'x': x_in,              # (4 или 5, 1248, 512)
            'y': y_traps,           # (1, 1248, 512)
            'mask_depth': mask_depth, 
            'mask_map': mask_map,
            'sample_idx': idx,
            'use_faults': self.use_faults
        }