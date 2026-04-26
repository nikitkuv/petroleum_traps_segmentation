import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict

from settings import settings
from utils.images_utils import create_map_mask, pad_image
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.dataset_utils import load_maps_into_ndarray, collect_samples, resolve_path


class GeologyTrapsDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        data_dir: str = None,
        target_h: int = None,
        target_w: int = None,
        augment: bool = True,
        use_faults: bool = None,
    ):
        self.file_list = file_list
        self.data_dir = data_dir or str(settings.data_path)
        self.target_h = target_h or settings.TARGET_HEIGHT
        self.target_w = target_w or settings.TARGET_WIDTH
        self.augment = augment
        self.use_faults = use_faults if use_faults is not None else settings.USE_FAULTS
        
        self.transforms = get_train_transforms() if augment else get_val_transforms()
        print(f"Transforms: {self.transforms}")
        print(f"Len transforms: {len(self.transforms)}")
        
        self.samples = self._parse_files(file_list)
        
        n_faults = sum(1 for s in self.samples if 'faults' in s)
        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"Mode: use_faults={self.use_faults}")
        print(f"Samples with fault files: {n_faults} / {len(self.samples)}")
        print(f"Augmentations: {'ON' if augment else 'OFF'}")
        print(f"Target size: {self.target_h}×{self.target_w}")
        print()
        
        n_files = 6 if self.use_faults else 5
        print(f"Required files per sample: {n_files} (rgb, depth_norm, isolines, [faults], traps)")

    def _parse_files(self, file_list: List[str]) -> List[Dict[str, str]]:
        samples = collect_samples(file_list)
        result = []

        for key, paths in samples.items():
            required_keys = ['rgb', 'depth_norm', 'isolines', 'traps']

            if not all(k in paths for k in required_keys):
                continue

            clean_paths = {
                k: resolve_path(paths[k], self.data_dir)
                for k in required_keys
            }
            
            # Добавляем faults только если он есть
            if self.use_faults and 'faults' in paths:
                clean_paths['faults'] = resolve_path(paths['faults'], self.data_dir)

            clean_paths['_sample_key'] = key
            result.append(clean_paths)

        return result
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_paths = self.samples[idx]
        
        rgb_img, depth_img, isolines_img, trap_mask, fault_mask = load_maps_into_ndarray(
            sample_paths=sample_paths, 
            use_faults=self.use_faults
        )

        sample_key = sample_paths.get('_sample_key', f"sample_{idx}")
        metadata = {'sample_key': sample_key}
        
        # Создаем маску карты
        map_mask = create_map_mask(rgb_img)
        
        # Нормализация
        rgb_norm = rgb_img.astype(np.float32) / 255.0
        depth_norm = depth_img.astype(np.float32) / 255.0
        isolines_norm = isolines_img.astype(np.float32) / 255.0
        
        # Паддинг
        rgb_padded = pad_image(rgb_norm, self.target_h, self.target_w)
        depth_padded = pad_image(depth_norm, self.target_h, self.target_w)
        isolines_padded = pad_image(isolines_norm, self.target_h, self.target_w)
        fault_mask_padded = pad_image(fault_mask, self.target_h, self.target_w)
        trap_mask_padded = pad_image(trap_mask, self.target_h, self.target_w)
        map_mask_padded = pad_image(map_mask, self.target_h, self.target_w)
        
        # Аугментации
        augmented = self.transforms(
            image=rgb_padded,
            depth=depth_padded,
            isolines=isolines_padded,
            faults=fault_mask_padded,
            traps=trap_mask_padded,
            mask_map=map_mask_padded
        )
        
        # Извлекаем тензоры
        x_rgb = augmented['image']           
        x_depth = augmented['depth']         
        x_isolines = augmented['isolines']   
        x_faults = augmented['faults']       
        
        y_traps = augmented['traps']         
        mask_map = augmented['mask_map']     
        
        # Добавляем канал для масок если нужно
        if x_depth.dim() == 2:
            x_depth = x_depth.unsqueeze(0)
        if x_isolines.dim() == 2:
            x_isolines = x_isolines.unsqueeze(0)
        if x_faults.dim() == 2:
            x_faults = x_faults.unsqueeze(0)
        if y_traps.dim() == 2:
            y_traps = y_traps.unsqueeze(0)
        if mask_map.dim() == 2:
            mask_map = mask_map.unsqueeze(0)

        # Объединяем входы
        if self.use_faults:
            x_in = torch.cat([x_rgb, x_depth, x_isolines, x_faults], dim=0)
        else:
            x_in = torch.cat([x_rgb, x_depth, x_isolines], dim=0)
        
        return {
            'x': x_in,
            'y': y_traps,
            'mask_map': mask_map,
            'sample_idx': idx,
            'use_faults': self.use_faults,
            'metadata': metadata
        }
    