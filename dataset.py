import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict

from settings import settings
from utils.images_utils import (
    load_image, 
    load_grayscale_image, 
    create_binary_mask, 
    create_map_mask, 
    pad_image
)
from utils.cps_utils import (
    read_cps_grid, 
    cps_to_rgb, 
    cps_to_grayscale, 
    cps_to_binary_mask,
)
from utils.augmentations import get_train_transforms, get_val_transforms


class GeologyTrapsDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        data_dir: str = None,
        cps_dir: str = None,
        target_h: int = None,
        target_w: int = None,
        augment: bool = True,
        use_faults: bool = None,
        data_source: str = None
    ):
        self.file_list = file_list
        self.data_source = data_source if data_source is not None else settings.DATA_SOURCE
        self.data_dir = data_dir or str(settings.data_path)
        self.cps_dir = cps_dir or str(settings.cps_path)
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
        print(f"Data source: {self.data_source.upper()}")
        print(f"Samples with fault files: {n_faults} / {len(self.samples)}")
        print(f"Augmentations: {'ON' if augment else 'OFF'}")
        print(f"Target size: {self.target_h}×{self.target_w}")
        
        # Информация о требуемых файлах
        if self.data_source == 'cps':
            n_files = 3 if self.use_faults else 2
            print(f"Required files per sample: {n_files} (structuralNOisoline, [faults], traps)")
        else:
            print(f"Required files per sample: 4 (rgb, depth_norm, [faults], traps)")
    
    def _parse_files(self, file_list: List[str]) -> List[Dict[str, str]]:
        """
        Группирует файлы по семплам.
        
        Формат названий: {number}_{x|y}_{type}_{name}.png
        Примеры:
            - 001_x_structuralNOisoline_H150.png → rgb
            - 001_x_structuralBlackWhite_H150.png → depth_norm
            - 001_x_faults_H150.png → faults
            - 001_y_traps_H150.png → traps
        
        Группировка производится по комбинации {number}_{name}, где:
        - number: номер карты (например, 001, 002)
        - name: название месторождения (например, H150, BZ24)
        
        Все файлы с одинаковыми number и name объединяются в один семпл.
        Разные number для одного месторождения (001_H150, 002_H150) 
        будут РАЗНЫМИ семплами.
        """
        import re
        from pathlib import Path
        
        samples = {}
        
        # Паттерн для парсинга: {number}_{x|y}_{type}_{name}
        pattern = r'^(\d+)_(x|y)_([^_]+)_(.+)$'
        
        for f in file_list:
            if self.data_source == 'cps':
                f_clean = f.replace('.cps', '').replace('.grd', '')
            else:
                f_clean = Path(f).stem  # убираем расширение .png
            
            match = re.match(pattern, f_clean)
            
            if match:
                number = match.group(1)
                role = match.group(2)  # 'x' или 'y'
                file_type = match.group(3)
                name = match.group(4)
                
                # Ключ для группировки: number + name
                key = f"{number}_{name}"
                
                if key not in samples:
                    samples[key] = {}
                
                # Определяем тип файла по role и type
                if role == 'x':
                    if file_type == 'structuralNOisoline':
                        samples[key]['rgb'] = f
                    elif file_type == 'structuralBlackWhite':
                        samples[key]['depth_norm'] = f
                    elif file_type == 'faults':
                        samples[key]['faults'] = f
                elif role == 'y':
                    if file_type == 'traps':
                        samples[key]['traps'] = f
            else:
                print(f"Warning: Skipping file with unexpected format: {f}")
        
        result = []
        for key, paths in samples.items():
            if self.data_source == 'cps':
                # CPS режим: depth_norm не требуется (генерируется из rgb)
                required_keys = ['rgb', 'traps']
                if self.use_faults:
                    required_keys.append('faults')
                
                if all(k in paths for k in required_keys):
                    clean_paths = {
                        k: os.path.join(self.cps_dir, v) for k, v in paths.items() 
                        if k in required_keys or k == 'depth_norm'
                    }
                    result.append(clean_paths)
            else:
                # PNG режим: все 4 файла (или 3 без faults)
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
        
        # Загрузка в зависимости от источника
        if self.data_source == 'cps':
            # CPS режим
            rgb_grid, rgb_meta = read_cps_grid(sample_paths['rgb'])
            traps_grid, traps_meta = read_cps_grid(sample_paths['traps'])
            
            # Конвертация structuralNOisoline → RGB + depth_norm
            rgb_img = cps_to_rgb(rgb_grid, cmap_name='purple_jet')
            depth_img = cps_to_grayscale(rgb_grid, invert=True)  # Из того же грида!
            
            # Faults если нужен
            if self.use_faults and 'faults' in sample_paths:
                faults_grid, _ = read_cps_grid(sample_paths['faults'])
                fault_mask = cps_to_binary_mask(faults_grid)
            else:
                fault_mask = np.zeros_like(depth_img, dtype=np.float32)
            
            # Traps mask
            trap_mask = cps_to_binary_mask(traps_grid)
            
            metadata = {
                'rgb': rgb_meta,
                'traps': traps_meta,
                'source': 'cps'
            }
        else:
            # PNG режим
            rgb_img = load_image(sample_paths['rgb'])
            depth_img = load_grayscale_image(sample_paths['depth_norm'])
            traps_img = load_grayscale_image(sample_paths['traps'])
            
            if self.use_faults and 'faults' in sample_paths:
                faults_img = load_grayscale_image(sample_paths['faults'])
                fault_mask = create_binary_mask(faults_img, invert=False)
            else:
                fault_mask = np.zeros_like(depth_img, dtype=np.float32)
            
            trap_mask = create_binary_mask(traps_img, invert=False)
            
            metadata = {
                'source': 'png'
            }
        
        # Создание масок
        map_mask = create_map_mask(rgb_img)
        
        if self.use_faults:
            depth_mask = map_mask * (1.0 - fault_mask)
        else:
            depth_mask = np.zeros_like(map_mask, dtype=np.float32)
        
        # Нормализация
        rgb_norm = rgb_img.astype(np.float32) / 255.0
        depth_norm = depth_img.astype(np.float32) / 255.0
        
        # Паддинг
        rgb_padded = pad_image(rgb_norm, self.target_h, self.target_w)
        depth_padded = pad_image(depth_norm, self.target_h, self.target_w)
        fault_mask_padded = pad_image(fault_mask, self.target_h, self.target_w)
        trap_mask_padded = pad_image(trap_mask, self.target_h, self.target_w)
        depth_mask_padded = pad_image(depth_mask, self.target_h, self.target_w)
        map_mask_padded = pad_image(map_mask, self.target_h, self.target_w)
        
        # Аугментации
        augmented = self.transforms(
            image=rgb_padded,
            depth=depth_padded,
            faults=fault_mask_padded,
            traps=trap_mask_padded,
            mask_depth=depth_mask_padded,
            mask_map=map_mask_padded
        )
        
        # Извлекаем тензоры
        x_rgb = augmented['image']           # (3, H, W)
        x_depth = augmented['depth']         # (H, W) или (1, H, W)
        x_faults = augmented['faults']       # (H, W)
        
        y_traps = augmented['traps']         # (H, W)
        mask_depth = augmented['mask_depth'] # (H, W)
        mask_map = augmented['mask_map']     # (H, W)
        
        # Добавляем канал для масок если нужно
        if x_depth.dim() == 2:
            x_depth = x_depth.unsqueeze(0)
        if x_faults.dim() == 2:
            x_faults = x_faults.unsqueeze(0)
        if y_traps.dim() == 2:
            y_traps = y_traps.unsqueeze(0)
        if mask_depth.dim() == 2:
            mask_depth = mask_depth.unsqueeze(0)
        if mask_map.dim() == 2:
            mask_map = mask_map.unsqueeze(0)

        # Объединяем входы
        if self.use_faults:
            x_in = torch.cat([x_rgb, x_depth, x_faults], dim=0)  # (5, H, W)
        else:
            x_in = torch.cat([x_rgb, x_depth], dim=0)            # (4, H, W)
        
        return {
            'x': x_in,
            'y': y_traps,
            'mask_depth': mask_depth,
            'mask_map': mask_map,
            'sample_idx': idx,
            'use_faults': self.use_faults,
            'data_source': self.data_source,
            'metadata': metadata
        }