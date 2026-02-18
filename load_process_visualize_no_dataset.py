import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List


TARGET_HEIGHT = 1248
TARGET_WIDTH = 512
BINARY_THRESHOLD = 128
DATA_DIR = './data/images/'


def load_image(path: str) -> np.ndarray:
    """Загружает изображение в RGB."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_grayscale_image(path: str) -> np.ndarray:
    """Загружает изображение в grayscale (1 канал)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

def create_binary_mask(img: np.ndarray, invert: bool = False) -> np.ndarray:
    """Создает бинарную маску 0/1 (float32)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    mask = (gray < BINARY_THRESHOLD).astype(np.float32)
    if invert:
        mask = 1.0 - mask
    return mask

def create_map_mask(structural_img: np.ndarray) -> np.ndarray:
    """Создает маску карты (1 внутри карты, 0 снаружи)."""
    if len(structural_img.shape) == 3:
        is_not_background = np.any(structural_img < 250, axis=2)
    else:
        is_not_background = structural_img < 250
    return is_not_background.astype(np.float32)

def pad_image(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Добавляет паддинг до target_h x target_w."""
    curr_h, curr_w = img.shape[0], img.shape[1]
    
    if curr_h > target_h or curr_w > target_w:
        raise ValueError(f"Image {curr_h}x{curr_w} exceeds target {target_h}x{target_w}")
    
    pad_top = (target_h - curr_h) // 2
    pad_bottom = target_h - curr_h - pad_top
    pad_left = (target_w - curr_w) // 2
    pad_right = target_w - curr_w - pad_left
    
    print(f"    Padding: Top={pad_top}, Bottom={pad_bottom}, Left={pad_left}, Right={pad_right}")
    
    padded = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )
    return padded

def process_sample(file_paths: Dict[str, str], target_h: int, target_w: int) -> Dict[str, np.ndarray]:
    """Обрабатывает один семпл: загрузка -> маски -> паддинг."""
    
    print("  Loading images...")
    rgb_img = load_image(file_paths['rgb'])           # (H, W, 3)
    faults_img = load_grayscale_image(file_paths['faults'])  # (H, W) - 1 канал
    depth_img = load_grayscale_image(file_paths['depth_norm']) # (H, W) - 1 канал
    traps_img = load_grayscale_image(file_paths['traps'])    # (H, W) - 1 канал
    
    # Проверка размеров (для grayscale shape[0:2])
    rgb_shape = (rgb_img.shape[0], rgb_img.shape[1])
    faults_shape = (faults_img.shape[0], faults_img.shape[1])
    depth_shape = (depth_img.shape[0], depth_img.shape[1])
    traps_shape = (traps_img.shape[0], traps_img.shape[1])
    
    print(f"  RGB size: {rgb_shape}")
    print(f"  Faults/Depth/Traps size: {faults_shape}")
    
    if not (rgb_shape == faults_shape == depth_shape == traps_shape):
        raise ValueError(f"Inconsistent shapes: RGB={rgb_shape}, Others={faults_shape}")
    
    orig_h, orig_w = rgb_shape
    print(f"  Original size: {orig_h}x{orig_w} (H×W)")
    
    print("  Creating masks...")
    # fault_mask: разломы (черные) = 1, остальное = 0
    fault_mask = create_binary_mask(faults_img, invert=False)
    # trap_mask: ловушки (черные) = 1, остальное = 0
    trap_mask = create_binary_mask(traps_img, invert=False)
    # map_mask: карта = 1, фон = 0
    map_mask = create_map_mask(rgb_img)
    # depth_mask: валидно для глубины = 1, разломы/фон = 0
    depth_mask = map_mask * (1.0 - fault_mask)
    
    print("  Normalizing images...")
    rgb_norm = rgb_img.astype(np.float32) / 255.0      # (H, W, 3)
    depth_norm = depth_img.astype(np.float32) / 255.0  # (H, W) - 1 канал!
    
    print(f"  Padding to {target_h}x{target_w}...")
    rgb_padded = pad_image(rgb_norm, target_h, target_w)
    depth_padded = pad_image(depth_norm, target_h, target_w)
    fault_mask_padded = pad_image(fault_mask, target_h, target_w)
    trap_mask_padded = pad_image(trap_mask, target_h, target_w)
    depth_mask_padded = pad_image(depth_mask, target_h, target_w)
    map_mask_padded = pad_image(map_mask, target_h, target_w)
    
    # Проверка что все маски 2D (H, W)
    assert len(depth_padded.shape) == 2, f"depth_padded must be 2D, got {depth_padded.shape}"
    assert len(fault_mask_padded.shape) == 2, f"fault_mask_padded must be 2D, got {fault_mask_padded.shape}"
    
    return {
        'x_rgb': rgb_padded,           # (H, W, 3)
        'x_depth': depth_padded,       # (H, W)
        'x_faults': fault_mask_padded, # (H, W)
        'y_traps': trap_mask_padded,   # (H, W)
        'mask_depth': depth_mask_padded,
        'mask_map': map_mask_padded,
        'orig_size': (orig_h, orig_w)
    }

def parse_files(file_list: List[str]) -> List[Dict[str, str]]:
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
            result.append({k: os.path.join(DATA_DIR, v) for k, v in paths.items()})
        else:
            print(f"Warning: Incomplete sample {key}")
    return result


def visualize_sample(processed: Dict, original: Dict, idx: int):
    """Визуализация: 4 картинки в ряду 1, 4 в ряду 2, 3 в ряду 3."""
    fig = plt.figure(figsize=(20, 15))  # Ещё немного увеличил высоту
    
    # ===== ROW 1: 4 изображения =====
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.imshow(original['rgb'])
    ax1.set_title(f'Original RGB\n{original["rgb"].shape[0]}×{original["rgb"].shape[1]}', fontsize=10, pad=15)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.imshow(processed['x_rgb'])
    ax2.set_title(f'Processed RGB (Padded)\n{TARGET_HEIGHT}×{TARGET_WIDTH}', fontsize=10, pad=15)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.imshow(original['depth_norm'], cmap='gray')
    ax3.set_title('Original Depth (B&W)\nBlack=High, White=Low', fontsize=10, pad=15)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.imshow(processed['x_depth'], cmap='gray')
    ax4.set_title('Processed Depth (Padded)\nBlack=High, White=Low', fontsize=10, pad=15)
    ax4.axis('off')
    
    # ===== ROW 2: 4 изображения =====
    ax5 = fig.add_subplot(3, 4, 5)
    ax5.imshow(original['faults'], cmap='gray')
    ax5.set_title('Original Fault Mask\nBlack=Faults', fontsize=10, pad=15)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.imshow(processed['x_faults'], cmap='gray')
    ax6.set_title('Processed Fault Mask\n1=Faults (White), 0=No Faults', fontsize=10, pad=15)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.imshow(original['traps'], cmap='gray')
    ax7.set_title('Original Trap Mask\nBlack=Traps', fontsize=10, pad=15)
    ax7.axis('off')
    
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.imshow(processed['y_traps'], cmap='gray')
    ax8.set_title('Processed Trap Mask (Y)\n1=Traps (White), 0=No Traps', fontsize=10, pad=15)
    ax8.axis('off')
    
    # ===== ROW 3: 3 изображения =====
    ax9 = fig.add_subplot(3, 3, 7)
    ax9.imshow(processed['mask_map'], cmap='gray')
    ax9.set_title('Map Mask\n1=Map (White), 0=Padding (Black)', fontsize=10, pad=15)
    ax9.axis('off')
    
    ax10 = fig.add_subplot(3, 3, 8)
    ax10.imshow(processed['mask_depth'], cmap='gray')
    ax10.set_title('Depth Mask\n1=Valid (White), 0=Faults/Pad (Black)', fontsize=10, pad=15)
    ax10.axis('off')
    
    ax11 = fig.add_subplot(3, 3, 9)
    ax11.axis('off')
    info_text = (
        f"✓ Preprocessing Complete\n\n"
        f"Input: {processed['x_rgb'].shape}\n"
        f"Target: {processed['y_traps'].shape}\n"
        f"Map Valid: {processed['mask_map'].sum():.0f} px\n"
        f"Depth Valid: {processed['mask_depth'].sum():.0f} px\n"
        f"Faults: {processed['x_faults'].sum():.0f} px\n"
        f"Traps: {processed['y_traps'].sum():.0f} px"
    )
    ax11.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=11,
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 🔧 Отступы: top=0.92 даёт больше места для заголовков первого ряда
    plt.subplots_adjust(hspace=0.35, wspace=0.15, top=0.92, bottom=0.05)
    plt.show()


if __name__ == "__main__":
    file_list = [
        '001_x_faults_H150.png', '001_x_structuralBlackWhite_H150.png',
        '001_x_structuralNOisoline_H150.png', '001_y_traps_H150.png',
        '002_x_faults_H150.png', '002_x_structuralBlackWhite_H150.png',
        '002_x_structuralNOisoline_H150.png', '002_y_traps_H150.png',
        '003_x_faults_H150.png', '003_x_structuralBlackWhite_H150.png',
        '003_x_structuralNOisoline_H150.png', '003_y_traps_H150.png'
    ]
    
    print("="*60)
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"TARGET SIZE: {TARGET_HEIGHT}×{TARGET_WIDTH} (H×W)")
    print(f"ORIGINAL SIZE: 1218×505 (H×W)")
    print("="*60)
    
    try:
        samples_paths = parse_files(file_list)
        
        if not samples_paths:
            print("\n✗ No samples found. Check DATA_DIR and file names.")
        else:
            print(f"\n✓ Found {len(samples_paths)} complete samples\n")
            sample = samples_paths[0]
            
            orig_viz = {
                'rgb': load_image(sample['rgb']),
                'depth_norm': load_grayscale_image(sample['depth_norm']),
                'faults': load_grayscale_image(sample['faults']),
                'traps': load_grayscale_image(sample['traps'])
            }
            
            processed = process_sample(sample, TARGET_HEIGHT, TARGET_WIDTH)
            
            visualize_sample(processed, orig_viz, idx=0)
            
            print("\nConverting to PyTorch tensors...")
            
            # RGB: (H, W, 3) → (3, H, W) → (1, 3, H, W)
            x_rgb = torch.from_numpy(processed['x_rgb']).permute(2, 0, 1)
            x_rgb = x_rgb.unsqueeze(0)
            
            # Depth: (H, W) → (1, H, W) → (1, 1, H, W)
            x_depth = torch.from_numpy(processed['x_depth'])
            x_depth = x_depth.unsqueeze(0).unsqueeze(0)
            
            # Faults: (H, W) → (1, H, W) → (1, 1, H, W)
            x_faults = torch.from_numpy(processed['x_faults'])
            x_faults = x_faults.unsqueeze(0).unsqueeze(0)
            
            # Проверка размерностей
            print(f"  x_rgb dimensions:   {x_rgb.dim()}D, shape: {x_rgb.shape}")
            print(f"  x_depth dimensions: {x_depth.dim()}D, shape: {x_depth.shape}")
            print(f"  x_faults dimensions:{x_faults.dim()}D, shape: {x_faults.shape}")
            
            assert x_rgb.dim() == 4, f"x_rgb must be 4D, got {x_rgb.dim()}D"
            assert x_depth.dim() == 4, f"x_depth must be 4D, got {x_depth.dim()}D"
            assert x_faults.dim() == 4, f"x_faults must be 4D, got {x_faults.dim()}D"
            
            # Объединяем (5 каналов: 3 RGB + 1 Depth + 1 Faults)
            x_in = torch.cat([x_rgb, x_depth, x_faults], dim=1)
            
            # Target
            y_out = torch.from_numpy(processed['y_traps'])
            y_out = y_out.unsqueeze(0).unsqueeze(0)
            
            # Маски
            mask_map = torch.from_numpy(processed['mask_map'])
            mask_map = mask_map.unsqueeze(0).unsqueeze(0)
            
            mask_depth = torch.from_numpy(processed['mask_depth'])
            mask_depth = mask_depth.unsqueeze(0).unsqueeze(0)
            
            print(f"\n{'='*60}")
            print(f"✓ Input Tensor:  {x_in.shape} (5 channels)")
            print(f"✓ Target Tensor: {y_out.shape} (1 channel)")
            print(f"✓ Map Mask:      {mask_map.shape}")
            print(f"✓ Depth Mask:    {mask_depth.shape}")
            print(f"{'='*60}")
            print(f"\n✓ Ready for UNet++ training!")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()