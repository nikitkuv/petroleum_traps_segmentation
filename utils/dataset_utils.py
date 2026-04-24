import numpy as np
from typing import Dict, Tuple, Optional, List
import re
from pathlib import Path
import os

from utils.images_utils import load_image, load_grayscale_image, create_binary_mask


def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    name_no_ext = Path(filename).stem
    pattern = r'^(\d+)_(x|y)_([^_]+)_(.+)$'
    match = re.match(pattern, name_no_ext)
    
    if match:
        return {
            'number': match.group(1),
            'role': match.group(2),
            'type': match.group(3),
            'name': match.group(4)
        }
    return None


def get_sample_key(parsed: Dict[str, str]) -> str:
    return f"{parsed['number']}_{parsed['name']}"


def collect_samples(file_list: List[str]) -> Dict[str, Dict[str, str]]:
    samples = {}
    
    for filename in file_list:
        parsed = parse_filename(filename)
        if not parsed:
            continue
            
        key = get_sample_key(parsed)
        
        if key not in samples:
            samples[key] = {}
            
        file_type = parsed['type']
        role = parsed['role']
        
        if role == 'x':
            if file_type == 'structuralNOisoline':
                samples[key]['rgb'] = filename
            elif file_type == 'structuralBlackWhite':
                samples[key]['depth_norm'] = filename
            elif file_type == 'isolines':
                samples[key]['isolines'] = filename
            elif file_type == 'faults':
                samples[key]['faults'] = filename
        elif role == 'y':
            if file_type == 'traps':
                samples[key]['traps'] = filename
                
    return samples


def resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path) or path.startswith('./') or path.startswith('../'):
        return path
    return os.path.join(base_dir, path)


def load_maps_into_ndarray(
    sample_paths: Dict, 
    use_faults: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb_img = load_image(sample_paths['rgb'])
    depth_img = load_grayscale_image(sample_paths['depth_norm'])
    isolines_img = load_grayscale_image(sample_paths['isolines'])
    traps_img = load_grayscale_image(sample_paths['traps'])
    
    if use_faults and 'faults' in sample_paths:
        faults_img = load_grayscale_image(sample_paths['faults'])
        fault_mask = create_binary_mask(faults_img, invert=False)
    else:
        fault_mask = np.zeros_like(depth_img, dtype=np.float32)
    
    trap_mask = create_binary_mask(traps_img, invert=False)

    return rgb_img, depth_img, isolines_img, trap_mask, fault_mask
