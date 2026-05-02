"""
Фикстуры и общие утилиты для тестов.
"""
import pytest
import os
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict


@pytest.fixture(scope="session")
def temp_dir():
    """Создает временную директорию для тестов."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_image_data(temp_dir):
    """Создает тестовые изображения для одного семпла."""
    sample_dir = os.path.join(temp_dir, "sample_001")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Создаем тестовые изображения
    height, width = 100, 80
    
    # RGB изображение (structuralNOisoline)
    rgb_img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    rgb_path = os.path.join(sample_dir, "001_x_structuralNOisoline_H150.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    
    # Depth изображение (structuralBlackWhite)
    depth_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    depth_path = os.path.join(sample_dir, "001_x_structuralBlackWhite_H150.png")
    cv2.imwrite(depth_path, depth_img)
    
    # Isolines изображение
    isolines_img = np.zeros((height, width), dtype=np.uint8)
    isolines_img[20:25, :] = 255  # Добавляем линии
    isolines_path = os.path.join(sample_dir, "001_x_isolines_H150.png")
    cv2.imwrite(isolines_path, isolines_img)
    
    # Faults изображение (опционально)
    faults_img = np.ones((height, width), dtype=np.uint8) * 255
    faults_img[40:45, :] = 0  # Разломы
    faults_path = os.path.join(sample_dir, "001_x_faults_H150.png")
    cv2.imwrite(faults_path, faults_img)
    
    # Traps изображение (таргет)
    traps_img = np.zeros((height, width), dtype=np.uint8)
    traps_img[30:60, 20:50] = 255  # Ловушки
    traps_path = os.path.join(sample_dir, "001_y_traps_H150.png")
    cv2.imwrite(traps_path, traps_img)
    
    return {
        'rgb': rgb_path,
        'depth': depth_path,
        'isolines': isolines_path,
        'faults': faults_path,
        'traps': traps_path,
        'sample_dir': sample_dir
    }


@pytest.fixture
def multiple_samples_data(temp_dir):
    """Создает несколько семплов для тестирования разделения данных."""
    samples_info = []
    
    # Семпл 1: H150
    sample1_dir = os.path.join(temp_dir, "sample1")
    os.makedirs(sample1_dir, exist_ok=True)
    height, width = 100, 80
    
    rgb_img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    rgb_path = os.path.join(sample1_dir, "001_x_structuralNOisoline_H150.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    
    depth_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    depth_path = os.path.join(sample1_dir, "001_x_structuralBlackWhite_H150.png")
    cv2.imwrite(depth_path, depth_img)
    
    isolines_img = np.zeros((height, width), dtype=np.uint8)
    isolines_img[20:25, :] = 255
    isolines_path = os.path.join(sample1_dir, "001_x_isolines_H150.png")
    cv2.imwrite(isolines_path, isolines_img)
    
    traps_img = np.zeros((height, width), dtype=np.uint8)
    traps_img[30:60, 20:50] = 255
    traps_path = os.path.join(sample1_dir, "001_y_traps_H150.png")
    cv2.imwrite(traps_path, traps_img)
    
    samples_info.append({
        'name': 'H150',
        'number': '001',
        'files': [rgb_path, depth_path, isolines_path, traps_path]
    })
    
    # Семпл 2: H200
    sample2_dir = os.path.join(temp_dir, "sample2")
    os.makedirs(sample2_dir, exist_ok=True)
    
    rgb_img2 = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    rgb_path2 = os.path.join(sample2_dir, "002_x_structuralNOisoline_H200.png")
    cv2.imwrite(rgb_path2, cv2.cvtColor(rgb_img2, cv2.COLOR_RGB2BGR))
    
    depth_img2 = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    depth_path2 = os.path.join(sample2_dir, "002_x_structuralBlackWhite_H200.png")
    cv2.imwrite(depth_path2, depth_img2)
    
    isolines_img2 = np.zeros((height, width), dtype=np.uint8)
    isolines_img2[10:15, :] = 255
    isolines_path2 = os.path.join(sample2_dir, "002_x_isolines_H200.png")
    cv2.imwrite(isolines_path2, isolines_img2)
    
    traps_img2 = np.zeros((height, width), dtype=np.uint8)
    traps_img2[20:40, 10:30] = 255
    traps_path2 = os.path.join(sample2_dir, "002_y_traps_H200.png")
    cv2.imwrite(traps_path2, traps_img2)
    
    samples_info.append({
        'name': 'H200',
        'number': '002',
        'files': [rgb_path2, depth_path2, isolines_path2, traps_path2]
    })
    
    # Семпл 3: H250
    sample3_dir = os.path.join(temp_dir, "sample3")
    os.makedirs(sample3_dir, exist_ok=True)
    
    rgb_img3 = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    rgb_path3 = os.path.join(sample3_dir, "003_x_structuralNOisoline_H250.png")
    cv2.imwrite(rgb_path3, cv2.cvtColor(rgb_img3, cv2.COLOR_RGB2BGR))
    
    depth_img3 = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    depth_path3 = os.path.join(sample3_dir, "003_x_structuralBlackWhite_H250.png")
    cv2.imwrite(depth_path3, depth_img3)
    
    isolines_img3 = np.zeros((height, width), dtype=np.uint8)
    isolines_img3[30:35, :] = 255
    isolines_path3 = os.path.join(sample3_dir, "003_x_isolines_H250.png")
    cv2.imwrite(isolines_path3, isolines_img3)
    
    traps_img3 = np.zeros((height, width), dtype=np.uint8)
    traps_img3[40:70, 30:60] = 255
    traps_path3 = os.path.join(sample3_dir, "003_y_traps_H250.png")
    cv2.imwrite(traps_path3, traps_img3)
    
    samples_info.append({
        'name': 'H250',
        'number': '003',
        'files': [rgb_path3, depth_path3, isolines_path3, traps_path3]
    })
    
    all_files = []
    for sample in samples_info:
        all_files.extend(sample['files'])
    
    return {
        'samples': samples_info,
        'all_files': all_files,
        'temp_dir': temp_dir
    }


@pytest.fixture
def sample_with_nodata(temp_dir):
    """Создает семпл с большим количеством фона (NoData)."""
    sample_dir = os.path.join(temp_dir, "nodata_sample")
    os.makedirs(sample_dir, exist_ok=True)
    
    height, width = 100, 80
    
    # RGB с большим фоном (>40% фона)
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_img[0:30, :] = 5  # Только 30% карты, остальное фон
    rgb_path = os.path.join(sample_dir, "004_x_structuralNOisoline_H300.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    
    depth_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    depth_path = os.path.join(sample_dir, "004_x_structuralBlackWhite_H300.png")
    cv2.imwrite(depth_path, depth_img)
    
    isolines_img = np.zeros((height, width), dtype=np.uint8)
    isolines_path = os.path.join(sample_dir, "004_x_isolines_H300.png")
    cv2.imwrite(isolines_path, isolines_img)
    
    traps_img = np.zeros((height, width), dtype=np.uint8)
    traps_path = os.path.join(sample_dir, "004_y_traps_H300.png")
    cv2.imwrite(traps_path, traps_img)
    
    return {
        'rgb': rgb_path,
        'depth': depth_path,
        'isolines': isolines_path,
        'traps': traps_path,
        'sample_dir': sample_dir
    }
