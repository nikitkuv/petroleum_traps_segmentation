"""
Юнит тесты для утилит обработки данных (utils/dataset_utils.py, utils/images_utils.py).
"""
import pytest
import numpy as np
import cv2
import os
from pathlib import Path

from utils.dataset_utils import (
    parse_filename,
    get_sample_key,
    collect_samples,
    resolve_path,
    load_maps_into_ndarray
)
from utils.images_utils import (
    load_image,
    load_grayscale_image,
    load_numpy_array,
    create_binary_mask,
    create_map_mask,
    pad_image
)


class TestParseFilename:
    """Тесты для функции parse_filename."""
    
    def test_valid_filename_structural_noisoline(self):
        """Тест парсинга имени файла structuralNOisoline."""
        result = parse_filename("001_x_structuralNOisoline_H150.png")
        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'x'
        assert result['type'] == 'structuralNOisoline'
        assert result['name'] == 'H150'
    
    def test_valid_filename_structural_blackwhite(self):
        """Тест парсинга имени файла structuralBlackWhite."""
        result = parse_filename("002_x_structuralBlackWhite_Ach3-2-1_toptop1.png")
        assert result is not None
        assert result['number'] == '002'
        assert result['role'] == 'x'
        assert result['type'] == 'structuralBlackWhite'
        assert result['name'] == 'Ach3-2-1_toptop1'
    
    def test_valid_filename_isolines(self):
        """Тест парсинга имени файла isolines."""
        result = parse_filename("003_x_isolines_BZ24.png")
        assert result is not None
        assert result['number'] == '003'
        assert result['role'] == 'x'
        assert result['type'] == 'isolines'
        assert result['name'] == 'BZ24'
    
    def test_valid_filename_faults(self):
        """Тест парсинга имени файла faults."""
        result = parse_filename("001_x_faults_H150.png")
        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'x'
        assert result['type'] == 'faults'
        assert result['name'] == 'H150'
    
    def test_valid_filename_traps(self):
        """Тест парсинга имени файла traps (target)."""
        result = parse_filename("001_y_traps_H150.png")
        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'y'
        assert result['type'] == 'traps'
        assert result['name'] == 'H150'
    
    def test_invalid_filename_no_extension(self):
        """Тест невалидного имени без расширения - функция работает и без .png."""
        # Функция использует Path.stem, поэтому расширение не обязательно
        result = parse_filename("001_x_structuralNOisoline_H150")
        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'x'
        assert result['type'] == 'structuralNOisoline'
        assert result['name'] == 'H150'
    
    def test_invalid_filename_wrong_format(self):
        """Тест невалидного имени с неправильным форматом."""
        result = parse_filename("invalid_file.png")
        assert result is None
    
    def test_invalid_filename_missing_parts(self):
        """Тест невалидного имени с недостающими частями."""
        result = parse_filename("001_H150.png")
        assert result is None
    
    def test_filename_with_npy_extension(self):
        """Тест парсинга имени файла с расширением .npy."""
        result = parse_filename("001_x_structuralBlackWhite_H150.npy")
        assert result is not None
        assert result['number'] == '001'
        assert result['role'] == 'x'
        assert result['type'] == 'structuralBlackWhite'
        assert result['name'] == 'H150'


class TestGetSampleKey:
    """Тесты для функции get_sample_key."""
    
    def test_sample_key_generation(self):
        """Тест генерации ключа семпла."""
        parsed = {
            'number': '001',
            'role': 'x',
            'type': 'structuralNOisoline',
            'name': 'H150'
        }
        key = get_sample_key(parsed)
        assert key == '001_H150'
    
    def test_sample_key_different_number(self):
        """Тест генерации ключа для другого номера."""
        parsed = {
            'number': '002',
            'role': 'x',
            'type': 'structuralNOisoline',
            'name': 'H150'
        }
        key = get_sample_key(parsed)
        assert key == '002_H150'
    
    def test_sample_key_complex_name(self):
        """Тест генерации ключа для сложного имени горизонта."""
        parsed = {
            'number': '001',
            'role': 'x',
            'type': 'structuralNOisoline',
            'name': 'Ach3-2-1_toptop1'
        }
        key = get_sample_key(parsed)
        assert key == '001_Ach3-2-1_toptop1'


class TestCollectSamples:
    """Тесты для функции collect_samples."""
    
    def test_collect_complete_sample(self, sample_image_data):
        """Тест сбора полного семпла."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['traps']
        ]
        
        samples = collect_samples(file_list)
        
        assert len(samples) == 1
        assert '001_H150' in samples
        assert 'rgb' in samples['001_H150']
        assert 'depth_norm' in samples['001_H150']
        assert 'isolines' in samples['001_H150']
        assert 'traps' in samples['001_H150']
    
    def test_collect_sample_with_faults(self, sample_image_data):
        """Тест сбора семпла с разломами."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['faults'],
            sample_image_data['traps']
        ]
        
        samples = collect_samples(file_list)
        
        assert len(samples) == 1
        assert '001_H150' in samples
        assert 'faults' in samples['001_H150']
    
    def test_collect_multiple_samples(self, multiple_samples_data):
        """Тест сбора нескольких семплов."""
        samples = collect_samples(multiple_samples_data['all_files'])
        
        assert len(samples) == 3
        assert '001_H150' in samples
        assert '002_H200' in samples
        assert '003_H250' in samples
    
    def test_collect_incomplete_sample(self):
        """Тест сбора неполного семпла (отсутствуют файлы)."""
        file_list = [
            "/fake/path/001_x_structuralNOisoline_H150.png",
            "/fake/path/001_x_isolines_H150.png"
            # Отсутствуют depth_norm и traps
        ]
        
        samples = collect_samples(file_list)
        
        assert len(samples) == 1
        assert '001_H150' in samples
        assert 'rgb' in samples['001_H150']
        assert 'isolines' in samples['001_H150']
        assert 'depth_norm' not in samples['001_H150']
        assert 'traps' not in samples['001_H150']
    
    def test_collect_skips_invalid_filenames(self):
        """Тест пропуска файлов с невалидными именами."""
        file_list = [
            "/fake/path/001_x_structuralNOisoline_H150.png",
            "/fake/path/invalid_file.png",
            "/fake/path/001_y_traps_H150.png"
        ]
        
        samples = collect_samples(file_list)
        
        assert len(samples) == 1
        assert '001_H150' in samples


class TestResolvePath:
    """Тесты для функции resolve_path."""
    
    def test_resolve_relative_path(self):
        """Тест разрешения относительного пути."""
        path = "file.png"
        base_dir = "/data"
        resolved = resolve_path(path, base_dir)
        assert resolved == "/data/file.png"
    
    def test_resolve_absolute_path(self):
        """Тест разрешения абсолютного пути."""
        path = "/absolute/path/file.png"
        base_dir = "/data"
        resolved = resolve_path(path, base_dir)
        assert resolved == "/absolute/path/file.png"
    
    def test_resolve_path_with_dot_slash(self):
        """Тест разрешения пути с ./."""
        path = "./relative/file.png"
        base_dir = "/data"
        resolved = resolve_path(path, base_dir)
        assert resolved == "./relative/file.png"
    
    def test_resolve_path_with_parent_dir(self):
        """Тест разрешения пути с ../."""
        path = "../parent/file.png"
        base_dir = "/data"
        resolved = resolve_path(path, base_dir)
        assert resolved == "../parent/file.png"


class TestLoadMapsIntoNdarray:
    """Тесты для функции load_maps_into_ndarray."""
    
    def test_load_without_faults(self, sample_image_data):
        """Тест загрузки карт без разломов."""
        sample_paths = {
            'rgb': sample_image_data['rgb'],
            'depth_norm': sample_image_data['depth'],
            'isolines': sample_image_data['isolines'],
            'traps': sample_image_data['traps']
        }
        
        rgb, depth, isolines, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=False
        )
        
        assert rgb.shape == (100, 80, 3)
        assert depth.shape == (100, 80)
        assert isolines.shape == (100, 80)
        assert traps.shape == (100, 80)
        assert faults.shape == (100, 80)
        assert np.all(faults == 0)  # Faults должна быть нулевой
    
    def test_load_with_faults(self, sample_image_data):
        """Тест загрузки карт с разломами."""
        sample_paths = {
            'rgb': sample_image_data['rgb'],
            'depth_norm': sample_image_data['depth'],
            'isolines': sample_image_data['isolines'],
            'faults': sample_image_data['faults'],
            'traps': sample_image_data['traps']
        }
        
        rgb, depth, isolines, traps, faults = load_maps_into_ndarray(
            sample_paths, use_faults=True
        )
        
        assert rgb.shape == (100, 80, 3)
        assert depth.shape == (100, 80)
        assert isolines.shape == (100, 80)
        assert traps.shape == (100, 80)
        assert faults.shape == (100, 80)
        # Проверяем что fault_mask имеет правильные значения (инвертировано)
        assert np.any(faults > 0)  # Есть области не-разломов
        assert np.any(faults == 0)  # Есть области разломов


class TestImageUtils:
    """Тесты для утилит изображений."""
    
    def test_create_binary_mask_grayscale(self):
        """Тест создания бинарной маски из grayscale изображения."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:80, 20:80] = 200  # Белая область
        
        mask = create_binary_mask(img, invert=False)
        
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32
        assert np.all(mask[20:80, 20:80] == 1.0)
        assert np.all(mask[0:20, :] == 0.0)
    
    def test_create_binary_mask_rgb(self):
        """Тест создания бинарной маски из RGB изображения."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = 200  # Белая область
        
        mask = create_binary_mask(img, invert=False)
        
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32
    
    def test_create_binary_mask_inverted(self):
        """Тест создания инвертированной бинарной маски."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[20:80, 20:80] = 200
        
        mask = create_binary_mask(img, invert=True)
        
        assert np.all(mask[20:80, 20:80] == 0.0)
        assert np.all(mask[0:20, :] == 1.0)
    
    def test_create_map_mask(self):
        """Тест создания маски карты."""
        # Изображение с фоном (<10) и картой (>10)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:90, 10:90] = 50  # Карта
        
        map_mask = create_map_mask(img)
        
        assert map_mask.shape == (100, 100)
        assert map_mask.dtype == np.float32
        assert np.all(map_mask[10:90, 10:90] == 1.0)
        assert np.all(map_mask[0:10, :] == 0.0)
    
    def test_pad_image_smaller(self):
        """Тест паддинга меньшего изображения."""
        img = np.ones((50, 40), dtype=np.float32)
        
        padded = pad_image(img, 60, 50)
        
        assert padded.shape == (60, 50)
        # Проверяем что оригинальные значения сохранились в центре
        assert np.all(padded[5:55, 5:45] == 1.0)
    
    def test_pad_image_exact_size(self):
        """Тест паддинга изображения точного размера."""
        img = np.ones((60, 50), dtype=np.float32)
        
        padded = pad_image(img, 60, 50)
        
        assert padded.shape == (60, 50)
        assert np.all(padded == 1.0)
    
    def test_pad_image_too_large_raises_error(self):
        """Тест ошибки при паддинге слишком большого изображения."""
        img = np.ones((100, 80), dtype=np.float32)
        
        with pytest.raises(ValueError):
            pad_image(img, 60, 50)
    
    def test_load_numpy_array(self, temp_dir):
        """Тест загрузки numpy массива."""
        arr = np.random.rand(100, 100).astype(np.float32)
        npy_path = os.path.join(temp_dir, "test.npy")
        np.save(npy_path, arr)
        
        loaded = load_numpy_array(npy_path)
        
        assert loaded.shape == arr.shape
        assert np.allclose(loaded, arr)
    
    def test_load_numpy_array_not_found(self):
        """Тест ошибки при загрузке несуществующего numpy файла."""
        with pytest.raises(FileNotFoundError):
            load_numpy_array("/nonexistent/path/file.npy")
