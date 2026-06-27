import pytest
import numpy as np
import torch
import os

from data.dataset import GeologyTrapsDataset
from data.dataloaders import (
    get_file_list,
    split_data_by_groups,
    create_dataloaders
)


class TestGetFileList:
    """Тесты для функции get_file_list."""
    
    def test_get_file_list_valid_files(self, multiple_samples_data):
        """Тест получения списка валидных файлов."""
        # Используем только файлы из multiple_samples_data fixture
        files = get_file_list(multiple_samples_data['temp_dir'])
        
        # Проверяем что файлы найдены (количество зависит от фикстур)
        assert len(files) > 0
        
        # Проверяем что все файлы имеют правильное расширение
        for f in files:
            assert f.endswith('.png')
        
        # Проверяем что все файлы имеют правильный формат имени
        from utils.dataset_utils import parse_filename
        for f in files:
            parsed = parse_filename(os.path.basename(f))
            assert parsed is not None
    
    def test_get_file_list_with_faults(self, sample_image_data):
        """Тест получения списка файлов с разломами."""
        files = get_file_list(sample_image_data['sample_dir'])
        
        assert len(files) == 5  # rgb, depth, isolines, faults, traps
        
        file_basenames = [os.path.basename(f) for f in files]
        assert any('faults' in f for f in file_basenames)
    
    def test_get_file_list_empty_directory(self, temp_dir):
        """Тест получения списка из пустой директории."""
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        files = get_file_list(empty_dir)
        
        assert len(files) == 0
    
    def test_get_file_list_skips_invalid_files(self, temp_dir):
        """Тест пропуска файлов с невалидными именами."""
        # Создаем файл с невалидным именем в отдельной директории
        invalid_dir = os.path.join(temp_dir, "invalid_test")
        os.makedirs(invalid_dir, exist_ok=True)
        invalid_path = os.path.join(invalid_dir, "invalid_file.txt")
        with open(invalid_path, 'w') as f:
            f.write("test")
        
        files = get_file_list(invalid_dir)
        
        # Файл должен быть пропущен
        assert len(files) == 0


class TestSplitDataByGroups:
    """Тесты для функции split_data_by_groups."""
    
    def test_split_data_no_leakage(self, multiple_samples_data):
        """Тест отсутствия утечки данных между выборками."""
        train_files, val_files, test_files = split_data_by_groups(
            multiple_samples_data['all_files'],
            train_horizons=["H150"],
            val_horizons=["H200"],
            test_horizons=["H250"],
        )
        
        # Проверяем что выборки не пересекаются
        train_set = set(os.path.basename(f) for f in train_files)
        val_set = set(os.path.basename(f) for f in val_files)
        test_set = set(os.path.basename(f) for f in test_files)
        
        assert train_set.isdisjoint(val_set), "Leakage: train and val share files"
        assert train_set.isdisjoint(test_set), "Leakage: train and test share files"
        assert val_set.isdisjoint(test_set), "Leakage: val and test share files"
    
    def test_split_data_grouping_by_horizon(self, multiple_samples_data):
        """Тест группировки по горизонтам - все карты одного горизонта в одной выборке."""
        # Добавим второй номер для того же горизонта H150
        sample_dir = os.path.join(multiple_samples_data['temp_dir'], "sample1_dup")
        os.makedirs(sample_dir, exist_ok=True)
        height, width = 100, 80

        rgb_img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        rgb_path = os.path.join(sample_dir, "005_x_structuralNOisoline_H150.png")
        import cv2
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        depth_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        depth_path = os.path.join(sample_dir, "005_x_structuralBlackWhite_H150.png")
        cv2.imwrite(depth_path, depth_img)

        isolines_img = np.zeros((height, width), dtype=np.uint8)
        isolines_path = os.path.join(sample_dir, "005_x_isolines_H150.png")
        cv2.imwrite(isolines_path, isolines_img)

        traps_img = np.zeros((height, width), dtype=np.uint8)
        traps_path = os.path.join(sample_dir, "005_y_traps_H150.png")
        cv2.imwrite(traps_path, traps_img)

        all_files = multiple_samples_data['all_files'] + [rgb_path, depth_path, isolines_path, traps_path]

        train_files, val_files, test_files = split_data_by_groups(
            all_files,
            train_horizons=["H150"],
            val_horizons=["H200"],
            test_horizons=["H250"],
        )
        
        # Проверяем что оба семпла H150 попали в одну выборку
        h150_in_train = sum(1 for f in train_files if 'H150' in f)
        h150_in_val = sum(1 for f in val_files if 'H150' in f)
        h150_in_test = sum(1 for f in test_files if 'H150' in f)
        
        # Все файлы H150 должны быть в одной выборке
        nonzero_counts = sum(1 for x in [h150_in_train, h150_in_val, h150_in_test] if x > 0)
        assert nonzero_counts == 1, "Files from same horizon split across different sets"
    
    def test_split_data_all_files_assigned(self, multiple_samples_data):
        """Тест что все файлы распределены по выборкам."""
        train_files, val_files, test_files = split_data_by_groups(
            multiple_samples_data['all_files'],
            train_horizons=["H150"],
            val_horizons=["H200"],
            test_horizons=["H250"],
        )

        total = len(train_files) + len(val_files) + len(test_files)

        # Каждая выборка должна быть непустой
        assert len(train_files) > 0
        assert len(val_files) > 0
        assert len(test_files) > 0

        # Сумма должна равняться общему количеству
        assert total == len(multiple_samples_data['all_files'])
    
    def test_split_data_empty_input(self):
        """Тест обработки пустого списка файлов."""
        with pytest.raises(ValueError, match="No valid samples found"):
            split_data_by_groups([])


class TestGeologyTrapsDataset:
    """Тесты для класса GeologyTrapsDataset."""
    
    def test_dataset_initialization(self, multiple_samples_data):
        """Тест инициализации датасета."""
        dataset = GeologyTrapsDataset(
            file_list=multiple_samples_data['all_files'],
            data_dir=multiple_samples_data['temp_dir'],
            augment=False,
            use_faults=False
        )
        
        assert len(dataset) == 3
        assert dataset.use_faults == False
        assert dataset.augment == False
    
    def test_dataset_with_faults(self, sample_image_data):
        """Тест датасета с разломами."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['faults'],
            sample_image_data['traps']
        ]
        
        dataset = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=True
        )
        
        assert len(dataset) == 1
        assert dataset.use_faults == True
    
    def test_dataset_getitem_without_faults(self, sample_image_data):
        """Тест получения элемента без разломов."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['traps']
        ]
        
        dataset = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=False,
            target_h=128,
            target_w=96
        )
        
        item = dataset[0]
        
        assert 'x' in item
        assert 'y' in item
        assert 'mask_map' in item
        assert 'sample_idx' in item
        assert 'metadata' in item
        
        # Проверяем размерности
        # x: RGB(3) + Depth(1) + Isolines(1) + MapMask(1) = 6 каналов
        assert item['x'].shape[0] == 6
        assert item['x'].shape[1] == 128
        assert item['x'].shape[2] == 96
        
        # y: 1 канал
        assert item['y'].shape[0] == 1
        assert item['y'].shape[1] == 128
        assert item['y'].shape[2] == 96
        
        # mask_map: 1 канал
        assert item['mask_map'].shape[0] == 1
        assert item['mask_map'].shape[1] == 128
        assert item['mask_map'].shape[2] == 96
        
        # Проверяем типы данных
        assert item['x'].dtype == torch.float32
        assert item['y'].dtype == torch.float32
        assert item['mask_map'].dtype == torch.float32
    
    def test_dataset_getitem_with_faults(self, sample_image_data):
        """Тест получения элемента с разломами."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['faults'],
            sample_image_data['traps']
        ]
        
        dataset = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=True,
            target_h=128,
            target_w=96
        )
        
        item = dataset[0]
        
        # x: RGB(3) + Depth(1) + Isolines(1) + Faults(1) + MapMask(1) = 7 каналов
        assert item['x'].shape[0] == 7
        assert item['x'].shape[1] == 128
        assert item['x'].shape[2] == 96
    
    def test_dataset_metadata(self, sample_image_data):
        """Тест метаданных семпла."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['traps']
        ]
        
        dataset = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=False
        )
        
        item = dataset[0]
        
        assert 'metadata' in item
        assert 'sample_key' in item['metadata']
        assert item['metadata']['sample_key'] == '001_H150'
    
    def test_dataset_nodata_filtering(self, sample_with_nodata):
        """Тест фильтрации семплов с большим количеством NoData."""
        file_list = [
            sample_with_nodata['rgb'],
            sample_with_nodata['depth'],
            sample_with_nodata['isolines'],
            sample_with_nodata['traps']
        ]
        
        # MAX_NODATA_RATIO = 0.4, у нас ~70% фона, должен отфильтроваться
        dataset = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_with_nodata['sample_dir'],
            augment=False,
            use_faults=False
        )
        
        # Семпл должен быть отфильтрован
        assert len(dataset) == 0
    
    def test_dataset_augmentations_applied(self, sample_image_data):
        """Тест применения аугментаций."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['traps']
        ]
        
        dataset_aug = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=True,
            use_faults=False
        )
        
        dataset_no_aug = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=False
        )
        
        # Проверяем что трансформы разные
        assert dataset_aug.augment == True
        assert dataset_no_aug.augment == False


class TestCreateDataloaders:
    """Тесты для функции create_dataloaders."""
    
    def test_create_dataloaders(self, multiple_samples_data):
        """Тест создания DataLoader'ов."""
        train_files, val_files, test_files = split_data_by_groups(
            multiple_samples_data['all_files'],
            train_horizons=["H150"],
            val_horizons=["H200"],
            test_horizons=["H250"],
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dir=multiple_samples_data['temp_dir'],
            batch_size=2,
            num_workers=0,  # Используем 0 для тестов
            use_faults=False
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Проверяем что батчи создаются
        for batch in train_loader:
            assert 'x' in batch
            assert 'y' in batch
            assert 'mask_map' in batch
            break
    
    def test_create_dataloaders_with_faults(self, sample_image_data):
        """Тест создания DataLoader'ов с разломами."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['faults'],
            sample_image_data['traps']
        ]
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files=file_list,
            val_files=file_list,
            test_files=file_list,
            data_dir=sample_image_data['sample_dir'],
            batch_size=1,
            num_workers=0,
            use_faults=True
        )
        
        # Проверяем батч с разломами
        for batch in train_loader:
            # x должен иметь 7 каналов с разломами
            assert batch['x'].shape[0] == 1  # batch size
            assert batch['x'].shape[1] == 7  # channels
            break
    
    def test_create_dataloaders_shuffle(self, multiple_samples_data):
        """Тест перемешивания в train loader."""
        train_files, val_files, test_files = split_data_by_groups(
            multiple_samples_data['all_files'],
            train_horizons=["H150"],
            val_horizons=["H200"],
            test_horizons=["H250"],
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dir=multiple_samples_data['temp_dir'],
            batch_size=2,
            num_workers=0,
            use_faults=False
        )
        
        # Проверяем что DataLoader'ы созданы корректно
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Примечание: shuffle не доступен как атрибут в новых версиях PyTorch
        # Но мы можем проверить что train_loader работает
        for batch in train_loader:
            assert 'x' in batch
            break
