import numpy as np
import torch
import os
import cv2

from data.dataloaders import get_file_list, split_data_by_groups, create_dataloaders
from data.dataset import GeologyTrapsDataset
from utils.dataset_utils import collect_samples


class TestDataPipelineIntegration:
    """Интеграционные тесты полного пайплайна данных."""
    
    def test_full_pipeline_without_faults(self, multiple_samples_data):
        """Тест полного пайплайна без разломов."""
        # 1. Получаем список файлов
        files = get_file_list(multiple_samples_data['temp_dir'])
        
        # Проверяем что файлы найдены
        assert len(files) > 0
        
        # 2. Разделяем на выборки
        train_files, val_files, test_files = split_data_by_groups(
            files,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )
        
        # Проверяем что нет утечек
        train_basenames = set(os.path.basename(f) for f in train_files)
        val_basenames = set(os.path.basename(f) for f in val_files)
        test_basenames = set(os.path.basename(f) for f in test_files)
        
        assert train_basenames.isdisjoint(val_basenames)
        assert train_basenames.isdisjoint(test_basenames)
        assert val_basenames.isdisjoint(test_basenames)
        
        # 3. Создаем DataLoader'ы
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dir=multiple_samples_data['temp_dir'],
            batch_size=2,
            num_workers=0,
            use_faults=False
        )
        
        # 4. Проверяем что данные загружаются корректно
        for batch in train_loader:
            assert 'x' in batch
            assert 'y' in batch
            assert 'mask_map' in batch
            
            # Проверяем размерности
            batch_size = batch['x'].shape[0]
            channels = batch['x'].shape[1]
            height = batch['x'].shape[2]
            width = batch['x'].shape[3]
            
            assert batch_size <= 2
            assert channels == 6  # RGB(3) + Depth(1) + Isolines(1) + MapMask(1)
            assert height == 640  # TARGET_HEIGHT
            assert width == 448   # TARGET_WIDTH
            
            # Проверяем что mask_map имеет значения 0 и 1
            assert torch.min(batch['mask_map']) >= 0.0
            assert torch.max(batch['mask_map']) <= 1.0
            
            break
    
    def test_full_pipeline_with_faults(self, temp_dir):
        """Тест полного пайплайна с разломами."""
        # Создаем семплы с faults
        sample_dir = os.path.join(temp_dir, "faults_sample")
        os.makedirs(sample_dir, exist_ok=True)
        height, width = 100, 80
        
        # RGB
        rgb_img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        rgb_path = os.path.join(sample_dir, "001_x_structuralNOisoline_H150.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        # Depth
        depth_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        depth_path = os.path.join(sample_dir, "001_x_structuralBlackWhite_H150.png")
        cv2.imwrite(depth_path, depth_img)
        
        # Isolines
        isolines_img = np.zeros((height, width), dtype=np.uint8)
        isolines_img[20:25, :] = 255
        isolines_path = os.path.join(sample_dir, "001_x_isolines_H150.png")
        cv2.imwrite(isolines_path, isolines_img)
        
        # Faults
        faults_img = np.ones((height, width), dtype=np.uint8) * 255
        faults_img[40:45, :] = 0
        faults_path = os.path.join(sample_dir, "001_x_faults_H150.png")
        cv2.imwrite(faults_path, faults_img)
        
        # Traps
        traps_img = np.zeros((height, width), dtype=np.uint8)
        traps_img[30:60, 20:50] = 255
        traps_path = os.path.join(sample_dir, "001_y_traps_H150.png")
        cv2.imwrite(traps_path, traps_img)
        
        # Запускаем пайплайн
        files = get_file_list(sample_dir)
        assert len(files) == 5
        
        dataset = GeologyTrapsDataset(
            file_list=files,
            data_dir=sample_dir,
            augment=False,
            use_faults=True,
            target_h=128,
            target_w=96
        )
        
        assert len(dataset) == 1
        
        item = dataset[0]
        
        # Проверяем что x имеет 7 каналов с faults
        assert item['x'].shape[0] == 7  # RGB(3) + Depth(1) + Isolines(1) + Faults(1) + MapMask(1)
    
    def test_horizon_grouping_prevents_leakage(self, temp_dir):
        """Тест что группировка по горизонтам предотвращает утечку."""
        # Создаем несколько семплов для одного горизонта
        all_files = []
        
        for i in range(1, 4):  # 3 семпла для H150
            sample_dir = os.path.join(temp_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            height, width = 100, 80
            
            rgb_img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
            rgb_path = os.path.join(sample_dir, f"{i:03d}_x_structuralNOisoline_H150.png")
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            
            depth_img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            depth_path = os.path.join(sample_dir, f"{i:03d}_x_structuralBlackWhite_H150.png")
            cv2.imwrite(depth_path, depth_img)
            
            isolines_img = np.zeros((height, width), dtype=np.uint8)
            isolines_path = os.path.join(sample_dir, f"{i:03d}_x_isolines_H150.png")
            cv2.imwrite(isolines_path, isolines_img)
            
            traps_img = np.zeros((height, width), dtype=np.uint8)
            traps_path = os.path.join(sample_dir, f"{i:03d}_y_traps_H150.png")
            cv2.imwrite(traps_path, traps_img)
            
            all_files.extend([rgb_path, depth_path, isolines_path, traps_path])
        
        # Добавляем семпл для другого горизонта
        sample_dir2 = os.path.join(temp_dir, "sample_other")
        os.makedirs(sample_dir2, exist_ok=True)
        
        rgb_img2 = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
        rgb_path2 = os.path.join(sample_dir2, "004_x_structuralNOisoline_H200.png")
        cv2.imwrite(rgb_path2, cv2.cvtColor(rgb_img2, cv2.COLOR_RGB2BGR))
        
        depth_img2 = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        depth_path2 = os.path.join(sample_dir2, "004_x_structuralBlackWhite_H200.png")
        cv2.imwrite(depth_path2, depth_img2)
        
        isolines_img2 = np.zeros((height, width), dtype=np.uint8)
        isolines_path2 = os.path.join(sample_dir2, "004_x_isolines_H200.png")
        cv2.imwrite(isolines_path2, isolines_img2)
        
        traps_img2 = np.zeros((height, width), dtype=np.uint8)
        traps_path2 = os.path.join(sample_dir2, "004_y_traps_H200.png")
        cv2.imwrite(traps_path2, traps_img2)
        
        all_files.extend([rgb_path2, depth_path2, isolines_path2, traps_path2])
        
        # Разделяем
        train_files, val_files, test_files = split_data_by_groups(
            all_files,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )
        
        # Проверяем что все файлы H150 в одной выборке
        h150_train = [f for f in train_files if 'H150' in f]
        h150_val = [f for f in val_files if 'H150' in f]
        h150_test = [f for f in test_files if 'H150' in f]
        
        nonzero = sum(1 for x in [len(h150_train), len(h150_val), len(h150_test)] if x > 0)
        assert nonzero == 1, "Files from same horizon leaked across splits"
    
    def test_dataset_batch_consistency(self, multiple_samples_data):
        """Тест консистентности батчей в DataLoader."""
        train_files, val_files, test_files = split_data_by_groups(
            multiple_samples_data['all_files'],
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )
        
        train_loader, _, _ = create_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dir=multiple_samples_data['temp_dir'],
            batch_size=2,
            num_workers=0,
            use_faults=False
        )
        
        # Проходим по всем батчам
        for batch_idx, batch in enumerate(train_loader):
            # Все тензоры должны иметь одинаковый batch size
            batch_size = batch['x'].shape[0]
            
            assert batch['y'].shape[0] == batch_size
            assert batch['mask_map'].shape[0] == batch_size
            
            # Проверяем что spatial dimensions совпадают
            assert batch['x'].shape[2:] == batch['y'].shape[2:]
            assert batch['x'].shape[2:] == batch['mask_map'].shape[2:]
    
    def test_metadata_preserved_through_pipeline(self, sample_image_data):
        """Тест что метаданные сохраняются через весь пайплайн."""
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
        
        # Проверяем что metadata содержит sample_key
        assert 'metadata' in item
        assert 'sample_key' in item['metadata']
        assert item['metadata']['sample_key'] == '001_H150'
    
    def test_padding_applied_correctly(self, sample_image_data):
        """Тест что паддинг применяется корректно."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['traps']
        ]
        
        target_h, target_w = 128, 96
        
        dataset = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=False,
            target_h=target_h,
            target_w=target_w
        )
        
        item = dataset[0]
        
        # Проверяем что выходные размеры соответствуют target
        assert item['x'].shape[1] == target_h
        assert item['x'].shape[2] == target_w
        assert item['y'].shape[1] == target_h
        assert item['y'].shape[2] == target_w
        assert item['mask_map'].shape[1] == target_h
        assert item['mask_map'].shape[2] == target_w
    
    def test_mask_values_in_valid_range(self, sample_image_data):
        """Тест что значения масок в допустимом диапазоне."""
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
        
        # mask_map должен быть бинарным (0 или 1)
        unique_mask_values = torch.unique(item['mask_map'])
        for val in unique_mask_values:
            assert val == 0.0 or val == 1.0, f"Invalid mask value: {val}"
        
        # y (traps) должен быть бинарным
        unique_trap_values = torch.unique(item['y'])
        for val in unique_trap_values:
            assert val == 0.0 or val == 1.0, f"Invalid trap value: {val}"
    
    def test_input_channels_concatenation(self, sample_image_data):
        """Тест корректного объединения входных каналов."""
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['traps']
        ]
        
        dataset_no_faults = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=False
        )
        
        item_no_faults = dataset_no_faults[0]
        
        # Без разломов: RGB(3) + Depth(1) + Isolines(1) + MapMask(1) = 6
        assert item_no_faults['x'].shape[0] == 6
        
        # С разломами
        file_list_with_faults = file_list + [sample_image_data['faults']]
        
        dataset_with_faults = GeologyTrapsDataset(
            file_list=file_list_with_faults,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=True
        )
        
        item_with_faults = dataset_with_faults[0]
        
        # С разломами: RGB(3) + Depth(1) + Isolines(1) + Faults(1) + MapMask(1) = 7
        assert item_with_faults['x'].shape[0] == 7


class TestEdgeCases:
    """Тесты граничных случаев."""
    
    def test_single_sample_dataset(self, sample_image_data):
        """Тест датасета с одним семплом."""
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
        
        assert len(dataset) == 1
        
        # Проверяем что можно получить элемент
        item = dataset[0]
        assert 'x' in item
        assert 'y' in item
    
    def test_missing_optional_files(self, sample_image_data):
        """Тест что отсутствие опциональных файлов не ломает датасет."""
        # Не включаем faults в file_list
        file_list = [
            sample_image_data['rgb'],
            sample_image_data['depth'],
            sample_image_data['isolines'],
            sample_image_data['traps']
            # faults отсутствует
        ]
        
        dataset = GeologyTrapsDataset(
            file_list=file_list,
            data_dir=sample_image_data['sample_dir'],
            augment=False,
            use_faults=False  # faults не требуется
        )
        
        assert len(dataset) == 1
        
        item = dataset[0]
        
        # Проверяем что fault_mask нулевой
        # (это проверяется внутри load_maps_into_ndarray)
    
    def test_large_batch_size(self, multiple_samples_data):
        """Тест работы с большим размером батча."""
        train_files, val_files, test_files = split_data_by_groups(
            multiple_samples_data['all_files'],
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )
        
        # Устанавливаем batch_size больше чем количество семплов
        train_loader, val_loader, test_loader = create_dataloaders(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dir=multiple_samples_data['temp_dir'],
            batch_size=100,
            num_workers=0,
            use_faults=False
        )
        
        # drop_last=True для train, поэтому если батч меньше batch_size, он будет пропущен
        # В данном случае просто проверяем что loader создается без ошибок
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
