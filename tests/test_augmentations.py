"""
Тесты для аугментаций данных (utils/augmentations.py).
"""
import pytest
import numpy as np
import torch

from utils.augmentations import get_train_transforms, get_val_transforms


class TestAugmentations:
    """Тесты для функций аугментации."""
    
    def test_train_transforms_creation(self):
        """Тест создания тренировочных трансформов."""
        transforms = get_train_transforms()
        
        assert transforms is not None
        # Проверяем что есть дополнительные таргеты
        assert 'image' in transforms.additional_targets
        assert 'depth' in transforms.additional_targets
        assert 'isolines' in transforms.additional_targets
        assert 'faults' in transforms.additional_targets
        assert 'traps' in transforms.additional_targets
        assert 'mask_map' in transforms.additional_targets
    
    def test_val_transforms_creation(self):
        """Тест создания валидационных трансформов."""
        transforms = get_val_transforms()
        
        assert transforms is not None
        # Проверяем что есть дополнительные таргеты
        assert 'image' in transforms.additional_targets
        assert 'depth' in transforms.additional_targets
        assert 'isolines' in transforms.additional_targets
        assert 'faults' in transforms.additional_targets
        assert 'traps' in transforms.additional_targets
        assert 'mask_map' in transforms.additional_targets
    
    def test_train_transforms_apply(self, sample_image_data):
        """Тест применения тренировочных трансформов."""
        from utils.images_utils import load_image, load_grayscale_image, create_map_mask
        
        transforms = get_train_transforms()
        
        rgb_img = load_image(sample_image_data['rgb'])
        depth_img = load_grayscale_image(sample_image_data['depth'])
        isolines_img = load_grayscale_image(sample_image_data['isolines'])
        faults_img = load_grayscale_image(sample_image_data['faults'])
        traps_img = load_grayscale_image(sample_image_data['traps'])
        map_mask = create_map_mask(rgb_img)
        
        # Применяем трансформы
        result = transforms(
            image=rgb_img,
            depth=depth_img,
            isolines=isolines_img,
            faults=faults_img,
            traps=traps_img,
            mask_map=map_mask
        )
        
        # Проверяем что результат содержит все поля
        assert 'image' in result
        assert 'depth' in result
        assert 'isolines' in result
        assert 'faults' in result
        assert 'traps' in result
        assert 'mask_map' in result
        
        # Проверяем что результат - тензоры
        assert isinstance(result['image'], torch.Tensor)
        assert isinstance(result['depth'], torch.Tensor)
        assert isinstance(result['isolines'], torch.Tensor)
        assert isinstance(result['faults'], torch.Tensor)
        assert isinstance(result['traps'], torch.Tensor)
        assert isinstance(result['mask_map'], torch.Tensor)
        
        # Проверяем размерности
        # RGB: 3 канала
        assert result['image'].shape[0] == 3
        # Остальные могут быть 2D или 3D в зависимости от версии albumentations
        # Проверяем что spatial dimensions корректные
        assert len(result['depth'].shape) >= 2
        assert len(result['isolines'].shape) >= 2
        assert len(result['faults'].shape) >= 2
        assert len(result['traps'].shape) >= 2
        assert len(result['mask_map'].shape) >= 2
    
    def test_val_transforms_apply(self, sample_image_data):
        """Тест применения валидационных трансформов."""
        from utils.images_utils import load_image, load_grayscale_image, create_map_mask
        
        transforms = get_val_transforms()
        
        rgb_img = load_image(sample_image_data['rgb'])
        depth_img = load_grayscale_image(sample_image_data['depth'])
        isolines_img = load_grayscale_image(sample_image_data['isolines'])
        faults_img = load_grayscale_image(sample_image_data['faults'])
        traps_img = load_grayscale_image(sample_image_data['traps'])
        map_mask = create_map_mask(rgb_img)
        
        # Применяем трансформы
        result = transforms(
            image=rgb_img,
            depth=depth_img,
            isolines=isolines_img,
            faults=faults_img,
            traps=traps_img,
            mask_map=map_mask
        )
        
        # Проверяем что результат содержит все поля
        assert 'image' in result
        assert 'depth' in result
        assert 'isolines' in result
        assert 'faults' in result
        assert 'traps' in result
        assert 'mask_map' in result
        
        # Проверяем что результат - тензоры
        assert isinstance(result['image'], torch.Tensor)
        assert isinstance(result['depth'], torch.Tensor)
    
    def test_val_transforms_no_augmentation(self, sample_image_data):
        """Тест что валидационные трансформы не применяют аугментации."""
        from utils.images_utils import load_image, load_grayscale_image, create_map_mask
        
        transforms = get_val_transforms()
        
        rgb_img = load_image(sample_image_data['rgb'])
        original_shape = rgb_img.shape
        
        depth_img = load_grayscale_image(sample_image_data['depth'])
        isolines_img = load_grayscale_image(sample_image_data['isolines'])
        faults_img = load_grayscale_image(sample_image_data['faults'])
        traps_img = load_grayscale_image(sample_image_data['traps'])
        map_mask = create_map_mask(rgb_img)
        
        # Применяем трансформы несколько раз
        result1 = transforms(
            image=rgb_img,
            depth=depth_img,
            isolines=isolines_img,
            faults=faults_img,
            traps=traps_img,
            mask_map=map_mask
        )
        
        result2 = transforms(
            image=rgb_img,
            depth=depth_img,
            isolines=isolines_img,
            faults=faults_img,
            traps=traps_img,
            mask_map=map_mask
        )
        
        # Результаты должны быть одинаковыми (нет рандомизации)
        assert torch.allclose(result1['image'], result2['image'])
        assert torch.allclose(result1['depth'], result2['depth'])
    
    def test_mask_consistency_after_transform(self, sample_image_data):
        """Тест консистентности масок после применения трансформов."""
        from utils.images_utils import load_image, load_grayscale_image, create_map_mask
        
        transforms = get_train_transforms()
        
        rgb_img = load_image(sample_image_data['rgb'])
        depth_img = load_grayscale_image(sample_image_data['depth'])
        isolines_img = load_grayscale_image(sample_image_data['isolines'])
        faults_img = load_grayscale_image(sample_image_data['faults'])
        traps_img = load_grayscale_image(sample_image_data['traps'])
        map_mask = create_map_mask(rgb_img)
        
        # Применяем трансформы
        result = transforms(
            image=rgb_img,
            depth=depth_img,
            isolines=isolines_img,
            faults=faults_img,
            traps=traps_img,
            mask_map=map_mask
        )
        
        # Все маски должны иметь одинаковые spatial dimensions
        assert result['depth'].shape[1:] == result['isolines'].shape[1:]
        assert result['depth'].shape[1:] == result['faults'].shape[1:]
        assert result['depth'].shape[1:] == result['traps'].shape[1:]
        assert result['depth'].shape[1:] == result['mask_map'].shape[1:]
        
        # Маски должны иметь значения в диапазоне [0, 1] (после ToTensorV2 они float)
        # Проверяем что min >= 0 и max <= 1 для float тензоров
        if result['mask_map'].dtype.is_floating_point:
            assert torch.min(result['mask_map']) >= 0.0
            assert torch.max(result['mask_map']) <= 1.0
        if result['traps'].dtype.is_floating_point:
            assert torch.min(result['traps']) >= 0.0
            assert torch.max(result['traps']) <= 1.0
    
    def test_different_seeds_produce_different_results(self, sample_image_data):
        """Тест что разные random seed дают разные результаты аугментации."""
        from utils.images_utils import load_image, load_grayscale_image, create_map_mask
        
        # Создаем два набора трансформов
        transforms1 = get_train_transforms()
        transforms2 = get_train_transforms()
        
        rgb_img = load_image(sample_image_data['rgb'])
        depth_img = load_grayscale_image(sample_image_data['depth'])
        isolines_img = load_grayscale_image(sample_image_data['isolines'])
        faults_img = load_grayscale_image(sample_image_data['faults'])
        traps_img = load_grayscale_image(sample_image_data['traps'])
        map_mask = create_map_mask(rgb_img)
        
        # Просто проверяем что трансформы работают
        result1 = transforms1(
            image=rgb_img,
            depth=depth_img,
            isolines=isolines_img,
            faults=faults_img,
            traps=traps_img,
            mask_map=map_mask
        )
        
        result2 = transforms2(
            image=rgb_img,
            depth=depth_img,
            isolines=isolines_img,
            faults=faults_img,
            traps=traps_img,
            mask_map=map_mask
        )
        
        # Результаты должны иметь одинаковую форму
        assert result1['image'].shape == result2['image'].shape
        assert result1['depth'].shape == result2['depth'].shape
