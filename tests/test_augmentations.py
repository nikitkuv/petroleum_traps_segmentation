import pytest
import numpy as np
import albumentations as A

from utils.augmentations import get_train_transforms, get_val_transforms


# Фикстура для создания случайных данных, имитирующих входные данные пайплайна
@pytest.fixture
def sample_data():
    """
    Создает словарь с данными, соответствующими ожидаемым ключам в датасете.
    Изображения имеют форму (H, W, C), маски - (H, W).
    """
    h, w = 256, 256
    data = {
        "image": np.random.randint(0, 255, (h, w, 3), dtype=np.uint8),       # RGB изображение
        "depth": np.random.rand(h, w).astype(np.float32),                   # Карта глубины (одноканальная)
        "faults": np.random.randint(0, 2, (h, w), dtype=np.uint8),          # Маска разломов
        "traps": np.random.randint(0, 2, (h, w), dtype=np.uint8),           # Маска ловушек
        "mask_depth": np.random.randint(0, 2, (h, w), dtype=np.uint8),      # Маска валидности глубины
        "mask_map": np.random.randint(0, 2, (h, w), dtype=np.uint8),        # Общая маска области
    }
    return data


class TestTrainTransforms:
    """Тесты для тренировочных аугментаций"""

    def test_transforms_creation(self):
        """Проверка, что трансформации успешно создаются и являются экземпляром A.Compose"""
        transforms = get_train_transforms()
        assert isinstance(transforms, A.Compose)
        assert len(transforms.transforms) > 0

    def test_required_transforms_present(self):
        """
        Проверка наличия критически важных трансформаций.
        Не зависит от конкретных шумов или яркости, которые могут меняться.
        """
        transforms = get_train_transforms()
        transform_types = [type(t).__name__ for t in transforms.transforms]

        # Обязательные элементы пайплайна
        assert "HorizontalFlip" in transform_types or "VerticalFlip" in transform_types, \
            "Ожидается наличие хотя бы одного типа флипа для аугментации"
        assert "ToTensorV2" in transform_types, "ToTensorV2 обязателен для конвертации в тензор PyTorch"

    def test_additional_targets_config(self):
        """Проверка, что все необходимые ключи настроены в additional_targets"""
        transforms = get_train_transforms()

        # albumentations автоматически добавляет 'image': 'image', поэтому проверяем вхождение
        expected_targets = {
            'depth': 'image',
            'faults': 'mask',
            'traps': 'mask',
            'mask_depth': 'mask',
            'mask_map': 'mask'
        }

        actual_targets = transforms.additional_targets

        # Проверяем, что все ожидаемые ключи присутствуют и имеют правильные значения
        for key, value in expected_targets.items():
            assert key in actual_targets, f"Ключ {key} отсутствует в additional_targets"
            assert actual_targets[key] == value, f"Для ключа {key} ожидается тип '{value}', но получено '{actual_targets[key]}'"

    def test_output_shapes_and_types(self, sample_data):
        """Проверка форматов выходных данных после применения аугментаций"""
        transforms = get_train_transforms()
        result = transforms(**sample_data)

        # ToTensorV2 переводит изображения в формат (C, H, W) и тип float32/uint8 в зависимости от реализации,
        # но обычно нормализует или оставляет uint8 -> float.
        # Главное проверить, что данные стали тензорами (numpy array с правильной размерностью для torch)

        # Image и Depth должны стать (C, H, W) или (1, H, W) для depth
        assert result["image"].ndim == 3, "Изображение должно быть в формате CHW"
        assert result["image"].shape[0] == 3, "RGB изображение должно иметь 3 канала"

        # Маски должны остаться (H, W) или стать (1, H, W) в зависимости от версии albumentations/ToTensorV2
        # В новых версиях ToTensorV2 маски часто остаются (H, W), если не указано иное,
        # но для consistency проверим, что они не потеряли данные.
        # Ключевое: они должны быть готовы к передаче в модель.
        assert result["faults"].ndim >= 2, "Маска должна иметь как минимум 2 измерения"
        assert result["traps"].shape == sample_data["traps"].shape or result["traps"].ndim == 3, \
            "Размерность маски трэпов изменилась непредсказуемо"


class TestValTransforms:
    """Тесты для валидационных аугментаций"""

    def test_transforms_creation(self):
        """Проверка создания валидационных трансформаций"""
        transforms = get_val_transforms()
        assert isinstance(transforms, A.Compose)

    def test_no_augmentations_in_val(self):
        """
        Проверка, что в валидации нет агрессивных аугментаций (флипы, шумы).
        Допускаются только нормализация и конвертация в тензор.
        """
        transforms = get_val_transforms()
        transform_names = [type(t).__name__ for t in transforms.transforms]

        forbidden_augs = ["HorizontalFlip", "VerticalFlip", "RandomBrightnessContrast", "GaussNoise", "Blur"]

        for aug in forbidden_augs:
            assert aug not in transform_names, f"Аугментация {aug} не должна присутствовать в валидационном пайплайне"

    def test_tensor_conversion(self, sample_data):
        """Проверка, что валидация корректно конвертирует данные в тензоры"""
        transforms = get_val_transforms()
        result = transforms(**sample_data)

        # Проверка наличия ключей
        for key in sample_data.keys():
            assert key in result, f"Ключ {key} отсутствует в результате валидационной трансформации"

        # Проверка размерности изображения (должно стать CHW)
        assert result["image"].ndim == 3
        assert result["image"].shape[0] == 3

    def test_deterministic_behavior(self, sample_data):
        """
        Валидационные трансформации должны быть детерминированными.
        Два прогона с одинаковыми данными должны дать идентичный результат.
        """
        transforms = get_val_transforms()

        res1 = transforms(**sample_data)
        res2 = transforms(**sample_data)

        assert np.array_equal(res1["image"], res2["image"]), "Валидационные трансформации должны быть детерминированными"
        assert np.array_equal(res1["faults"], res2["faults"]), "Маски в валидации должны обрабатываться детерминировано"

    def test_additional_targets_config(self):
        """Проверка конфигурации additional_targets для валидации"""
        transforms = get_val_transforms()

        # albumentations автоматически добавляет 'image': 'image', поэтому проверяем вхождение
        expected_targets = {
            'depth': 'image',
            'faults': 'mask',
            'traps': 'mask',
            'mask_depth': 'mask',
            'mask_map': 'mask'
        }

        actual_targets = transforms.additional_targets

        # Проверяем, что все ожидаемые ключи присутствуют и имеют правильные значения
        for key, value in expected_targets.items():
            assert key in actual_targets, f"Ключ {key} отсутствует в additional_targets"
            assert actual_targets[key] == value, f"Для ключа {key} ожидается тип '{value}', но получено '{actual_targets[key]}'"
