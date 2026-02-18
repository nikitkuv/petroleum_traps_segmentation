import albumentations as A
from albumentations.pytorch import ToTensorV2
from settings import settings


def get_train_transforms():
    """Аугментации для обучения."""
    p = settings.AUGMENT_PROB
    
    return A.Compose([
        # Геометрические трансформации
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        
        # Цветовые трансформации (только для RGB)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        ], p=0.3),
        
        # Шум и размытие
        A.OneOf([
            A.GaussNoise(sigma=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.3),
        
        # Конвертация в тензор
        ToTensorV2(),
    ], additional_targets={
        'depth': 'image',
        'faults': 'mask',
        'traps': 'mask',
        'mask_depth': 'mask',
        'mask_map': 'mask'
    })


def get_val_transforms():
    """Аугментации для валидации (только тензор)."""
    return A.Compose([
        ToTensorV2(),
    ], additional_targets={
        'depth': 'image',
        'faults': 'mask',
        'traps': 'mask',
        'mask_depth': 'mask',
        'mask_map': 'mask'
    })