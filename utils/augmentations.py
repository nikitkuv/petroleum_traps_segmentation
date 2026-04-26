import albumentations as A
from albumentations.pytorch import ToTensorV2

from settings import settings


def get_train_transforms():
    
    p = settings.AUGMENT_PROB
    
    return A.Compose([
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.ElasticTransform(
            p=0.3, 
            alpha=60, 
            sigma=60 * 0.05, 
            alpha_affine=60 * 0.03,
            border_mode=0,
            value=0,
            mask_value=0
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
        ], p=0.3),
        ToTensorV2(),
        
    ], additional_targets={
        'image': 'image',
        'depth': 'mask',
        'isolines': 'mask',
        'faults': 'mask',
        'traps': 'mask',
        'mask_map': 'mask'
    })


def get_val_transforms():
    return A.Compose([
        ToTensorV2(),
    ], additional_targets={
        'image': 'image',
        'depth': 'mask',
        'isolines': 'mask',
        'faults': 'mask',
        'traps': 'mask',
        'mask_map': 'mask'
    })
