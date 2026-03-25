import segmentation_models_pytorch as smp

from settings import settings


def get_model():
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=settings.in_channels,
        classes=1,
        activation=None
    )