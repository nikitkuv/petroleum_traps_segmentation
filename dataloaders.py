from torch.utils.data import DataLoader

from dataset import GeologyTrapsDataset
from utils.data_utils import get_all_files, split_dataset
from settings import settings


def create_dataloaders():
    files = get_all_files()
    train_files, val_files, test_files = split_dataset(files)

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_ds = GeologyTrapsDataset(
        train_files,
        augment=True,
        use_faults=settings.USE_FAULTS,
        data_source=settings.DATA_SOURCE
    )

    val_ds = GeologyTrapsDataset(
        val_files,
        augment=False,
        use_faults=settings.USE_FAULTS,
        data_source=settings.DATA_SOURCE
    )

    test_ds = GeologyTrapsDataset(
        test_files,
        augment=False,
        use_faults=settings.USE_FAULTS,
        data_source=settings.DATA_SOURCE
    )

    return (
        DataLoader(train_ds, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_WORKERS),
        DataLoader(val_ds, batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=settings.NUM_WORKERS),
        DataLoader(test_ds, batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=settings.NUM_WORKERS),
    )