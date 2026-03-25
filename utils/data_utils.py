import os
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from settings import settings


def get_all_files():
    if settings.is_cps:
        files = os.listdir(settings.cps_path)
        return [f for f in files if f.endswith(('.cps', '.grd'))]
    else:
        files = os.listdir(settings.data_path)
        return [f for f in files if f.endswith('.png')]


def extract_horizon_key(filename: str, data_source: str):
    if data_source == 'cps':
        name = filename.replace('.cps', '').replace('.grd', '')
    else:
        name = filename.replace('.png', '')

    parts = name.split('_')
    if len(parts) < 4:
        return None

    return "-".join(parts[3:]).lower().replace(" ", "")


def split_dataset(files, val_ratio=0.15, test_ratio=0.15, seed=42):
    groups = {}

    for f in files:
        key = extract_horizon_key(f, settings.DATA_SOURCE)
        if key is None:
            continue
        groups.setdefault(key, []).append(f)

    group_keys = list(groups.keys())

    train_keys, test_keys = train_test_split(
        group_keys, test_size=test_ratio, random_state=seed
    )

    train_keys, val_keys = train_test_split(
        train_keys,
        test_size=val_ratio / (1 - test_ratio),
        random_state=seed
    )

    def collect(keys):
        return [f for k in keys for f in groups[k]]

    return collect(train_keys), collect(val_keys), collect(test_keys)