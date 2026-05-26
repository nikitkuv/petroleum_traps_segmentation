# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Petroleum structural trap segmentation using U-Net++ on seismic horizon structural maps. The pipeline takes CPS grid files (depth + trap maps), converts them to tiled PNG/npy images, and trains a binary segmentation model to identify structural traps.

## Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=. --cov-report=html

# Run a single test file
pytest tests/test_losses.py -v

# Run only smoke tests (fast)
pytest tests/ -m "smoke" -v

# Convert CPS grids to tiles (prerequisite for training)
python data/convert_cps_to_tiles.py

# Quick overfit check (2 samples, 100 epochs)
python run_overfit_check.py

# Full training with W&B logging
python run_training.py

# Evaluate trained model on test set
python evaluate_model.py
```

## Architecture

### Data flow
CPS grids (`data/cps/`) → `data/convert_cps_to_tiles.py` → full images (`data/images_cps_full/`) → tiled images (`data/images_cps/`) → `GeologyTrapsDataset` → DataLoader

### Key module relationships
- `settings.py` — Single pydantic-settings config. All modules import `settings` from here. `IN_CHANNELS` is computed from `USE_RGB` and `USE_FAULTS`.
- `training/pipeline.py` — Orchestrates the full pipeline: data loading → split → leakage check → dataloaders → model → loss → optimizer → train → evaluate → visualize.
- `data/dataloaders.py` — Splits data by horizon groups (not random) to prevent leakage. `split_data_by_groups()` groups by `{name}` field in filenames and asserts no overlap between splits.
- `data/dataset.py` (`GeologyTrapsDataset`) — Filters tiles by NoData ratio, builds input tensor by concatenating channels in order: RGB(3) + depth(1) + isolines(1) + faults(1 optional) + map_mask(1).
- `utils/dataset_utils.py` — Filename parsing: `{number}_{x|y}_{type}_{name}.{png|npy}`. Types: `structuralNOisoline`=rgb, `structuralBlackWhite`=depth, `isolines`, `faults`, `traps`.
- `models/unetplusplus.py` — Wraps `smp.UnetPlusPlus` with ResNet34 encoder. Modifies first conv layer to accept variable input channels while preserving ImageNet weights for RGB channels.
- `losses/losses.py` — `CombinedLoss` = `MaskedBCELoss` + `MaskedDiceLoss` (default 0.5/0.5 weights). All losses support `mask_map` to ignore regions outside the map and faults.
- `training/train.py` — Training loop with W&B logging, gradient tracking, early stopping, gradient accumulation.
- `training/gradient_tracker.py` — Monitors gradient explosion/vanishing during training.

### Input channels (6 or 7)
With `USE_RGB=True` (default) and `USE_FAULTS=True` (default):
- Channels 0-2: RGB structural map (no isolines)
- Channel 3: Normalized depth (grayscale, float32)
- Channel 4: Isolines (binary)
- Channel 5: Faults (binary, optional)
- Channel 6: Map mask (1=valid, 0=off-map or fault)

### Data split strategy
Data is split by horizon name (the `{name}` part of filenames) to prevent leakage — all tiles from the same structural horizon stay in the same split. Split ratios: 70/15/15 (configurable in settings).

## Testing

Tests use pytest with fixtures in `tests/conftest.py` that generate synthetic tile images. Test files follow the naming pattern `test_{module}.py`. No external data is needed — all tests create temporary datasets.

## Key conventions

- All configuration is centralized in `settings.py` using pydantic-settings. No scattered magic numbers.
- Depth maps are saved as `.npy` (float32) not PNG to avoid precision loss.
- The `map_mask` channel marks invalid regions (off-map + faults) and is used by masked losses to ignore those areas.
- Code comments and identifiers are in Russian (variable names in English).
- Checkpoint filenames encode training config: `{version}_cps_tiles_{resolution}_{faults/no_faults}_e-{epochs}_bs-{batch}_lr-{lr}({encoder_mult})_wd-{wd}_{scheduler}.pth`
