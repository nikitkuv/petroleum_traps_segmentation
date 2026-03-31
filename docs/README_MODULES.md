# Модульная структура пайплайна обучения U-Net++

## Описание проекта

Проект для сегментации замкнутых структурных ловушек по структурным картам глубин кровли геологического горизонта с использованием U-Net++ с предобученным энкодером ResNet34.

**Задача**: Выделить замкнутые структурные ловушки - закрашенные части карт по последней замкнутой изолинии, выше которых существует замкнутая возвышенность.

## Структура проекта

```
/workspace/
├── pipeline.py              # Главный файл для запуска полного пайплайна
├── dataset.py               # Dataset класс для загрузки данных
├── settings.py              # Конфигурация и гиперпараметры
│
├── data/                    # Модуль работы с данными
│   ├── __init__.py
│   └── dataloaders.py       # Загрузка файлов, разделение на выборки, создание DataLoader
│
├── models/                  # Модуль моделей
│   ├── __init__.py
│   └── unetplusplus.py      # Загрузка U-Net++, модификация входных каналов, чекпоинты
│
├── losses/                  # Модуль функций потерь
│   ├── __init__.py
│   └── losses.py            # MaskedBCE, MaskedDice, CombinedLoss
│
├── optimizers/              # Модуль оптимизаторов
│   ├── __init__.py
│   └── optimizers.py        # AdamW с differential LR, планировщики, статистика градиентов
│
├── metrics/                 # Модуль метрик
│   ├── __init__.py
│   └── metrics.py           # Dice, IoU, Recall, Precision, F1, FP/FN area
│
├── visualization/           # Модуль визуализации
│   ├── __init__.py
│   └── visualize.py         # Визуализация обучения и тестовых результатов
│
├── training/                # Модуль обучения
│   ├── __init__.py
│   ├── overfit_check.py     # Проверка overfit на 1-2 картах
│   └── train.py             # Fine-tuning с W&B мониторингом
│
├── evaluation/              # Модуль оценки
│   ├── __init__.py
│   └── evaluate.py          # Тестирование и визуализация результатов
│
├── utils/                   # Утилиты
│   ├── images_utils.py      # Утилиты для PNG (загрузка, маски, паддинг)
│   ├── augmentations.py     # Аугментации (Albumentations)
│   └── cps_utils.py         # Утилиты для CPS (если используются)
│
└── docs/                    # Документация
    ├── PLAN.md              # План проекта и требования
    ├── README_FORMAT.md     # Формат имен файлов
    ├── README_MODULES.md    # Этот файл
    └── TRAINING_GUIDE.md    # Руководство по обучению
```

## Формат названий изображений

Формат: `{number}_{x|y}_{type}_{name}.png`

### Типы файлов:
- `{number}_x_structuralNOisoline_{name}.png` → rgb (RGB карта без изолиний и разломов)
- `{number}_x_structuralBlackWhite_{name}.png` → depth_norm (нормализованная глубина)
- `{number}_x_faults_{name}.png` → faults (карта разломов, опционально)
- `{number}_y_traps_{name}.png` → traps (целевая маска ловушек)

### Примеры:
- `001_x_structuralNOisoline_H150.png` - RGB карта для горизонта H150
- `002_x_structuralNOisoline_H150.png` - Еще одна RGB карта для H150
- `001_x_structuralBlackWhite_H150.png` - Глубина для H150
- `001_y_traps_H150.png` - Ловушки для H150

Группировка производится по комбинации `{number}_{name}` - все файлы с одинаковым номером и названием горизонта попадают в один семпл. Разные номера для одного горизонта (например, 001_H150, 002_H150) будут разными семплами, но при разделении на выборки группировка происходит по `{name}` (горизонту), чтобы данные из одного горизонта не попадали одновременно в train и test.

## Быстрый старт

### 1. Проверка overfit (быстрая проверка корректности)

```python
from training.pipeline import run_full_pipeline
from settings import settings

test_metrics = run_full_pipeline(
    data_dir=settings.DATA_DIR,
    use_faults=False,  # Без разломов
    data_source='png',
    overfit_check_mode=True,  # Режим проверки overfit
    n_epochs=100,
    batch_size=4,
    learning_rate=1e-4
)
```

### 2. Полное обучение с W&B мониторингом

```python
from training.pipeline import run_full_pipeline
from settings import settings

test_metrics = run_full_pipeline(
    data_dir=settings.DATA_DIR,
    use_faults=False,  # Без разломов
    data_source='png',
    overfit_check_mode=False,  # Полное обучение
    wandb_project='geology-traps-segmentation',
    wandb_run_name='unetplusplus_rgb_depth_no_faults',
    n_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    early_stopping_patience=15,
    encoder_lr_multiplier=0.1  # Differential LR
)
```

## Отдельное использование модулей

### Загрузка данных

```python
from data.dataloaders import get_file_list, split_data_by_groups, create_dataloaders

# Получить список файлов
files = get_file_list('./data/images/', data_source='png')

# Разделить на выборки
train_files, val_files, test_files = split_data_by_groups(
    files, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15
)

# Создать DataLoader
train_loader, val_loader, test_loader = create_dataloaders(
    train_files, val_files, test_files,
    batch_size=4,
    use_faults=False
)
```

### Загрузка модели

```python
from models.unetplusplus import load_unetplusplus, load_model_checkpoint

# Создать модель
model = load_unetplusplus(
    in_channels=4,  # RGB + depth
    classes=1,
    encoder_name='resnet34',
    encoder_weights='imagenet'
)

# Загрузить чекпоинт
model = load_model_checkpoint(model, './checkpoints/best_model.pth')
```

### Функции потерь

```python
from losses.losses import CombinedLoss

criterion = CombinedLoss(
    bce_weight=0.5,
    dice_weight=0.5,
    use_map_mask=True,
    use_depth_mask=False  # True только если use_faults=True
)

# Использование
loss, loss_metrics = criterion(predictions, targets, mask_map=mask_map)
```

### Оптимизатор

```python
from optimizers.optimizers import create_optimizer_and_scheduler

optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    learning_rate=1e-4,
    weight_decay=1e-4,
    scheduler_type='reduce_lr_plateau',
    encoder_lr_multiplier=0.1  # Encoder LR = 0.1 * decoder LR
)
```

### Метрики

```python
from metrics.metrics import MetricsCalculator

metrics_calc = MetricsCalculator(threshold=0.5)

metrics = metrics_calc.compute_all(predictions, targets, mask=mask_map)
# metrics: {'iou', 'dice', 'recall', 'precision', 'f1', 'fp_area', 'fn_area'}
```

### Визуализация

```python
from visualization.visualize import visualize_training_results, visualize_test_results

# Во время обучения
visualize_training_results(batch, predictions, epoch, save_path='./logs/')

# На тесте
visualize_test_results(batch, predictions, sample_indices=[0,1,2,3], alpha=0.4)
```

### Обучение

```python
from training.train import train_with_wandb
from training.overfit_check import overfit_check

# Overfit проверка
overfit_check(model, train_loader, criterion, optimizer, n_epochs=100)

# Полное обучение
history = train_with_wandb(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    n_epochs=100,
    wandb_project='my-project',
    early_stopping_patience=15
)
```

### Оценка на тесте

```python
from evaluation.evaluate import evaluate_on_test, visualize_test_predictions

# Метрики на тесте
test_metrics = evaluate_on_test(model, test_loader, criterion)

# Визуализация
visualize_test_predictions(model, test_loader, sample_indices=[0,1,2,3])
```

## Конфигурация (settings.py)

Основные параметры:
- `DATA_SOURCE`: Источник данных ('png' или 'cps')
- `USE_FAULTS`: Использовать ли разломы (True/False)
- `DATA_DIR`: Путь к данным
- `BATCH_SIZE`: Размер батча (по умолчанию 4)
- `NUM_EPOCHS`: Количество эпох (по умолчанию 100)
- `LEARNING_RATE`: Базовая скорость обучения (1e-4)
- `TARGET_HEIGHT`: Целевая высота (1248)
- `TARGET_WIDTH`: Целевая ширина (512)
- `DEVICE`: Устройство (cuda/cpu)
- `CHECKPOINT_DIR`: Путь для сохранения чекпоинтов

Вычисляемые свойства:
- `in_channels`: 5 если use_faults=True, иначе 4

## Режимы данных

### Без разломов (use_faults=False)
- Входные каналы: 4 (RGB + depth_norm)
- Маски: только map_mask (игнорирование фона)
- Encoder: ResNet34 ImageNet pretrained

### С разломами (use_faults=True)
- Входные каналы: 5 (RGB + depth_norm + fault_mask)
- Маски: map_mask + depth_mask (игнорирование фона и областей под разломами)
- Encoder: ResNet34 ImageNet pretrained

## Особенности архитектуры

- **Модель**: U-Net++ с энкодером ResNet34
- **Входные каналы**: 4 или 5 (модифицируется первый слой conv1)
- **Инициализация**: Предобученные веса ImageNet для первых 3 каналов (RGB), остальные инициализируются средним значением RGB весов
- **Differential LR**: Encoder обучается с LR × 0.1, Decoder с базовым LR
- **Gradient Accumulation**: Поддерживается для больших моделей
