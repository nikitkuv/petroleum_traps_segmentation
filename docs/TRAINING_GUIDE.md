# Пайплайн обучения U-Net++ для сегментации геологических ловушек

## Описание

Полный пайплайн для fine-tuning модели U-Net++ с предобученным энкодером ResNet34 для задачи сегментации замкнутых структурных ловушек по геологическим картам.

**Задача**: Выделить замкнутые структурные ловушки - закрашенные части карт по последней замкнутой изолинии, выше которых существует замкнутая возвышенность.

**Варианты данных**:
- PNG: RGB без изолиний + depth_norm + map_mask (без разломов)
- PNG: RGB без изолиний + fault_mask + depth_norm + depth_mask + map_mask (с разломами)

## Структура проекта

### Основные модули

| Модуль | Файл | Описание |
|--------|------|----------|
| Pipeline | `pipeline.py` | Главный файл для запуска полного пайплайна |
| Dataset | `dataset.py` | Dataset класс для загрузки данных |
| Settings | `settings.py` | Конфигурация и гиперпараметры |
| Dataloaders | `data/dataloaders.py` | Загрузка файлов, разделение на выборки |
| Model | `models/unetplusplus.py` | U-Net++ с модификацией входных каналов |
| Losses | `losses/losses.py` | MaskedBCE, MaskedDice, CombinedLoss |
| Optimizers | `optimizers/optimizers.py` | AdamW с differential LR, планировщики |
| Metrics | `metrics/metrics.py` | Dice, IoU, Recall, Precision, F1, FP/FN area |
| Visualization | `visualization/visualize.py` | Визуализация результатов |
| Training | `training/train.py` | Fine-tuning с W&B мониторингом |
| Overfit Check | `training/overfit_check.py` | Проверка overfit на 1-2 картах |
| Evaluation | `evaluation/evaluate.py` | Тестирование и оценка |

### Вспомогательные утилиты

| Утилита | Файл | Описание |
|---------|------|----------|
| Images Utils | `utils/images_utils.py` | Загрузка PNG, создание масок, паддинг |
| Augmentations | `utils/augmentations.py` | Аугментации (Albumentations) |
| CPS Utils | `utils/cps_utils.py` | Работа с CPS гридами (опционально) |

## Быстрый старт

### Базовое использование

```python
from training.pipeline import run_full_pipeline
from settings import settings

# Запуск полного пайплайна
test_metrics = run_full_pipeline(
    data_dir=settings.DATA_DIR,
    use_faults=False,  # Без разломов
    data_source='png',
    overfit_check_mode=False,
    wandb_project='geology-traps-segmentation',
    wandb_run_name='unetplusplus_rgb_depth_no_faults',
    n_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    early_stopping_patience=15
)
```

### Режим проверки Overfit

```python
# Проверка на способность модели переобучиться на 1-2 картах
test_metrics = run_full_pipeline(
    data_dir=settings.DATA_DIR,
    use_faults=False,
    data_source='png',
    overfit_check_mode=True,  # Включить режим overfit check
    n_epochs=100,
    batch_size=4,
    learning_rate=1e-4
)
```

## Пошаговое руководство

### Шаг 1: Подготовка данных

#### Формат имен файлов
```
{number}_{x|y}_{type}_{name}.png
```

Примеры:
- `001_x_structuralNOisoline_H150.png` - RGB карта
- `001_x_structuralBlackWhite_H150.png` - Depth нормализованный
- `001_x_faults_H150.png` - Разломы (опционально)
- `001_y_traps_H150.png` - Ловушки (таргет)

#### Структура директорий
```
data/images/
├── 001_x_structuralNOisoline_H150.png
├── 001_x_structuralBlackWhite_H150.png
├── 001_y_traps_H150.png
├── 002_x_structuralNOisoline_H150.png
└── ...
```

```python
from data.dataloaders import get_file_list, split_data_by_groups, create_dataloaders

# Получить список файлов
file_list = get_file_list('./data/images/', data_source='png')

# Разделить на выборки (группировка по name)
train_files, val_files, test_files = split_data_by_groups(
    file_list,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Создать dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_files, val_files, test_files,
    batch_size=4,
    use_faults=False,
    data_source='png'
)
```

### Шаг 2: Загрузка модели

```python
from models.unetplusplus import load_unetplusplus
from settings import settings

model = load_unetplusplus(
    in_channels=4,  # RGB (3) + depth_norm (1)
    classes=1,
    encoder_name='resnet34',
    encoder_weights='imagenet',
    device='cuda'
)
```

**Особенности архитектуры**:
- U-Net++ с энкодером ResNet34 (ImageNet pretrained)
- Первые 3 канала используют предобученные веса
- Дополнительные каналы (depth, faults) инициализируются средним значением RGB весов
- Decoder инициализируется случайно

### Шаг 3: Настройка функции потерь

```python
from losses.losses import CombinedLoss

# Для режима без разломов
criterion = CombinedLoss(
    bce_weight=0.5,
    dice_weight=0.5,
    use_map_mask=True,
    use_depth_mask=False
)

# Для режима с разломами
criterion_with_faults = CombinedLoss(
    bce_weight=0.5,
    dice_weight=0.5,
    use_map_mask=True,
    use_depth_mask=True
)
```

**Компоненты лосса**:
- **MaskedBCELoss**: Binary Cross-Entropy с поддержкой масок
- **MaskedDiceLoss**: Dice loss с поддержкой масок
- **Маски**: 
  - `map_mask`: игнорирует фон за пределами карты
  - `depth_mask`: игнорирует области под разломами (только для режима с разломами)

### Шаг 4: Настройка оптимизатора

```python
from optimizers.optimizers import create_optimizer_and_scheduler

optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    learning_rate=1e-4,
    weight_decay=1e-4,
    scheduler_type='reduce_lr_plateau',
    encoder_lr_multiplier=0.1  # Differential LR
)
```

**Differential Learning Rate**:
- Encoder (предобученный): lr × 0.1 = 1e-5
- Decoder (новый): lr = 1e-4
- Это помогает сохранить предобученные признаки энкодера

### Шаг 5: Обучение

#### Вариант A: Overfit Check (рекомендуется сначала)

```python
from training.overfit_check import overfit_check

overfit_check(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda',
    n_epochs=100,
    save_path='./logs/overfit_check/'
)
```

**Цель**: Убедиться, что модель может переобучиться на 1-2 картах. Если нет - проблема в данных или пайплайне.

#### Вариант B: Полное обучение с W&B

```python
from training.train import train_with_wandb

history = train_with_wandb(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda',
    n_epochs=100,
    early_stopping_patience=15,
    gradient_accumulation_steps=1,
    wandb_project='geology-traps-segmentation',
    wandb_run_name='unetplusplus_rgb_depth',
    checkpoint_path='./checkpoints/',
    log_gradients=True
)
```

**W&B логирование**:
- Train/Val loss (BCE, Dice, total)
- Train/Val метрики (Dice, IoU, Recall, Precision, F1)
- Learning rate
- Gradient norm
- Визуализации предсказаний
- Best model checkpoint

### Шаг 6: Оценка на тесте

```python
from models.unetplusplus import load_model_checkpoint
from evaluation.evaluate import evaluate_on_test, visualize_test_predictions

# Загрузить лучшую модель
model = load_model_checkpoint(model, './checkpoints/best_model.pth')

# Оценить на тесте
test_metrics = evaluate_on_test(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device='cuda',
    threshold=0.5
)

# Визуализировать результаты
visualize_test_predictions(
    model=model,
    test_loader=test_loader,
    device='cuda',
    sample_indices=[0, 1, 2, 3],
    save_path='./logs/test_visualizations/',
    alpha=0.4
)
```

## Гиперпараметры по умолчанию

| Параметр | Значение | Описание |
|----------|----------|----------|
| optimizer | AdamW | Оптимизатор |
| learning_rate | 1e-4 | Базовая скорость обучения |
| encoder_lr_multiplier | 0.1 | Множитель LR для энкодера |
| weight_decay | 1e-4 | L2 регуляризация |
| batch_size | 4 | Размер батча |
| epochs | 100 | Количество эпох |
| scheduler | ReduceLROnPlateau | Планировщик LR |
| early_stopping_patience | 15 | Патанс для ранней остановки |
| bce_weight | 0.5 | Вес BCE лосса |
| dice_weight | 0.5 | Вес Dice лосса |
| target_height | 1248 | Целевая высота после паддинга |
| target_width | 512 | Целевая ширина после паддинга |

## Метрики

### Основные метрики

| Метрика | Ориентир | Описание |
|---------|----------|----------|
| Dice | > 0.7-0.8 | Коэффициент схожести |
| IoU | > 0.55-0.65 | Intersection over Union |
| Recall | > 0.75-0.85 | Полнота (TP / (TP + FN)) |
| Precision | - | Точность (TP / (TP + FP)) |
| F1 | - | Гармоническое среднее Precision и Recall |
| FP Area | < 0.3-0.5 | Доля ложноположительных |
| FN Area | < 0.3-0.5 | Доля ложноотрицательных |

## Визуализации

Пайплайн автоматически сохраняет визуализации:

1. **Training visualizations** (`./logs/visualizations/`):
   - Оригинальная RGB карта
   - Ground truth traps
   - Predicted traps
   - Overlay предсказания на RGB

2. **Validation visualizations** (`./logs/val_visualizations/`):
   - Аналогично training, но на валидационных данных

3. **Test visualizations** (`./logs/test_visualizations/`):
   - RGB карта
   - Ground truth
   - Prediction overlay с прозрачностью

4. **Overfit history** (`./logs/overfit_check/`):
   - Графики loss, Dice, IoU
   - Визуализации прогресса обучения

## Требования

```
torch>=2.0
torchvision>=0.15
segmentation-models-pytorch>=0.3.3
albumentations
opencv-python
scikit-learn
matplotlib
wandb
tqdm
pydantic-settings
```

## Рекомендации

### 1. Начните с overfit check
Убедитесь, что модель может переобучиться на 1-2 картах перед полным обучением. Это поможет выявить проблемы в данных или пайплайне.

### 2. Мониторьте градиенты
Если градиенты слишком большие или маленькие,调整 learning rate или добавьте gradient clipping.

### 3. Используйте early stopping
Предотвратит переобучение и сэкономит время.

### 4. Проверяйте визуализации
Визуальная оценка часто важнее числовых метрик. Обращайте внимание на:
- Соответствие предсказаний реальным ловушкам
- Отсутствие предсказаний на фоне
- Качество границ

### 5. Differential LR
Энкодер обучается медленнее декодера (lr × 0.1), что помогает сохранить предобученные признаки.

### 6. Режимы данных

#### Без разломов (рекомендуется начать с этого)
- Входные каналы: 4 (RGB + depth_norm)
- Маски: только map_mask
- Проще для отладки

#### С разломами
- Входные каналы: 5 (RGB + depth_norm + fault_mask)
- Маски: map_mask + depth_mask
- Может улучшить качество, если разломы важны для задачи

## Мониторинг и отладка

### Gradient Tracking

Пайплайн включает продвинутый трекинг градиентов для диагностики проблем обучения:

```python
from training.gradient_tracker import GradientTracker

# Автоматический трекинг в train_with_wandb
history = train_with_wandb(
    model=model,
    # ... другие параметры
    log_gradients=True,  # Включить логирование градиентов
    gradient_clip_value=1.0  # Опционально: gradient clipping
)
```

**Что отслеживается**:
- Norm градиентов по слоям (encoder, decoder, first conv)
- Статистика градиентов (mean, std, min, max)
- Графики в W&B для визуализации динамики

**Интерпретация**:
- **Градиенты > 10**: Возможна нестабильность, уменьшите LR или добавьте gradient clipping
- **Градиенты < 1e-6**: Возможное vanishing gradient, проверьте архитектуру или увеличьте LR
- **Скачки градиентов**: Проверьте данные на аномалии

### Анализ аномалий

Для выявления проблемных карт в датасете используйте `analyze_anomalies`:

```python
from training.pipeline import analyze_anomalies

# Найти карты с аномально высоким loss
anomalies = analyze_anomalies(
    model=model,
    dataloader=train_loader,
    criterion=criterion,
    device='cuda',
    top_k=10  # Показать топ-10 худших карт
)

# Выводит названия файлов и соответствующие loss
for file_name, loss in anomalies:
    print(f"{file_name}: loss={loss:.4f}")
```

Это помогает выявить:
- Неправильно размеченные данные
- Артефакты на картах
- Выбросы в распределении данных

## Возможные улучшения (TODO)

- [ ] Tversky loss (α=0.7, β=0.3) или Focal loss при дисбалансе классов
- [ ] DeepLabV3+ или HRNet энкодеры
- [ ] Multi-task learning: auxiliary head на contour/boundary
- [ ] Test-time augmentation (TTA)
- [ ] Post-processing: морфологические операции для очистки предсказаний
