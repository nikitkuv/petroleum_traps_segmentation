# Пайплайн обучения U-Net++ для сегментации геологических ловушек

## Описание

Полный пайплайн для fine-tuning модели U-Net++ с предобученным энкодером ResNet34 для задачи сегментации замкнутых структурных ловушек по геологическим картам.

**Вариант данных:** PNG с RGB без изолиний + depth_norm + map_mask (без разломов)

## Структура файла `train_pipeline.py`

Файл содержит все необходимые компоненты для обучения модели:

### 1. Загрузка данных и разделение на выборки
- `get_file_list()` - получение списка файлов данных
- `split_data_by_groups()` - разделение на train/val/test с группировкой по исходным картам
- `create_dataloaders()` - создание DataLoader для каждой выборки

### 2. Модель
- `load_unetplusplus()` - загрузка предобученной U-Net++ с модификацией входных каналов
- `load_model_checkpoint()` - загрузка весов из чекпоинта

### 3. Лоссы
- `MaskedBCELoss` - BCE loss с поддержкой масок
- `MaskedDiceLoss` - Dice loss с поддержкой масок
- `CombinedLoss` - комбинированный лосс (BCE + Dice)

### 4. Оптимизатор и гиперпараметры
- `create_optimizer_and_scheduler()` - создание AdamW оптимизатора с differential learning rate и планировщика

### 5. Метрики
- `MetricsCalculator` - вычисление всех метрик (Dice, IoU, Recall, Precision, F1, FP/FN area)

### 6. Визуализация
- `visualize_training_results()` - визуализация результатов обучения (RGB, GT, prediction, overlay)
- `visualize_test_results()` - визуализация результатов на тестовых данных

### 7. Overfit проверка
- `overfit_check()` - проверка обучения на 1-2 картах без валидации

### 8. Обучение с W&B мониторингом
- `train_with_wandb()` - fine-tuning с логированием в wandb (лоссы, метрики, градиенты, визуализации)

### 9. Тестирование
- `evaluate_on_test()` - оценка модели на тестовых данных

### 10. Полный пайплайн
- `run_full_pipeline()` - запуск всего процесса обучения и тестирования

## Быстрый старт

### Базовое использование

```python
from train_pipeline import run_full_pipeline
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

### Отдельные шаги

#### 1. Подготовка данных

```python
from train_pipeline import get_file_list, split_data_by_groups, create_dataloaders

# Получить список файлов
file_list = get_file_list('./data/images/', data_source='png')

# Разделить на выборки
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

#### 2. Загрузка модели

```python
from train_pipeline import load_unetplusplus
from settings import settings

model = load_unetplusplus(
    in_channels=4,  # RGB (3) + depth_norm (1)
    classes=1,
    encoder_name='resnet34',
    encoder_weights='imagenet',
    device='cuda'
)
```

#### 3. Настройка обучения

```python
from train_pipeline import CombinedLoss, create_optimizer_and_scheduler

# Лосс
criterion = CombinedLoss(
    bce_weight=0.5,
    dice_weight=0.5,
    use_map_mask=True,
    use_depth_mask=False  # Только если есть разломы
)

# Оптимизатор и планировщик
optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    learning_rate=1e-4,
    scheduler_type='reduce_lr_plateau',
    encoder_lr_multiplier=0.1  # Differential LR
)
```

#### 4. Обучение

```python
from train_pipeline import train_with_wandb

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
    wandb_project='geology-traps-segmentation',
    checkpoint_path='./checkpoints/'
)
```

#### 5. Тестирование

```python
from train_pipeline import load_model_checkpoint, evaluate_on_test, visualize_test_results

# Загрузить лучшую модель
model = load_model_checkpoint(model, './checkpoints/best_model.pth')

# Оценить на тесте
test_metrics = evaluate_on_test(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device='cuda'
)

# Визуализировать результаты
with torch.no_grad():
    batch = next(iter(test_loader))
    x = batch['x'].to('cuda')
    predictions = model(x)
    
    visualize_test_results(
        batch=batch,
        predictions=predictions,
        sample_indices=[0, 1, 2, 3],
        save_path='./logs/test_visualizations/'
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

## W&B интеграция

Для использования wandb:

```bash
wandb login
```

Пайплайн логирует:
- Train/Val loss (BCE, Dice, total)
- Train/Val метрики (Dice, IoU, Recall, Precision, F1)
- Learning rate
- Gradient norm
- Визуализации предсказаний
- Best model checkpoint

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

## Структура данных

Ожидаемая структура файлов:

```
data/images/
├── 001_fieldA_x_structuralNOisoline.png  # RGB без изолиний
├── 001_fieldA_x_structuralBlackWhite.png # depth_norm
├── 001_fieldA_y_traps.png                # Ground truth traps
├── 002_fieldB_x_structuralNOisoline.png
├── 002_fieldB_x_structuralBlackWhite.png
├── 002_fieldB_y_traps.png
...
```

Формат именования: `{number}_{name}_x_{type}.png` или `{number}_{name}_y_{type}.png`

## Рекомендации

1. **Начните с overfit check**: Убедитесь, что модель может переобучиться на 1-2 картах перед полным обучением.

2. **Мониторьте градиенты**: Если градиенты слишком большие или маленькие,调整 learning rate или добавьте gradient clipping.

3. **Используйте early stopping**: Предотвратит переобучение и сэкономит время.

4. **Проверяйте визуализации**: Визуальная оценка часто важнее числовых метрик.

5. **Differential LR**: Энкодер обучается медленнее декодера (lr * 0.1), что помогает сохранить предобученные признаки.

## Авторские заметки

- Код поддерживает как режим с разломами, так и без них
- Маска map_mask используется всегда для игнорирования фона
- Маска depth_mask используется только при наличии разломов
- Реализована поддержка gradient accumulation для больших моделей
- Все функции имеют подробные docstrings
