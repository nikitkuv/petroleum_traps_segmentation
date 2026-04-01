# Gradient Anomaly Tracking

## Обзор

Эта система позволяет отслеживать аномалии нормы градиентов во время обучения и автоматически сохранять проблемные батчи для последующего анализа.

## Как это работает

### 1. **GradientNormTracker** (`training/gradient_tracker.py`)

Класс для мониторинга нормы градиентов с использованием:
- **Welford's algorithm** для численно стабильного вычисления running mean и std
- **Двойной порог детектирования**:
  - Абсолютный порог (по умолчанию `grad_abs_threshold=10.0`)
  - Динамический порог: `running_mean + std_multiplier * running_std`

### 2. **Автоматическое сохранение батчей**

При обнаружении аномалии:
- Сохраняется весь батч данных (x, y, mask_map) в `.pt` формате
- Создается preview изображений для быстрого просмотра
- Ведется JSON лог всех аномалий с метаданными

### 3. **Пер-семпл анализ** (`compute_per_sample_grad_norms`)

Функция для постфактум анализа сохраненных батчей:
- Вычисляет норму градиента для каждого семпла отдельно
- Помогает идентифицировать конкретные проблемные изображения

## Использование

### В training pipeline:

```python
from training.train import train_with_wandb

history = train_with_wandb(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    # Параметры трекинга аномалий:
    grad_anomaly_tracking=True,      # Включить трекинг
    grad_abs_threshold=10.0,         # Абсолютный порог grad_norm
    grad_std_multiplier=3.0,         # Сколько std для динамического порога
    save_anomaly_batches=True,       # Сохранять проблемные батчи
    max_anomalies_to_save=20         # Максимум аномалий для сохранения
)
```

### Анализ после обучения:

```bash
# Просмотр сводки по всем аномалиям
python training/analyze_anomalies.py --anomaly_dir ./gradient_anomalies/

# Анализ топ-5 батчей с highest grad norms
python training/analyze_anomalies.py --anomaly_dir ./gradient_anomalies/ --top_n 5 --model_path ./checkpoints/best_model.pth

# Анализ всех сохраненных батчей
python training/analyze_anomalies.py --anomaly_dir ./gradient_anomalies/ --analyze_all
```

## Структура сохраняемых данных

```
./gradient_anomalies/
├── anomalies_list.json              # Полный лог всех аномалий
├── epoch0_batch42_grad12.34_batch.pt    # Сохраненный батч
├── epoch0_batch42_grad12.34_preview.png # Preview изображений
├── epoch1_batch15_grad11.56_batch.pt
├── epoch1_batch15_grad11.56_preview.png
└── detailed_analysis.json           # Результаты пер-семпл анализа
```

## Формат anomalies_list.json

```json
{
  "total_anomalies": 5,
  "tracker_config": {
    "abs_threshold": 10.0,
    "std_multiplier": 3.0,
    "min_samples_for_std": 20
  },
  "final_stats": {
    "running_mean": 5.2,
    "running_std": 1.8,
    "count": 500
  },
  "anomalies": [
    {
      "epoch": 0,
      "batch_idx": 42,
      "grad_norm": 12.34,
      "running_mean": 5.1,
      "running_std": 1.7
    }
  ]
}
```

## Рекомендации по настройке порогов

### Для вашего случая (grad_norm ~4-6 в среднем, вылеты до 10-15):

1. **Начальные настройки**:
   ```python
   grad_abs_threshold=10.0    # Ловить вылеты выше 10
   grad_std_multiplier=3.0    # +3 sigma от running mean
   ```

2. **Если слишком много аномалий**:
   - Увеличить `grad_abs_threshold` до 12.0 или 15.0
   - Увеличить `grad_std_multiplier` до 4.0 или 5.0

3. **Если слишком мало аномалий**:
   - Уменьшить `grad_abs_threshold` до 8.0
   - Уменьшить `grad_std_multiplier` до 2.5

## Performance considerations

### Во время обучения:
- **Минимальный overhead**: Welford's algorithm - O(1) per batch
- **Сохранение батчей**: Только при детектировании аномалий
- **Preview изображений**: Асинхронно, не блокирует training

### Постфактум анализ:
- **compute_per_sample_grad_norms**: Требует N forward-backward passes (где N = batch_size)
- **Рекомендация**: Анализировать только топ-5 батчей с highest grad norms

## WandB интеграция

Система автоматически логирует в wandb:
- `grad_norm` - текущая норма градиента
- `grad_norm_running_mean` - running average
- `grad_norm_running_std` - running стандартное отклонение
- `grad_norm_threshold` - текущий порог детектирования
- `gradient_anomaly_detected` - флаг аномалии (1 когда detected)
- `gradient_anomaly_grad_norm` - значение grad_norm при аномалии
- `gradient_anomaly_batch_idx` - индекс проблемного батча

В конце обучения:
- `gradient_anomalies_total` - всего аномалий
- `gradient_anomaly_rate` - процент аномальных батчей
- `gradient_norm_final_mean` - финальный running mean
- `gradient_norm_final_std` - финальный running std

## Пример workflow

1. **Запуск обучения** с включенным трекингом:
   ```python
   train_with_wandb(..., grad_anomaly_tracking=True)
   ```

2. **Мониторинг в реальном времени** через wandb dashboard

3. **После обучения** - анализ сохраненных батчей:
   ```bash
   python training/analyze_anomalies.py --top_n 5
   ```

4. **Исследование проблемных семплов**:
   - Открыть `detailed_analysis.json`
   - Посмотреть какие семплы дают наибольший вклад в градиент
   - Визуально inspect сохраненные preview images
   - Проверить качество аннотаций для проблемных семплов

5. **Действия на основе анализа**:
   - Исключить проблемные семплы из датасета
   - Исправить incorrect annotations
   - Добавить data augmentation для подобных случаев
   - Настроить learning rate schedule
