# Рабочие процессы (Workflows)

## Добавление новых CPS гридов

### Шаг 1: Визуализация CPS гридов

Проверка качества конвертации в RGB и изолинии:

```bash
python visualization/visualize_full_sample.py
```

Визуализирует:
- RGB карта
- Depth нормализованный
- Изолинии
- Разломы (если есть)
- Ловушки (таргет)
- Маску валидной области

### Шаг 2: Конвертация CPS в PNG тайлы

Полная конвертация всех CPS гридов в PNG:

```bash
python data/convert_cps_to_tiles.py
```

**Режимы работы**:
- `ONLY_SAVE = False, ONLY_SPLIT = False` (по умолчанию): Полный цикл (конвертация + тайлинг)
- `ONLY_SAVE = True`: Только сохранение полных изображений
- `ONLY_SPLIT = True`: Только разбиение на тайлы (использует существующие полные изображения)

**Результат**:
- Полные PNG изображения в `data/images_cps_full/`
- Разбитые на тайлы в `data/images_cps/`

**Типы генерируемых PNG**:
- `x_structuralNOisoline_{horizon}.png` — RGB карта (purple_jet)
- `x_structuralBlackWhite_{horizon}.png` — Grayscale depth
- `x_isolines_{horizon}.png` — Карта изолиний
- `x_faults_{horizon}.png` — Маска разломов
- `y_traps_{horizon}.png` — Карта ловушек (таргет)

---

## Проверка семплов и датасета

### Шаг 1: Проверка размеров изображений

```bash
python data_validation/check_image_sizes.py
```

Проверяет размеры всех PNG файлов в директории, выявляет аномалии.

### Шаг 2: Валидация датасета

```bash
python data_validation/validate_dataset.py
```

Проверяет:
- Наличие всех необходимых файлов для каждого семпла
- Корректность именования файлов
- Соответствие формату данных
- Отсутствие leakage между train/val/test

### Шаг 3: Визуализация примера датасета

```bash
python visualization/visualize_dataset_sample.py
```

Визуализирует случайный семпл со всеми каналами:
- RGB карта
- Depth нормализованный
- Изолинии
- Разломы (если есть)
- Ловушки (таргет)
- Маску валидной области

---

## Тестирование

### Запуск всех тестов

```bash
pytest tests/ -v
```

### Запуск с покрытием кода

```bash
pytest tests/ --cov=. --cov-report=html
```

Отчет откроется в `htmlcov/index.html`.

### Запуск smoke-тестов (быстрые)

```bash
pytest tests/ -m "smoke" -v
```

### Запуск отдельных категорий тестов

```bash
# Валидация данных
pytest tests/test_data_validation.py -v

# Dataset и DataLoader
pytest tests/test_dataset.py -v

# Обучение и модель
pytest tests/test_training.py -v

# Полный пайплайн (медленные тесты)
pytest tests/test_pipeline.py -v -m "not slow"
```

---

## Overfit Check

Проверка способности модели переобучиться на 1-2 картах для отладки пайплайна.

### Шаг 1: Отключение аугментаций

Выставить `AUGMENT_TRAIN = False` в `settings.py`:

```bash
AUGMENT_TRAIN=False
```

### Шаг 2: Запуск overfit check

```bash
python run_overfit_check.py
```

Или через pytest:

```bash
pytest run_overfit_check.py
```

**Ожидаемый результат**:
- Loss должен стремиться к 0 за 50-100 эпох
- Dice/IoU должны расти до ~0.95-1.0

Если модель не переобучается — проблема в данных или пайплайне.

---

## Обучение модели

### Быстрый старт

```bash
python run_training.py
```

### С кастомными параметрами

```bash
python run_training.py \
  --data_source cps_tiles \
  --use_faults False \
  --epochs 100 \
  --batch_size 4 \
  --learning_rate 3e-4
```

### Мониторинг

- **W&B Dashboard**: Графики loss, метрик, learning rate
- **Gradient Tracking**: Автоматическое детектирование аномалий градиентов
- **Визуализации**: Сохраняются в `logs/visualizations/`

---

## Оценка модели

### Тестирование на test выборке

```bash
python evaluate_model.py
```

### Визуализация результатов

```bash
python visualization/visualize_test_predictions.py
```

Результаты сохраняются в `logs/test_visualizations/`.

---

## Анализ аномалий

### Просмотр аномальных батчей

```bash
python training/analyze_anomalies.py --anomaly_dir ./gradient_anomalies/
```

### Анализ топ-N батчей с highest grad norms

```bash
python training/analyze_anomalies.py --anomaly_dir ./gradient_anomalies/ --top_n 5 --model_path ./checkpoints/best_model.pth
```

### Детальный анализ всех батчей

```bash
python training/analyze_anomalies.py --anomaly_dir ./gradient_anomalies/ --analyze_all
```
