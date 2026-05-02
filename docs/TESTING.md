# Тестирование проекта Geology Traps Segmentation

## Обзор

Этот документ описывает систему тестирования для проекта сегментации геологических ловушек. 
Тесты покрывают все ключевые компоненты проекта:

1. **Валидация входных данных** (PNG и CPS форматы)
2. **Загрузка данных в GeologyTrapsDataset**
3. **Аугментации данных**
4. **Функции потерь (Losses)**
5. **Метрики качества (Metrics)**
6. **Оптимизаторы и планировщики**
7. **Трекинг градиентов**
8. **Модели (U-Net++)**
9. **Процесс обучения и evaluation**
10. **Интеграционные тесты полного пайплайна**

## Структура тестов

```
tests/
├── __init__.py                    # Инициализация пакета тестов
├── conftest.py                    # Общие фикстуры и конфигурация pytest
├── test_dataset_utils.py          # Юнит тесты утилит загрузки данных
├── test_dataset_dataloader.py     # Юнит тесты Dataset и DataLoader
├── test_data_integration.py       # Интеграционные тесты пайплайна данных
├── test_augmentations.py          # Тесты аугментаций данных
├── test_losses.py                 # Тесты функций потерь
├── test_metrics.py                # Тесты калькулятора метрик
├── test_optimizers.py             # Тесты оптимизаторов и scheduler'ов
├── test_gradient_tracker.py       # Тесты трекинга градиентов
├── test_models.py                 # Тесты архитектуры модели
├── test_training_fixtures.py      # Фикстуры для тестов обучения
└── test_training_integration.py   # Интеграционные тесты обучения
```

## Установка зависимостей

Для запуска тестов установите дополнительные зависимости:

```bash
pip install pytest pytest-cov
```

Или обновите все зависимости:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Запуск тестов

### Все тесты

```bash
pytest tests/ -v
```

### Только smoke-тесты (быстрые)

```bash
pytest tests/ -m "smoke" -v
```

### Тесты с покрытием кода

```bash
pytest tests/ --cov=. --cov-report=html
```

Отчет откроется в `htmlcov/index.html`.

### Отдельные категории тестов

```bash
# Валидация данных и утилиты
pytest tests/test_dataset_utils.py -v

# Dataset и DataLoader
pytest tests/test_dataset_dataloader.py -v

# Интеграционные тесты данных
pytest tests/test_data_integration.py -v

# Аугментации
pytest tests/test_augmentations.py -v

# Функции потерь
pytest tests/test_losses.py -v

# Метрики
pytest tests/test_metrics.py -v

# Оптимизаторы и scheduler'ы
pytest tests/test_optimizers.py -v

# Трекинг градиентов
pytest tests/test_gradient_tracker.py -v

# Модели
pytest tests/test_models.py -v

# Интеграционные тесты обучения
pytest tests/test_training_integration.py -v
```

### Один конкретный тест

```bash
pytest tests/test_dataset_dataloader.py::TestGeologyTrapsDataset::test_dataset_getitem_with_faults -v
```

## Маркеры тестов

Тесты используют следующие маркеры:

| Маркер | Описание | Когда использовать |
|--------|----------|-------------------|
| `smoke` | Быстрые тесты базовой функциональности | Для быстрой проверки после изменений |
| `slow` | Медленные тесты (полное обучение) | Для nightly builds или перед релизом |
| `integration` | Интеграционные тесты | Для проверки взаимодействия компонентов |

Пример запуска без медленных тестов:

```bash
pytest tests/ -v -m "not slow"
```

## Категории тестов

### 1. Утилиты данных (`test_dataset_utils.py`)

Юнит тесты вспомогательных функций обработки данных:

- **TestParseFilename**: Парсинг имен файлов различных типов (structuralNOisoline, structuralBlackWhite, isolines, faults, traps)
- **TestGetSampleKey**: Генерация ключей семплов из имени файла
- **TestCollectSamples**: Сбор семплов из списка файлов, группировка по типам карт
- **TestResolvePath**: Разрешение путей (относительные/абсолютные)
- **TestLoadMapsIntoNdarray**: Загрузка карт в ndarray с/без разломов
- **TestImageUtils**: Создание бинарных масок, map_mask, паддинг, загрузка numpy массивов

Пример теста:
```python
def test_parse_filename_isolines(self):
    filename = "well_123_Bobrikovsky_structuralNOisoline.png"
    result = parse_filename(filename)
    assert result["type"] == "iso"
    assert result["well"] == "123"
    assert result["horizon"] == "Bobrikovsky"
```

### 2. Dataset и DataLoader (`test_dataset_dataloader.py`)

Юнит тесты основных классов загрузки данных:

- **TestGetFileList**: Получение списка файлов из директории, фильтрация невалидных имен
- **TestSplitDataByGroups**: Разделение данных с группировкой по горизонтам, проверка отсутствия утечек
- **TestGeologyTrapsDataset**: Инициализация, getitem с/без разломов, метаданные, фильтрация NoData, аугментации
- **TestCreateDataloaders**: Создание DataLoader'ов, проверка батчей

Пример теста:
```python
def test_split_data_no_leakage(self, multiple_samples_data):
    """Тест что данные одного горизонта не попадают в разные выборки."""
    files = get_file_list(...)
    samples = collect_samples(files)
    train_files, val_files, test_files = split_data_by_groups(samples)
    
    # Проверка что горизонты не пересекаются
    train_horizons = {f.split('_')[2] for f in train_files}
    val_horizons = {f.split('_')[2] for f in val_files}
    assert train_horizons.isdisjoint(val_horizons)
```

### 3. Интеграционные тесты данных (`test_data_integration.py`)

Тесты полного пайплайна от файлов до DataLoader:

- **TestDataPipelineIntegration**: Полный пайплайн без/с разломами
- **Группировка по горизонтам**: Предотвращение утечки данных между выборками
- **Консистентность батчей**: Проверка размеров и типов данных в батчах
- **Сохранение метаданных**: Проверка что метаданные проходят через весь пайплайн
- **Корректность паддинга**: Проверка применения паддинга к изображениям
- **Диапазоны значений масок**: Валидация значений выходных масок
- **Объединение каналов**: Проверка количества входных каналов (6 без faults, 7 с faults)
- **TestEdgeCases**: Граничные случаи (один семпл, отсутствие опциональных файлов, большой batch_size)

### 4. Аугментации (`test_augmentations.py`)

Тесты функций аугментации данных:

- Создание train/val трансформов
- Применение трансформов к данным
- Проверка что val трансформы не применяют рандомизацию
- Консистентность масок после трансформов
- Разные random seed дают разные результаты

### 5. Функции потерь (`test_losses.py`)

Тесты функций потерь:

- **MaskedBCELoss**: Бинарная кросс-энтропия с маской
- **MaskedDiceLoss**: Dice loss с маской
- **CombinedLoss**: Комбинация BCE + Dice
- Проверка работы с различными типами масок
- Тесты граничных случаев (пустая маска, вся маска)

### 6. Метрики (`test_metrics.py`)

Тесты калькулятора метрик:

- **IoU (Intersection over Union)**: С маской и без
- **Dice Coefficient**: Коэффициент схожести
- **Recall/Precision**: Полнота и точность
- **F1 Score**: Гармоническое среднее
- **FP/FN Area**: Площадь ложных срабатываний/пропусков
- Проверка работы с edge cases (пустые предсказания, идеальное совпадение)

### 7. Оптимизаторы (`test_optimizers.py`)

Тесты создания оптимизаторов и scheduler'ов:

- **create_optimizer_and_scheduler**: Создание Adam/SGD оптимизаторов
- Типы scheduler'ов (CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR)
- **get_gradient_stats**: Статистики градиентов (norm, min, max)
- Проверка параметров оптимизации (lr, weight_decay)

### 8. Трекинг градиентов (`test_gradient_tracker.py`)

Тесты системы отслеживания аномалий градиентов:

- **GradientNormTracker**: Трекер норм градиентов
- **compute_per_sample_grad_norms**: Вычисление норм градиентов на семпл
- Обнаружение exploding/vanishing градиентов
- Статистики по эпохам
- Логирование и визуализация

### 9. Модели (`test_models.py`)

Тесты архитектуры модели:

- **U-Net++**: Инициализация модели
- Forward pass с различными размерами входов
- Проверка количества каналов входа/выхода
- Работа с/без deep supervision
- Сохранение/загрузка весов модели

### 10. Интеграционные тесты обучения (`test_training_integration.py`)

Тесты полного цикла обучения:

- **TrainingStepIntegration**: Один шаг обучения
- **EvaluationPipeline**: Полный цикл evaluation
- **CheckpointSaving**: Сохранение и загрузка чекпоинтов
- **MetricsCalculation**: Расчет метрик на валидации
- **GradClipIntegration**: Интеграция gradient clipping
- **MultiEpochSimulation**: Симуляция нескольких эпох

## Как понять, что тесты прошли успешно

### Успешный запуск

```
============================= test session starts ==============================
platform linux -- Python 3.x.x, pytest-x.x.x
collected 162 items

tests/test_dataset_utils.py ....................                         [ 12%]
tests/test_dataset_dataloader.py ..................                      [ 23%]
tests/test_data_integration.py .........                                 [ 29%]
tests/test_augmentations.py ........                                     [ 34%]
tests/test_losses.py ......................                              [ 47%]
tests/test_metrics.py ..........................                         [ 63%]
tests/test_optimizers.py .................                               [ 74%]
tests/test_gradient_tracker.py ....................                      [ 86%]
tests/test_models.py ......                                              [ 90%]
tests/test_training_integration.py ...............                       [100%]

======================== 162 passed in 45.23s =============================
```

✅ Все тесты прошли: `XX passed`

### Проваленные тесты

```
============================= test session starts ==============================
collected 162 items

tests/test_losses.py ....F.....                                         [ 20%]

=================================== FAILURES ===================================
_________________ TestMaskedBCELoss.test_loss_computation ____________________

self = <tests.test_losses.TestMaskedBCELoss object at 0x...>

    def test_loss_computation(self):
        output = torch.randn(4, 1, 128, 128)
        target = torch.randint(0, 2, (4, 1, 128, 128)).float()
        mask_map = torch.ones_like(target)
        
        loss_fn = MaskedBCELoss()
        loss = loss_fn(output, target, mask_map)
>       assert loss > 0
E       AssertionError: assert tensor(0.) > 0

tests/test_losses.py:XX: AssertionError
=========================== short test summary info ============================
FAILED tests/test_losses.py::TestMaskedBCELoss::test_loss_computation
========================= 1 failed, 161 passed in 42.34s ========================
```

❌ Тесты провалены: смотрите `FAILURES` и `AssertionError`

### Ошибки

```
==================================== ERRORS ====================================
_______________ ERROR at setup of test_something _______________

    @pytest.fixture
    def sample_batch():
>       return load_data("/nonexistent")
E       FileNotFoundError: [Errno 2] No such file or directory

tests/test_training_fixtures.py:XX: FileNotFoundError
=========================== short test summary info ============================
ERROR tests/test_training_fixtures.py::test_something - FileNotFoundError
========================== 1 error in 0.5s ====================================
```

⚠️ Ошибки (ERROR): проблемы в фикстурах или настройке, не в самих тестах

## Интерпретация результатов

### Покрытие кода

После запуска с `--cov`:

```
Name                                Stmts   Miss  Cover
-------------------------------------------------------
data/dataset.py                       150     20    87%
data/dataloaders.py                   100      5    95%
data/utils.py                         120     15    88%
utils/augmentations.py                 80      8    90%
losses/losses.py                       60      2    97%
metrics/metrics.py                    100      5    95%
optimizers/optimizers.py               70      3    96%
training/gradient_tracker.py           90      7    92%
models/unetplusplus.py                 80     10    88%
-------------------------------------------------------
TOTAL                                 850     75    91%
```

Цель: >85% покрытия для критических модулей.

### Длительность тестов

- **Smoke тесты**: < 10 секунд
- **Юнит тесты**: < 1 минута
- **Интеграционные тесты**: 1-3 минуты
- **Slow тесты**: 3-10 минут

## CI/CD интеграция

### GitHub Actions пример

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run smoke tests
        run: pytest tests/ -m "smoke" -v
      
      - name: Run all tests with coverage
        run: pytest tests/ --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Добавление новых тестов

### Шаблон теста

```python
import pytest
import torch
from your_module import your_function

class TestYourFeature:
    """Tests for your feature."""
    
    def test_happy_path(self):
        """Test normal case."""
        result = your_function(input_data)
        assert result == expected_value
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
    
    @pytest.mark.smoke
    def test_quick_check(self):
        """Quick smoke test."""
        assert your_function(simple_input) is not None
    
    @pytest.mark.parametrize("input_size,expected_channels", [
        ((128, 128), 6),
        ((256, 256), 6),
        ((512, 512), 6),
    ])
    def test_different_sizes(self, input_size, expected_channels):
        """Parameterized test for different input sizes."""
        x = torch.randn(1, 3, *input_size)
        output = your_model(x)
        assert output.shape[1] == expected_channels
```

### Best practices

1. **Используйте фикстуры** для общих данных (см. `conftest.py`)
2. **Маркируйте тесты** appropriately (`smoke`, `slow`, `integration`)
3. **Тестируйте один аспект** за раз
4. **Используйте параметризацию** для похожих тестов
5. **Изолируйте тесты** - каждый тест должен работать независимо
6. **Используйте temp_dir** для временных файлов
7. **Мокайте внешние зависимости** (wandb, filesystem)

## Отладка упавших тестов

### Запуск с отладочной информацией

```bash
pytest tests/test_file.py::test_name -v -s
```

Флаг `-s` показывает print statements.

### Post-mortem отладка

```bash
pytest tests/test_file.py --pdb
```

Останавливается на первом провале для интерактивной отладки.

### Логирование

```bash
pytest tests/ --log-cli-level=INFO
```

### Запуск конкретного теста несколько раз

```bash
pytest tests/test_augmentations.py::test_different_seeds_produce_different_results --count=5
```

Требует `pytest-repeat`: `pip install pytest-repeat`

## Ответ на вопрос про validate_dataset.py

**Нужен ли `validate_dataset/validate_dataset.py` если есть тесты?**

**Да, нужен!** Это разные инструменты:

| Аспект | `validate_dataset.py` | Тесты (`tests/`) |
|--------|----------------------|------------------|
| **Цель** | Валидация реальных данных пользователя | Валидация кода проекта |
| **Когда** | Перед обучением на новых данных | При разработке и CI/CD |
| **Что проверяет** | Конкретные файлы данных | Логику кода |
| **Аудитория** | Пользователи проекта | Разработчики |

**Рекомендация:**
- Оставьте `validate_dataset.py` для пользователей
- Используйте тесты для разработки
- Можно добавить тесты, которые проверяют сам `validate_dataset.py`

## Troubleshooting

### Тесты не находят модули

```bash
# Убедитесь, что вы в корне проекта
cd /workspace

# Запускайте pytest из корня
pytest tests/
```

### Проблемы с CUDA

Все тесты настроены на CPU. Если нужны GPU тесты:

```python
@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Memory issues

Для тестов с большими данными:

```bash
pytest tests/ -v --maxfail=1  # Остановиться после первой ошибки
```

### Случайные failures из-за random seed

Если тесты иногда падают из-за рандомизации:

```bash
# Запустить с фиксированным seed
pytest tests/ --randomly-seed=42
```

## Текущая статистика тестов

На момент последнего обновления:

- **Всего тестов**: 162
- **Юнит тесты**: ~120
- **Интеграционные тесты**: ~42
- **Покрытие кода**: ~90% (для основных модулей)
- **Время выполнения**: ~45 секунд

### Распределение по модулям:

| Модуль | Количество тестов | Файл |
|--------|------------------|------|
| Утилиты данных | 32 | test_dataset_utils.py |
| Dataset/DataLoader | 18 | test_dataset_dataloader.py |
| Интеграция данных | 10 | test_data_integration.py |
| Аугментации | 8 | test_augmentations.py |
| Функции потерь | 22 | test_losses.py |
| Метрики | 26 | test_metrics.py |
| Оптимизаторы | 17 | test_optimizers.py |
| Трекинг градиентов | 20 | test_gradient_tracker.py |
| Модели | ~6 | test_models.py |
| Интеграция обучения | 10 | test_training_integration.py |

## Контакты и поддержка

При возникновении проблем:
1. Проверьте логи ошибок
2. Запустите с `-v -s` флагами
3. Проверьте версию зависимостей
4. Убедитесь, что все данные на месте
5. Посмотрите примеры работающих тестов в `tests/`
