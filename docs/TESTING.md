# Тестирование проекта Geology Traps Segmentation

## Обзор

Этот документ описывает систему тестирования для проекта сегментации геологических ловушек. 
Тесты покрывают все ключевые компоненты проекта:

1. **Валидация входных данных** (PNG и CPS форматы)
2. **Загрузка данных в GeologyTrapsDataset**
3. **Проверка готовности к обучению**
4. **Процесс обучения**

## Структура тестов

```
tests/
├── __init__.py              # Инициализация пакета тестов
├── conftest.py              # Общие фикстуры и конфигурация pytest
├── test_data_validation.py  # Тесты утилит загрузки и обработки данных
├── test_dataset.py          # Тесты GeologyTrapsDataset и DataLoader
├── test_training.py         # Тесты модели, лоссов, оптимизаторов
└── test_pipeline.py         # Интеграционные тесты полного пайплайна
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
# Валидация данных
pytest tests/test_data_validation.py -v

# Dataset и DataLoader
pytest tests/test_dataset.py -v

# Обучение и модель
pytest tests/test_training.py -v

# Полный пайплайн (медленные тесты)
pytest tests/test_pipeline.py -v -m "not slow"
```

### Один конкретный тест

```bash
pytest tests/test_dataset.py::TestDatasetGetItem::test_getitem_returns_required_keys -v
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

### 1. Валидация данных (`test_data_validation.py`)

Проверяет корректность обработки входных данных:

- **ImageLoading**: Загрузка RGB и grayscale изображений
- **MaskCreation**: Создание бинарных масок
- **Padding**: Паддинг изображений до целевого размера
- **CPSUtils**: Чтение и конвертация CPS файлов
- **FileNamingValidation**: Валидация имен файлов

Пример теста:
```python
def test_load_image_returns_rgb(self, sample_rgb_image):
    img = load_image(str(sample_rgb_image))
    assert img.shape == (100, 100, 3)
    assert img.dtype == np.uint8
```

### 2. Dataset (`test_dataset.py`)

Проверяет загрузку данных в GeologyTrapsDataset:

- **FileListCollection**: Сбор и группировка файлов
- **DatasetInitialization**: Инициализация датасета
- **DatasetGetItem**: Извлечение семплов
- **DataLoaderCreation**: Создание DataLoader

Пример теста:
```python
def test_getitem_input_channels_with_faults(self, complete_sample):
    dataset = GeologyTrapsDataset(..., use_faults=True)
    sample = dataset[0]
    assert sample['x'].shape[0] == 5  # RGB(3) + depth(1) + faults(1)
```

### 3. Training readiness (`test_training.py`)

Проверяет готовность к обучению:

- **ModelLoading**: Загрузка и архитектура модели
- **LossFunctions**: Функции потерь
- **MetricsCalculator**: Расчет метрик
- **OptimizerAndScheduler**: Оптимизаторы и планировщики
- **TrainingReadiness**: Базовые проверки training loop
- **CheckpointSaving**: Сохранение/загрузка чекпоинтов

Пример теста:
```python
def test_complete_training_iteration(self):
    output = model(x)
    loss, metrics = criterion(output, y, mask_map=mask_map)
    loss.backward()
    optimizer.step()
    # Проверка градиентов и обновления весов
```

### 4. Pipeline (`test_pipeline.py`)

Интеграционные тесты полного пайплайна:

- **PipelineIntegration**: Полный цикл обучения
- **OverfitCheck**: Режим проверки overfit
- **Evaluation**: Оценка на тестовых данных

## Как понять, что тесты прошли успешно

### Успешный запуск

```
============================= test session starts ==============================
platform linux -- Python 3.x.x, pytest-x.x.x
collected 50 items

tests/test_data_validation.py ............                               [ 24%]
tests/test_dataset.py ................                                   [ 56%]
tests/test_training.py ...............                                   [ 86%]
tests/test_pipeline.py .......                                           [100%]

======================== 50 passed in 15.23s =============================
```

✅ Все тесты прошли: `XX passed`

### Проваленные тесты

```
============================= test session starts ==============================
collected 50 items

tests/test_dataset.py ....F.....                                         [ 20%]

=================================== FAILURES ===================================
_________________ TestDatasetGetItem.test_getitem_channels ____________________

self = <tests.test_dataset.TestDatasetGetItem object at 0x...>

    def test_getitem_channels(self, complete_sample):
        sample = dataset[0]
>       assert sample['x'].shape[0] == 5
E       AssertionError: assert 4 == 5
E        +  where 4 = torch.Size([4, 128, 128]).shape[0]

tests/test_dataset.py:XX: AssertionError
=========================== short test summary info ============================
FAILED tests/test_dataset.py::TestDatasetGetItem::test_getitem_channels
========================= 1 failed, 49 passed in 12.34s ========================
```

❌ Тесты провалены: смотрите `FAILURES` и `AssertionError`

### Ошибки

```
==================================== ERRORS ====================================
_______________ ERROR at setup of test_something _______________

    @pytest.fixture
    def sample_data():
>       return load_data("/nonexistent")
E       FileNotFoundError: [Errno 2] No such file or directory

tests/test_file.py:XX: FileNotFoundError
=========================== short test summary info ============================
ERROR tests/test_file.py::test_something - FileNotFoundError
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
models/unetplusplus.py                 80     10    88%
losses/losses.py                       60      2    97%
-------------------------------------------------------
TOTAL                                 390     37    91%
```

Цель: >85% покрытия для критических модулей.

### Длительность тестов

- **Smoke тесты**: < 10 секунд
- **Обычные тесты**: < 1 минута
- **Slow тесты**: 1-5 минут

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
```

### Best practices

1. **Используйте фикстуры** для общих данных
2. **Маркируйте тесты** appropriately (`smoke`, `slow`)
3. **Тестируйте один аспект** за раз
4. **Используйте параметризацию** для похожих тестов:

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double(input, expected):
    assert double(input) == expected
```

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
cd /path/to/project

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

## Контакты и поддержка

При возникновении проблем:
1. Проверьте логи ошибок
2. Запустите с `-v -s` флагами
3. Проверьте версию зависимостей
4. Убедитесь, что все данные на месте
