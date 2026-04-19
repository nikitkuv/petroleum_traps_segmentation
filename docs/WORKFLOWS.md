# Добавление новых cps гридов
1. data\remove_extra_symbols_from_filename.py
2. visualization\visualize_cps_rgb_vs_isolines.py
3. visualization\visualize_closed_isolines.py
4. data\convert_cps_to_tiles.py

# Проверка семплов и датасета
1. data\check_image_sizes.py
2. data\cleanup_files.py
3. data\validate_dataset.py
4. visualization\visualize_dataset_sample.py

# Тесты
```bash
pytest tests/ --cov=. --cov-report=html
```

# Overfit check
1. Выставить `AUGMENT_TRAIN = False` в `settings.py`
2. Запускаем проверку пайплайна на переобучение
```bash
pytest run_overfit_check.py
```