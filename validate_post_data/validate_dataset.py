import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List, Dict
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from settings import settings
from dataset import GeologyTrapsDataset


class DatasetValidator:
    """Валидатор для проверки корректности Dataset."""
    
    def __init__(self, file_list: List[str], data_dir: str = None, data_source: str = None):
        self.file_list = file_list
        self.data_dir = data_dir or str(settings.data_path)
        self.data_source = data_source if data_source is not None else settings.DATA_SOURCE
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
    def scan_directory(self) -> List[str]:
        """Сканирует директорию и находит все файлы."""
        
        if self.data_source == 'cps':
            # CPS: ищем файлы без расширения или .cps
            data_path = Path(self.data_dir)
            files = []
            for ext in ['*.cps', '*.grd', '*']:
                files.extend([f.name.replace('.cps', '').replace('.grd', '') 
                            for f in data_path.glob(ext) if f.is_file()])
            files = sorted(list(set(files)))
        else:
            # PNG: ищем только .png
            data_path = Path(self.data_dir)
            files = sorted([f.name for f in data_path.glob("*.png")])
        
        print(f"📁 Found {len(files)} files in {self.data_dir}")
        
        self.file_list = files
        
        return files
    
    # ... остальной код без изменений ...
    
    def analyze_files(self, file_list: List[str]) -> Dict:
        """Анализирует структуру файлов."""
        analysis = {
            'total_files': len(file_list),
            'unique_cards': set(),
            'cards_with_faults': set(),
            'cards_without_faults': set(),
            'missing_files': defaultdict(list),
            'file_types': defaultdict(int)
        }
        
        for f in file_list:
            parts = f.split('_')
            if len(parts) >= 4:
                # Убираем .png из названия для CPS совместимости
                card_id = f"{parts[0]}_{'_'.join(parts[3:]).replace('.png', '').replace('.cps', '')}"
                analysis['unique_cards'].add(card_id)
                
                subtype = parts[2]
                if 'faults' in subtype:
                    analysis['cards_with_faults'].add(card_id)
                    analysis['file_types']['faults'] += 1
                elif 'structuralBlackWhite' in subtype:
                    analysis['file_types']['depth_norm'] += 1
                elif 'structuralNOisoline' in subtype:
                    analysis['file_types']['rgb'] += 1
                elif 'traps' in subtype:
                    analysis['file_types']['traps'] += 1
        
        analysis['cards_without_faults'] = analysis['unique_cards'] - analysis['cards_with_faults']
        analysis['unique_cards'] = len(analysis['unique_cards'])
        analysis['cards_with_faults'] = len(analysis['cards_with_faults'])
        analysis['cards_without_faults'] = len(analysis['cards_without_faults'])
        
        return analysis
    
    def validate_dataset(self, use_faults: bool = None) -> Dict:
        """Проверяет Dataset на ошибки."""
        use_faults = use_faults if use_faults is not None else settings.USE_FAULTS
        
        print("\n" + "=" * 70)
        print(f"🔍 DATASET VALIDATION (use_faults={use_faults})")
        print("=" * 70)
        
        # 🔧 ИСПРАВЛЕНИЕ: Если file_list пустой — сканируем директорию
        if len(self.file_list) == 0:
            print("⚠️  file_list is empty, scanning directory...")
            self.scan_directory()
        
        # 1. Анализ файлов
        print("\n📊 FILE ANALYSIS:")
        print("-" * 70)
        analysis = self.analyze_files(self.file_list)
        
        print(f"  Total files:           {analysis['total_files']}")
        print(f"  Unique cards:          {analysis['unique_cards']}")
        print(f"  Cards with faults:     {analysis['cards_with_faults']}")
        print(f"  Cards without faults:  {analysis['cards_without_faults']}")
        print(f"\n  File types:")
        for ftype, count in analysis['file_types'].items():
            print(f"    - {ftype}: {count}")
        
        # 🔧 ИСПРАВЛЕНИЕ: Проверка на пустой датасет
        if analysis['unique_cards'] == 0:
            print("\n❌ No valid cards found! Check file naming convention.")
            print("   Expected format: NNN_x_type_HORIZON.png")
            print("   Example: 001_x_structuralNOisoline_H150.png")
            return {
                'success': False,
                'errors': ['No valid cards found'],
                'total_samples': 0,
                'successful_samples': 0,
                'failed_samples': 0,
                'warnings': self.warnings,
                'shape_stats': {},
                'value_stats': {},
                'file_analysis': analysis
            }
        
        # 2. Создание Dataset
        print("\n📦 CREATING DATASET:")
        print("-" * 70)
        try:
            dataset = GeologyTrapsDataset(
                file_list=self.file_list,
                data_dir=self.data_dir,
                augment=False,
                use_faults=use_faults
            )
            print(f"  ✓ Dataset created successfully")
            print(f"  ✓ Total samples: {len(dataset)}")
        except Exception as e:
            self.errors.append(f"Dataset creation failed: {str(e)}")
            print(f"  ❌ Dataset creation failed: {str(e)}")
            return {
                'success': False,
                'errors': self.errors,
                'total_samples': 0,
                'successful_samples': 0,
                'failed_samples': 0,
                'warnings': self.warnings,
                'shape_stats': {},
                'value_stats': {},
                'file_analysis': analysis
            }
        
        # 🔧 ИСПРАВЛЕНИЕ: Проверка на пустой датасет после создания
        if len(dataset) == 0:
            print("\n❌ Dataset has 0 samples! Check file naming and _parse_files logic.")
            return {
                'success': False,
                'errors': ['Dataset has 0 samples'],
                'total_samples': 0,
                'successful_samples': 0,
                'failed_samples': 0,
                'warnings': self.warnings,
                'shape_stats': {},
                'value_stats': {},
                'file_analysis': analysis
            }
        
        # 3. Проверка каждого семпла
        print("\n🔍 VALIDATING EACH SAMPLE:")
        print("-" * 70)
        
        expected_channels = 5 if use_faults else 4
        successful_samples = 0
        failed_samples = 0
        channel_mismatches = 0
        
        shape_stats = {
            'x_shapes': set(),
            'y_shapes': set(),
            'mask_map_shapes': set(),
            'mask_depth_shapes': set()
        }
        
        value_stats = {
            'x_min': float('inf'),
            'x_max': float('-inf'),
            'y_min': float('inf'),
            'y_max': float('-inf'),
            'traps_pixels': 0,
            'total_pixels': 0
        }
        
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                
                # Проверка структуры
                required_keys = ['x', 'y', 'mask_depth', 'mask_map', 'sample_idx', 'use_faults']
                for key in required_keys:
                    if key not in sample:
                        self.errors.append(f"Sample {idx}: Missing key '{key}'")
                        failed_samples += 1
                        continue
                
                # Проверка каналов
                actual_channels = sample['x'].shape[0]
                if actual_channels != expected_channels:
                    self.errors.append(
                        f"Sample {idx}: Expected {expected_channels} channels, got {actual_channels}"
                    )
                    channel_mismatches += 1
                    failed_samples += 1
                else:
                    successful_samples += 1
                
                # Сбор статистики форм
                shape_stats['x_shapes'].add(tuple(sample['x'].shape))
                shape_stats['y_shapes'].add(tuple(sample['y'].shape))
                shape_stats['mask_map_shapes'].add(tuple(sample['mask_map'].shape))
                if sample['mask_depth'] is not None:
                    shape_stats['mask_depth_shapes'].add(tuple(sample['mask_depth'].shape))
                
                # Сбор статистики значений
                value_stats['x_min'] = min(value_stats['x_min'], sample['x'].min().item())
                value_stats['x_max'] = max(value_stats['x_max'], sample['x'].max().item())
                value_stats['y_min'] = min(value_stats['y_min'], sample['y'].min().item())
                value_stats['y_max'] = max(value_stats['y_max'], sample['y'].max().item())
                value_stats['traps_pixels'] += (sample['y'] > 0).sum().item()
                value_stats['total_pixels'] += sample['y'].numel()
                
                # Прогресс
                if (idx + 1) % 50 == 0 or idx == len(dataset) - 1:
                    print(f"  Processed {idx + 1}/{len(dataset)} samples "
                          f"({(idx + 1) / len(dataset) * 100:.1f}%)")
                
            except Exception as e:
                self.errors.append(f"Sample {idx}: {str(e)}")
                failed_samples += 1
                print(f"  ❌ Sample {idx} failed: {str(e)}")
        
        # 4. Проверка DataLoader
        print("\n📦 VALIDATING DATALOADER:")
        print("-" * 70)
        
        try:
            loader = DataLoader(
                dataset,
                batch_size=settings.BATCH_SIZE,
                shuffle=False,
                num_workers=settings.NUM_WORKERS,
                pin_memory=True
            )
            
            batch_count = 0
            total_batch_samples = 0
            
            for batch_idx, batch in enumerate(loader):
                # Проверка формы батча
                if batch['x'].shape[0] != batch['y'].shape[0]:
                    self.errors.append(f"Batch {batch_idx}: X and Y batch size mismatch")
                
                batch_count += 1
                total_batch_samples += batch['x'].shape[0]
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches "
                          f"({total_batch_samples} samples)")
            
            print(f"  ✓ DataLoader validation passed")
            print(f"  ✓ Total batches: {batch_count}")
            print(f"  ✓ Total samples in batches: {total_batch_samples}")
            
        except Exception as e:
            self.errors.append(f"DataLoader validation failed: {str(e)}")
            print(f"  ❌ DataLoader validation failed: {str(e)}")
        
        # 5. Итоговый отчёт
        print("\n" + "=" * 70)
        print("📋 VALIDATION REPORT")
        print("=" * 70)
        
        # 🔧 ИСПРАВЛЕНИЕ: Защита от деления на ноль
        total = len(dataset)
        success_pct = (successful_samples / total * 100) if total > 0 else 0
        fail_pct = (failed_samples / total * 100) if total > 0 else 0
        
        print(f"\n✅ SUCCESSFUL SAMPLES: {successful_samples}/{total} ({success_pct:.1f}%)")
        print(f"❌ FAILED SAMPLES: {failed_samples}/{total} ({fail_pct:.1f}%)")
        print(f"⚠️  CHANNEL MISMATCHES: {channel_mismatches}")
        
        print(f"\n📊 SHAPE STATISTICS:")
        print(f"  X shapes: {len(shape_stats['x_shapes'])} unique")
        for shape in shape_stats['x_shapes']:
            print(f"    - {shape}")
        print(f"  Y shapes: {len(shape_stats['y_shapes'])} unique")
        for shape in shape_stats['y_shapes']:
            print(f"    - {shape}")
        
        print(f"\n📊 VALUE STATISTICS:")
        if value_stats['x_min'] != float('inf'):
            print(f"  X range: [{value_stats['x_min']:.4f}, {value_stats['x_max']:.4f}]")
            print(f"  Y range: [{value_stats['y_min']:.4f}, {value_stats['y_max']:.4f}]")
            print(f"  Traps coverage: {value_stats['traps_pixels'] / value_stats['total_pixels'] * 100:.2f}%")
        else:
            print("  No value statistics (no samples processed)")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for err in self.errors[:10]:  # Показать первые 10
                print(f"  - {err}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        
        # 6. Итоговый статус
        print("\n" + "=" * 70)
        if len(self.errors) == 0 and failed_samples == 0:
            print("✅ VALIDATION PASSED! Dataset is ready for training!")
            success = True
        else:
            print("❌ VALIDATION FAILED! Fix errors before training!")
            success = False
        print("=" * 70)
        
        return {
            'success': success,
            'total_samples': len(dataset),
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'errors': self.errors,
            'warnings': self.warnings,
            'shape_stats': shape_stats,
            'value_stats': value_stats,
            'file_analysis': analysis
        }


def main():
    """Главная функция валидации."""
    print("\n" + "=" * 70)
    print("🚀 GEOLOGY TRAPS DATASET VALIDATOR")
    print("=" * 70)
    
    # Создание директорий
    settings.create_dirs()
    
    # Сканирование директории
    validator = DatasetValidator(file_list=[], data_dir=settings.DATA_DIR)
    all_files = validator.scan_directory()
    
    if len(all_files) == 0:
        print("❌ No PNG files found in data directory!")
        return
    
    # 🔧 ИСПРАВЛЕНИЕ: Сохраняем файлы в validator
    validator.file_list = all_files
    
    # Валидация в режиме БЕЗ разломов
    print("\n" + "=" * 70)
    print("MODE 1: VALIDATION WITHOUT FAULTS (use_faults=False)")
    print("=" * 70)
    settings.USE_FAULTS = False
    result_no_faults = validator.validate_dataset(use_faults=False)
    
    # Если первая валидация провалилась — не продолжаем
    if not result_no_faults['success']:
        print("\n❌ First validation failed. Fix errors before continuing.")
        return
    
    # Валидация в режиме С разломами (опционально)
    print("\n\n")
    run_with_faults = input("Run validation WITH faults (use_faults=True)? [y/N]: ").strip().lower()
    
    if run_with_faults == 'y':
        print("\n" + "=" * 70)
        print("MODE 2: VALIDATION WITH FAULTS (use_faults=True)")
        print("=" * 70)
        settings.USE_FAULTS = True
        validator_with_faults = DatasetValidator(file_list=all_files, data_dir=settings.DATA_DIR)
        result_with_faults = validator_with_faults.validate_dataset(use_faults=True)
    else:
        result_with_faults = None
    
    # Сохранение отчёта
    report_path = settings.logs_path / 'dataset_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GEOLOGY TRAPS DATASET VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Validation Date: {torch.__version__}\n")
        f.write(f"Data Directory: {settings.DATA_DIR}\n\n")
        
        f.write("MODE 1: WITHOUT FAULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Success: {result_no_faults['success']}\n")
        f.write(f"Total Samples: {result_no_faults['total_samples']}\n")
        f.write(f"Successful: {result_no_faults['successful_samples']}\n")
        f.write(f"Failed: {result_no_faults['failed_samples']}\n")
        f.write(f"Errors: {len(result_no_faults['errors'])}\n\n")
        
        if result_with_faults:
            f.write("MODE 2: WITH FAULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Success: {result_with_faults['success']}\n")
            f.write(f"Total Samples: {result_with_faults['total_samples']}\n")
            f.write(f"Successful: {result_with_faults['successful_samples']}\n")
            f.write(f"Failed: {result_with_faults['failed_samples']}\n")
            f.write(f"Errors: {len(result_with_faults['errors'])}\n")
    
    print(f"\n📄 Report saved to: {report_path}")
    
    return result_no_faults, result_with_faults


if __name__ == "__main__":
    main()