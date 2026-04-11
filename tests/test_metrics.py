import pytest
import torch

from metrics.metrics import MetricsCalculator


class TestMetricsCalculator:
    """Тесты для MetricsCalculator."""

    @pytest.fixture
    def calculator(self):
        """Фикстура с калькулятором метрик."""
        return MetricsCalculator(threshold=0.5, smooth=1e-6)

    @pytest.fixture
    def sample_data(self):
        """Пример данных для тестов."""
        # Perfect predictions
        perfect_preds = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)
        perfect_targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)

        # Zero predictions
        zero_preds = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]], dtype=torch.float32)
        zero_targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)

        # Partial predictions
        partial_preds = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
        partial_targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)

        return {
            'perfect': (perfect_preds, perfect_targets),
            'zero': (zero_preds, zero_targets),
            'partial': (partial_preds, partial_targets)
        }

    def test_perfect_predictions(self, calculator, sample_data):
        """Тест идеальных предсказаний - все метрики должны быть близки к 1.0."""
        preds, targets = sample_data['perfect']
        metrics = calculator.compute_all(preds, targets)

        assert metrics['iou'] == pytest.approx(1.0, rel=1e-3)
        assert metrics['dice'] == pytest.approx(1.0, rel=1e-3)
        assert metrics['recall'] == pytest.approx(1.0, rel=1e-3)
        assert metrics['precision'] == pytest.approx(1.0, rel=1e-3)
        assert metrics['f1'] == pytest.approx(1.0, rel=1e-3)
        assert metrics['fp_area'] == pytest.approx(0.0, abs=1e-3)
        assert metrics['fn_area'] == pytest.approx(0.0, abs=1e-3)

    def test_zero_predictions(self, calculator, sample_data):
        """Тест нулевых предсказаний - метрики должны быть низкими."""
        preds, targets = sample_data['zero']
        metrics = calculator.compute_all(preds, targets)

        # При нулевых предсказаниях IoU и Dice должны быть близки к 0
        assert metrics['iou'] == pytest.approx(0.0, abs=0.1)
        assert metrics['dice'] == pytest.approx(0.0, abs=0.1)
        # Recall должен быть 0 (нет TP)
        assert metrics['recall'] == pytest.approx(0.0, abs=0.1)
        # Precision может быть низким или undefined (обработано smooth)
        assert metrics['precision'] >= 0.0
        assert metrics['fn_area'] > 0.5  # Много FN

    def test_partial_predictions(self, calculator, sample_data):
        """Тест частичных предсказаний."""
        preds, targets = sample_data['partial']
        metrics = calculator.compute_all(preds, targets)

        # 2 из 4 пикселей предсказаны правильно
        assert 0.3 < metrics['iou'] < 0.7
        assert 0.3 < metrics['dice'] < 0.7
        assert metrics['recall'] == pytest.approx(0.5, rel=0.1)  # 2/4
        assert metrics['precision'] == pytest.approx(1.0, rel=0.1)  # Все предсказанные верны

    def test_compute_iou单独(self, calculator, sample_data):
        """Тест отдельного вычисления IoU."""
        preds, targets = sample_data['perfect']
        iou = calculator.compute_iou(preds, targets)
        assert iou == pytest.approx(1.0, rel=1e-3)

    def test_compute_dice单独(self, calculator, sample_data):
        """Тест отдельного вычисления Dice."""
        preds, targets = sample_data['perfect']
        dice = calculator.compute_dice(preds, targets)
        assert dice == pytest.approx(1.0, rel=1e-3)

    def test_with_mask(self, calculator):
        """Тест с маской для фильтрации областей."""
        # Предсказания и таргеты
        preds = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
        targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32)

        # Маска игнорирует правую половину
        mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)

        metrics_with_mask = calculator.compute_all(preds, targets, mask=mask)
        metrics_without_mask = calculator.compute_all(preds, targets, mask=None)

        # С маской метрики должны быть лучше (игнорируем FN справа)
        assert metrics_with_mask['dice'] > metrics_without_mask['dice']

    def test_batch_processing(self, calculator):
        """Тест обработки батча."""
        batch_size = 4
        preds = torch.ones(batch_size, 1, 10, 10)
        targets = torch.ones(batch_size, 1, 10, 10)

        metrics = calculator.compute_all(preds, targets)

        assert metrics['iou'] == pytest.approx(1.0, rel=1e-3)
        assert metrics['dice'] == pytest.approx(1.0, rel=1e-3)

    def test_4d_sigmoid_input(self, calculator):
        """Тест входных данных с логитами (4D тензор)."""
        # Логиты (до сигмоиды)
        logits = torch.tensor([[[[2.0, -2.0], [2.0, -2.0]]]], dtype=torch.float32)
        targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)

        metrics = calculator.compute_all(logits, targets)

        # После порога 0.5, sigmoid(2.0) > 0.5, sigmoid(-2.0) < 0.5
        assert metrics['precision'] == pytest.approx(1.0, rel=0.1)
        assert metrics['recall'] == pytest.approx(1.0, rel=0.1)

    def test_edge_case_all_zeros(self, calculator):
        """Тест граничного случая - все нули."""
        preds = torch.zeros(1, 1, 10, 10)
        targets = torch.zeros(1, 1, 10, 10)

        metrics = calculator.compute_all(preds, targets)

        # Когда всё 0, метрики должны быть обработаны через smooth
        assert metrics['iou'] > 0  # smooth предотвращает деление на 0
        assert metrics['dice'] > 0

    def test_different_thresholds(self):
        """Тест различных порогов."""
        calc_low = MetricsCalculator(threshold=0.3)
        calc_high = MetricsCalculator(threshold=0.7)

        # Предсказания с вероятностью 0.5
        preds = torch.ones(1, 1, 10, 10) * 0.5
        targets = torch.ones(1, 1, 10, 10)

        metrics_low = calc_low.compute_all(preds, targets)
        metrics_high = calc_high.compute_all(preds, targets)

        # Низкий порог даст лучшие метрики для pred=0.5
        assert metrics_low['dice'] > metrics_high['dice']

    def test_fp_fn_area_calculation(self, calculator):
        """Тест расчёта площадей FP и FN."""
        # Создаём ситуацию с явными FP и FN
        preds = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32)
        targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)

        metrics = calculator.compute_all(preds, targets)

        # Должны быть и FP и FN
        assert metrics['fp_area'] > 0
        assert metrics['fn_area'] > 0
