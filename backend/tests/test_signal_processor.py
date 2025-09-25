import pytest
import numpy as np
from signal_processor import SignalProcessor

class TestSignalProcessor:
    def test_noise_removal(self):
        # Генерация тестовых данных с шумом
        times = np.linspace(0, 10, 1000)
        flux = np.sin(times) + 0.1 * np.random.normal(size=1000)

        processor = SignalProcessor(flux)
        processor.remove_noise('wavelet')

        # Проверяем, что шум уменьшился
        assert processor.clean_curve is not None
        assert len(processor.clean_curve) == len(flux)

    def test_transit_detection(self):
        # Тест обнаружения транзитов
        times = np.linspace(0, 10, 1000)
        flux = np.ones(1000)
        flux[450:550] = 0.95  # Добавляем транзит

        processor = SignalProcessor(flux)
        processor.remove_noise('wavelet')
        processor.detect_transits(threshold=2.0, max_candidates=5)

        # Должен найти транзит
        assert len(processor.transits) > 0
        assert 450 <= processor.transits[0] <= 550

    def test_periodicity_analysis(self):
        # Тест анализа периодичности
        times = np.linspace(0, 10, 1000)
        flux = np.sin(2 * np.pi * times) + 0.1 * np.random.normal(size=1000)

        processor = SignalProcessor(flux)
        processor.remove_noise('wavelet')
        processor.analyze_periodicity()

        # Должен найти период ~1
        period = processor.features.get('period')
        assert 0.8 <= period <= 1.2

    def test_signal_classification(self):
        # Тест классификации сигналов
        times = np.linspace(0, 10, 1000)
        flux = np.ones(1000)
        flux[450:550] = 0.95  # Четкий транзит

        processor = SignalProcessor(flux)
        processor.remove_noise('wavelet')
        processor.classify_signal()

        # Должен классифицировать как планету
        probabilities = processor.features['probabilities']
        assert len(probabilities) == 3  # [планета, звезда, шум]
        assert probabilities[0] > 0.5  # Вероятность планеты > 50%
