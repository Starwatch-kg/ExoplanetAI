import numpy as np
from scipy import signal
from scipy import stats
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models
from transit_classifier import TransitClassifier

class SignalProcessor:
    """
    Класс для продвинутой обработки астрономических сигналов
    """
    def __init__(self, light_curve):
        self.light_curve = light_curve
        self.clean_curve = None
        self.features = {}
        self.classifier = TransitClassifier()  # Инициализация модели
        
    def remove_noise(self, method='wavelet'):
        """
        Удаление шумов из кривой блеска
        
        Параметры:
            method: 'wavelet' (вейвлет-фильтрация) или 'savgol' (фильтр Савицкого-Голея)
        """
        if method == 'wavelet':
            # Вейвлет-фильтрация для удаления теллурических линий и космических лучей
            coeffs = pywt.wavedec(self.light_curve, 'db4', level=5)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(self.light_curve)))
            coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
            self.clean_curve = pywt.waverec(coeffs, 'db4')
        
        elif method == 'savgol':
            # Фильтр Савицкого-Голея для сглаживания
            self.clean_curve = signal.savgol_filter(
                self.light_curve, 
                window_length=21, 
                polyorder=2
            )
        return self
        
    def detect_transits(self, threshold=5):
        """
        Обнаружение транзитов с помощью согласованного фильтра
        
        Параметры:
            threshold: порог обнаружения (в сигмах)
        """
        # Создаем шаблон транзита
        transit_template = np.zeros(100)
        transit_template[40:60] = -0.01  # Неглубокий транзит 1%
        
        # Применяем согласованный фильтр
        matched_filter = np.correlate(
            self.clean_curve - np.mean(self.clean_curve),
            transit_template,
            mode='same'
        )
        
        # Нормализуем и находим значимые пики
        matched_filter /= np.max(np.abs(matched_filter))
        self.transits = np.where(np.abs(matched_filter) > threshold * np.std(matched_filter))[0]
        self.features['transits'] = self.transits
        return self
        
    def analyze_periodicity(self):
        """
        Анализ периодичности сигнала с помощью периодограммы
        """
        # Вычисление периодограммы Ломба-Скаргля
        frequencies = np.linspace(0.01, 10, 10000)
        periodogram = signal.lombscargle(
            np.arange(len(self.clean_curve)),
            self.clean_curve,
            frequencies
        )
        
        # Находим наиболее значимый период
        dominant_freq = frequencies[np.argmax(periodogram)]
        self.features['period'] = 1 / dominant_freq
        return self
        
    def extract_features(self):
        """
        Извлечение признаков для классификации
        """
        # Простые статистические признаки
        self.features['mean'] = np.mean(self.clean_curve)
        self.features['std'] = np.std(self.clean_curve)
        self.features['skew'] = stats.skew(self.clean_curve)  
        self.features['kurtosis'] = stats.kurtosis(self.clean_curve)
        return self
        
    def classify_signal(self):
        """Классификация сигнала с помощью CNN"""
        if self.clean_curve is None:
            self.remove_noise('wavelet')
            
        # Используем центральный сегмент кривой
        segment_size = 100
        start = max(0, len(self.clean_curve)//2 - segment_size//2)
        segment = self.clean_curve[start:start+segment_size]
        
        # Классификация
        class_id, probabilities = self.classifier.predict(segment)
        class_names = ['planet', 'star', 'noise']
        
        self.features['classification'] = class_names[class_id]
        self.features['probabilities'] = probabilities.tolist()
        return self

# Пример использования
if __name__ == "__main__":
    # Генерация тестовых данных
    time = np.linspace(0, 10, 1000)
    flux = np.sin(time) + 0.1 * np.random.normal(size=1000)
    
    # Обработка сигнала
    processor = SignalProcessor(flux)\
        .remove_noise('wavelet')\
        .detect_transits()\
        .analyze_periodicity()\
        .extract_features()\
        .classify_signal()
        
    print("Обнаружены транзиты на позициях:", processor.transits)
    print("Доминирующий период:", processor.features['period'])
    print("Классификация:", processor.features['classification'])
    print("Вероятности:", processor.features['probabilities'])
