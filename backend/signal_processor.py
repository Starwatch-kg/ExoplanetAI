import numpy as np
from scipy import signal
from scipy import stats
import pywt
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from transit_classifier import TransitClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

class SignalProcessor:
    """
    Класс для продвинутой обработки астрономических сигналов
    """
    def __init__(self, light_curve):
        self.light_curve = np.array(light_curve)
        self.clean_curve = None
        self.features = {}
        self.transits = []
        
        # Инициализация классификатора только если доступен
        if CLASSIFIER_AVAILABLE and TF_AVAILABLE:
            try:
                self.classifier = TransitClassifier()
            except Exception:
                self.classifier = None
        else:
            self.classifier = None
        
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
        """Классификация сигнала с помощью CNN или статистических методов"""
        if self.clean_curve is None:
            self.remove_noise('wavelet')
            
        if self.classifier is not None:
            # Используем CNN классификатор
            segment_size = 100
            start = max(0, len(self.clean_curve)//2 - segment_size//2)
            segment = self.clean_curve[start:start+segment_size]
            
            # Дополняем или обрезаем до нужного размера
            if len(segment) < segment_size:
                segment = np.pad(segment, (0, segment_size - len(segment)), 'constant')
            elif len(segment) > segment_size:
                segment = segment[:segment_size]
            
            try:
                class_id, probabilities = self.classifier.predict(segment)
                class_names = ['planet', 'star', 'noise']
                
                self.features['classification'] = class_names[class_id]
                self.features['probabilities'] = probabilities.tolist()
            except Exception as e:
                # Fallback к статистической классификации
                self._statistical_classification()
        else:
            # Статистическая классификация
            self._statistical_classification()
            
        return self
    
    def _statistical_classification(self):
        """Статистическая классификация на основе признаков"""
        # Простая эвристическая классификация
        std_ratio = self.features.get('std', 0) / (self.features.get('mean', 1) + 1e-8)
        
        # Проверяем наличие транзитов
        has_transits = len(self.transits) > 0
        
        # Проверяем периодичность
        period = self.features.get('period', 0)
        is_periodic = period > 0 and period < len(self.clean_curve) / 3
        
        if has_transits and is_periodic and std_ratio < 0.1:
            # Вероятно планета
            self.features['classification'] = 'planet'
            self.features['probabilities'] = [0.7, 0.2, 0.1]
        elif std_ratio > 0.2 or abs(self.features.get('skew', 0)) > 2:
            # Вероятно шум
            self.features['classification'] = 'noise'
            self.features['probabilities'] = [0.1, 0.2, 0.7]
        else:
            # Вероятно звездная переменность
            self.features['classification'] = 'star'
            self.features['probabilities'] = [0.2, 0.7, 0.1]

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
