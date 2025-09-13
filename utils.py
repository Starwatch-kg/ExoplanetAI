"""
Модуль утилит для работы с данными, метриками и преобразованиями.

Этот модуль содержит вспомогательные функции для обработки данных,
вычисления метрик и различных преобразований.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any
from scipy import signal, stats
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings

# Настройка логирования
logger = logging.getLogger(__name__)

# Подавление предупреждений
warnings.filterwarnings('ignore')


class BoxLeastSquares:
    """
    Реализация алгоритма Box Least Squares для поиска периодических транзитов.
    
    Этот класс реализует классический алгоритм BLS для поиска транзитов
    экзопланет в кривых блеска.
    """
    
    def __init__(self, period_range: Tuple[float, float] = (0.5, 50.0),
                 nperiods: int = 1000,
                 oversample_factor: int = 5):
        """
        Инициализация BLS.
        
        Args:
            period_range: Диапазон периодов для поиска (дни).
            nperiods: Количество периодов для тестирования.
            oversample_factor: Коэффициент передискретизации.
        """
        self.period_range = period_range
        self.nperiods = nperiods
        self.oversample_factor = oversample_factor
        
        # Создание сетки периодов
        self.periods = np.logspace(
            np.log10(period_range[0]), 
            np.log10(period_range[1]), 
            nperiods
        )
        
        logger.info(f"BLS инициализирован: {nperiods} периодов в диапазоне {period_range}")
    
    def compute_periodogram(self, times: np.ndarray, 
                           fluxes: np.ndarray,
                           errors: Optional[np.ndarray] = None) -> Dict:
        """
        Вычисляет периодограмму BLS.
        
        Args:
            times: Временные метки.
            fluxes: Потоки.
            errors: Ошибки измерений (опционально).
            
        Returns:
            Dict: Результаты BLS анализа.
        """
        logger.info("Вычисление BLS периодограммы")
        
        if errors is None:
            errors = np.ones_like(fluxes) * np.std(fluxes) / np.sqrt(len(fluxes))
        
        # Нормализация данных
        mean_flux = np.mean(fluxes)
        normalized_fluxes = fluxes - mean_flux
        
        # Вычисление периодограммы
        powers = []
        best_params = []
        
        for period in self.periods:
            power, params = self._compute_power_for_period(
                times, normalized_fluxes, errors, period
            )
            powers.append(power)
            best_params.append(params)
        
        powers = np.array(powers)
        
        # Поиск лучшего периода
        best_idx = np.argmax(powers)
        best_period = self.periods[best_idx]
        best_power = powers[best_idx]
        best_params_dict = best_params[best_idx]
        
        logger.info(f"Лучший период BLS: {best_period:.3f} дней, мощность: {best_power:.3f}")
        
        return {
            'periods': self.periods,
            'powers': powers,
            'best_period': best_period,
            'best_power': best_power,
            'best_params': best_params_dict
        }
    
    def _compute_power_for_period(self, times: np.ndarray,
                                 fluxes: np.ndarray,
                                 errors: np.ndarray,
                                 period: float) -> Tuple[float, Dict]:
        """
        Вычисляет мощность BLS для конкретного периода.
        
        Args:
            times: Временные метки.
            fluxes: Потоки.
            errors: Ошибки измерений.
            period: Период для тестирования.
            
        Returns:
            Tuple[float, Dict]: Мощность и параметры транзита.
        """
        # Фазы
        phases = (times % period) / period
        
        # Поиск оптимальных параметров транзита
        best_power = 0.0
        best_params = {}
        
        # Тестируем разные глубины и длительности
        depths = np.linspace(0.001, 0.1, 20)
        durations = np.linspace(0.01, 0.3, 10)
        
        for depth in depths:
            for duration in durations:
                # Создание модели транзита
                transit_model = self._create_transit_model(phases, depth, duration)
                
                # Вычисление мощности
                power = self._compute_bls_power(fluxes, errors, transit_model)
                
                if power > best_power:
                    best_power = power
                    best_params = {
                        'depth': depth,
                        'duration': duration,
                        'phase': 0.0  # Упрощение - всегда начинаем с фазы 0
                    }
        
        return best_power, best_params
    
    def _create_transit_model(self, phases: np.ndarray,
                             depth: float,
                             duration: float) -> np.ndarray:
        """
        Создает модель транзита для заданных параметров.
        
        Args:
            phases: Фазы орбиты.
            depth: Глубина транзита.
            duration: Длительность транзита.
            
        Returns:
            np.ndarray: Модель транзита.
        """
        model = np.ones_like(phases)
        
        # Определяем область транзита (вокруг фазы 0.5)
        transit_center = 0.5
        transit_half_width = duration / 2
        
        # Маска для транзита
        transit_mask = np.abs(phases - transit_center) <= transit_half_width
        
        # Применяем транзит
        model[transit_mask] = 1.0 - depth
        
        return model
    
    def _compute_bls_power(self, fluxes: np.ndarray,
                          errors: np.ndarray,
                          model: np.ndarray) -> float:
        """
        Вычисляет мощность BLS для модели транзита.
        
        Args:
            fluxes: Наблюдаемые потоки.
            errors: Ошибки измерений.
            model: Модель транзита.
            
        Returns:
            float: Мощность BLS.
        """
        # Вычисление взвешенной суммы квадратов
        weights = 1.0 / (errors ** 2)
        total_weight = np.sum(weights)
        
        # Среднее значение
        weighted_mean = np.sum(weights * fluxes) / total_weight
        
        # Остатки
        residuals = fluxes - weighted_mean
        
        # Мощность BLS
        numerator = np.sum(weights * residuals * (model - np.mean(model))) ** 2
        denominator = np.sum(weights * (model - np.mean(model)) ** 2)
        
        if denominator > 0:
            power = numerator / denominator
        else:
            power = 0.0
        
        return power


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Вычисляет метрики классификации.
    
    Args:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.
        y_scores: Предсказанные вероятности (опционально).
        
    Returns:
        Dict[str, float]: Словарь с метриками.
    """
    logger.info("Вычисление метрик классификации")
    
    metrics = {}
    
    # Основные метрики
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Accuracy
    metrics['accuracy'] = np.mean(y_true == y_pred)
    
    # ROC AUC (если есть вероятности)
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    # Дополнительные метрики
    metrics['true_positives'] = np.sum((y_true == 1) & (y_pred == 1))
    metrics['false_positives'] = np.sum((y_true == 0) & (y_pred == 1))
    metrics['true_negatives'] = np.sum((y_true == 0) & (y_pred == 0))
    metrics['false_negatives'] = np.sum((y_true == 1) & (y_pred == 0))
    
    logger.info(f"Метрики вычислены: Precision={metrics['precision']:.3f}, "
               f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    
    return metrics


def normalize_data(data: np.ndarray, 
                  method: str = 'zscore',
                  axis: Optional[int] = None) -> np.ndarray:
    """
    Нормализация данных различными методами.
    
    Args:
        data: Данные для нормализации.
        method: Метод нормализации ('zscore', 'minmax', 'robust').
        axis: Ось для нормализации.
        
    Returns:
        np.ndarray: Нормализованные данные.
    """
    logger.info(f"Нормализация данных методом: {method}")
    
    if method == 'zscore':
        # Z-score нормализация
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        
    elif method == 'minmax':
        # Min-Max нормализация
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        
    elif method == 'robust':
        # Robust нормализация (медиана и MAD)
        median = np.median(data, axis=axis, keepdims=True)
        mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
        normalized = (data - median) / (mad + 1e-8)
        
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")
    
    logger.info("Нормализация завершена")
    return normalized


def remove_outliers(data: np.ndarray,
                   method: str = 'iqr',
                   threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Удаление выбросов из данных.
    
    Args:
        data: Данные для обработки.
        method: Метод удаления выбросов ('iqr', 'zscore', 'modified_zscore').
        threshold: Порог для определения выбросов.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Очищенные данные и маска выбросов.
    """
    logger.info(f"Удаление выбросов методом: {method}")
    
    if method == 'iqr':
        # Interquartile Range метод
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (data >= lower_bound) & (data <= upper_bound)
        
    elif method == 'zscore':
        # Z-score метод
        z_scores = np.abs(stats.zscore(data))
        mask = z_scores < threshold
        
    elif method == 'modified_zscore':
        # Modified Z-score метод
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        mask = np.abs(modified_z_scores) < threshold
        
    else:
        raise ValueError(f"Неизвестный метод удаления выбросов: {method}")
    
    cleaned_data = data[mask]
    
    logger.info(f"Удалено {np.sum(~mask)} выбросов из {len(data)} точек")
    return cleaned_data, mask


def apply_smoothing(data: np.ndarray,
                   method: str = 'savgol',
                   window_length: int = 11,
                   **kwargs) -> np.ndarray:
    """
    Применение сглаживания к данным.
    
    Args:
        data: Данные для сглаживания.
        method: Метод сглаживания ('savgol', 'moving_average', 'gaussian').
        window_length: Размер окна сглаживания.
        **kwargs: Дополнительные параметры.
        
    Returns:
        np.ndarray: Сглаженные данные.
    """
    logger.info(f"Применение сглаживания методом: {method}")
    
    if method == 'savgol':
        # Savitzky-Golay фильтр
        polyorder = kwargs.get('polyorder', 3)
        smoothed = signal.savgol_filter(data, window_length, polyorder)
        
    elif method == 'moving_average':
        # Скользящее среднее
        smoothed = np.convolve(data, np.ones(window_length)/window_length, mode='same')
        
    elif method == 'gaussian':
        # Гауссовский фильтр
        sigma = kwargs.get('sigma', window_length / 6)
        smoothed = signal.gaussian_filter1d(data, sigma)
        
    else:
        raise ValueError(f"Неизвестный метод сглаживания: {method}")
    
    logger.info("Сглаживание завершено")
    return smoothed


def detect_periodicity(data: np.ndarray,
                      times: Optional[np.ndarray] = None,
                      method: str = 'fft') -> Dict[str, Any]:
    """
    Детекция периодичности в данных.
    
    Args:
        data: Временной ряд данных.
        times: Временные метки (опционально).
        method: Метод детекции ('fft', 'autocorr', 'lomb_scargle').
        
    Returns:
        Dict[str, Any]: Результаты детекции периодичности.
    """
    logger.info(f"Детекция периодичности методом: {method}")
    
    if method == 'fft':
        # FFT анализ
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        
        # Поиск доминирующих частот
        power_spectrum = np.abs(fft) ** 2
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        if dominant_freq > 0:
            period = 1.0 / dominant_freq
        else:
            period = 0.0
        
        results = {
            'period': period,
            'frequency': dominant_freq,
            'power': power_spectrum[dominant_freq_idx],
            'method': 'fft'
        }
        
    elif method == 'autocorr':
        # Автокорреляционный анализ
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Поиск пиков (исключая нулевой лаг)
        peaks, _ = signal.find_peaks(autocorr[1:])
        if len(peaks) > 0:
            dominant_lag = peaks[0] + 1
            period = dominant_lag
        else:
            period = 0.0
        
        results = {
            'period': period,
            'lag': dominant_lag if len(peaks) > 0 else 0,
            'autocorr_max': np.max(autocorr[1:]) if len(peaks) > 0 else 0,
            'method': 'autocorr'
        }
        
    elif method == 'lomb_scargle':
        # Lomb-Scargle периодограмма
        if times is None:
            times = np.arange(len(data))
        
        # Подготовка частот
        freqs = np.linspace(0.01, 0.5, 1000)
        
        # Lomb-Scargle периодограмма
        power = signal.lombscargle(times, data, freqs)
        
        # Поиск доминирующей частоты
        dominant_freq_idx = np.argmax(power)
        dominant_freq = freqs[dominant_freq_idx]
        period = 1.0 / dominant_freq
        
        results = {
            'period': period,
            'frequency': dominant_freq,
            'power': power[dominant_freq_idx],
            'method': 'lomb_scargle'
        }
        
    else:
        raise ValueError(f"Неизвестный метод детекции периодичности: {method}")
    
    logger.info(f"Найден период: {results['period']:.3f}")
    return results


def create_train_test_split(data: np.ndarray,
                          labels: np.ndarray,
                          test_size: float = 0.2,
                          random_state: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """
    Разделение данных на обучающую и тестовую выборки.
    
    Args:
        data: Данные для разделения.
        labels: Метки данных.
        test_size: Доля тестовой выборки.
        random_state: Случайное состояние.
        
    Returns:
        Tuple[np.ndarray, ...]: Разделенные данные.
    """
    logger.info(f"Разделение данных: test_size={test_size}")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(data)
    n_test = int(n_samples * test_size)
    
    # Случайная перестановка индексов
    indices = np.random.permutation(n_samples)
    
    # Разделение на train/test
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Разделение данных
    X_train = data[train_indices]
    X_test = data[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    logger.info(f"Разделение завершено: train={len(X_train)}, test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def validate_data_quality(data: np.ndarray,
                         times: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Валидация качества данных.
    
    Args:
        data: Данные для валидации.
        times: Временные метки (опционально).
        
    Returns:
        Dict[str, Any]: Результаты валидации.
    """
    logger.info("Валидация качества данных")
    
    quality_report = {}
    
    # Основная статистика
    quality_report['length'] = len(data)
    quality_report['mean'] = np.mean(data)
    quality_report['std'] = np.std(data)
    quality_report['min'] = np.min(data)
    quality_report['max'] = np.max(data)
    
    # Проверка на NaN и бесконечности
    quality_report['has_nan'] = np.any(np.isnan(data))
    quality_report['has_inf'] = np.any(np.isinf(data))
    
    # Проверка на выбросы
    z_scores = np.abs(stats.zscore(data))
    quality_report['outliers_count'] = np.sum(z_scores > 3)
    quality_report['outliers_fraction'] = quality_report['outliers_count'] / len(data)
    
    # Проверка временных меток
    if times is not None:
        quality_report['time_span'] = times[-1] - times[0]
        quality_report['cadence'] = np.median(np.diff(times))
        quality_report['has_time_gaps'] = np.any(np.diff(times) > 2 * quality_report['cadence'])
    
    # Общая оценка качества
    quality_score = 1.0
    if quality_report['has_nan'] or quality_report['has_inf']:
        quality_score -= 0.3
    if quality_report['outliers_fraction'] > 0.1:
        quality_score -= 0.2
    if times is not None and quality_report['has_time_gaps']:
        quality_score -= 0.1
    
    quality_report['quality_score'] = max(0.0, quality_score)
    
    logger.info(f"Качество данных: {quality_report['quality_score']:.2f}")
    return quality_report
