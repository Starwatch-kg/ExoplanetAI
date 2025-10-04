"""
Извлечение признаков для ML анализа - выделен из EnsembleSearchService
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq

from core.constants import MLConstants, TransitConstants
from core.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Специализированный извлекатель признаков для ML анализа
    
    Отвечает только за:
    - Извлечение статистических признаков
    - Частотные признаки (FFT, спектральные)
    - Морфологические признаки кривых блеска
    - Нелинейные динамические признаки
    """
    
    def __init__(self):
        self.name = "Feature Extractor"
        self.version = "2.0.0"
        self.feature_categories = [
            'statistical',
            'frequency',
            'morphological', 
            'nonlinear'
        ]
    
    async def extract_all_features(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        categories: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Извлечение всех категорий признаков
        
        Args:
            time: Временные метки
            flux: Значения потока
            categories: Категории признаков для извлечения
            
        Returns:
            Словарь с признаками по категориям
        """
        try:
            if len(time) < MLConstants.MIN_DATA_POINTS:
                raise AnalysisError(
                    f"Insufficient data points: {len(time)} < {MLConstants.MIN_DATA_POINTS}"
                )
            
            # Подготовка данных
            time_clean, flux_clean = self._prepare_data(time, flux)
            
            # Выбор категорий
            if categories is None:
                categories = self.feature_categories
            
            # Извлечение признаков по категориям
            all_features = {}
            
            for category in categories:
                if category in self.feature_categories:
                    try:
                        features = await self._extract_category_features(
                            category, time_clean, flux_clean
                        )
                        all_features[category] = features
                    except Exception as e:
                        logger.warning(f"Failed to extract {category} features: {e}")
                        all_features[category] = np.array([])
            
            # Объединение всех признаков
            combined_features = self._combine_features(all_features)
            
            return {
                'individual_categories': all_features,
                'combined': combined_features,
                'feature_names': self._get_feature_names(),
                'data_quality': self._assess_feature_quality(time_clean, flux_clean)
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise AnalysisError(f"Feature extraction failed: {str(e)}")
    
    def _prepare_data(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для извлечения признаков"""
        # Удаление NaN и сортировка
        mask = np.isfinite(time) & np.isfinite(flux)
        time_clean = time[mask]
        flux_clean = flux[mask]
        
        sort_idx = np.argsort(time_clean)
        time_clean = time_clean[sort_idx]
        flux_clean = flux_clean[sort_idx]
        
        # Нормализация потока
        flux_median = np.median(flux_clean)
        if flux_median != 0:
            flux_clean = flux_clean / flux_median
        
        return time_clean, flux_clean
    
    async def _extract_category_features(
        self, 
        category: str, 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> np.ndarray:
        """Извлечение признаков определенной категории"""
        
        if category == 'statistical':
            return self._extract_statistical_features(flux)
        elif category == 'frequency':
            return self._extract_frequency_features(time, flux)
        elif category == 'morphological':
            return self._extract_morphological_features(time, flux)
        elif category == 'nonlinear':
            return self._extract_nonlinear_features(flux)
        else:
            raise AnalysisError(f"Unknown feature category: {category}")
    
    def _extract_statistical_features(self, flux: np.ndarray) -> np.ndarray:
        """Извлечение статистических признаков"""
        features = []
        
        try:
            # Основные статистики
            features.extend([
                np.mean(flux),
                np.median(flux),
                np.std(flux),
                np.var(flux),
                stats.skew(flux),
                stats.kurtosis(flux)
            ])
            
            # Квантили
            quantiles = np.percentile(flux, [5, 25, 75, 95])
            features.extend(quantiles)
            
            # Межквартильный размах
            iqr = quantiles[2] - quantiles[1]  # Q75 - Q25
            features.append(iqr)
            
            # Размах
            features.append(np.max(flux) - np.min(flux))
            
            # Коэффициент вариации
            cv = np.std(flux) / np.mean(flux) if np.mean(flux) != 0 else 0
            features.append(cv)
            
            # Медианное абсолютное отклонение
            mad = np.median(np.abs(flux - np.median(flux)))
            features.append(mad)
            
            # Количество пиков и впадин
            peaks, _ = signal.find_peaks(flux, height=np.mean(flux) + np.std(flux))
            valleys, _ = signal.find_peaks(-flux, height=-np.mean(flux) + np.std(flux))
            features.extend([len(peaks), len(valleys)])
            
        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Statistical features calculation failed: {e}")
            # Заполняем нулями при ошибке
            features = [0.0] * 15
        
        return np.array(features)
    
    def _extract_frequency_features(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Извлечение частотных признаков"""
        features = []
        
        try:
            # Интерполяция на равномерную сетку для FFT
            dt = np.median(np.diff(time))
            time_uniform = np.arange(time.min(), time.max(), dt)
            flux_interp = np.interp(time_uniform, time, flux)
            
            # Удаление тренда
            flux_detrended = signal.detrend(flux_interp)
            
            # FFT
            fft_values = fft(flux_detrended)
            frequencies = fftfreq(len(flux_detrended), dt)
            
            # Берем только положительные частоты
            positive_freq_mask = frequencies > 0
            frequencies = frequencies[positive_freq_mask]
            power_spectrum = np.abs(fft_values[positive_freq_mask])**2
            
            # Нормализация спектра
            power_spectrum = power_spectrum / np.sum(power_spectrum)
            
            # Спектральные моменты
            spectral_centroid = np.sum(frequencies * power_spectrum)
            features.append(spectral_centroid)
            
            # Спектральная ширина
            spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * power_spectrum))
            features.append(spectral_spread)
            
            # Спектральная асимметрия
            spectral_skewness = np.sum(((frequencies - spectral_centroid) ** 3) * power_spectrum) / (spectral_spread ** 3)
            features.append(spectral_skewness)
            
            # Спектральный эксцесс
            spectral_kurtosis = np.sum(((frequencies - spectral_centroid) ** 4) * power_spectrum) / (spectral_spread ** 4)
            features.append(spectral_kurtosis)
            
            # Спектральная энтропия
            spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-12))
            features.append(spectral_entropy)
            
            # Доминирующая частота
            dominant_freq_idx = np.argmax(power_spectrum)
            dominant_frequency = frequencies[dominant_freq_idx]
            features.append(dominant_frequency)
            
            # Мощность в различных частотных диапазонах
            # Низкие частоты (долгопериодические изменения)
            low_freq_mask = frequencies < (1.0 / 10.0)  # Периоды > 10 дней
            low_freq_power = np.sum(power_spectrum[low_freq_mask])
            features.append(low_freq_power)
            
            # Средние частоты (транзитные периоды)
            mid_freq_mask = (frequencies >= (1.0 / 10.0)) & (frequencies <= (1.0 / 0.5))
            mid_freq_power = np.sum(power_spectrum[mid_freq_mask])
            features.append(mid_freq_power)
            
            # Высокие частоты (шум)
            high_freq_mask = frequencies > (1.0 / 0.5)  # Периоды < 0.5 дня
            high_freq_power = np.sum(power_spectrum[high_freq_mask])
            features.append(high_freq_power)
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Frequency features calculation failed: {e}")
            features = [0.0] * 9
        
        return np.array(features)
    
    def _extract_morphological_features(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Извлечение морфологических признаков кривой блеска"""
        features = []
        
        try:
            # Производная (скорость изменения)
            dt = np.diff(time)
            dflux = np.diff(flux)
            velocity = dflux / dt
            
            # Статистики производной
            features.extend([
                np.mean(np.abs(velocity)),
                np.std(velocity),
                np.max(np.abs(velocity))
            ])
            
            # Вторая производная (ускорение)
            if len(velocity) > 1:
                acceleration = np.diff(velocity) / dt[:-1]
                features.extend([
                    np.mean(np.abs(acceleration)),
                    np.std(acceleration)
                ])
            else:
                features.extend([0.0, 0.0])
            
            # Количество пересечений среднего
            mean_flux = np.mean(flux)
            crossings = np.sum(np.diff(np.sign(flux - mean_flux)) != 0)
            features.append(crossings)
            
            # Длительность выбросов (outliers)
            threshold = np.mean(flux) + 2 * np.std(flux)
            outlier_mask = np.abs(flux) > threshold
            outlier_duration = np.sum(outlier_mask) / len(flux)
            features.append(outlier_duration)
            
            # Асимметрия кривой блеска
            flux_centered = flux - np.median(flux)
            positive_area = np.sum(flux_centered[flux_centered > 0])
            negative_area = np.sum(np.abs(flux_centered[flux_centered < 0]))
            asymmetry = (positive_area - negative_area) / (positive_area + negative_area + 1e-12)
            features.append(asymmetry)
            
            # Компактность (отношение площади к периметру)
            total_variation = np.sum(np.abs(dflux))
            compactness = len(flux) / (total_variation + 1e-12)
            features.append(compactness)
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Morphological features calculation failed: {e}")
            features = [0.0] * 9
        
        return np.array(features)
    
    def _extract_nonlinear_features(self, flux: np.ndarray) -> np.ndarray:
        """Извлечение нелинейных динамических признаков"""
        features = []
        
        try:
            # Приближенная энтропия (Approximate Entropy)
            def _approximate_entropy(data, m=2, r=None):
                if r is None:
                    r = 0.2 * np.std(data)
                
                def _maxdist(xi, xj, m):
                    return max([abs(ua - va) for ua, va in zip(xi, xj)])
                
                def _phi(m):
                    patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
                    C = np.zeros(len(patterns))
                    
                    for i, pattern_i in enumerate(patterns):
                        template_match_count = sum([1 for pattern_j in patterns 
                                                  if _maxdist(pattern_i, pattern_j, m) <= r])
                        C[i] = template_match_count / float(len(patterns))
                    
                    phi = np.mean(np.log(C))
                    return phi
                
                return _phi(m) - _phi(m + 1)
            
            if len(flux) > 10:
                approx_entropy = _approximate_entropy(flux)
                features.append(approx_entropy)
            else:
                features.append(0.0)
            
            # Показатель Ляпунова (упрощенная оценка)
            if len(flux) > 2:
                diffs = np.diff(flux)
                lyapunov_approx = np.mean(np.log(np.abs(diffs) + 1e-10))
                features.append(lyapunov_approx)
            else:
                features.append(0.0)
            
            # Корреляционная размерность (упрощенная)
            if len(flux) > 5:
                # Вложение в фазовое пространство
                embedding_dim = min(3, len(flux) // 2)
                embedded = np.array([flux[i:i+embedding_dim] for i in range(len(flux) - embedding_dim + 1)])
                
                # Корреляционная сумма
                distances = []
                for i in range(len(embedded)):
                    for j in range(i+1, len(embedded)):
                        dist = np.linalg.norm(embedded[i] - embedded[j])
                        distances.append(dist)
                
                if distances:
                    correlation_dim = np.mean(distances)
                    features.append(correlation_dim)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
            # Энтропия Шеннона
            hist, _ = np.histogram(flux, bins=min(20, len(flux)//5))
            hist = hist / np.sum(hist)  # Нормализация
            hist = hist[hist > 0]  # Удаляем нулевые значения
            if len(hist) > 0:
                shannon_entropy = -np.sum(hist * np.log2(hist))
                features.append(shannon_entropy)
            else:
                features.append(0.0)
            
            # Фрактальная размерность (метод подсчета коробок)
            if len(flux) > 10:
                # Упрощенная оценка через вариацию на разных масштабах
                scales = [1, 2, 4, 8]
                variations = []
                
                for scale in scales:
                    if scale < len(flux):
                        downsampled = flux[::scale]
                        if len(downsampled) > 1:
                            variation = np.sum(np.abs(np.diff(downsampled)))
                            variations.append(variation)
                
                if len(variations) >= 2:
                    # Логарифмический наклон как оценка фрактальной размерности
                    log_scales = np.log(scales[:len(variations)])
                    log_variations = np.log(np.array(variations) + 1e-12)
                    fractal_dim = -np.polyfit(log_scales, log_variations, 1)[0]
                    features.append(fractal_dim)
                else:
                    features.append(1.0)
            else:
                features.append(1.0)
            
        except (ValueError, IndexError, ZeroDivisionError) as e:
            logger.warning(f"Nonlinear features calculation failed: {e}")
            features = [0.0] * 5
        
        return np.array(features)
    
    def _combine_features(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Объединение всех признаков в один вектор"""
        combined = []
        
        for category in self.feature_categories:
            if category in feature_dict:
                features = feature_dict[category]
                if len(features) > 0:
                    combined.extend(features)
        
        return np.array(combined)
    
    def _get_feature_names(self) -> List[str]:
        """Получение названий признаков"""
        names = []
        
        # Статистические признаки
        statistical_names = [
            'mean', 'median', 'std', 'var', 'skew', 'kurtosis',
            'q5', 'q25', 'q75', 'q95', 'iqr', 'range', 'cv', 'mad',
            'n_peaks', 'n_valleys'
        ]
        names.extend([f'stat_{name}' for name in statistical_names])
        
        # Частотные признаки
        frequency_names = [
            'spectral_centroid', 'spectral_spread', 'spectral_skewness', 
            'spectral_kurtosis', 'spectral_entropy', 'dominant_frequency',
            'low_freq_power', 'mid_freq_power', 'high_freq_power'
        ]
        names.extend([f'freq_{name}' for name in frequency_names])
        
        # Морфологические признаки
        morphological_names = [
            'mean_velocity', 'std_velocity', 'max_velocity',
            'mean_acceleration', 'std_acceleration', 'mean_crossings',
            'outlier_duration', 'asymmetry', 'compactness'
        ]
        names.extend([f'morph_{name}' for name in morphological_names])
        
        # Нелинейные признаки
        nonlinear_names = [
            'approximate_entropy', 'lyapunov_exponent', 'correlation_dimension',
            'shannon_entropy', 'fractal_dimension'
        ]
        names.extend([f'nonlin_{name}' for name in nonlinear_names])
        
        return names
    
    def _assess_feature_quality(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, float]:
        """Оценка качества данных для извлечения признаков"""
        
        # Временное покрытие
        time_span = time.max() - time.min()
        
        # Количество точек
        n_points = len(time)
        
        # Равномерность временной сетки
        dt_values = np.diff(time)
        dt_uniformity = 1.0 - (np.std(dt_values) / np.mean(dt_values))
        
        # Уровень шума
        noise_level = np.std(flux) / np.mean(np.abs(flux))
        
        # Динамический диапазон
        dynamic_range = (np.max(flux) - np.min(flux)) / np.mean(flux)
        
        # Общая оценка качества
        quality_factors = [
            min(time_span / 30.0, 1.0),  # Нормализация к 30 дням
            min(n_points / MLConstants.RECOMMENDED_DATA_POINTS, 1.0),
            dt_uniformity,
            max(0.0, 1.0 - noise_level),  # Меньше шума = лучше
            min(dynamic_range / 0.1, 1.0)  # Нормализация к 10%
        ]
        
        overall_quality = np.mean(quality_factors)
        
        return {
            'time_span_days': float(time_span),
            'n_points': int(n_points),
            'temporal_uniformity': float(dt_uniformity),
            'noise_level': float(noise_level),
            'dynamic_range': float(dynamic_range),
            'overall_quality': float(overall_quality)
        }
