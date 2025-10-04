"""
Advanced Feature Extraction for Exoplanet Classification
Продвинутое извлечение признаков для классификации экзопланет

Включает:
- Параметры транзита (period, depth, duration, radius)
- Статистические признаки (mean, std, skewness, kurtosis)
- Частотные признаки (FFT, Lomb-Scargle periodogram)
- Морфологические признаки формы транзита
- Признаки качества данных
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats, signal
from scipy.optimize import minimize_scalar
from astropy.timeseries import LombScargle
try:
    from astropy.stats import median_abs_deviation
except ImportError:
    from scipy.stats import median_abs_deviation
import logging

logger = logging.getLogger(__name__)


class ExoplanetFeatureExtractor:
    """
    Комплексный экстрактор признаков для классификации экзопланет
    """
    
    def __init__(self):
        self.feature_names = []
        self._initialize_feature_names()
    
    def _initialize_feature_names(self):
        """Инициализация названий признаков"""
        # Статистические признаки
        self.feature_names.extend([
            'flux_mean', 'flux_std', 'flux_var', 'flux_skewness', 'flux_kurtosis',
            'flux_median', 'flux_mad', 'flux_iqr', 'flux_range',
            'flux_p05', 'flux_p25', 'flux_p75', 'flux_p95'
        ])
        
        # Признаки транзита
        self.feature_names.extend([
            'transit_depth', 'transit_duration', 'transit_period', 'transit_epoch',
            'transit_snr', 'transit_shape_factor', 'transit_asymmetry',
            'ingress_duration', 'egress_duration', 'flat_bottom_duration'
        ])
        
        # Частотные признаки
        self.feature_names.extend([
            'dominant_frequency', 'frequency_power', 'spectral_centroid',
            'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate',
            'periodogram_peak_power', 'periodogram_peak_frequency'
        ])
        
        # Морфологические признаки
        self.feature_names.extend([
            'v_shape_score', 'u_shape_score', 'box_shape_score',
            'secondary_eclipse_depth', 'phase_curve_amplitude',
            'odd_even_depth_difference', 'transit_timing_variation'
        ])
        
        # Признаки качества данных
        self.feature_names.extend([
            'data_span', 'cadence_median', 'cadence_std', 'gap_fraction',
            'outlier_fraction', 'noise_level', 'systematic_trend'
        ])
    
    def extract_statistical_features(self, flux: np.ndarray) -> Dict[str, float]:
        """
        Извлечение статистических признаков
        
        Args:
            flux: Массив значений потока
            
        Returns:
            Словарь со статистическими признаками
        """
        features = {}
        
        # Базовые статистики
        features['flux_mean'] = np.mean(flux)
        features['flux_std'] = np.std(flux)
        features['flux_var'] = np.var(flux)
        features['flux_median'] = np.median(flux)
        features['flux_mad'] = median_abs_deviation(flux, scale='normal')
        features['flux_range'] = np.ptp(flux)
        
        # Перцентили
        percentiles = np.percentile(flux, [5, 25, 75, 95])
        features['flux_p05'] = percentiles[0]
        features['flux_p25'] = percentiles[1]
        features['flux_p75'] = percentiles[2]
        features['flux_p95'] = percentiles[3]
        features['flux_iqr'] = percentiles[2] - percentiles[1]
        
        # Моменты распределения
        try:
            features['flux_skewness'] = stats.skew(flux)
            features['flux_kurtosis'] = stats.kurtosis(flux)
        except:
            features['flux_skewness'] = 0.0
            features['flux_kurtosis'] = 0.0
        
        return features
    
    def extract_transit_features(self, 
                                time: np.ndarray,
                                flux: np.ndarray,
                                period: float,
                                epoch: float,
                                duration: float) -> Dict[str, float]:
        """
        Извлечение признаков транзита
        
        Args:
            time, flux: Данные кривой блеска
            period: Орбитальный период
            epoch: Эпоха транзита
            duration: Длительность транзита
            
        Returns:
            Словарь с признаками транзита
        """
        features = {}
        
        # Фазовая кривая
        phase = ((time - epoch) % period) / period
        phase[phase > 0.5] -= 1.0
        
        # Сортируем по фазе для анализа формы
        sort_idx = np.argsort(phase)
        phase_sorted = phase[sort_idx]
        flux_sorted = flux[sort_idx]
        
        # Базовые параметры транзита
        features['transit_period'] = period
        features['transit_epoch'] = epoch
        features['transit_duration'] = duration
        
        # Глубина и SNR
        in_transit = np.abs(phase) < (duration / period / 2)
        out_transit = np.abs(phase) > (duration / period)
        
        if np.sum(in_transit) > 0 and np.sum(out_transit) > 0:
            flux_in = np.median(flux[in_transit])
            flux_out = np.median(flux[out_transit])
            depth = (flux_out - flux_in) / flux_out
            noise = np.std(flux[out_transit])
            snr = depth / noise if noise > 0 else 0
            
            features['transit_depth'] = depth
            features['transit_snr'] = snr
        else:
            features['transit_depth'] = 0
            features['transit_snr'] = 0
        
        # Анализ формы транзита
        transit_features = self._analyze_transit_shape(phase_sorted, flux_sorted, duration/period)
        features.update(transit_features)
        
        # Поиск вторичного затмения
        secondary_features = self._detect_secondary_eclipse(phase_sorted, flux_sorted)
        features.update(secondary_features)
        
        return features
    
    def _analyze_transit_shape(self, 
                              phase: np.ndarray, 
                              flux: np.ndarray,
                              transit_width: float) -> Dict[str, float]:
        """
        Анализ формы транзита
        
        Args:
            phase: Фазовый массив (отсортированный)
            flux: Поток (отсортированный по фазе)
            transit_width: Ширина транзита в фазовых единицах
            
        Returns:
            Признаки формы транзита
        """
        features = {}
        
        # Находим транзитную область
        transit_mask = np.abs(phase) < transit_width / 2
        
        if np.sum(transit_mask) < 10:  # Недостаточно точек
            return {
                'transit_shape_factor': 0,
                'transit_asymmetry': 0,
                'ingress_duration': 0,
                'egress_duration': 0,
                'flat_bottom_duration': 0,
                'v_shape_score': 0,
                'u_shape_score': 0,
                'box_shape_score': 0
            }
        
        transit_phase = phase[transit_mask]
        transit_flux = flux[transit_mask]
        
        # Нормализуем транзит
        baseline = np.median(flux[~transit_mask])
        transit_flux_norm = transit_flux / baseline
        
        # Находим минимум транзита
        min_idx = np.argmin(transit_flux_norm)
        min_phase = transit_phase[min_idx]
        min_flux = transit_flux_norm[min_idx]
        
        # Асимметрия транзита
        left_flux = transit_flux_norm[transit_phase < min_phase]
        right_flux = transit_flux_norm[transit_phase > min_phase]
        
        if len(left_flux) > 0 and len(right_flux) > 0:
            asymmetry = np.mean(left_flux) - np.mean(right_flux)
        else:
            asymmetry = 0
        
        features['transit_asymmetry'] = asymmetry
        
        # Форм-фактор (отношение глубины к ширине)
        depth = 1 - min_flux
        width = np.ptp(transit_phase)
        features['transit_shape_factor'] = depth / width if width > 0 else 0
        
        # Анализ ingress/egress
        ingress_egress = self._analyze_ingress_egress(transit_phase, transit_flux_norm)
        features.update(ingress_egress)
        
        # Оценка формы (V, U, или Box)
        shape_scores = self._classify_transit_shape(transit_phase, transit_flux_norm)
        features.update(shape_scores)
        
        return features
    
    def _analyze_ingress_egress(self, 
                               phase: np.ndarray, 
                               flux: np.ndarray) -> Dict[str, float]:
        """
        Анализ ingress и egress транзита
        """
        features = {}
        
        try:
            # Находим точки 10% и 90% глубины
            min_flux = np.min(flux)
            max_flux = np.max(flux)
            depth_10 = max_flux - 0.1 * (max_flux - min_flux)
            depth_90 = max_flux - 0.9 * (max_flux - min_flux)
            
            # Ingress: от 10% до 90% глубины (левая сторона)
            left_mask = phase < np.median(phase)
            if np.sum(left_mask) > 5:
                left_phase = phase[left_mask]
                left_flux = flux[left_mask]
                
                ingress_start = None
                ingress_end = None
                
                for i in range(len(left_flux)):
                    if left_flux[i] <= depth_10 and ingress_start is None:
                        ingress_start = left_phase[i]
                    if left_flux[i] <= depth_90:
                        ingress_end = left_phase[i]
                
                if ingress_start is not None and ingress_end is not None:
                    features['ingress_duration'] = abs(ingress_end - ingress_start)
                else:
                    features['ingress_duration'] = 0
            else:
                features['ingress_duration'] = 0
            
            # Egress: от 90% до 10% глубины (правая сторона)
            right_mask = phase > np.median(phase)
            if np.sum(right_mask) > 5:
                right_phase = phase[right_mask]
                right_flux = flux[right_mask]
                
                egress_start = None
                egress_end = None
                
                for i in range(len(right_flux)):
                    if right_flux[i] <= depth_90 and egress_start is None:
                        egress_start = right_phase[i]
                    if right_flux[i] <= depth_10:
                        egress_end = right_phase[i]
                
                if egress_start is not None and egress_end is not None:
                    features['egress_duration'] = abs(egress_end - egress_start)
                else:
                    features['egress_duration'] = 0
            else:
                features['egress_duration'] = 0
            
            # Длительность плоского дна (между 90% точками)
            bottom_mask = flux <= depth_90
            if np.sum(bottom_mask) > 0:
                bottom_phases = phase[bottom_mask]
                features['flat_bottom_duration'] = np.ptp(bottom_phases)
            else:
                features['flat_bottom_duration'] = 0
                
        except Exception as e:
            logger.warning(f"Ingress/egress analysis failed: {e}")
            features.update({
                'ingress_duration': 0,
                'egress_duration': 0,
                'flat_bottom_duration': 0
            })
        
        return features
    
    def _classify_transit_shape(self, 
                               phase: np.ndarray, 
                               flux: np.ndarray) -> Dict[str, float]:
        """
        Классификация формы транзита (V, U, Box)
        """
        features = {}
        
        try:
            # Нормализуем к [0, 1]
            flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
            
            # V-образная форма: острый минимум
            # Считаем кривизну в минимуме
            min_idx = np.argmin(flux_norm)
            if min_idx > 2 and min_idx < len(flux_norm) - 3:
                curvature = flux_norm[min_idx-2] + flux_norm[min_idx+2] - 2*flux_norm[min_idx]
                v_score = max(0, curvature)
            else:
                v_score = 0
            
            # U-образная форма: плавный минимум
            # Считаем ширину области минимума
            min_val = np.min(flux_norm)
            near_min_mask = flux_norm <= min_val + 0.1
            u_score = np.sum(near_min_mask) / len(flux_norm)
            
            # Box-образная форма: плоское дно
            # Считаем стандартное отклонение в области минимума
            if np.sum(near_min_mask) > 3:
                box_score = 1.0 / (1.0 + np.std(flux_norm[near_min_mask]))
            else:
                box_score = 0
            
            features['v_shape_score'] = v_score
            features['u_shape_score'] = u_score
            features['box_shape_score'] = box_score
            
        except Exception as e:
            logger.warning(f"Shape classification failed: {e}")
            features.update({
                'v_shape_score': 0,
                'u_shape_score': 0,
                'box_shape_score': 0
            })
        
        return features
    
    def _detect_secondary_eclipse(self, 
                                 phase: np.ndarray, 
                                 flux: np.ndarray) -> Dict[str, float]:
        """
        Поиск вторичного затмения
        """
        features = {}
        
        try:
            # Ищем вторичное затмение около фазы 0.5
            secondary_mask = np.abs(phase - 0.5) < 0.1
            if np.sum(secondary_mask) > 10:
                secondary_flux = flux[secondary_mask]
                baseline_flux = np.median(flux[np.abs(phase) > 0.3])
                
                # Глубина вторичного затмения
                secondary_depth = (np.median(secondary_flux) - baseline_flux) / baseline_flux
                features['secondary_eclipse_depth'] = secondary_depth
            else:
                features['secondary_eclipse_depth'] = 0
            
            # Амплитуда фазовой кривой
            phase_curve_amp = np.ptp(flux) / np.median(flux)
            features['phase_curve_amplitude'] = phase_curve_amp
            
        except Exception as e:
            logger.warning(f"Secondary eclipse detection failed: {e}")
            features.update({
                'secondary_eclipse_depth': 0,
                'phase_curve_amplitude': 0
            })
        
        return features
    
    def extract_frequency_features(self, 
                                  time: np.ndarray, 
                                  flux: np.ndarray) -> Dict[str, float]:
        """
        Извлечение частотных признаков
        
        Args:
            time, flux: Данные временного ряда
            
        Returns:
            Частотные признаки
        """
        features = {}
        
        try:
            # FFT анализ
            fft_features = self._compute_fft_features(time, flux)
            features.update(fft_features)
            
            # Lomb-Scargle периодограмма
            ls_features = self._compute_lomb_scargle_features(time, flux)
            features.update(ls_features)
            
            # Zero crossing rate
            zcr = self._compute_zero_crossing_rate(flux)
            features['zero_crossing_rate'] = zcr
            
        except Exception as e:
            logger.warning(f"Frequency feature extraction failed: {e}")
            features.update({
                'dominant_frequency': 0,
                'frequency_power': 0,
                'spectral_centroid': 0,
                'spectral_bandwidth': 0,
                'spectral_rolloff': 0,
                'zero_crossing_rate': 0,
                'periodogram_peak_power': 0,
                'periodogram_peak_frequency': 0
            })
        
        return features
    
    def _compute_fft_features(self, 
                             time: np.ndarray, 
                             flux: np.ndarray) -> Dict[str, float]:
        """
        Вычисление FFT признаков
        """
        features = {}
        
        # Интерполируем на равномерную сетку
        dt = np.median(np.diff(time))
        time_uniform = np.arange(time[0], time[-1], dt)
        flux_uniform = np.interp(time_uniform, time, flux)
        
        # FFT
        fft_vals = np.fft.fft(flux_uniform - np.mean(flux_uniform))
        freqs = np.fft.fftfreq(len(flux_uniform), dt)
        
        # Берем только положительные частоты
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        power_pos = np.abs(fft_vals[pos_mask])**2
        
        if len(power_pos) > 0:
            # Доминирующая частота
            max_idx = np.argmax(power_pos)
            features['dominant_frequency'] = freqs_pos[max_idx]
            features['frequency_power'] = power_pos[max_idx]
            
            # Спектральный центроид
            features['spectral_centroid'] = np.sum(freqs_pos * power_pos) / np.sum(power_pos)
            
            # Спектральная ширина
            centroid = features['spectral_centroid']
            features['spectral_bandwidth'] = np.sqrt(
                np.sum(((freqs_pos - centroid)**2) * power_pos) / np.sum(power_pos)
            )
            
            # Спектральный rolloff (95% энергии)
            cumsum_power = np.cumsum(power_pos)
            rolloff_idx = np.where(cumsum_power >= 0.95 * cumsum_power[-1])[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = freqs_pos[rolloff_idx[0]]
            else:
                features['spectral_rolloff'] = freqs_pos[-1]
        else:
            features.update({
                'dominant_frequency': 0,
                'frequency_power': 0,
                'spectral_centroid': 0,
                'spectral_bandwidth': 0,
                'spectral_rolloff': 0
            })
        
        return features
    
    def _compute_lomb_scargle_features(self, 
                                      time: np.ndarray, 
                                      flux: np.ndarray) -> Dict[str, float]:
        """
        Вычисление признаков Lomb-Scargle периодограммы
        """
        features = {}
        
        try:
            # Частотная сетка
            dt = np.median(np.diff(time))
            freq_min = 1.0 / (time[-1] - time[0])
            freq_max = 1.0 / (2 * dt)  # Частота Найквиста
            
            frequencies = np.logspace(
                np.log10(freq_min), 
                np.log10(freq_max), 
                1000
            )
            
            # Lomb-Scargle периодограмма
            ls = LombScargle(time, flux - np.mean(flux))
            power = ls.power(frequencies)
            
            # Пик периодограммы
            max_idx = np.argmax(power)
            features['periodogram_peak_frequency'] = frequencies[max_idx]
            features['periodogram_peak_power'] = power[max_idx]
            
        except Exception as e:
            logger.warning(f"Lomb-Scargle computation failed: {e}")
            features.update({
                'periodogram_peak_frequency': 0,
                'periodogram_peak_power': 0
            })
        
        return features
    
    def _compute_zero_crossing_rate(self, flux: np.ndarray) -> float:
        """
        Вычисление zero crossing rate
        """
        try:
            # Центрируем сигнал
            flux_centered = flux - np.mean(flux)
            
            # Считаем пересечения нуля
            zero_crossings = np.where(np.diff(np.signbit(flux_centered)))[0]
            zcr = len(zero_crossings) / len(flux)
            
            return zcr
        except:
            return 0.0
    
    def extract_data_quality_features(self, 
                                     time: np.ndarray, 
                                     flux: np.ndarray,
                                     flux_err: np.ndarray) -> Dict[str, float]:
        """
        Извлечение признаков качества данных
        
        Args:
            time, flux, flux_err: Данные кривой блеска
            
        Returns:
            Признаки качества данных
        """
        features = {}
        
        # Временной охват
        features['data_span'] = time[-1] - time[0]
        
        # Каденция
        cadences = np.diff(time)
        features['cadence_median'] = np.median(cadences)
        features['cadence_std'] = np.std(cadences)
        
        # Доля пропусков (большие промежутки)
        median_cadence = features['cadence_median']
        large_gaps = cadences > 3 * median_cadence
        features['gap_fraction'] = np.sum(large_gaps) / len(cadences)
        
        # Доля выбросов
        flux_median = np.median(flux)
        flux_mad = median_abs_deviation(flux, scale='normal')
        outliers = np.abs(flux - flux_median) > 5 * flux_mad
        features['outlier_fraction'] = np.sum(outliers) / len(flux)
        
        # Уровень шума
        features['noise_level'] = np.median(flux_err) / flux_median
        
        # Систематический тренд
        try:
            # Линейная регрессия для оценки тренда
            slope, intercept, r_value, p_value, std_err = stats.linregress(time, flux)
            features['systematic_trend'] = abs(slope) * features['data_span'] / flux_median
        except:
            features['systematic_trend'] = 0
        
        return features
    
    def extract_features(self, 
                        time: np.ndarray,
                        flux: np.ndarray,
                        flux_err: np.ndarray) -> np.ndarray:
        """
        Основной метод извлечения признаков
        
        Args:
            time: Временные данные
            flux: Данные потока  
            flux_err: Ошибки потока
            
        Returns:
            Массив признаков
        """
        try:
            # Сначала делаем простой BLS анализ для получения параметров
            from astropy.timeseries import BoxLeastSquares
            
            bls = BoxLeastSquares(time, flux)
            periods = np.linspace(1.0, 20.0, 1000)
            bls_result = bls.power(periods)
            
            best_period = periods[np.argmax(bls_result.power)]
            epoch = time[0]  # Простая оценка
            duration = best_period * 0.1  # 10% от периода
            
            # Используем полный набор признаков
            features_dict = self.extract_all_features(time, flux, flux_err, best_period, epoch, duration)
            
            # Преобразуем в массив numpy
            feature_values = []
            for feature_name in self.feature_names:
                if feature_name in features_dict:
                    value = features_dict[feature_name]
                    # Обработка NaN и inf значений
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    feature_values.append(value)
                else:
                    feature_values.append(0.0)
            
            return np.array(feature_values)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Возвращаем нулевые признаки при ошибке
            return np.zeros(len(self.feature_names))
    
    def extract_all_features(self, 
                            time: np.ndarray,
                            flux: np.ndarray,
                            flux_err: np.ndarray,
                            period: float,
                            epoch: float,
                            duration: float) -> Dict[str, float]:
        """
        Извлечение всех признаков
        
        Args:
            time, flux, flux_err: Данные кривой блеска
            period, epoch, duration: Параметры транзита
            
        Returns:
            Полный набор признаков
        """
        logger.info("Extracting all features")
        
        all_features = {}
        
        # Статистические признаки
        stat_features = self.extract_statistical_features(flux)
        all_features.update(stat_features)
        
        # Признаки транзита
        transit_features = self.extract_transit_features(time, flux, period, epoch, duration)
        all_features.update(transit_features)
        
        # Частотные признаки
        freq_features = self.extract_frequency_features(time, flux)
        all_features.update(freq_features)
        
        # Признаки качества данных
        quality_features = self.extract_data_quality_features(time, flux, flux_err)
        all_features.update(quality_features)
        
        # Дополнительные признаки
        additional_features = self._extract_additional_features(time, flux, period)
        all_features.update(additional_features)
        
        logger.info(f"Extracted {len(all_features)} features")
        return all_features
    
    def _extract_additional_features(self, 
                                    time: np.ndarray,
                                    flux: np.ndarray,
                                    period: float) -> Dict[str, float]:
        """
        Дополнительные специфичные признаки
        """
        features = {}
        
        try:
            # Odd-even depth difference (для поиска ложных позитивов)
            features['odd_even_depth_difference'] = self._compute_odd_even_difference(
                time, flux, period
            )
            
            # Transit timing variation
            features['transit_timing_variation'] = self._compute_ttv(time, flux, period)
            
        except Exception as e:
            logger.warning(f"Additional feature extraction failed: {e}")
            features.update({
                'odd_even_depth_difference': 0,
                'transit_timing_variation': 0
            })
        
        return features
    
    def _compute_odd_even_difference(self, 
                                    time: np.ndarray,
                                    flux: np.ndarray,
                                    period: float) -> float:
        """
        Вычисление разности глубин четных и нечетных транзитов
        """
        try:
            # Находим транзиты
            epoch = time[0]  # Примерная эпоха
            transit_times = []
            
            t = epoch
            while t < time[-1]:
                if t >= time[0]:
                    transit_times.append(t)
                t += period
            
            if len(transit_times) < 4:
                return 0
            
            odd_depths = []
            even_depths = []
            
            for i, tt in enumerate(transit_times):
                # Находим данные вокруг транзита
                mask = np.abs(time - tt) < period * 0.1
                if np.sum(mask) > 10:
                    transit_flux = flux[mask]
                    baseline = np.median(flux[~mask])
                    depth = (baseline - np.min(transit_flux)) / baseline
                    
                    if i % 2 == 0:
                        even_depths.append(depth)
                    else:
                        odd_depths.append(depth)
            
            if len(odd_depths) > 0 and len(even_depths) > 0:
                return abs(np.median(odd_depths) - np.median(even_depths))
            else:
                return 0
                
        except:
            return 0
    
    def _compute_ttv(self, 
                    time: np.ndarray,
                    flux: np.ndarray,
                    period: float) -> float:
        """
        Вычисление Transit Timing Variation
        """
        try:
            # Упрощенная оценка TTV через стандартное отклонение
            # периодов между транзитами
            
            # Находим минимумы (приближенные времена транзитов)
            from scipy.signal import find_peaks
            
            # Инвертируем поток для поиска минимумов как пиков
            inv_flux = -flux
            peaks, _ = find_peaks(inv_flux, distance=int(period/np.median(np.diff(time))))
            
            if len(peaks) < 3:
                return 0
            
            transit_times = time[peaks]
            periods = np.diff(transit_times)
            
            # TTV как стандартное отклонение периодов
            ttv = np.std(periods) / np.mean(periods) if len(periods) > 1 else 0
            
            return ttv
            
        except:
            return 0
    
    def get_feature_vector(self, features_dict: Dict[str, float]) -> np.ndarray:
        """
        Преобразование словаря признаков в вектор
        
        Args:
            features_dict: Словарь с признаками
            
        Returns:
            Вектор признаков в правильном порядке
        """
        feature_vector = []
        
        for feature_name in self.feature_names:
            if feature_name in features_dict:
                value = features_dict[feature_name]
                # Обработка NaN и inf
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(value)
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    def get_feature_names(self) -> List[str]:
        """Получить список названий признаков"""
        return self.feature_names.copy()
