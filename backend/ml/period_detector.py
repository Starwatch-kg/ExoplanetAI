"""
Детектор периодических сигналов - выделен из EnsembleSearchService
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

from core.constants import TransitConstants, MLConstants
from core.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class PeriodDetector:
    """
    Специализированный детектор периодических сигналов
    
    Отвечает только за:
    - Детекцию периодов различными методами
    - Анализ гармоник и алиасов
    - Валидацию найденных периодов
    """
    
    def __init__(self):
        self.name = "Period Detector"
        self.version = "2.0.0"
        self.methods = ['lomb_scargle', 'autocorrelation', 'fft', 'phase_dispersion']
    
    async def detect_periods(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        methods: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Детекция периодов различными методами
        
        Args:
            time: Временные метки
            flux: Значения потока
            methods: Список методов для использования
            
        Returns:
            Результаты детекции периодов
        """
        try:
            if len(time) < MLConstants.MIN_DATA_POINTS:
                raise AnalysisError(
                    f"Insufficient data points: {len(time)} < {MLConstants.MIN_DATA_POINTS}"
                )
            
            # Подготовка данных
            time_clean, flux_clean = self._prepare_data(time, flux)
            
            # Выбор методов
            if methods is None:
                methods = self.methods
            
            # Применение всех методов
            results = {}
            for method in methods:
                if method in self.methods:
                    try:
                        method_result = await self._apply_method(method, time_clean, flux_clean)
                        results[method] = method_result
                    except Exception as e:
                        logger.warning(f"Method {method} failed: {e}")
                        results[method] = None
            
            # Консенсус между методами
            consensus_result = self._find_consensus(results, time_clean, flux_clean)
            
            return {
                'individual_methods': results,
                'consensus': consensus_result,
                'data_quality': self._assess_data_quality(time_clean, flux_clean)
            }
            
        except Exception as e:
            logger.error(f"Period detection failed: {e}")
            raise AnalysisError(f"Period detection failed: {str(e)}")
    
    def _prepare_data(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для анализа периодов"""
        # Удаление NaN и сортировка
        mask = np.isfinite(time) & np.isfinite(flux)
        time_clean = time[mask]
        flux_clean = flux[mask]
        
        sort_idx = np.argsort(time_clean)
        time_clean = time_clean[sort_idx]
        flux_clean = flux_clean[sort_idx]
        
        # Нормализация потока
        flux_median = np.median(flux_clean)
        flux_clean = (flux_clean - flux_median) / flux_median
        
        return time_clean, flux_clean
    
    async def _apply_method(
        self, 
        method: str, 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Dict[str, any]:
        """Применение конкретного метода детекции"""
        
        if method == 'lomb_scargle':
            return self._lomb_scargle_periodogram(time, flux)
        elif method == 'autocorrelation':
            return self._autocorrelation_analysis(time, flux)
        elif method == 'fft':
            return self._fft_analysis(time, flux)
        elif method == 'phase_dispersion':
            return self._phase_dispersion_minimization(time, flux)
        else:
            raise AnalysisError(f"Unknown method: {method}")
    
    def _lomb_scargle_periodogram(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, any]:
        """Анализ методом Lomb-Scargle периодограммы"""
        try:
            from scipy.signal import lombscargle
            
            # Частотная сетка
            min_freq = 1.0 / TransitConstants.MAX_PERIOD_DAYS
            max_freq = 1.0 / TransitConstants.MIN_PERIOD_DAYS
            frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), 10000)
            
            # Вычисление периодограммы
            power = lombscargle(time, flux, frequencies, normalize=True)
            
            # Поиск пиков
            peaks, properties = signal.find_peaks(
                power, 
                height=np.mean(power) + 2*np.std(power),
                distance=50
            )
            
            # Сортировка по мощности
            if len(peaks) > 0:
                peak_powers = power[peaks]
                sorted_indices = np.argsort(peak_powers)[::-1]
                
                best_periods = []
                for i in sorted_indices[:5]:  # Топ 5 периодов
                    period = 1.0 / frequencies[peaks[i]]
                    period_power = peak_powers[i]
                    
                    best_periods.append({
                        'period': float(period),
                        'power': float(period_power),
                        'frequency': float(frequencies[peaks[i]]),
                        'significance': float(period_power / np.mean(power))
                    })
                
                return {
                    'method': 'lomb_scargle',
                    'best_period': best_periods[0]['period'] if best_periods else None,
                    'all_periods': best_periods,
                    'periodogram': {
                        'frequencies': frequencies.tolist(),
                        'power': power.tolist()
                    }
                }
            else:
                return {
                    'method': 'lomb_scargle',
                    'best_period': None,
                    'all_periods': [],
                    'periodogram': {
                        'frequencies': frequencies.tolist(),
                        'power': power.tolist()
                    }
                }
                
        except Exception as e:
            raise AnalysisError(f"Lomb-Scargle analysis failed: {str(e)}")
    
    def _autocorrelation_analysis(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, any]:
        """Анализ автокорреляционной функции"""
        try:
            # Интерполяция на равномерную сетку
            dt = np.median(np.diff(time))
            time_uniform = np.arange(time.min(), time.max(), dt)
            flux_interp = np.interp(time_uniform, time, flux)
            
            # Автокорреляция
            autocorr = np.correlate(flux_interp, flux_interp, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Нормализация
            autocorr = autocorr / autocorr[0]
            
            # Поиск пиков в автокорреляции
            lags = np.arange(len(autocorr)) * dt
            
            # Исключаем центральный пик
            search_start = int(TransitConstants.MIN_PERIOD_DAYS / dt)
            search_end = min(len(autocorr), int(TransitConstants.MAX_PERIOD_DAYS / dt))
            
            if search_start < search_end:
                peaks, _ = signal.find_peaks(
                    autocorr[search_start:search_end],
                    height=0.1,  # Минимальная корреляция
                    distance=int(0.5 / dt)  # Минимальное расстояние между пиками
                )
                
                if len(peaks) > 0:
                    # Корректировка индексов
                    peaks = peaks + search_start
                    peak_periods = lags[peaks]
                    peak_correlations = autocorr[peaks]
                    
                    # Сортировка по корреляции
                    sorted_indices = np.argsort(peak_correlations)[::-1]
                    
                    best_periods = []
                    for i in sorted_indices[:5]:
                        best_periods.append({
                            'period': float(peak_periods[i]),
                            'correlation': float(peak_correlations[i]),
                            'lag_index': int(peaks[i])
                        })
                    
                    return {
                        'method': 'autocorrelation',
                        'best_period': best_periods[0]['period'] if best_periods else None,
                        'all_periods': best_periods,
                        'autocorrelation': {
                            'lags': lags.tolist(),
                            'values': autocorr.tolist()
                        }
                    }
            
            return {
                'method': 'autocorrelation',
                'best_period': None,
                'all_periods': [],
                'autocorrelation': {
                    'lags': lags.tolist(),
                    'values': autocorr.tolist()
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"Autocorrelation analysis failed: {str(e)}")
    
    def _fft_analysis(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, any]:
        """Анализ методом быстрого преобразования Фурье"""
        try:
            # Интерполяция на равномерную сетку
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
            power = np.abs(fft_values[positive_freq_mask])**2
            
            # Фильтрация по диапазону периодов
            min_freq = 1.0 / TransitConstants.MAX_PERIOD_DAYS
            max_freq = 1.0 / TransitConstants.MIN_PERIOD_DAYS
            
            freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
            frequencies = frequencies[freq_mask]
            power = power[freq_mask]
            
            if len(frequencies) > 0:
                # Поиск пиков
                peaks, _ = signal.find_peaks(
                    power,
                    height=np.mean(power) + 2*np.std(power),
                    distance=10
                )
                
                if len(peaks) > 0:
                    peak_powers = power[peaks]
                    peak_frequencies = frequencies[peaks]
                    peak_periods = 1.0 / peak_frequencies
                    
                    # Сортировка по мощности
                    sorted_indices = np.argsort(peak_powers)[::-1]
                    
                    best_periods = []
                    for i in sorted_indices[:5]:
                        best_periods.append({
                            'period': float(peak_periods[i]),
                            'power': float(peak_powers[i]),
                            'frequency': float(peak_frequencies[i])
                        })
                    
                    return {
                        'method': 'fft',
                        'best_period': best_periods[0]['period'] if best_periods else None,
                        'all_periods': best_periods,
                        'spectrum': {
                            'frequencies': frequencies.tolist(),
                            'power': power.tolist()
                        }
                    }
            
            return {
                'method': 'fft',
                'best_period': None,
                'all_periods': [],
                'spectrum': {
                    'frequencies': frequencies.tolist() if len(frequencies) > 0 else [],
                    'power': power.tolist() if len(power) > 0 else []
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"FFT analysis failed: {str(e)}")
    
    def _phase_dispersion_minimization(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, any]:
        """Метод минимизации фазовой дисперсии (PDM)"""
        try:
            # Сетка периодов для тестирования
            periods = np.logspace(
                np.log10(TransitConstants.MIN_PERIOD_DAYS),
                np.log10(TransitConstants.MAX_PERIOD_DAYS),
                1000
            )
            
            dispersions = []
            
            for period in periods:
                # Фазировка данных
                phases = (time % period) / period
                
                # Сортировка по фазе
                sort_indices = np.argsort(phases)
                sorted_flux = flux[sort_indices]
                
                # Разбиение на бины по фазе
                n_bins = min(20, len(flux) // 5)  # Адаптивное количество бинов
                if n_bins < 3:
                    dispersions.append(np.inf)
                    continue
                
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_dispersions = []
                
                for i in range(n_bins):
                    bin_mask = (phases >= bin_edges[i]) & (phases < bin_edges[i + 1])
                    if np.sum(bin_mask) > 1:
                        bin_flux = flux[bin_mask]
                        bin_dispersions.append(np.var(bin_flux))
                
                if bin_dispersions:
                    # PDM статистика
                    total_dispersion = np.mean(bin_dispersions)
                    dispersions.append(total_dispersion)
                else:
                    dispersions.append(np.inf)
            
            dispersions = np.array(dispersions)
            
            # Поиск минимумов дисперсии
            finite_mask = np.isfinite(dispersions)
            if np.any(finite_mask):
                valid_dispersions = dispersions[finite_mask]
                valid_periods = periods[finite_mask]
                
                # Нормализация
                dispersions_norm = valid_dispersions / np.mean(valid_dispersions)
                
                # Поиск минимумов
                minima, _ = signal.find_peaks(
                    -dispersions_norm,  # Инвертируем для поиска минимумов
                    height=-0.8,  # Ищем значения меньше 80% от среднего
                    distance=50
                )
                
                if len(minima) > 0:
                    best_periods = []
                    for i in minima:
                        best_periods.append({
                            'period': float(valid_periods[i]),
                            'dispersion': float(valid_dispersions[i]),
                            'normalized_dispersion': float(dispersions_norm[i])
                        })
                    
                    # Сортировка по дисперсии (меньше = лучше)
                    best_periods.sort(key=lambda x: x['dispersion'])
                    
                    return {
                        'method': 'phase_dispersion',
                        'best_period': best_periods[0]['period'] if best_periods else None,
                        'all_periods': best_periods[:5],
                        'dispersion_curve': {
                            'periods': valid_periods.tolist(),
                            'dispersions': dispersions_norm.tolist()
                        }
                    }
            
            return {
                'method': 'phase_dispersion',
                'best_period': None,
                'all_periods': [],
                'dispersion_curve': {
                    'periods': periods.tolist(),
                    'dispersions': dispersions.tolist()
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"Phase dispersion minimization failed: {str(e)}")
    
    def _find_consensus(
        self, 
        results: Dict[str, Dict], 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Dict[str, any]:
        """Поиск консенсуса между различными методами"""
        
        # Собираем все найденные периоды
        all_periods = []
        method_weights = {
            'lomb_scargle': 1.0,
            'autocorrelation': 0.8,
            'fft': 0.7,
            'phase_dispersion': 0.9
        }
        
        for method, result in results.items():
            if result and result.get('best_period'):
                period = result['best_period']
                weight = method_weights.get(method, 0.5)
                
                # Добавляем метрику качества из каждого метода
                if method == 'lomb_scargle':
                    quality = result.get('all_periods', [{}])[0].get('significance', 0)
                elif method == 'autocorrelation':
                    quality = result.get('all_periods', [{}])[0].get('correlation', 0)
                elif method == 'fft':
                    quality = result.get('all_periods', [{}])[0].get('power', 0)
                elif method == 'phase_dispersion':
                    # Для PDM меньшая дисперсия = лучше, инвертируем
                    dispersion = result.get('all_periods', [{}])[0].get('normalized_dispersion', 1)
                    quality = 1.0 / (1.0 + dispersion)
                else:
                    quality = 0.5
                
                all_periods.append({
                    'period': period,
                    'method': method,
                    'weight': weight,
                    'quality': quality,
                    'score': weight * quality
                })
        
        if not all_periods:
            return {
                'consensus_period': None,
                'confidence': 0.0,
                'supporting_methods': [],
                'period_clusters': []
            }
        
        # Кластеризация периодов (группировка близких значений)
        clusters = self._cluster_periods(all_periods)
        
        # Выбор лучшего кластера
        best_cluster = max(clusters, key=lambda c: c['total_score'])
        
        return {
            'consensus_period': best_cluster['mean_period'],
            'confidence': best_cluster['confidence'],
            'supporting_methods': best_cluster['methods'],
            'period_clusters': clusters,
            'individual_scores': all_periods
        }
    
    def _cluster_periods(self, periods_data: List[Dict]) -> List[Dict]:
        """Кластеризация периодов по близости значений"""
        
        if not periods_data:
            return []
        
        clusters = []
        tolerance = 0.05  # 5% толерантность для группировки
        
        for period_info in periods_data:
            period = period_info['period']
            
            # Ищем существующий кластер
            assigned = False
            for cluster in clusters:
                cluster_mean = cluster['mean_period']
                if abs(period - cluster_mean) / cluster_mean < tolerance:
                    # Добавляем в существующий кластер
                    cluster['periods'].append(period_info)
                    cluster['total_score'] += period_info['score']
                    cluster['methods'].append(period_info['method'])
                    
                    # Пересчитываем средний период
                    all_periods = [p['period'] for p in cluster['periods']]
                    cluster['mean_period'] = np.mean(all_periods)
                    
                    assigned = True
                    break
            
            if not assigned:
                # Создаем новый кластер
                clusters.append({
                    'mean_period': period,
                    'periods': [period_info],
                    'total_score': period_info['score'],
                    'methods': [period_info['method']],
                    'confidence': 0.0  # Будет рассчитана позже
                })
        
        # Расчет уверенности для каждого кластера
        for cluster in clusters:
            n_methods = len(set(cluster['methods']))  # Уникальные методы
            n_total_methods = len(self.methods)
            
            # Уверенность зависит от количества поддерживающих методов и их скоров
            method_diversity = n_methods / n_total_methods
            score_strength = cluster['total_score'] / len(cluster['periods'])
            
            cluster['confidence'] = (method_diversity * 0.6 + score_strength * 0.4)
        
        # Сортировка по общему скору
        clusters.sort(key=lambda c: c['total_score'], reverse=True)
        
        return clusters
    
    def _assess_data_quality(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, any]:
        """Оценка качества данных для периодического анализа"""
        
        # Временное покрытие
        time_span = time.max() - time.min()
        
        # Каденс (среднее время между наблюдениями)
        cadence = np.median(np.diff(time))
        
        # Количество точек
        n_points = len(time)
        
        # Пропуски в данных
        expected_points = int(time_span / cadence)
        completeness = n_points / expected_points if expected_points > 0 else 0
        
        # Шум в данных
        flux_std = np.std(flux)
        flux_median = np.median(flux)
        noise_level = flux_std / abs(flux_median) if flux_median != 0 else np.inf
        
        # Оценка качества
        quality_score = 0.0
        
        # Бонусы за хорошие характеристики
        if time_span > 10 * TransitConstants.MAX_PERIOD_DAYS:  # Достаточное покрытие
            quality_score += 0.3
        
        if n_points > MLConstants.RECOMMENDED_DATA_POINTS:  # Достаточно точек
            quality_score += 0.3
        
        if completeness > 0.8:  # Хорошая полнота
            quality_score += 0.2
        
        if noise_level < 0.01:  # Низкий шум
            quality_score += 0.2
        
        return {
            'time_span_days': float(time_span),
            'cadence_days': float(cadence),
            'n_points': int(n_points),
            'completeness': float(completeness),
            'noise_level': float(noise_level),
            'quality_score': float(quality_score)
        }
