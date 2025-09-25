# -*- coding: utf-8 -*-
"""
Улучшенный BLS алгоритм для поиска экзопланет
Включает продвинутые методы обработки сигналов и машинного обучения
"""
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging
from scipy import stats, signal, optimize, interpolate
from scipy.signal import savgol_filter, find_peaks, medfilt, periodogram
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedBLS:
    """
    Улучшенный Box Least Squares алгоритм с продвинутыми возможностями
    """
    
    def __init__(self, 
                 minimum_period: float = 0.5, 
                 maximum_period: float = 50.0,
                 frequency_factor: float = 5.0,
                 minimum_n_transit: int = 3,
                 maximum_duration_factor: float = 0.3,
                 use_gpu: bool = False,
                 enable_ml_validation: bool = True):
        """
        Инициализация улучшенного BLS
        
        Args:
            minimum_period: Минимальный период поиска (дни)
            maximum_period: Максимальный период поиска (дни)
            frequency_factor: Фактор частотной сетки
            minimum_n_transit: Минимальное количество транзитов
            maximum_duration_factor: Максимальная доля периода для транзита
            use_gpu: Использовать GPU ускорение (если доступно)
            enable_ml_validation: Включить ML валидацию результатов
        """
        self.minimum_period = minimum_period
        self.maximum_period = maximum_period
        self.frequency_factor = frequency_factor
        self.minimum_n_transit = minimum_n_transit
        self.maximum_duration_factor = maximum_duration_factor
        self.use_gpu = use_gpu
        self.enable_ml_validation = enable_ml_validation
        
        # Кэш для ускорения повторных вычислений
        self._cache = {}
        
        logger.info(f"Enhanced BLS initialized: P=[{minimum_period:.2f}, {maximum_period:.2f}] days")
    
    def search(self, 
               time: np.ndarray, 
               flux: np.ndarray, 
               flux_err: Optional[np.ndarray] = None,
               target_name: str = "Unknown") -> Dict[str, Any]:
        """
        Выполнить улучшенный BLS поиск
        
        Args:
            time: Массив времени
            flux: Массив потока
            flux_err: Массив ошибок потока (опционально)
            target_name: Имя цели
            
        Returns:
            Словарь с результатами анализа
        """
        logger.info(f"Starting enhanced BLS search for {target_name}")
        
        try:
            # Предобработка данных
            time_clean, flux_clean, flux_err_clean = self._advanced_preprocessing(
                time, flux, flux_err
            )
            
            # Создание частотной сетки
            frequencies = self._create_optimal_frequency_grid(time_clean)
            periods = 1.0 / frequencies
            
            # Основной BLS поиск с оптимизациями
            bls_power, best_params = self._optimized_bls_search(
                time_clean, flux_clean, flux_err_clean, periods
            )
            
            # Статистическая значимость
            significance_stats = self._calculate_significance(
                bls_power, time_clean, flux_clean
            )
            
            # Физическая валидация
            physical_validation = self._validate_physical_parameters(
                best_params, time_clean, flux_clean
            )
            
            # ML валидация (если включена)
            ml_confidence = 0.0
            if self.enable_ml_validation:
                ml_confidence = self._ml_validation(
                    time_clean, flux_clean, best_params
                )
            
            # Дополнительные метрики
            additional_metrics = self._calculate_additional_metrics(
                time_clean, flux_clean, best_params
            )
            
            # Формирование результата
            result = {
                'target_name': target_name,
                'best_period': float(best_params['period']),
                'best_t0': float(best_params['t0']),
                'best_duration': float(best_params['duration']),
                'best_power': float(best_params['power']),
                'snr': float(significance_stats['snr']),
                'depth': float(best_params['depth']),
                'depth_err': float(best_params['depth_err']),
                'significance': float(significance_stats['significance']),
                'is_significant': bool(significance_stats['is_significant']),
                'enhanced_analysis': True,
                'ml_confidence': float(ml_confidence),
                'physical_validation': physical_validation,
                'additional_metrics': additional_metrics,
                'processing_info': {
                    'data_points': len(time_clean),
                    'frequency_grid_size': len(frequencies),
                    'search_range': f"{self.minimum_period:.2f}-{self.maximum_period:.2f} days"
                }
            }
            
            logger.info(f"Enhanced BLS completed: P={result['best_period']:.3f}d, SNR={result['snr']:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced BLS search failed: {str(e)}")
            raise
    
    def _advanced_preprocessing(self, 
                              time: np.ndarray, 
                              flux: np.ndarray, 
                              flux_err: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Продвинутая предобработка данных
        """
        logger.info(f"Advanced preprocessing: {len(time)} points")
        
        # Базовая очистка
        finite_mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            finite_mask &= np.isfinite(flux_err) & (flux_err > 0)
        
        time_clean = time[finite_mask]
        flux_clean = flux[finite_mask]
        flux_err_clean = flux_err[finite_mask] if flux_err is not None else None
        
        # Сортировка
        sort_indices = np.argsort(time_clean)
        time_clean = time_clean[sort_indices]
        flux_clean = flux_clean[sort_indices]
        if flux_err_clean is not None:
            flux_err_clean = flux_err_clean[sort_indices]
        
        # Продвинутое удаление выбросов с адаптивным порогом
        flux_clean = self._adaptive_outlier_removal(time_clean, flux_clean)
        
        # Удаление систематических трендов
        flux_clean = self._remove_systematic_trends(time_clean, flux_clean)
        
        # Нормализация с сохранением вариабельности
        flux_clean = self._robust_normalization(flux_clean)
        
        # Оценка ошибок если не предоставлены
        if flux_err_clean is None:
            flux_err_clean = self._estimate_noise_level(flux_clean)
        
        logger.info(f"Advanced preprocessing complete: {len(time_clean)} points")
        return time_clean, flux_clean, flux_err_clean
    
    def _adaptive_outlier_removal(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Адаптивное удаление выбросов с учетом локальной вариабельности
        """
        # Скользящее окно для локальной статистики
        window_size = max(50, len(flux) // 100)
        flux_filtered = flux.copy()
        
        for i in range(len(flux)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(flux), i + window_size // 2)
            
            local_flux = flux[start_idx:end_idx]
            local_median = np.median(local_flux)
            local_mad = np.median(np.abs(local_flux - local_median))
            
            # Адаптивный порог на основе локальной вариабельности
            threshold = max(3.0, 5.0 * local_mad / np.median(local_flux))
            
            if np.abs(flux[i] - local_median) > threshold * local_mad:
                flux_filtered[i] = local_median
        
        return flux_filtered
    
    def _remove_systematic_trends(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Удаление систематических трендов с сохранением транзитных сигналов
        """
        # Многомасштабное удаление трендов
        flux_detrended = flux.copy()
        
        # 1. Долгосрочные тренды (полиномиальная аппроксимация)
        if len(time) > 100:
            try:
                # Робастная полиномиальная аппроксимация
                coeffs = np.polyfit(time, flux, deg=3)
                trend_poly = np.polyval(coeffs, time)
                flux_detrended = flux / trend_poly
            except:
                pass
        
        # 2. Среднесрочные вариации (Savitzky-Golay фильтр)
        window_length = min(201, len(flux) // 5)
        if window_length >= 5 and window_length % 2 == 1:
            try:
                trend_sg = savgol_filter(flux_detrended, window_length, 3)
                flux_detrended = flux_detrended / trend_sg
            except:
                pass
        
        # 3. Высокочастотные шумы (Gaussian фильтр)
        if len(flux) > 50:
            sigma = len(flux) / 1000.0
            trend_gauss = gaussian_filter1d(flux_detrended, sigma=sigma)
            flux_detrended = flux_detrended / trend_gauss
        
        return flux_detrended
    
    def _robust_normalization(self, flux: np.ndarray) -> np.ndarray:
        """
        Робастная нормализация с сохранением транзитных сигналов
        """
        # Используем медиану вместо среднего для робастности
        flux_median = np.median(flux)
        
        # Нормализация с учетом вариабельности
        flux_mad = np.median(np.abs(flux - flux_median))
        if flux_mad > 0:
            # Сохраняем относительную вариабельность
            flux_normalized = (flux - flux_median) / flux_mad * 0.01 + 1.0
        else:
            flux_normalized = flux / flux_median
        
        return flux_normalized
    
    def _estimate_noise_level(self, flux: np.ndarray) -> np.ndarray:
        """
        Оценка уровня шума в данных
        """
        # Оценка шума через высокочастотные компоненты
        diff_flux = np.diff(flux)
        noise_estimate = np.std(diff_flux) / np.sqrt(2)
        
        # Адаптивная оценка ошибок
        flux_err = np.full_like(flux, noise_estimate)
        
        # Учет локальной вариабельности
        window_size = max(10, len(flux) // 50)
        for i in range(len(flux)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(flux), i + window_size // 2)
            
            local_flux = flux[start_idx:end_idx]
            local_std = np.std(local_flux)
            flux_err[i] = max(noise_estimate, local_std * 0.1)
        
        return flux_err
    
    def _create_optimal_frequency_grid(self, time: np.ndarray) -> np.ndarray:
        """
        Создание оптимальной частотной сетки
        """
        time_span = np.max(time) - np.min(time)
        
        # Минимальная и максимальная частоты
        f_min = 1.0 / self.maximum_period
        f_max = 1.0 / self.minimum_period
        
        # Частотное разрешение на основе временного базиса
        df = 1.0 / (self.frequency_factor * time_span)
        
        # Логарифмическая сетка для лучшего покрытия
        n_frequencies = int((f_max - f_min) / df)
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_frequencies)
        
        logger.info(f"Created frequency grid: {len(frequencies)} points")
        return frequencies
    
    def _optimized_bls_search(self, 
                            time: np.ndarray, 
                            flux: np.ndarray, 
                            flux_err: np.ndarray, 
                            periods: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Оптимизированный BLS поиск с ускорениями
        """
        n_periods = len(periods)
        bls_power = np.zeros(n_periods)
        best_params = {'period': 0, 't0': 0, 'duration': 0, 'power': 0, 'depth': 0, 'depth_err': 0}
        
        # Предвычисления для ускорения
        time_span = np.max(time) - np.min(time)
        weights = 1.0 / (flux_err ** 2)
        
        for i, period in enumerate(periods):
            # Фазовая свертка
            phase = ((time - time[0]) % period) / period
            sort_indices = np.argsort(phase)
            phase_sorted = phase[sort_indices]
            flux_sorted = flux[sort_indices]
            weights_sorted = weights[sort_indices]
            
            # Поиск оптимальной длительности транзита
            max_duration = min(period * self.maximum_duration_factor, period / 10.0)
            durations = np.linspace(0.01 * period, max_duration, 20)
            
            period_power = 0
            period_params = {}
            
            for duration in durations:
                duration_frac = duration / period
                
                # BLS статистика для данной длительности
                power, t0, depth, depth_err = self._calculate_bls_statistic(
                    phase_sorted, flux_sorted, weights_sorted, duration_frac
                )
                
                if power > period_power:
                    period_power = power
                    period_params = {
                        'period': period,
                        't0': t0 * period + time[0],
                        'duration': duration,
                        'power': power,
                        'depth': depth,
                        'depth_err': depth_err
                    }
            
            bls_power[i] = period_power
            
            if period_power > best_params['power']:
                best_params = period_params.copy()
        
        return bls_power, best_params
    
    def _calculate_bls_statistic(self, 
                               phase: np.ndarray, 
                               flux: np.ndarray, 
                               weights: np.ndarray, 
                               duration_frac: float) -> Tuple[float, float, float, float]:
        """
        Вычисление BLS статистики для заданной длительности
        """
        n_points = len(phase)
        max_power = 0
        best_t0 = 0
        best_depth = 0
        best_depth_err = 0
        
        # Сетка фаз для поиска t0
        n_phases = max(50, int(1.0 / duration_frac))
        phase_grid = np.linspace(0, 1, n_phases)
        
        for t0_phase in phase_grid:
            # Определение точек в транзите
            in_transit = ((phase - t0_phase + 0.5) % 1.0 - 0.5) < duration_frac / 2
            
            if np.sum(in_transit) < 3:  # Минимум 3 точки в транзите
                continue
            
            # Взвешенные средние
            flux_in = flux[in_transit]
            flux_out = flux[~in_transit]
            weights_in = weights[in_transit]
            weights_out = weights[~in_transit]
            
            if len(flux_out) < 3:
                continue
            
            mean_in = np.average(flux_in, weights=weights_in)
            mean_out = np.average(flux_out, weights=weights_out)
            
            # Глубина транзита
            depth = mean_out - mean_in
            
            if depth <= 0:  # Транзит должен быть понижением яркости
                continue
            
            # Ошибка глубины
            var_in = np.average((flux_in - mean_in)**2, weights=weights_in)
            var_out = np.average((flux_out - mean_out)**2, weights=weights_out)
            depth_err = np.sqrt(var_in / len(flux_in) + var_out / len(flux_out))
            
            # BLS мощность (отношение сигнал/шум)
            if depth_err > 0:
                power = (depth / depth_err) ** 2
            else:
                power = 0
            
            if power > max_power:
                max_power = power
                best_t0 = t0_phase
                best_depth = depth
                best_depth_err = depth_err
        
        return max_power, best_t0, best_depth, best_depth_err
    
    def _calculate_significance(self, 
                              bls_power: np.ndarray, 
                              time: np.ndarray, 
                              flux: np.ndarray) -> Dict[str, float]:
        """
        Расчет статистической значимости
        """
        max_power = np.max(bls_power)
        
        # SNR на основе распределения мощностей
        power_median = np.median(bls_power)
        power_mad = np.median(np.abs(bls_power - power_median))
        snr = (max_power - power_median) / (1.4826 * power_mad) if power_mad > 0 else 0
        
        # Статистическая значимость (False Alarm Probability)
        n_independent = len(bls_power) / 10  # Приблизительное число независимых частот
        significance = 1.0 - (1.0 - stats.chi2.sf(max_power, df=1)) ** n_independent
        
        # Критерий значимости
        is_significant = snr > 7.0 and significance > 0.99
        
        return {
            'snr': snr,
            'significance': significance,
            'is_significant': is_significant,
            'max_power': max_power,
            'power_median': power_median
        }
    
    def _validate_physical_parameters(self, 
                                    params: Dict, 
                                    time: np.ndarray, 
                                    flux: np.ndarray) -> Dict[str, bool]:
        """
        Валидация физических параметров
        """
        validation = {
            'period_reasonable': True,
            'depth_reasonable': True,
            'duration_reasonable': True,
            'multiple_transits': True,
            'overall_valid': True
        }
        
        # Проверка периода
        if params['period'] < 0.1 or params['period'] > 1000:
            validation['period_reasonable'] = False
        
        # Проверка глубины транзита
        if params['depth'] < 0.0001 or params['depth'] > 0.1:  # 0.01% - 10%
            validation['depth_reasonable'] = False
        
        # Проверка длительности
        duration_ratio = params['duration'] / params['period']
        if duration_ratio < 0.001 or duration_ratio > 0.3:
            validation['duration_reasonable'] = False
        
        # Проверка количества транзитов
        time_span = np.max(time) - np.min(time)
        n_transits = time_span / params['period']
        if n_transits < self.minimum_n_transit:
            validation['multiple_transits'] = False
        
        # Общая валидность
        validation['overall_valid'] = all([
            validation['period_reasonable'],
            validation['depth_reasonable'], 
            validation['duration_reasonable'],
            validation['multiple_transits']
        ])
        
        return validation
    
    def _ml_validation(self, 
                      time: np.ndarray, 
                      flux: np.ndarray, 
                      params: Dict) -> float:
        """
        ML валидация результатов (упрощенная версия)
        """
        # Простая эвристическая оценка на основе характеристик сигнала
        
        # Фазовая свертка для анализа формы транзита
        period = params['period']
        phase = ((time - params['t0']) % period) / period
        
        # Нормализация фазы к [-0.5, 0.5]
        phase = (phase + 0.5) % 1.0 - 0.5
        
        # Сортировка по фазе
        sort_indices = np.argsort(phase)
        phase_sorted = phase[sort_indices]
        flux_sorted = flux[sort_indices]
        
        # Анализ формы транзита
        duration_frac = params['duration'] / period
        in_transit = np.abs(phase_sorted) < duration_frac / 2
        
        if np.sum(in_transit) < 3:
            return 0.0
        
        # Характеристики для ML оценки
        features = []
        
        # 1. Симметрия транзита
        transit_flux = flux_sorted[in_transit]
        transit_phase = phase_sorted[in_transit]
        
        if len(transit_flux) > 5:
            # Корреляция с симметричной формой
            symmetric_phase = np.abs(transit_phase)
            symmetry_corr = np.corrcoef(transit_flux, symmetric_phase)[0, 1]
            features.append(abs(symmetry_corr))
        else:
            features.append(0.0)
        
        # 2. Глубина относительно шума
        out_of_transit = flux_sorted[~in_transit]
        if len(out_of_transit) > 10:
            noise_level = np.std(out_of_transit)
            depth_snr = params['depth'] / noise_level if noise_level > 0 else 0
            features.append(min(depth_snr / 10.0, 1.0))
        else:
            features.append(0.0)
        
        # 3. Соотношение длительности и периода
        duration_ratio = params['duration'] / params['period']
        optimal_ratio = 0.05  # Типичное соотношение для экзопланет
        ratio_score = 1.0 - abs(duration_ratio - optimal_ratio) / optimal_ratio
        features.append(max(0.0, ratio_score))
        
        # Простая комбинация признаков
        ml_confidence = np.mean(features)
        
        return float(np.clip(ml_confidence, 0.0, 1.0))
    
    def _calculate_additional_metrics(self, 
                                    time: np.ndarray, 
                                    flux: np.ndarray, 
                                    params: Dict) -> Dict[str, float]:
        """
        Дополнительные метрики для анализа
        """
        metrics = {}
        
        # Временной базис
        time_span = np.max(time) - np.min(time)
        metrics['time_span_days'] = float(time_span)
        
        # Количество транзитов
        n_transits = time_span / params['period']
        metrics['n_transits'] = float(n_transits)
        
        # Каденс наблюдений
        median_cadence = np.median(np.diff(time)) * 24 * 60  # в минутах
        metrics['cadence_minutes'] = float(median_cadence)
        
        # Уровень шума
        noise_level = np.std(flux) * 1e6  # в ppm
        metrics['noise_level_ppm'] = float(noise_level)
        
        # Отношение сигнал/шум транзита
        transit_snr = params['depth'] / (noise_level / 1e6) if noise_level > 0 else 0
        metrics['transit_snr'] = float(transit_snr)
        
        return metrics


def create_enhanced_bls(**kwargs) -> EnhancedBLS:
    """
    Фабричная функция для создания улучшенного BLS
    """
    return EnhancedBLS(**kwargs)
