"""
Профессиональный BLS алгоритм для поиска транзитов экзопланет
Основан на астрономических стандартах и научных публикациях
"""
import numpy as np
from typing import Tuple, Dict, List, Optional
import logging
from scipy import stats, signal, optimize, interpolate
from scipy.signal import savgol_filter, find_peaks, medfilt
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProfessionalBLS:
    """
    Профессиональный Box Least Squares алгоритм
    Реализует полный астрономический стандарт поиска транзитов
    """
    
    def __init__(self, 
                 minimum_period: float = 0.5, 
                 maximum_period: float = 50.0,
                 frequency_factor: float = 5.0,
                 minimum_n_transit: int = 3,
                 maximum_duration_factor: float = 0.3):
        """
        Инициализация BLS с астрономическими параметрами
        
        Args:
            minimum_period: Минимальный период поиска (дни)
            maximum_period: Максимальный период поиска (дни)  
            frequency_factor: Фактор частотной сетки
            minimum_n_transit: Минимальное количество транзитов
            maximum_duration_factor: Максимальная доля периода для транзита
        """
        self.minimum_period = minimum_period
        self.maximum_period = maximum_period
        self.frequency_factor = frequency_factor
        self.minimum_n_transit = minimum_n_transit
        self.maximum_duration_factor = maximum_duration_factor
        
        # Астрономические константы
        self.EARTH_RADIUS_KM = 6371.0
        self.SUN_RADIUS_KM = 696340.0
        self.AU_KM = 149597870.7
        
    def preprocess_lightcurve(self, time: np.ndarray, flux: np.ndarray, 
                             flux_err: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Профессиональная предобработка кривой блеска
        """
        logger.info(f"Starting lightcurve preprocessing: {len(time)} points")
        
        # Проверка входных данных
        if len(time) != len(flux):
            raise ValueError("Time and flux arrays must have same length")
        
        if len(time) < 100:
            raise ValueError("Insufficient data points for reliable analysis")
        
        # Удаление NaN и бесконечных значений
        finite_mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            finite_mask &= np.isfinite(flux_err) & (flux_err > 0)
        
        time_clean = time[finite_mask]
        flux_clean = flux[finite_mask]
        flux_err_clean = flux_err[finite_mask] if flux_err is not None else None
        
        logger.info(f"After NaN removal: {len(time_clean)} points")
        
        # Сортировка по времени
        sort_indices = np.argsort(time_clean)
        time_clean = time_clean[sort_indices]
        flux_clean = flux_clean[sort_indices]
        if flux_err_clean is not None:
            flux_err_clean = flux_err_clean[sort_indices]
        
        # Удаление дубликатов времени
        unique_mask = np.diff(time_clean, prepend=time_clean[0]-1) > 1e-10
        time_clean = time_clean[unique_mask]
        flux_clean = flux_clean[unique_mask]
        if flux_err_clean is not None:
            flux_err_clean = flux_err_clean[unique_mask]
        
        # Статистическое удаление выбросов (итеративное 3-sigma clipping)
        for iteration in range(3):
            flux_median = np.median(flux_clean)
            flux_mad = np.median(np.abs(flux_clean - flux_median))
            sigma_estimate = 1.4826 * flux_mad
            
            outlier_threshold = 3.0 * sigma_estimate
            good_mask = np.abs(flux_clean - flux_median) < outlier_threshold
            
            if np.sum(~good_mask) == 0:
                break
                
            time_clean = time_clean[good_mask]
            flux_clean = flux_clean[good_mask]
            if flux_err_clean is not None:
                flux_err_clean = flux_err_clean[good_mask]
        
        logger.info(f"After outlier removal: {len(time_clean)} points")
        
        # Удаление долгосрочного тренда
        flux_detrended = self._remove_stellar_trends(time_clean, flux_clean)
        
        # Нормализация
        flux_normalized = flux_detrended / np.median(flux_detrended)
        
        # Оценка ошибок если не предоставлены
        if flux_err_clean is None:
            flux_err_clean = self._estimate_photometric_errors(flux_normalized)
        else:
            flux_err_clean = flux_err_clean / np.median(flux_detrended)
        
        logger.info(f"Preprocessing complete: {len(time_clean)} points ready for BLS")
        
        return time_clean, flux_normalized, flux_err_clean
    
    def _remove_stellar_trends(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Удаление звездных трендов и инструментальных эффектов
        """
        # Медианная фильтрация для удаления кратковременных выбросов
        if len(flux) > 51:
            flux_filtered = medfilt(flux, kernel_size=5)
        else:
            flux_filtered = flux.copy()
        
        # Savitzky-Golay фильтр для удаления долгосрочных трендов
        window_length = min(101, len(flux) // 10)
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= 5:
            try:
                trend = savgol_filter(flux_filtered, window_length, 3)
                flux_detrended = flux / trend
            except:
                # Fallback: полиномиальное удаление тренда
                flux_detrended = self._polynomial_detrend(time, flux, degree=3)
        else:
            flux_detrended = flux / np.median(flux)
        
        return flux_detrended
    
    def _polynomial_detrend(self, time: np.ndarray, flux: np.ndarray, degree: int = 3) -> np.ndarray:
        """
        Полиномиальное удаление тренда
        """
        try:
            # Нормализация времени для численной стабильности
            time_norm = (time - time[0]) / (time[-1] - time[0])
            
            # Робастная полиномиальная подгонка
            coeffs = np.polyfit(time_norm, flux, degree)
            trend = np.polyval(coeffs, time_norm)
            
            return flux / trend
        except:
            return flux / np.median(flux)
    
    def _estimate_photometric_errors(self, flux: np.ndarray) -> np.ndarray:
        """
        Оценка фотометрических ошибок на основе статистики
        """
        # Оценка шума через скользящее стандартное отклонение
        window_size = min(50, len(flux) // 10)
        flux_errors = np.zeros_like(flux)
        
        for i in range(len(flux)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(flux), i + window_size // 2)
            local_std = np.std(flux[start_idx:end_idx])
            flux_errors[i] = max(local_std, 1e-6)  # Минимальная ошибка
        
        return flux_errors
    
    def compute_bls_periodogram(self, time: np.ndarray, flux: np.ndarray, 
                               flux_err: np.ndarray) -> Dict:
        """
        Вычисление полного BLS периодограммы
        """
        logger.info("Computing BLS periodogram...")
        
        # Создание частотной сетки
        frequency_grid = self._create_frequency_grid(time)
        period_grid = 1.0 / frequency_grid
        
        # Создание сетки длительностей
        duration_grid = self._create_duration_grid(period_grid)
        
        logger.info(f"BLS grid: {len(period_grid)} periods × {len(duration_grid)} durations")
        
        # Инициализация массивов результатов
        n_periods = len(period_grid)
        n_durations = len(duration_grid)
        
        bls_power = np.zeros((n_periods, n_durations))
        bls_depth = np.zeros((n_periods, n_durations))
        bls_t0 = np.zeros((n_periods, n_durations))
        bls_snr = np.zeros((n_periods, n_durations))
        
        # Предвычисления для оптимизации
        flux_mean = np.mean(flux)
        flux_var = np.var(flux)
        weights = 1.0 / (flux_err ** 2)
        
        # Основной цикл BLS
        for i, period in enumerate(period_grid):
            if i % 100 == 0:
                logger.info(f"BLS progress: {i}/{n_periods} periods")
            
            # Фазовая свертка
            phases = self._fold_lightcurve(time, period)
            
            for j, duration in enumerate(duration_grid):
                # Вычисление BLS статистики для данного периода и длительности
                power, depth, t0, snr = self._compute_bls_statistics(
                    phases, flux, weights, duration, period, time[0]
                )
                
                bls_power[i, j] = power
                bls_depth[i, j] = depth
                bls_t0[i, j] = t0
                bls_snr[i, j] = snr
        
        # Поиск глобального максимума
        best_idx = np.unravel_index(np.argmax(bls_power), bls_power.shape)
        best_period_idx, best_duration_idx = best_idx
        
        best_period = period_grid[best_period_idx]
        best_duration = duration_grid[best_duration_idx]
        best_power = bls_power[best_idx]
        best_depth = bls_depth[best_idx]
        best_t0 = bls_t0[best_idx]
        best_snr = bls_snr[best_idx]
        
        # Статистический анализ
        significance = self._compute_statistical_significance(
            best_power, bls_power, len(time)
        )
        
        # Оценка ошибок
        depth_error = self._estimate_parameter_errors(
            time, flux, flux_err, best_period, best_duration, best_t0
        )
        
        logger.info(f"BLS complete: Best period={best_period:.6f}d, SNR={best_snr:.2f}")
        
        return {
            "best_period": best_period,
            "best_duration": best_duration,
            "best_power": best_power,
            "best_depth": abs(best_depth),
            "best_t0": best_t0,
            "snr": best_snr,
            "significance": significance,
            "depth_error": depth_error,
            "bls_spectrum": bls_power,
            "period_grid": period_grid,
            "duration_grid": duration_grid,
            "n_points": len(time),
            "time_span": time[-1] - time[0]
        }
    
    def _create_frequency_grid(self, time: np.ndarray) -> np.ndarray:
        """
        Создание оптимальной частотной сетки для BLS
        """
        time_span = time[-1] - time[0]
        
        # Минимальная и максимальная частоты
        min_frequency = 1.0 / self.maximum_period
        max_frequency = 1.0 / self.minimum_period
        
        # Частотное разрешение основано на времени наблюдений
        frequency_resolution = 1.0 / (self.frequency_factor * time_span)
        
        # Создание логарифмически распределенной сетки
        n_frequencies = int((max_frequency - min_frequency) / frequency_resolution)
        n_frequencies = min(n_frequencies, 10000)  # Ограничение для производительности
        
        frequency_grid = np.logspace(
            np.log10(min_frequency), 
            np.log10(max_frequency), 
            n_frequencies
        )
        
        return frequency_grid
    
    def _create_duration_grid(self, period_grid: np.ndarray) -> np.ndarray:
        """
        Создание сетки длительностей транзитов
        """
        # Длительности как доли периода
        duration_fractions = np.logspace(-3, np.log10(self.maximum_duration_factor), 50)
        
        # Берем медианный период для расчета абсолютных длительностей
        median_period = np.median(period_grid)
        duration_grid = duration_fractions * median_period
        
        # Ограничиваем разумными пределами
        duration_grid = duration_grid[
            (duration_grid >= 0.01) & (duration_grid <= 1.0)
        ]
        
        return duration_grid
    
    def _fold_lightcurve(self, time: np.ndarray, period: float) -> np.ndarray:
        """
        Фазовая свертка кривой блеска
        """
        phases = ((time - time[0]) % period) / period
        return phases
    
    def _compute_bls_statistics(self, phases: np.ndarray, flux: np.ndarray, 
                               weights: np.ndarray, duration: float, 
                               period: float, t0_ref: float) -> Tuple[float, float, float, float]:
        """
        Вычисление BLS статистики для конкретного периода и длительности
        """
        duration_phase = duration / period
        
        # Сетка фаз для поиска оптимального времени транзита
        n_phase_bins = min(200, len(phases) // 3)
        phase_centers = np.linspace(0, 1, n_phase_bins)
        
        best_power = 0
        best_depth = 0
        best_t0 = 0
        best_snr = 0
        
        for phase_center in phase_centers:
            # Определение точек в транзите
            phase_diff = np.minimum(
                np.abs(phases - phase_center),
                np.abs(phases - phase_center + 1),
                np.abs(phases - phase_center - 1)
            )
            
            in_transit = phase_diff <= duration_phase / 2
            
            n_in_transit = np.sum(in_transit)
            if n_in_transit < 3:  # Минимум точек в транзите
                continue
            
            # Взвешенные статистики
            flux_in = flux[in_transit]
            flux_out = flux[~in_transit]
            weights_in = weights[in_transit]
            weights_out = weights[~in_transit]
            
            if len(flux_out) < 3:
                continue
            
            # Взвешенные средние
            mean_in = np.average(flux_in, weights=weights_in)
            mean_out = np.average(flux_out, weights=weights_out)
            
            # Глубина транзита
            depth = mean_out - mean_in
            
            if depth <= 0:  # Транзит должен быть понижением
                continue
            
            # BLS мощность (Signal Residue)
            n_total = len(flux)
            q = n_in_transit / n_total  # Доля времени в транзите
            
            if q <= 0 or q >= 1:
                continue
            
            # Классическая BLS статистика
            signal_residue = (depth ** 2) * n_in_transit * (n_total - n_in_transit) / n_total
            
            # Нормализация на дисперсию
            var_in = np.average((flux_in - mean_in) ** 2, weights=weights_in)
            var_out = np.average((flux_out - mean_out) ** 2, weights=weights_out)
            total_variance = (var_in * n_in_transit + var_out * (n_total - n_in_transit)) / n_total
            
            if total_variance <= 0:
                continue
            
            power = signal_residue / (total_variance + 1e-10)
            
            # SNR оценка
            snr = depth / np.sqrt(total_variance / n_in_transit + 1e-10)
            
            if power > best_power:
                best_power = power
                best_depth = depth
                best_t0 = phase_center * period + t0_ref
                best_snr = snr
        
        return best_power, best_depth, best_t0, best_snr
    
    def _compute_statistical_significance(self, best_power: float, 
                                        bls_spectrum: np.ndarray, 
                                        n_points: int) -> float:
        """
        Вычисление статистической значимости обнаружения
        """
        # Статистика спектра
        spectrum_flat = bls_spectrum.flatten()
        spectrum_median = np.median(spectrum_flat)
        spectrum_mad = np.median(np.abs(spectrum_flat - spectrum_median))
        
        if spectrum_mad == 0:
            return 0.0
        
        # Z-score относительно фона
        z_score = (best_power - spectrum_median) / (1.4826 * spectrum_mad)
        
        # Поправка на множественные сравнения (Bonferroni)
        n_trials = len(spectrum_flat)
        p_value = stats.norm.sf(z_score) * n_trials
        
        # Значимость
        significance = max(0, min(1, 1 - p_value))
        
        return significance
    
    def _estimate_parameter_errors(self, time: np.ndarray, flux: np.ndarray, 
                                  flux_err: np.ndarray, period: float, 
                                  duration: float, t0: float) -> float:
        """
        Оценка ошибок параметров транзита
        """
        # Простая оценка через фотометрический шум
        typical_error = np.median(flux_err)
        n_in_transit = np.sum(
            np.abs(((time - t0) % period) / period - 0.5) < duration / period / 2
        )
        
        if n_in_transit > 0:
            depth_error = typical_error / np.sqrt(n_in_transit)
        else:
            depth_error = typical_error
        
        return depth_error
    
    def detect_transits(self, time: np.ndarray, flux: np.ndarray, 
                       flux_err: np.ndarray = None, **kwargs) -> Dict:
        """
        Главная функция обнаружения транзитов
        """
        try:
            logger.info(f"Starting professional BLS analysis on {len(time)} data points")
            
            # Предобработка
            time_clean, flux_clean, flux_err_clean = self.preprocess_lightcurve(
                time, flux, flux_err
            )
            
            # BLS анализ
            bls_results = self.compute_bls_periodogram(
                time_clean, flux_clean, flux_err_clean
            )
            
            # Проверка значимости
            snr_threshold = kwargs.get('snr_threshold', 7.0)
            is_significant = (
                bls_results["snr"] >= snr_threshold and 
                bls_results["significance"] > 0.001 and
                bls_results["best_power"] > 10.0
            )
            
            # Физическая валидация
            is_physical = self._validate_transit_physics(bls_results)
            
            final_result = {
                **bls_results,
                "is_significant": is_significant and is_physical,
                "snr_threshold": snr_threshold,
                "preprocessing_stats": {
                    "original_points": len(time),
                    "cleaned_points": len(time_clean),
                    "outliers_removed": len(time) - len(time_clean)
                }
            }
            
            logger.info(
                f"BLS analysis complete: Period={bls_results['best_period']:.6f}d, "
                f"SNR={bls_results['snr']:.2f}, Significant={is_significant and is_physical}"
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Professional BLS analysis failed: {e}")
            raise
    
    def _validate_transit_physics(self, bls_results: Dict) -> bool:
        """
        Валидация физической реалистичности транзита
        """
        period = bls_results["best_period"]
        duration = bls_results["best_duration"]
        depth = bls_results["best_depth"]
        
        # Проверка разумности параметров
        if not (0.1 <= period <= 1000):  # Период в разумных пределах
            return False
        
        if not (0.001 <= duration <= period * 0.5):  # Длительность не больше половины периода
            return False
        
        if not (1e-6 <= depth <= 0.5):  # Глубина в разумных пределах
            return False
        
        # Проверка отношения длительности к периоду
        duration_ratio = duration / period
        if duration_ratio > 0.3:  # Слишком длинный транзит
            return False
        
        return True

# Глобальный экземпляр
professional_bls = ProfessionalBLS()
