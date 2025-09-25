"""
Продвинутый BLS (Box Least Squares) алгоритм для поиска транзитов экзопланет
Основан на научных статьях и реальных астрономических методах
"""
import numpy as np
from typing import Tuple, Dict, List
import logging
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedBLS:
    """
    Продвинутый BLS алгоритм для обнаружения транзитов
    """
    
    def __init__(self, minimum_period: float = 0.5, maximum_period: float = 50.0):
        self.minimum_period = minimum_period
        self.maximum_period = maximum_period
        
    def preprocess_lightcurve(self, time: np.ndarray, flux: np.ndarray, 
                             flux_err: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Предобработка кривой блеска: удаление выбросов, нормализация, сглаживание тренда
        """
        # Удаляем NaN и inf значения
        mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            mask &= np.isfinite(flux_err) & (flux_err > 0)
        
        time_clean = time[mask]
        flux_clean = flux[mask]
        flux_err_clean = flux_err[mask] if flux_err is not None else np.ones_like(flux_clean) * np.std(flux_clean)
        
        if len(time_clean) < 100:
            raise ValueError("Недостаточно данных для анализа")
        
        # Удаляем выбросы (3-sigma clipping)
        flux_median = np.median(flux_clean)
        flux_std = np.std(flux_clean)
        outlier_mask = np.abs(flux_clean - flux_median) < 3 * flux_std
        
        time_clean = time_clean[outlier_mask]
        flux_clean = flux_clean[outlier_mask]
        flux_err_clean = flux_err_clean[outlier_mask]
        
        # Удаляем долгосрочный тренд с помощью Savitzky-Golay фильтра
        if len(flux_clean) > 51:
            window_length = min(51, len(flux_clean) // 10)
            if window_length % 2 == 0:
                window_length += 1
            
            try:
                trend = savgol_filter(flux_clean, window_length, 3)
                flux_detrended = flux_clean / trend
            except:
                # Fallback: простое удаление тренда
                flux_detrended = flux_clean / np.median(flux_clean)
        else:
            flux_detrended = flux_clean / np.median(flux_clean)
        
        # Нормализация к единице
        flux_detrended = flux_detrended / np.median(flux_detrended)
        
        return time_clean, flux_detrended, flux_err_clean
    
    def compute_bls_spectrum(self, time: np.ndarray, flux: np.ndarray, 
                           flux_err: np.ndarray, period_grid: np.ndarray,
                           duration_grid: np.ndarray) -> Dict:
        """
        Вычисление BLS спектра для сетки периодов и длительностей
        """
        n_periods = len(period_grid)
        n_durations = len(duration_grid)
        
        # Массивы для хранения результатов
        bls_power = np.zeros((n_periods, n_durations))
        bls_depth = np.zeros((n_periods, n_durations))
        bls_t0 = np.zeros((n_periods, n_durations))
        
        # Общие параметры
        n_points = len(flux)
        flux_mean = np.mean(flux)
        flux_var = np.var(flux)
        
        for i, period in enumerate(period_grid):
            for j, duration in enumerate(duration_grid):
                
                # Фазовая свертка
                phases = ((time - time[0]) % period) / period
                
                # Сортируем по фазе
                sort_indices = np.argsort(phases)
                phases_sorted = phases[sort_indices]
                flux_sorted = flux[sort_indices]
                weights_sorted = 1.0 / (flux_err[sort_indices] ** 2)
                
                # Ищем оптимальную фазу транзита
                power, depth, t0_phase = self._optimize_transit_phase(
                    phases_sorted, flux_sorted, weights_sorted, duration
                )
                
                bls_power[i, j] = power
                bls_depth[i, j] = depth
                bls_t0[i, j] = t0_phase * period + time[0]
        
        # Находим глобальный максимум
        max_idx = np.unravel_index(np.argmax(bls_power), bls_power.shape)
        best_period_idx, best_duration_idx = max_idx
        
        best_period = period_grid[best_period_idx]
        best_duration = duration_grid[best_duration_idx]
        best_power = bls_power[max_idx]
        best_depth = bls_depth[max_idx]
        best_t0 = bls_t0[max_idx]
        
        # Вычисляем статистическую значимость
        snr = self._compute_snr(best_power, bls_power, n_points)
        significance = self._compute_significance(snr, n_periods * n_durations)
        
        return {
            "best_period": best_period,
            "best_duration": best_duration,
            "best_power": best_power,
            "best_depth": best_depth,
            "best_t0": best_t0,
            "snr": snr,
            "significance": significance,
            "bls_spectrum": bls_power,
            "period_grid": period_grid,
            "duration_grid": duration_grid,
            "depth_err": self._estimate_depth_error(best_depth, flux_err, n_points)
        }
    
    def _optimize_transit_phase(self, phases: np.ndarray, flux: np.ndarray, 
                               weights: np.ndarray, duration: float) -> Tuple[float, float, float]:
        """
        Оптимизация фазы транзита для данного периода и длительности
        """
        n_phase_bins = min(200, len(phases) // 5)
        phase_grid = np.linspace(0, 1, n_phase_bins)
        
        best_power = 0
        best_depth = 0
        best_phase = 0
        
        duration_phase = duration  # duration уже в фазовых единицах
        
        for phase_center in phase_grid:
            # Определяем точки в транзите
            phase_diff = np.abs((phases - phase_center + 0.5) % 1 - 0.5)
            in_transit = phase_diff <= duration_phase / 2
            
            if np.sum(in_transit) < 3:  # Минимум 3 точки в транзите
                continue
            
            # Взвешенные средние внутри и вне транзита
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
            
            if depth <= 0:  # Транзит должен быть понижением яркости
                continue
            
            # BLS статистика (упрощенная версия)
            n_in = len(flux_in)
            n_out = len(flux_out)
            n_total = n_in + n_out
            
            if n_in == 0 or n_out == 0:
                continue
            
            # Статистика хи-квадрат для оценки качества подгонки
            var_in = np.average((flux_in - mean_in)**2, weights=weights_in)
            var_out = np.average((flux_out - mean_out)**2, weights=weights_out)
            
            # BLS мощность (Signal Residue)
            signal_residue = depth**2 * n_in * n_out / n_total
            power = signal_residue / (var_in + var_out + 1e-10)
            
            if power > best_power:
                best_power = power
                best_depth = depth
                best_phase = phase_center
        
        return best_power, best_depth, best_phase
    
    def _compute_snr(self, best_power: float, bls_spectrum: np.ndarray, n_points: int) -> float:
        """
        Вычисление отношения сигнал/шум
        """
        # Медиана и MAD (Median Absolute Deviation) спектра
        spectrum_flat = bls_spectrum.flatten()
        spectrum_median = np.median(spectrum_flat)
        spectrum_mad = np.median(np.abs(spectrum_flat - spectrum_median))
        
        # SNR как отклонение от медианы в единицах MAD
        snr = (best_power - spectrum_median) / (1.4826 * spectrum_mad + 1e-10)
        
        return max(0, snr)
    
    def _compute_significance(self, snr: float, n_trials: int) -> float:
        """
        Вычисление статистической значимости с учетом множественных сравнений
        """
        # Поправка Бонферрони для множественных сравнений
        p_value = stats.norm.sf(snr) * n_trials  # односторонний тест
        significance = max(0, min(1, 1 - p_value))
        
        return significance
    
    def _estimate_depth_error(self, depth: float, flux_err: np.ndarray, n_points: int) -> float:
        """
        Оценка ошибки глубины транзита
        """
        # Простая оценка: ошибка пропорциональна шуму и обратно пропорциональна sqrt(N)
        typical_error = np.median(flux_err)
        depth_error = typical_error / np.sqrt(max(1, n_points / 10))
        
        return depth_error
    
    def detect_transits(self, time: np.ndarray, flux: np.ndarray, 
                       flux_err: np.ndarray = None,
                       period_min: float = None, period_max: float = None,
                       duration_min: float = 0.01, duration_max: float = 0.3,
                       snr_threshold: float = 7.0) -> Dict:
        """
        Основная функция обнаружения транзитов
        """
        try:
            # Используем параметры класса если не заданы
            if period_min is None:
                period_min = self.minimum_period
            if period_max is None:
                period_max = self.maximum_period
            
            # Предобработка данных
            time_clean, flux_clean, flux_err_clean = self.preprocess_lightcurve(
                time, flux, flux_err
            )
            
            logger.info(f"Preprocessed lightcurve: {len(time_clean)} points")
            
            # Создаем сетки для поиска
            # Логарифмическая сетка периодов для лучшего покрытия
            n_periods = min(1000, max(100, int((period_max - period_min) * 10)))
            period_grid = np.logspace(np.log10(period_min), np.log10(period_max), n_periods)
            
            # Линейная сетка длительностей
            n_durations = 20
            duration_grid = np.linspace(duration_min, duration_max, n_durations)
            
            logger.info(f"BLS grid: {len(period_grid)} periods × {len(duration_grid)} durations")
            
            # Выполняем BLS анализ
            bls_results = self.compute_bls_spectrum(
                time_clean, flux_clean, flux_err_clean,
                period_grid, duration_grid
            )
            
            # Проверяем значимость обнаружения
            is_significant = (bls_results["snr"] >= snr_threshold and 
                            bls_results["significance"] > 0.001)
            
            # Формируем результат
            result = {
                "best_period": round(bls_results["best_period"], 6),
                "best_power": round(bls_results["best_power"], 6),
                "best_duration": round(bls_results["best_duration"], 6),
                "best_t0": round(bls_results["best_t0"], 6),
                "snr": round(bls_results["snr"], 2),
                "depth": round(abs(bls_results["best_depth"]), 6),
                "depth_err": round(bls_results["depth_err"], 6),
                "significance": round(bls_results["significance"], 4),
                "is_significant": is_significant,
                "n_points_used": len(time_clean),
                "periods_tested": len(period_grid),
                "durations_tested": len(duration_grid),
                "preprocessing": {
                    "outliers_removed": len(time) - len(time_clean),
                    "detrending_applied": True,
                    "normalization_applied": True
                }
            }
            
            logger.info(f"BLS completed: SNR={result['snr']:.1f}, "
                       f"Significance={result['significance']:.3f}, "
                       f"Significant={is_significant}")
            
            return result
            
        except Exception as e:
            logger.error(f"BLS analysis failed: {e}")
            raise

# Глобальный экземпляр для использования в API
advanced_bls = AdvancedBLS()
