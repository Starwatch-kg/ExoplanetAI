<<<<<<< HEAD
import asyncio
import logging
import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from scipy import stats
from astropy.stats import BoxLeastSquares
from astropy import units as u
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from models.search_models import (
    LightCurveData, 
    BLSResult, 
    TransitCandidate, 
    SearchResult, 
    SearchRequest
)

logger = logging.getLogger(__name__)

class BLSService:
    """Сервис для выполнения BLS анализа и поиска транзитов"""
    
    def __init__(self):
        self.min_observations_in_transit = 3
        self.oversample_factor = 5
        self.max_period_grid_size = 50000
        
    async def run_bls_analysis(
        self,
        lightcurve_data: LightCurveData,
        period_min: float = 0.5,
        period_max: float = 50.0,
        duration_min: float = 0.01,
        duration_max: float = 0.5,
        snr_threshold: float = 7.0
    ) -> SearchResult:
        """
        Выполнение полного BLS анализа
        """
        start_time = time.time()
        
        try:
            logger.info(f"Начат BLS анализ для {lightcurve_data.target_name}")
            
            # Подготовка данных
            time_array, flux_array, flux_err_array = await self._prepare_data(lightcurve_data)
            
            # Выполнение BLS
            bls_result = await self._run_bls(
                time_array, flux_array, flux_err_array,
                period_min, period_max, duration_min, duration_max
            )
            
            # Поиск кандидатов
            candidates = await self._find_candidates(
                time_array, flux_array, flux_err_array,
                bls_result, snr_threshold, lightcurve_data.target_name
            )
            
            # Валидация кандидатов
            validated_candidates = await self._validate_candidates(
                candidates, time_array, flux_array
            )
            
            processing_time = time.time() - start_time
            
            # Создание объекта запроса для результата
            search_request = SearchRequest(
                target_name=lightcurve_data.target_name,
                catalog="TIC",  # По умолчанию
                mission=lightcurve_data.mission,
                period_min=period_min,
                period_max=period_max,
                duration_min=duration_min,
                duration_max=duration_max,
                snr_threshold=snr_threshold
            )
            
            result = SearchResult(
                target_name=lightcurve_data.target_name,
                lightcurve=lightcurve_data,
                bls_result=bls_result,
                candidates=validated_candidates,
                total_candidates=len(validated_candidates),
                processing_time=processing_time,
                search_parameters=search_request
            )
            
            logger.info(f"BLS анализ завершен. Найдено {len(validated_candidates)} кандидатов")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в BLS анализе: {e}")
            raise
    
    async def _prepare_data(self, lightcurve_data: LightCurveData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Подготовка данных для BLS анализа"""
        try:
            time_array = np.array(lightcurve_data.time)
            flux_array = np.array(lightcurve_data.flux)
            
            # Обработка ошибок потока
            if lightcurve_data.flux_err:
                flux_err_array = np.array(lightcurve_data.flux_err)
            else:
                # Оценка ошибок на основе разброса данных
                flux_err_array = np.full_like(flux_array, np.std(flux_array) * 0.1)
            
            # Удаление NaN и бесконечных значений
            mask = np.isfinite(time_array) & np.isfinite(flux_array) & np.isfinite(flux_err_array)
            mask &= (flux_err_array > 0)  # Положительные ошибки
            
            time_array = time_array[mask]
            flux_array = flux_array[mask]
            flux_err_array = flux_err_array[mask]
            
            # Сортировка по времени
            sort_idx = np.argsort(time_array)
            time_array = time_array[sort_idx]
            flux_array = flux_array[sort_idx]
            flux_err_array = flux_err_array[sort_idx]
            
            # Нормализация потока (если еще не нормализован)
            if np.abs(np.median(flux_array) - 1.0) > 0.1:
                flux_array = flux_array / np.median(flux_array)
                flux_err_array = flux_err_array / np.median(flux_array)
            
            logger.info(f"Подготовлено {len(time_array)} точек данных")
            return time_array, flux_array, flux_err_array
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке данных: {e}")
            raise
    
    async def _run_bls(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        period_min: float,
        period_max: float,
        duration_min: float,
        duration_max: float
    ) -> BLSResult:
        """Выполнение BLS алгоритма"""
        try:
            # Создание BLS объекта
            bls = BoxLeastSquares(time * u.day, flux)
            
            # Определение сетки периодов
            baseline = np.max(time) - np.min(time)
            frequency_factor = self.oversample_factor
            
            # Ограничение размера сетки для производительности
            n_periods = min(
                int(frequency_factor * baseline * (1/period_min - 1/period_max)),
                self.max_period_grid_size
            )
            
            periods = np.linspace(period_min, period_max, n_periods) * u.day
            
            # Определение сетки длительностей
            durations = np.linspace(duration_min, duration_max, 20) * u.day
            
            logger.info(f"BLS сетка: {len(periods)} периодов, {len(durations)} длительностей")
            
            # Выполнение BLS
            periodogram = bls.power(periods, durations)
            
            # Поиск максимума
            best_idx = np.argmax(periodogram.power)
            best_period = periods[best_idx].value
            best_power = periodogram.power[best_idx]
            
            # Получение параметров лучшего кандидата
            stats_result = bls.compute_stats(
                best_period * u.day,
                durations,
                periodogram.objective[best_idx]
            )
            
            best_t0 = stats_result['transit_time'].value
            best_duration = stats_result['duration'].value
            best_depth = stats_result['depth']
            
            return BLSResult(
                periods=periods.value.tolist(),
                power=periodogram.power.tolist(),
                best_period=best_period,
                best_power=float(best_power),
                best_t0=float(best_t0),
                best_duration=float(best_duration),
                best_depth=float(best_depth)
            )
            
        except Exception as e:
            logger.error(f"Ошибка в BLS алгоритме: {e}")
            raise
    
    async def _find_candidates(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        bls_result: BLSResult,
        snr_threshold: float,
        target_name: str
    ) -> List[TransitCandidate]:
        """Поиск кандидатов в транзиты"""
        try:
            candidates = []
            
            # Поиск локальных максимумов в BLS периодограмме
            power_array = np.array(bls_result.power)
            periods_array = np.array(bls_result.periods)
            
            # Находим пики выше порога
            power_threshold = np.percentile(power_array, 95)  # Топ 5% по мощности
            peak_indices = self._find_peaks(power_array, height=power_threshold)
            
            for peak_idx in peak_indices:
                period = periods_array[peak_idx]
                power = power_array[peak_idx]
                
                # Вычисление SNR и других параметров
                snr = await self._calculate_snr(time, flux, flux_err, period)
                
                if snr >= snr_threshold:
                    # Подгонка транзитной модели
                    transit_params = await self._fit_transit_model(time, flux, flux_err, period)
                    
                    if transit_params:
                        # Вычисление физических параметров
                        physical_params = await self._calculate_physical_parameters(
                            transit_params, target_name
                        )
                        
                        # Оценка вероятностей
                        probabilities = await self._calculate_probabilities(
                            time, flux, transit_params, snr
                        )
                        
                        candidate = TransitCandidate(
                            period=float(period),
                            t0=float(transit_params['t0']),
                            duration=float(transit_params['duration'] * 24),  # в часах
                            depth=float(transit_params['depth'] * 1e6),  # в ppm
                            snr=float(snr),
                            sde=float(power * snr),  # Приближенная SDE
                            bls_power=float(power),
                            planet_radius=physical_params.get('planet_radius'),
                            semi_major_axis=physical_params.get('semi_major_axis'),
                            equilibrium_temp=physical_params.get('equilibrium_temp'),
                            false_alarm_probability=probabilities['false_alarm'],
                            planet_probability=probabilities['planet_prob']
                        )
                        
                        candidates.append(candidate)
            
            # Сортировка по SNR
            candidates.sort(key=lambda x: x.snr, reverse=True)
            
            return candidates[:10]  # Возвращаем топ-10 кандидатов
            
        except Exception as e:
            logger.error(f"Ошибка при поиске кандидатов: {e}")
            return []
    
    def _find_peaks(self, data: np.ndarray, height: float = None, distance: int = 5) -> List[int]:
        """Поиск пиков в данных"""
        peaks = []
        
        for i in range(distance, len(data) - distance):
            if height is not None and data[i] < height:
                continue
                
            # Проверка, является ли точка локальным максимумом
            is_peak = True
            for j in range(i - distance, i + distance + 1):
                if j != i and data[j] >= data[i]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    async def _calculate_snr(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        flux_err: np.ndarray, 
        period: float
    ) -> float:
        """Вычисление отношения сигнал/шум"""
        try:
            # Простая оценка SNR на основе разброса данных
            noise_level = np.std(flux)
            
            # Фолдинг данных по периоду
            phase = ((time - time[0]) % period) / period
            
            # Поиск транзитного сигнала
            phase_bins = np.linspace(0, 1, 100)
            binned_flux = []
            
            for i in range(len(phase_bins) - 1):
                mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
                if np.sum(mask) > 0:
                    binned_flux.append(np.mean(flux[mask]))
                else:
                    binned_flux.append(1.0)
            
            binned_flux = np.array(binned_flux)
            
            # Оценка глубины транзита
            transit_depth = 1.0 - np.min(binned_flux)
            
            # SNR как отношение глубины к шуму
            snr = transit_depth / noise_level if noise_level > 0 else 0
            
            return max(snr, 0.1)  # Минимальное значение SNR
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении SNR: {e}")
            return 0.1
    
    async def _fit_transit_model(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        flux_err: np.ndarray, 
        period: float
    ) -> Optional[Dict[str, float]]:
        """Подгонка простой транзитной модели"""
        try:
            # Фолдинг данных
            phase = ((time - time[0]) % period) / period
            
            # Поиск минимума (центр транзита)
            phase_bins = np.linspace(0, 1, 200)
            binned_flux = []
            
            for i in range(len(phase_bins) - 1):
                mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
                if np.sum(mask) > 0:
                    binned_flux.append(np.mean(flux[mask]))
                else:
                    binned_flux.append(1.0)
            
            binned_flux = np.array(binned_flux)
            
            # Поиск минимума
            min_idx = np.argmin(binned_flux)
            transit_phase = phase_bins[min_idx]
            
            # Оценка параметров транзита
            depth = 1.0 - np.min(binned_flux)
            
            # Оценка длительности (ширина на полувысоте)
            half_depth = 1.0 - depth / 2
            duration_mask = binned_flux < half_depth
            duration_phases = phase_bins[:-1][duration_mask]
            
            if len(duration_phases) > 0:
                duration = (np.max(duration_phases) - np.min(duration_phases)) * period
            else:
                duration = 0.1  # Минимальная длительность
            
            # Время первого транзита
            t0 = time[0] + transit_phase * period
            
            return {
                't0': t0,
                'period': period,
                'duration': duration,
                'depth': depth
            }
            
        except Exception as e:
            logger.error(f"Ошибка при подгонке транзитной модели: {e}")
            return None
    
    async def _calculate_physical_parameters(
        self, 
        transit_params: Dict[str, float], 
        target_name: str
    ) -> Dict[str, Optional[float]]:
        """Вычисление физических параметров планеты"""
        try:
            # Приближенные оценки (требуют данных о звезде)
            # Здесь используем типичные значения для звезд главной последовательности
            
            stellar_radius = 1.0  # R_sun (по умолчанию)
            stellar_mass = 1.0    # M_sun (по умолчанию)
            
            # Радиус планеты из глубины транзита
            depth = transit_params['depth']
            planet_radius = np.sqrt(depth) * stellar_radius * 109.2  # в радиусах Земли
            
            # Большая полуось из третьего закона Кеплера
            period_years = transit_params['period'] / 365.25
            semi_major_axis = (stellar_mass * period_years**2)**(1/3)  # в AU
            
            # Равновесная температура (приближенно)
            stellar_temp = 5778  # K (как у Солнца)
            equilibrium_temp = stellar_temp * np.sqrt(stellar_radius / (2 * semi_major_axis * 215))
            
            return {
                'planet_radius': float(planet_radius) if planet_radius > 0 else None,
                'semi_major_axis': float(semi_major_axis) if semi_major_axis > 0 else None,
                'equilibrium_temp': float(equilibrium_temp) if equilibrium_temp > 0 else None
            }
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении физических параметров: {e}")
            return {'planet_radius': None, 'semi_major_axis': None, 'equilibrium_temp': None}
    
    async def _calculate_probabilities(
        self, 
        time: np.ndarray, 
        flux: np.ndarray, 
        transit_params: Dict[str, float], 
        snr: float
    ) -> Dict[str, float]:
        """Вычисление вероятностей"""
        try:
            # Простая оценка вероятности ложного срабатывания
            # Основана на SNR и количестве независимых периодов
            
            baseline = np.max(time) - np.min(time)
            n_independent_periods = baseline / transit_params['period']
            
            # Вероятность ложного срабатывания (приближенная)
            false_alarm_prob = stats.norm.sf(snr) * n_independent_periods
            false_alarm_prob = min(false_alarm_prob, 0.99)
            
            # Вероятность планеты (эмпирическая формула)
            planet_prob = 1.0 / (1.0 + np.exp(-(snr - 7) / 2))
            planet_prob = max(0.01, min(0.99, planet_prob))
            
            return {
                'false_alarm': float(false_alarm_prob),
                'planet_prob': float(planet_prob)
            }
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении вероятностей: {e}")
            return {'false_alarm': 0.5, 'planet_prob': 0.5}
    
    async def _validate_candidates(
        self, 
        candidates: List[TransitCandidate], 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> List[TransitCandidate]:
        """Валидация кандидатов"""
        try:
            validated = []
            
            for candidate in candidates:
                # Проверки качества
                is_valid = True
                
                # Проверка минимального SNR
                if candidate.snr < 5.0:
                    is_valid = False
                
                # Проверка разумности периода
                if candidate.period < 0.1 or candidate.period > 1000:
                    is_valid = False
                
                # Проверка глубины транзита
                if candidate.depth < 10 or candidate.depth > 50000:  # ppm
                    is_valid = False
                
                # Проверка длительности
                if candidate.duration < 0.1 or candidate.duration > 24:  # часы
                    is_valid = False
                
                if is_valid:
                    validated.append(candidate)
            
            return validated
            
        except Exception as e:
            logger.error(f"Ошибка при валидации кандидатов: {e}")
            return candidates
=======
"""
Unified BLS (Box Least Squares) Service
Объединенный сервис BLS анализа для обнаружения транзитов
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from numba import jit, prange
import time

try:
    from core.logging_config import get_logger
    from core.telemetry import trace_bls_analysis, add_span_attributes
    from core.metrics import metrics_collector
except ImportError:
    # Fallback imports
    def get_logger(name):
        return logging.getLogger(name)
    
    # Mock functions
    def trace_bls_analysis(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def add_span_attributes(*args, **kwargs):
        pass
    
    class MockMetrics:
        def record_bls_analysis(self, *args, **kwargs): pass
        def record_error(self, *args, **kwargs): pass
    
    metrics_collector = MockMetrics()

logger = get_logger(__name__)

@dataclass
class BLSResult:
    """BLS analysis result"""
    best_period: float
    best_t0: float
    best_duration: float
    best_power: float
    snr: float
    depth: float
    depth_err: float
    significance: float
    is_significant: bool
    periods: np.ndarray
    powers: np.ndarray
    enhanced_analysis: bool = False
    ml_confidence: float = 0.0
    physical_validation: bool = True

@jit(nopython=True, parallel=True)
def _compute_bls_power_optimized(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    durations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized BLS power computation using Numba JIT
    Векторизованное вычисление BLS мощности
    """
    n_periods = len(periods)
    n_durations = len(durations)
    n_points = len(time)
    
    powers = np.zeros((n_periods, n_durations))
    best_t0s = np.zeros((n_periods, n_durations))
    best_depths = np.zeros((n_periods, n_durations))
    
    # Precompute flux statistics
    flux_mean = np.mean(flux)
    flux_var = np.var(flux)
    
    for i in prange(n_periods):
        period = periods[i]
        
        for j in range(n_durations):
            duration = durations[j]
            
            # Phase fold the data
            phases = ((time % period) / period) % 1.0
            
            # Grid search over transit center
            n_phase_bins = min(100, int(period / duration * 10))
            phase_grid = np.linspace(0, 1, n_phase_bins)
            
            max_power = 0.0
            best_t0 = 0.0
            best_depth = 0.0
            
            for k in range(n_phase_bins):
                phase_center = phase_grid[k]
                
                # Define transit window
                half_duration = duration / (2 * period)
                
                # Handle phase wrapping
                if phase_center - half_duration < 0:
                    in_transit = ((phases >= phase_center - half_duration + 1) | 
                                (phases <= phase_center + half_duration))
                elif phase_center + half_duration > 1:
                    in_transit = ((phases >= phase_center - half_duration) | 
                                (phases <= phase_center + half_duration - 1))
                else:
                    in_transit = ((phases >= phase_center - half_duration) & 
                                (phases <= phase_center + half_duration))
                
                n_in = np.sum(in_transit)
                n_out = n_points - n_in
                
                if n_in < 3 or n_out < 3:
                    continue
                
                # Compute in-transit and out-of-transit means
                flux_in = np.mean(flux[in_transit])
                flux_out = np.mean(flux[~in_transit])
                
                # Compute BLS power (signal-to-noise ratio)
                if flux_var > 0:
                    depth = flux_out - flux_in
                    power = depth * np.sqrt(n_in * n_out / n_points) / np.sqrt(flux_var)
                    
                    if power > max_power:
                        max_power = power
                        best_t0 = phase_center * period
                        best_depth = depth
            
            powers[i, j] = max_power
            best_t0s[i, j] = best_t0
            best_depths[i, j] = best_depth
    
    return powers, best_t0s, best_depths

class UnifiedBLSService:
    """Unified BLS analysis service"""
    
    def __init__(self):
        self.logger = get_logger("bls.service")
    
    def detect_transits(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period_min: float = 0.5,
        period_max: float = 20.0,
        duration_min: float = 0.05,
        duration_max: float = 0.3,
        snr_threshold: float = 7.0,
        use_enhanced: bool = True,
        star_info: Optional[Dict[str, Any]] = None
    ) -> BLSResult:
        """
        Detect transits using BLS algorithm
        
        Args:
            time: Time array (days)
            flux: Normalized flux array
            period_min: Minimum period to search (days)
            period_max: Maximum period to search (days)
            duration_min: Minimum transit duration (days)
            duration_max: Maximum transit duration (days)
            snr_threshold: SNR threshold for significance
            use_enhanced: Use enhanced analysis
            star_info: Stellar information for physical validation
        
        Returns:
            BLS analysis result
        """
        with trace_bls_analysis(
            target_name=star_info.get("target_id", "unknown") if star_info else "unknown",
            catalog=star_info.get("catalog", "unknown") if star_info else "unknown",
            mission="unknown"
        ) as span:
            
            start_time = time.time()
            
            # Validate inputs
            if len(time) != len(flux):
                raise ValueError("Time and flux arrays must have same length")
            
            if len(time) < 100:
                raise ValueError("Need at least 100 data points for BLS analysis")
            
            # Preprocess data
            time, flux = self._preprocess_data(time, flux)
            
            # Generate search grids
            periods = self._generate_period_grid(period_min, period_max, len(time))
            durations = self._generate_duration_grid(
                duration_min, duration_max, periods, star_info
            )
            
            self.logger.info(
                f"BLS grid: {len(periods)} periods × {len(durations)} durations = {len(periods) * len(durations)} combinations"
            )
            
            # Compute BLS powers
            powers, best_t0s, best_depths = _compute_bls_power_optimized(
                time, flux, periods, durations
            )
            
            # Find best result
            best_idx = np.unravel_index(np.argmax(powers), powers.shape)
            best_period = periods[best_idx[0]]
            best_duration = durations[best_idx[1]]
            best_power = powers[best_idx]
            best_t0 = best_t0s[best_idx]
            best_depth = best_depths[best_idx]
            
            # Compute statistics
            snr = self._compute_snr(time, flux, best_period, best_t0, best_duration)
            depth_err = self._estimate_depth_error(flux, best_period, best_duration)
            significance = self._compute_significance(powers, best_power)
            
            # Physical validation
            is_significant = (
                snr >= snr_threshold and
                significance > 0.01 and
                self._validate_physical_parameters(
                    best_period, best_duration, best_depth, star_info
                )
            )
            
            # Enhanced analysis
            ml_confidence = 0.0
            if use_enhanced and is_significant:
                ml_confidence = self._enhanced_analysis(
                    time, flux, best_period, best_t0, best_duration
                )
            
            processing_time = time.time() - start_time
            
            # Add telemetry attributes
            if span:
                add_span_attributes({
                    "bls.periods_tested": len(periods),
                    "bls.durations_tested": len(durations),
                    "bls.best_period": best_period,
                    "bls.best_snr": snr,
                    "bls.significance": significance,
                    "bls.processing_time_ms": processing_time * 1000
                })
            
            # Record metrics
            metrics_collector.record_exoplanet_analysis(
                catalog=star_info.get("catalog", "unknown") if star_info else "unknown",
                mission="BLS",
                status="success",
                duration=processing_time,
                candidates_found=1 if is_significant else 0,
                max_snr=snr
            )
            
            self.logger.info(
                f"BLS analysis completed in {processing_time:.2f}s: "
                f"P={best_period:.3f}d, SNR={snr:.1f}, sig={significance:.4f}"
            )
            
            return BLSResult(
                best_period=best_period,
                best_t0=best_t0,
                best_duration=best_duration,
                best_power=best_power,
                snr=snr,
                depth=best_depth,
                depth_err=depth_err,
                significance=significance,
                is_significant=is_significant,
                periods=periods,
                powers=np.max(powers, axis=1),
                enhanced_analysis=use_enhanced,
                ml_confidence=ml_confidence,
                physical_validation=True
            )
    
    def _preprocess_data(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess time and flux data"""
        # Remove NaN and infinite values
        valid = np.isfinite(time) & np.isfinite(flux)
        time = time[valid]
        flux = flux[valid]
        
        # Sort by time
        sort_idx = np.argsort(time)
        time = time[sort_idx]
        flux = flux[sort_idx]
        
        # Normalize flux
        flux = flux / np.median(flux)
        
        # Remove outliers (simple sigma clipping)
        median_flux = np.median(flux)
        mad = np.median(np.abs(flux - median_flux))
        sigma = 1.4826 * mad
        outliers = np.abs(flux - median_flux) > 5 * sigma
        
        if np.sum(outliers) > 0:
            flux[outliers] = median_flux
            self.logger.info(f"Removed {np.sum(outliers)} outliers")
        
        return time, flux
    
    def _generate_period_grid(
        self, 
        period_min: float, 
        period_max: float, 
        n_points: int
    ) -> np.ndarray:
        """Generate optimal period grid"""
        # Frequency spacing based on data span
        time_span = period_max  # Approximate
        df = 1.0 / (10 * time_span)  # Oversampling factor of 10
        
        f_min = 1.0 / period_max
        f_max = 1.0 / period_min
        
        # Logarithmic spacing in frequency
        n_periods = min(1000, int((f_max - f_min) / df))
        frequencies = np.linspace(f_min, f_max, n_periods)
        periods = 1.0 / frequencies
        
        return np.sort(periods)
    
    def _generate_duration_grid(
        self,
        duration_min: float,
        duration_max: float,
        periods: np.ndarray,
        star_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Generate duration grid with physical constraints"""
        # Base duration grid
        n_durations = 10
        durations = np.linspace(duration_min, duration_max, n_durations)
        
        # Apply physical constraints if star info available
        if star_info:
            stellar_radius = star_info.get("radius", 1.0)  # Solar radii
            
            # Maximum duration based on stellar radius and period
            max_physical_durations = []
            for period in periods:
                # Approximate maximum duration for central transit
                max_duration = 0.2 * period * np.sqrt(stellar_radius)
                max_physical_durations.append(min(max_duration, duration_max))
            
            # Use minimum of physical and requested maximum
            duration_max_physical = np.min(max_physical_durations)
            durations = np.linspace(duration_min, duration_max_physical, n_durations)
        
        return durations
    
    def _compute_snr(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
        duration: float
    ) -> float:
        """Compute signal-to-noise ratio"""
        # Phase fold
        phases = ((time - t0) % period) / period
        phases[phases > 0.5] -= 1.0
        
        # Define transit
        in_transit = np.abs(phases) < (duration / period / 2)
        
        if np.sum(in_transit) < 3:
            return 0.0
        
        # Compute depths
        flux_in = np.mean(flux[in_transit])
        flux_out = np.mean(flux[~in_transit])
        depth = flux_out - flux_in
        
        # Estimate noise
        noise = np.std(flux[~in_transit]) / np.sqrt(np.sum(in_transit))
        
        return depth / noise if noise > 0 else 0.0
    
    def _estimate_depth_error(
        self,
        flux: np.ndarray,
        period: float,
        duration: float
    ) -> float:
        """Estimate depth measurement error"""
        # Simple estimate based on photometric noise
        noise_level = np.std(flux)
        n_transits = len(flux) * duration / period
        
        if n_transits > 0:
            return noise_level / np.sqrt(n_transits)
        else:
            return noise_level
    
    def _compute_significance(self, powers: np.ndarray, best_power: float) -> float:
        """Compute statistical significance"""
        # False alarm probability based on power distribution
        all_powers = powers.flatten()
        all_powers = all_powers[np.isfinite(all_powers)]
        
        if len(all_powers) == 0:
            return 0.0
        
        # Fraction of powers greater than best power
        n_greater = np.sum(all_powers >= best_power)
        significance = 1.0 - (n_greater - 1) / len(all_powers)
        
        return max(0.0, significance)
    
    def _validate_physical_parameters(
        self,
        period: float,
        duration: float,
        depth: float,
        star_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate physical plausibility of parameters"""
        # Basic sanity checks
        if period <= 0 or duration <= 0 or depth <= 0:
            return False
        
        # Duration should be reasonable fraction of period
        if duration > 0.3 * period:
            return False
        
        # Depth should be reasonable (not too deep for stellar eclipse)
        if depth > 0.1:  # 10% depth is very deep for planet
            return False
        
        # Additional checks with stellar information
        if star_info:
            stellar_radius = star_info.get("radius", 1.0)
            
            # Estimate planet radius from depth
            planet_radius_ratio = np.sqrt(depth)
            planet_radius_earth = planet_radius_ratio * stellar_radius * 109.1  # Earth radii
            
            # Reject if planet would be larger than Jupiter
            if planet_radius_earth > 15:  # ~1.3 Jupiter radii
                return False
        
        return True
    
    def _enhanced_analysis(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        period: float,
        t0: float,
        duration: float
    ) -> float:
        """Enhanced analysis using additional metrics"""
        # Phase fold the data
        phases = ((time - t0) % period) / period
        phases[phases > 0.5] -= 1.0
        
        # Sort by phase
        sort_idx = np.argsort(phases)
        phases_sorted = phases[sort_idx]
        flux_sorted = flux[sort_idx]
        
        # Compute additional metrics
        metrics = []
        
        # 1. Transit shape consistency
        in_transit = np.abs(phases_sorted) < (duration / period / 2)
        if np.sum(in_transit) > 5:
            transit_flux = flux_sorted[in_transit]
            transit_phases = phases_sorted[in_transit]
            
            # Check for V-shaped transit (expected for planet)
            phase_bins = np.linspace(-duration/period/2, duration/period/2, 5)
            binned_flux = []
            for i in range(len(phase_bins) - 1):
                mask = (transit_phases >= phase_bins[i]) & (transit_phases < phase_bins[i+1])
                if np.sum(mask) > 0:
                    binned_flux.append(np.mean(transit_flux[mask]))
            
            if len(binned_flux) >= 3:
                # V-shape score (center should be deepest)
                center_idx = len(binned_flux) // 2
                if center_idx > 0 and center_idx < len(binned_flux) - 1:
                    v_shape_score = (
                        (binned_flux[0] + binned_flux[-1]) / 2 - binned_flux[center_idx]
                    ) / np.std(flux)
                    metrics.append(min(1.0, max(0.0, v_shape_score / 3.0)))
        
        # 2. Odd-even consistency (planets should have consistent transits)
        odd_transits = []
        even_transits = []
        
        transit_times = []
        current_time = t0
        while current_time < np.max(time):
            if current_time >= np.min(time):
                transit_times.append(current_time)
            current_time += period
        
        for i, tt in enumerate(transit_times):
            transit_mask = np.abs(time - tt) < duration / 2
            if np.sum(transit_mask) > 3:
                transit_depth = 1.0 - np.mean(flux[transit_mask])
                if i % 2 == 0:
                    even_transits.append(transit_depth)
                else:
                    odd_transits.append(transit_depth)
        
        if len(odd_transits) > 0 and len(even_transits) > 0:
            odd_mean = np.mean(odd_transits)
            even_mean = np.mean(even_transits)
            consistency = 1.0 - abs(odd_mean - even_mean) / (odd_mean + even_mean + 1e-6)
            metrics.append(consistency)
        
        # Return average of available metrics
        if metrics:
            return np.mean(metrics)
        else:
            return 0.5  # Neutral confidence

# Global instance
bls_service = UnifiedBLSService()

# Backward compatibility function
def detect_transits_bls(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float = 0.5,
    period_max: float = 20.0,
    duration_min: float = 0.05,
    duration_max: float = 0.3,
    snr_threshold: float = 7.0,
    use_enhanced: bool = True,
    star_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Backward compatible BLS function"""
    result = bls_service.detect_transits(
        time=time,
        flux=flux,
        period_min=period_min,
        period_max=period_max,
        duration_min=duration_min,
        duration_max=duration_max,
        snr_threshold=snr_threshold,
        use_enhanced=use_enhanced,
        star_info=star_info
    )
    
    return {
        "best_period": result.best_period,
        "best_t0": result.best_t0,
        "best_duration": result.best_duration,
        "best_power": result.best_power,
        "snr": result.snr,
        "depth": result.depth,
        "depth_err": result.depth_err,
        "significance": result.significance,
        "is_significant": result.is_significant,
        "enhanced_analysis": result.enhanced_analysis,
        "ml_confidence": result.ml_confidence,
        "physical_validation": result.physical_validation
    }
>>>>>>> 975c3a7 (Версия 1.5.1)
