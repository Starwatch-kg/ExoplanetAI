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
