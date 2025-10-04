"""
Анализатор транзитных сигналов - выделен из EnsembleSearchService
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.timeseries import BoxLeastSquares

from core.constants import TransitConstants, MLConstants
from core.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class TransitAnalyzer:
    """
    Специализированный анализатор транзитных сигналов
    
    Отвечает только за:
    - Детекцию транзитов методом BLS
    - Анализ параметров транзитов
    - Валидацию транзитных кандидатов
    """
    
    def __init__(self):
        self.name = "Transit Analyzer"
        self.version = "2.0.0"
    
    async def detect_transits(
        self, 
        time: np.ndarray, 
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Детекция транзитов методом Box Least Squares
        
        Args:
            time: Временные метки
            flux: Значения потока
            flux_err: Ошибки потока (опционально)
            
        Returns:
            Результаты BLS анализа
        """
        try:
            if len(time) < MLConstants.MIN_DATA_POINTS:
                raise AnalysisError(
                    f"Insufficient data points: {len(time)} < {MLConstants.MIN_DATA_POINTS}"
                )
            
            # Подготовка данных для BLS
            time_clean, flux_clean = self._prepare_data(time, flux)
            
            # BLS анализ
            bls_result = self._run_bls_analysis(time_clean, flux_clean)
            
            # Валидация результатов
            validated_result = self._validate_transit_candidates(bls_result, time_clean, flux_clean)
            
            return validated_result
            
        except Exception as e:
            logger.error(f"Transit detection failed: {e}")
            raise AnalysisError(f"Transit detection failed: {str(e)}")
    
    def _prepare_data(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для BLS анализа"""
        # Удаление NaN значений
        mask = np.isfinite(time) & np.isfinite(flux)
        time_clean = time[mask]
        flux_clean = flux[mask]
        
        # Нормализация потока
        flux_median = np.median(flux_clean)
        flux_clean = flux_clean / flux_median
        
        # Сортировка по времени
        sort_idx = np.argsort(time_clean)
        time_clean = time_clean[sort_idx]
        flux_clean = flux_clean[sort_idx]
        
        return time_clean, flux_clean
    
    def _run_bls_analysis(self, time: np.ndarray, flux: np.ndarray) -> Dict[str, any]:
        """Выполнение BLS анализа"""
        try:
            # Создание BLS объекта
            bls = BoxLeastSquares(time, flux)
            
            # Диапазон периодов для поиска
            period_grid = np.logspace(
                np.log10(TransitConstants.MIN_PERIOD_DAYS),
                np.log10(TransitConstants.MAX_PERIOD_DAYS),
                1000
            )
            
            # Выполнение BLS
            periodogram = bls.autopower(
                period_grid,
                minimum_n_transit=2,
                frequency_factor=1.0
            )
            
            # Поиск лучшего периода
            best_period_idx = np.argmax(periodogram.power)
            best_period = periodogram.period[best_period_idx]
            best_power = periodogram.power[best_period_idx]
            
            # Статистика транзита
            transit_info = bls.compute_stats(
                best_period,
                periodogram.duration[best_period_idx],
                periodogram.transit_time[best_period_idx]
            )
            
            return {
                'period': float(best_period),
                'power': float(best_power),
                'duration': float(periodogram.duration[best_period_idx]),
                'transit_time': float(periodogram.transit_time[best_period_idx]),
                'depth': float(transit_info['depth']),
                'snr': float(transit_info.get('snr', 0)),
                'periodogram': {
                    'period': periodogram.period.tolist(),
                    'power': periodogram.power.tolist()
                }
            }
            
        except Exception as e:
            raise AnalysisError(f"BLS analysis failed: {str(e)}")
    
    def _validate_transit_candidates(
        self, 
        bls_result: Dict[str, any], 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Dict[str, any]:
        """Валидация транзитных кандидатов"""
        
        period = bls_result['period']
        depth = bls_result['depth']
        duration = bls_result['duration']
        snr = bls_result['snr']
        
        # Проверки валидности
        validations = {
            'period_valid': (
                TransitConstants.MIN_PERIOD_DAYS <= period <= TransitConstants.MAX_PERIOD_DAYS
            ),
            'depth_valid': (
                TransitConstants.MIN_TRANSIT_DEPTH <= depth <= TransitConstants.MAX_TRANSIT_DEPTH
            ),
            'duration_valid': (
                TransitConstants.MIN_DURATION_HOURS <= duration * 24 <= TransitConstants.MAX_DURATION_HOURS
            ),
            'snr_sufficient': snr >= 5.0,  # Минимальный SNR для детекции
        }
        
        # Дополнительные проверки
        validations.update(self._additional_validations(bls_result, time, flux))
        
        # Общая валидность
        is_valid = all(validations.values())
        confidence = self._calculate_confidence(bls_result, validations)
        
        return {
            **bls_result,
            'validations': validations,
            'is_valid_candidate': is_valid,
            'confidence': confidence,
            'quality_score': self._calculate_quality_score(bls_result, validations)
        }
    
    def _additional_validations(
        self, 
        bls_result: Dict[str, any], 
        time: np.ndarray, 
        flux: np.ndarray
    ) -> Dict[str, bool]:
        """Дополнительные проверки качества"""
        
        period = bls_result['period']
        
        # Проверка на достаточное количество транзитов
        observation_span = time.max() - time.min()
        expected_transits = observation_span / period
        
        # Проверка на периодичность сигнала
        phase_coverage = self._check_phase_coverage(time, period)
        
        return {
            'sufficient_transits': expected_transits >= 2,
            'good_phase_coverage': phase_coverage > 0.5,
            'no_instrumental_period': not self._is_instrumental_period(period),
        }
    
    def _check_phase_coverage(self, time: np.ndarray, period: float) -> float:
        """Проверка покрытия фазы"""
        phases = (time % period) / period
        phase_bins = np.linspace(0, 1, 20)
        hist, _ = np.histogram(phases, bins=phase_bins)
        coverage = np.sum(hist > 0) / len(phase_bins)
        return coverage
    
    def _is_instrumental_period(self, period: float) -> bool:
        """Проверка на инструментальные периоды"""
        # Известные инструментальные периоды для TESS/Kepler
        instrumental_periods = [
            0.5,    # Полусуточный
            1.0,    # Суточный
            13.7,   # TESS орбитальный период
            27.4,   # Двойной TESS период
        ]
        
        tolerance = 0.1  # 10% толерантность
        
        for inst_period in instrumental_periods:
            if abs(period - inst_period) / inst_period < tolerance:
                return True
        
        return False
    
    def _calculate_confidence(
        self, 
        bls_result: Dict[str, any], 
        validations: Dict[str, bool]
    ) -> float:
        """Расчет уверенности в детекции"""
        
        # Базовая уверенность от SNR
        snr = bls_result.get('snr', 0)
        snr_confidence = min(snr / 10.0, 1.0)  # Нормализация к 1.0
        
        # Бонус за прохождение валидаций
        validation_score = sum(validations.values()) / len(validations)
        
        # Бонус за глубину транзита
        depth = bls_result.get('depth', 0)
        depth_confidence = min(depth / 0.01, 1.0)  # 1% глубина = максимум
        
        # Итоговая уверенность
        confidence = (snr_confidence * 0.5 + validation_score * 0.3 + depth_confidence * 0.2)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_quality_score(
        self, 
        bls_result: Dict[str, any], 
        validations: Dict[str, bool]
    ) -> float:
        """Расчет общего качества детекции"""
        
        power = bls_result.get('power', 0)
        snr = bls_result.get('snr', 0)
        validation_score = sum(validations.values()) / len(validations)
        
        # Комбинированный скор
        quality = (power * 0.4 + snr * 0.4 + validation_score * 0.2) / 3.0
        
        return float(np.clip(quality, 0.0, 1.0))
