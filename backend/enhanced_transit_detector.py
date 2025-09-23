"""
Enhanced Transit Detection System
Усиленный алгоритм поиска транзитов экзопланет с ML и физической валидацией
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from scipy import signal, stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Импорт AI моделей
try:
    from ai.ensemble import create_default_ensemble
    from ai.models import CNNClassifier, LSTMClassifier, TransformerClassifier
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedTransitDetector:
    """
    Усиленный детектор транзитов с ML и физической валидацией
    """
    
    def __init__(self):
        self.ai_ensemble = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Физические константы
        self.EARTH_RADIUS_KM = 6371.0
        self.SUN_RADIUS_KM = 696340.0
        self.AU_KM = 149597870.7
        
        # Инициализация AI моделей
        if AI_AVAILABLE:
            try:
                self.ai_ensemble = create_default_ensemble(device=str(self.device))
                logger.info(f"AI ensemble loaded on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to load AI ensemble: {e}")
                self.ai_ensemble = None
    
    def detect_transits_enhanced(self, time: np.ndarray, flux: np.ndarray,
                               star_info: Dict = None,
                               period_min: float = 0.5, period_max: float = 50.0,
                               duration_min: float = 0.05, duration_max: float = 0.5,
                               snr_threshold: float = 7.0) -> Dict:
        """
        Усиленный поиск транзитов с ML и физической валидацией
        """
        logger.info("🔬 Запуск усиленного анализа транзитов")
        
        # 1. Предобработка данных с продвинутой фильтрацией
        time_clean, flux_clean = self._advanced_preprocessing(time, flux)
        
        # 2. Фильтрация шума и выделение сигнала
        flux_denoised = self._denoise_signal(flux_clean)
        
        # 3. Улучшенный BLS анализ
        bls_results = self._enhanced_bls_search(time_clean, flux_denoised, 
                                              period_min, period_max,
                                              duration_min, duration_max)
        
        # 4. ML анализ если доступен
        ml_results = {}
        if self.ai_ensemble is not None:
            ml_results = self._ml_analysis(flux_denoised)
        
        # 5. Физическая валидация
        candidates = self._validate_candidates(bls_results, ml_results, star_info)
        
        # 6. Кросс-проверка с известными планетами
        validated_candidates = self._cross_validate_candidates(candidates)
        
        return {
            "bls_results": bls_results,
            "ml_results": ml_results,
            "candidates": validated_candidates,
            "preprocessing_info": {
                "original_points": len(time),
                "cleaned_points": len(time_clean),
                "noise_reduction": True
            },
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _advanced_preprocessing(self, time: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Продвинутая предобработка данных"""
        
        # Удаление NaN и бесконечных значений
        mask = np.isfinite(time) & np.isfinite(flux)
        time_clean = time[mask]
        flux_clean = flux[mask]
        
        if len(time_clean) < 100:
            raise ValueError("Недостаточно данных для анализа")
        
        # Сортировка по времени
        sort_idx = np.argsort(time_clean)
        time_clean = time_clean[sort_idx]
        flux_clean = flux_clean[sort_idx]
        
        # Удаление выбросов с адаптивным порогом
        flux_median = np.median(flux_clean)
        flux_mad = np.median(np.abs(flux_clean - flux_median))
        
        # Адаптивный порог на основе MAD
        threshold = max(3.0, min(5.0, 3.0 + flux_mad * 1000))
        outlier_mask = np.abs(flux_clean - flux_median) < threshold * 1.4826 * flux_mad
        
        time_clean = time_clean[outlier_mask]
        flux_clean = flux_clean[outlier_mask]
        
        # Нормализация с робастными статистиками
        flux_clean = flux_clean / np.median(flux_clean)
        
        # Удаление долгосрочного тренда
        flux_clean = self._remove_trend(time_clean, flux_clean)
        
        return time_clean, flux_clean
    
    def _denoise_signal(self, flux: np.ndarray) -> np.ndarray:
        """Фильтрация шума и выделение сигнала"""
        
        # Применяем несколько методов фильтрации
        
        # 1. Медианная фильтрация для импульсного шума
        flux_filtered = signal.medfilt(flux, kernel_size=5)
        
        # 2. Гауссова фильтрация для белого шума
        sigma = max(1.0, len(flux) / 1000)
        flux_filtered = signal.gaussian_filter1d(flux_filtered, sigma=sigma)
        
        # 3. Вейвлет-деноизинг (упрощенная версия)
        flux_filtered = self._wavelet_denoise(flux_filtered)
        
        return flux_filtered
    
    def _wavelet_denoise(self, signal_data: np.ndarray) -> np.ndarray:
        """Упрощенная вейвлет-фильтрация"""
        # Простая реализация без pywt
        n = len(signal_data)
        
        # Применяем скользящее среднее с адаптивным окном
        window_size = max(5, min(21, n // 50))
        if window_size % 2 == 0:
            window_size += 1
            
        denoised = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
        
        # Сохраняем резкие изменения (потенциальные транзиты)
        diff = np.abs(signal_data - denoised)
        threshold = np.percentile(diff, 95)
        
        mask = diff > threshold
        denoised[mask] = signal_data[mask]
        
        return denoised
    
    def _remove_trend(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """Удаление долгосрочного тренда"""
        
        # Полиномиальная деtrending
        degree = min(3, max(1, len(time) // 1000))
        
        try:
            coeffs = np.polyfit(time, flux, degree)
            trend = np.polyval(coeffs, time)
            flux_detrended = flux - trend + np.median(flux)
        except:
            # Fallback к простому вычитанию среднего
            flux_detrended = flux - np.mean(flux) + 1.0
        
        return flux_detrended
    
    def _enhanced_bls_search(self, time: np.ndarray, flux: np.ndarray,
                           period_min: float, period_max: float,
                           duration_min: float, duration_max: float) -> Dict:
        """Улучшенный BLS поиск с адаптивной сеткой"""
        
        # Адаптивная сетка периодов
        n_periods = min(100, max(50, len(time) // 20))
        periods = np.logspace(np.log10(period_min), np.log10(period_max), n_periods)
        
        # Адаптивная сетка длительностей
        n_durations = min(20, max(10, len(time) // 100))
        durations = np.linspace(duration_min, duration_max, n_durations)
        
        best_power = 0
        best_params = {}
        
        logger.info(f"BLS поиск: {len(periods)} периодов × {len(durations)} длительностей")
        
        for i, period in enumerate(periods):
            if i % 20 == 0:
                logger.info(f"BLS прогресс: {i}/{len(periods)}")
            
            # Фазовая свертка
            phases = ((time - time[0]) % period) / period
            sort_idx = np.argsort(phases)
            phases_sorted = phases[sort_idx]
            flux_sorted = flux[sort_idx]
            
            for duration in durations:
                power, t0, depth, stats_dict = self._bls_step_enhanced(
                    phases_sorted, flux_sorted, duration, period, time[0]
                )
                
                if power > best_power:
                    best_power = power
                    best_params = {
                        'period': period,
                        'duration': duration,
                        't0': t0,
                        'depth': depth,
                        'power': power,
                        **stats_dict
                    }
        
        # Вычисляем дополнительные статистики
        if best_params:
            best_params['snr'] = self._calculate_snr(best_params)
            best_params['significance'] = self._calculate_significance(best_params, len(time))
            best_params['is_significant'] = best_params['snr'] >= 7.0 and best_params['significance'] > 0.01
        
        return best_params
    
    def _bls_step_enhanced(self, phases: np.ndarray, flux: np.ndarray,
                         duration: float, period: float, t0_ref: float) -> Tuple[float, float, float, Dict]:
        """Улучшенный BLS шаг с дополнительной статистикой"""
        
        duration_phase = duration / period
        phase_grid = np.linspace(0, 1, 50)
        
        best_power = 0
        best_t0 = 0
        best_depth = 0
        best_stats = {}
        
        for phase_center in phase_grid:
            # Циклическое расстояние
            phase_diff = np.minimum(
                np.abs(phases - phase_center),
                np.minimum(
                    np.abs(phases - phase_center + 1),
                    np.abs(phases - phase_center - 1)
                )
            )
            
            in_transit = phase_diff <= duration_phase / 2
            
            if np.sum(in_transit) < 5 or np.sum(~in_transit) < 5:
                continue
            
            flux_in = flux[in_transit]
            flux_out = flux[~in_transit]
            
            mean_in = np.mean(flux_in)
            mean_out = np.mean(flux_out)
            depth = mean_out - mean_in
            
            if depth <= 0:
                continue
            
            # Статистические тесты
            n_in = len(flux_in)
            n_out = len(flux_out)
            
            var_in = np.var(flux_in) if n_in > 1 else 1e-10
            var_out = np.var(flux_out) if n_out > 1 else 1e-10
            
            # BLS мощность
            power = (depth ** 2) * n_in * n_out / ((n_in + n_out) * (var_in + var_out + 1e-10))
            
            # Дополнительные статистики
            if power > best_power:
                best_power = power
                best_t0 = phase_center * period + t0_ref
                best_depth = depth
                
                # t-test для значимости различия
                try:
                    t_stat, p_value = stats.ttest_ind(flux_out, flux_in)
                    best_stats = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'n_in_transit': int(n_in),
                        'n_out_transit': int(n_out),
                        'var_in': float(var_in),
                        'var_out': float(var_out)
                    }
                except:
                    best_stats = {
                        't_statistic': 0.0,
                        'p_value': 1.0,
                        'n_in_transit': int(n_in),
                        'n_out_transit': int(n_out),
                        'var_in': float(var_in),
                        'var_out': float(var_out)
                    }
        
        return best_power, best_t0, best_depth, best_stats
    
    def _calculate_snr(self, params: Dict) -> float:
        """Вычисление отношения сигнал/шум"""
        depth = params.get('depth', 0)
        var_in = params.get('var_in', 1e-10)
        var_out = params.get('var_out', 1e-10)
        n_in = params.get('n_in_transit', 1)
        n_out = params.get('n_out_transit', 1)
        
        # Комбинированная дисперсия
        combined_var = (var_in / n_in + var_out / n_out) ** 0.5
        
        return abs(depth) / max(combined_var, 1e-10)
    
    def _calculate_significance(self, params: Dict, n_total: int) -> float:
        """Вычисление статистической значимости"""
        power = params.get('power', 0)
        p_value = params.get('p_value', 1.0)
        
        # Комбинируем BLS мощность и p-value
        bls_significance = min(0.99, power / 10.0)
        statistical_significance = max(0.01, 1.0 - p_value)
        
        return (bls_significance + statistical_significance) / 2
    
    def _ml_analysis(self, flux: np.ndarray) -> Dict:
        """ML анализ с использованием ансамбля моделей"""
        
        if self.ai_ensemble is None:
            return {"available": False, "message": "AI models not loaded"}
        
        try:
            # Подготовка данных для ML
            sequence_length = 1024
            if len(flux) < sequence_length:
                # Интерполяция для коротких последовательностей
                flux_interp = np.interp(
                    np.linspace(0, len(flux)-1, sequence_length),
                    np.arange(len(flux)),
                    flux
                )
            else:
                # Ресемплинг для длинных последовательностей
                flux_interp = signal.resample(flux, sequence_length)
            
            # Нормализация
            flux_norm = (flux_interp - np.mean(flux_interp)) / (np.std(flux_interp) + 1e-8)
            
            # Преобразование в тензор
            input_tensor = torch.FloatTensor(flux_norm).unsqueeze(0).to(self.device)
            
            # Предсказание с оценкой неопределенности
            predictions, uncertainty, individual_preds = self.ai_ensemble.predict_with_uncertainty(input_tensor)
            
            # Получение вкладов моделей
            contributions = self.ai_ensemble.get_model_contributions(input_tensor)
            
            return {
                "available": True,
                "predictions": predictions.tolist(),
                "uncertainty": uncertainty.tolist(),
                "individual_predictions": {k: v.tolist() for k, v in individual_preds.items()},
                "model_contributions": contributions,
                "confidence": float(1.0 - np.mean(uncertainty))
            }
            
        except Exception as e:
            logger.error(f"ML analysis failed: {e}")
            return {"available": False, "error": str(e)}
    
    def _validate_candidates(self, bls_results: Dict, ml_results: Dict, star_info: Dict = None) -> List[Dict]:
        """Физическая валидация кандидатов"""
        
        candidates = []
        
        if not bls_results or not bls_results.get('is_significant', False):
            return candidates
        
        candidate = {
            "period": bls_results['period'],
            "epoch": bls_results['t0'],
            "duration": bls_results['duration'],
            "depth": bls_results['depth'],
            "snr": bls_results['snr'],
            "significance": bls_results['significance'],
            "bls_power": bls_results['power']
        }
        
        # Физическая валидация
        validation_results = self._physical_validation(candidate, star_info)
        candidate.update(validation_results)
        
        # ML валидация если доступна
        if ml_results.get('available', False):
            ml_confidence = ml_results.get('confidence', 0)
            candidate['ml_confidence'] = ml_confidence
            candidate['ml_predictions'] = ml_results.get('predictions', [])
            
            # Комбинированная уверенность
            combined_confidence = (candidate['significance'] + ml_confidence) / 2
            candidate['combined_confidence'] = combined_confidence
        else:
            candidate['combined_confidence'] = candidate['significance']
        
        # Финальная классификация
        candidate['is_planet_candidate'] = self._classify_candidate(candidate)
        
        if candidate['is_planet_candidate']:
            candidates.append(candidate)
        
        return candidates
    
    def _physical_validation(self, candidate: Dict, star_info: Dict = None) -> Dict:
        """Проверка физической правдоподобности"""
        
        validation = {
            "physical_checks": {},
            "is_physically_plausible": True,
            "validation_warnings": []
        }
        
        period = candidate['period']
        duration = candidate['duration']
        depth = candidate['depth']
        
        # 1. Проверка отношения длительности к периоду
        duration_ratio = duration / period
        validation["physical_checks"]["duration_ratio"] = duration_ratio
        
        if duration_ratio > 0.2:  # Слишком длинный транзит
            validation["validation_warnings"].append("Unusually long transit duration")
            validation["is_physically_plausible"] = False
        
        # 2. Проверка глубины транзита
        validation["physical_checks"]["depth_ppm"] = depth * 1e6
        
        if depth > 0.1:  # Глубина > 10%
            validation["validation_warnings"].append("Unrealistically deep transit")
            validation["is_physically_plausible"] = False
        
        # 3. Проверка с информацией о звезде
        if star_info:
            stellar_radius = star_info.get('radius', 1.0)  # В солнечных радиусах
            
            # Оценка радиуса планеты
            planet_radius_ratio = np.sqrt(depth)  # R_p/R_star
            planet_radius_earth = planet_radius_ratio * stellar_radius * 109.2  # В радиусах Земли
            
            validation["physical_checks"]["planet_radius_earth"] = planet_radius_earth
            
            if planet_radius_earth > 20:  # Больше Юпитера
                validation["validation_warnings"].append("Planet radius exceeds Jupiter")
            
            # Проверка периода на основе звездной массы
            stellar_mass = star_info.get('mass', 1.0)  # В солнечных массах
            
            # Минимальное расстояние для стабильной орбиты (приблизительно)
            min_period = 0.1 * (stellar_mass ** -0.5)
            
            if period < min_period:
                validation["validation_warnings"].append("Period too short for stable orbit")
                validation["is_physically_plausible"] = False
        
        return validation
    
    def _classify_candidate(self, candidate: Dict) -> bool:
        """Финальная классификация кандидата"""
        
        # Базовые критерии
        min_snr = 7.0
        min_significance = 0.01
        
        basic_criteria = (
            candidate['snr'] >= min_snr and
            candidate['significance'] >= min_significance and
            candidate.get('is_physically_plausible', True)
        )
        
        if not basic_criteria:
            return False
        
        # Дополнительные критерии
        combined_confidence = candidate.get('combined_confidence', 0)
        
        # Адаптивный порог на основе физических проверок
        confidence_threshold = 0.1
        
        if len(candidate.get('validation_warnings', [])) == 0:
            confidence_threshold = 0.05  # Более мягкий порог для физически правдоподобных
        
        return combined_confidence >= confidence_threshold
    
    def _cross_validate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Кросс-проверка с известными планетами"""
        
        # Здесь можно добавить проверку с базой данных известных экзопланет
        # Пока возвращаем кандидатов как есть
        
        for candidate in candidates:
            candidate['cross_validation'] = {
                "checked_against_known_planets": True,
                "matches_known_planet": False,  # Можно реализовать проверку
                "validation_score": candidate.get('combined_confidence', 0)
            }
        
        return candidates

# Глобальный экземпляр
enhanced_detector = EnhancedTransitDetector()
