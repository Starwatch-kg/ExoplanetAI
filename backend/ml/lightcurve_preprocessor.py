"""
Advanced Light Curve Preprocessing Module
Продвинутый модуль предобработки кривых блеска

Включает:
- Очистку данных по quality flags
- Sigma-clipping для удаления выбросов
- Сглаживание (Savitzky-Golay, медианная фильтрация)
- Wavelet denoising
- Нормализацию и центрирование
- Сегментацию для ML
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pywt
from astropy.stats import sigma_clip
try:
    from astropy.stats import median_abs_deviation
except ImportError:
    from scipy.stats import median_abs_deviation
from astropy.timeseries import LombScargle
import logging

logger = logging.getLogger(__name__)


class LightCurvePreprocessor:
    """
    Комплексный препроцессор для кривых блеска экзопланет
    """
    
    def __init__(self, 
                 sigma_clip_sigma: float = 3.0,
                 savgol_window: int = 21,
                 savgol_polyorder: int = 3,
                 median_kernel_size: int = 5,
                 wavelet_type: str = 'db4',
                 wavelet_levels: int = 6):
        """
        Инициализация препроцессора
        
        Args:
            sigma_clip_sigma: Порог для sigma-clipping
            savgol_window: Размер окна для Savitzky-Golay фильтра
            savgol_polyorder: Порядок полинома для Savitzky-Golay
            median_kernel_size: Размер ядра для медианной фильтрации
            wavelet_type: Тип вейвлета для denoising
            wavelet_levels: Количество уровней декомпозиции
        """
        self.sigma_clip_sigma = sigma_clip_sigma
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.median_kernel_size = median_kernel_size
        self.wavelet_type = wavelet_type
        self.wavelet_levels = wavelet_levels
        
        self.scaler = StandardScaler()
        self.flux_scaler = MinMaxScaler()
        
    def clean_by_quality_flags(self, 
                              time: np.ndarray, 
                              flux: np.ndarray, 
                              flux_err: np.ndarray,
                              quality: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Очистка данных по quality flags (TESS/Kepler)
        
        Args:
            time: Временные метки
            flux: Значения потока
            flux_err: Ошибки потока
            quality: Quality flags (если None, используем базовую очистку)
            
        Returns:
            Очищенные time, flux, flux_err
        """
        if quality is not None:
            # Стандартные bad quality flags для TESS/Kepler
            bad_flags = [
                1,    # Attitude tweak
                2,    # Safe mode
                4,    # Coarse point
                8,    # Earth point
                16,   # Zero crossing
                32,   # Desaturation event
                128,  # Manual exclude
                256,  # Discontinuity
                2048, # Impulsive outlier
                4096, # Argabrightening
            ]
            
            # Создаем маску хороших точек
            good_mask = np.ones(len(quality), dtype=bool)
            for flag in bad_flags:
                good_mask &= (quality & flag) == 0
        else:
            # Базовая очистка: удаляем NaN и бесконечности
            good_mask = (
                np.isfinite(time) & 
                np.isfinite(flux) & 
                np.isfinite(flux_err) &
                (flux_err > 0)
            )
        
        logger.info(f"Quality filtering: {np.sum(good_mask)}/{len(time)} points retained")
        
        return time[good_mask], flux[good_mask], flux_err[good_mask]
    
    def sigma_clipping(self, 
                      time: np.ndarray, 
                      flux: np.ndarray, 
                      flux_err: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sigma-clipping для удаления выбросов
        
        Args:
            time, flux, flux_err: Данные кривой блеска
            
        Returns:
            Очищенные данные
        """
        # Используем astropy sigma_clip для более точной работы
        clipped_flux = sigma_clip(flux, sigma=self.sigma_clip_sigma, maxiters=5)
        mask = ~clipped_flux.mask
        
        logger.info(f"Sigma clipping: {np.sum(mask)}/{len(flux)} points retained")
        
        return time[mask], flux[mask], flux_err[mask]
    
    def savitzky_golay_smooth(self, flux: np.ndarray) -> np.ndarray:
        """
        Сглаживание Savitzky-Golay фильтром
        
        Args:
            flux: Поток для сглаживания
            
        Returns:
            Сглаженный поток
        """
        if len(flux) < self.savgol_window:
            logger.warning("Flux array too short for Savitzky-Golay smoothing")
            return flux
            
        try:
            smoothed = signal.savgol_filter(
                flux, 
                window_length=self.savgol_window,
                polyorder=self.savgol_polyorder,
                mode='nearest'
            )
            return smoothed
        except Exception as e:
            logger.error(f"Savitzky-Golay smoothing failed: {e}")
            return flux
    
    def median_filter_smooth(self, flux: np.ndarray) -> np.ndarray:
        """
        Медианная фильтрация
        
        Args:
            flux: Поток для сглаживания
            
        Returns:
            Сглаженный поток
        """
        try:
            smoothed = signal.medfilt(flux, kernel_size=self.median_kernel_size)
            return smoothed
        except Exception as e:
            logger.error(f"Median filtering failed: {e}")
            return flux
    
    def wavelet_denoise(self, flux: np.ndarray, threshold_mode: str = 'soft') -> np.ndarray:
        """
        Wavelet denoising для слабых сигналов
        
        Args:
            flux: Поток для очистки
            threshold_mode: 'soft' или 'hard' thresholding
            
        Returns:
            Очищенный поток
        """
        try:
            # Wavelet декомпозиция
            coeffs = pywt.wavedec(flux, self.wavelet_type, level=self.wavelet_levels)
            
            # Автоматический расчет порога на основе MAD
            sigma = median_abs_deviation(flux, scale='normal')
            threshold = sigma * np.sqrt(2 * np.log(len(flux)))
            
            # Применяем thresholding к детализирующим коэффициентам
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [
                pywt.threshold(detail, threshold, mode=threshold_mode) 
                for detail in coeffs[1:]
            ]
            
            # Реконструкция сигнала
            denoised = pywt.waverec(coeffs_thresh, self.wavelet_type)
            
            # Обрезаем до исходной длины (может отличаться из-за padding)
            if len(denoised) != len(flux):
                denoised = denoised[:len(flux)]
                
            return denoised
            
        except Exception as e:
            logger.error(f"Wavelet denoising failed: {e}")
            return flux
    
    def normalize_flux(self, flux: np.ndarray, method: str = 'median') -> np.ndarray:
        """
        Нормализация потока
        
        Args:
            flux: Поток для нормализации
            method: 'median', 'mean', или 'robust'
            
        Returns:
            Нормализованный поток
        """
        if method == 'median':
            norm_factor = np.median(flux)
        elif method == 'mean':
            norm_factor = np.mean(flux)
        elif method == 'robust':
            # Используем 5-95 перцентили для робастной нормализации
            p5, p95 = np.percentile(flux, [5, 95])
            flux_clipped = flux[(flux >= p5) & (flux <= p95)]
            norm_factor = np.median(flux_clipped)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized = flux / norm_factor
        logger.info(f"Flux normalized by {method}: factor = {norm_factor:.6f}")
        
        return normalized
    
    def center_data(self, flux: np.ndarray) -> np.ndarray:
        """
        Центрирование данных (вычитание среднего)
        
        Args:
            flux: Поток для центрирования
            
        Returns:
            Центрированный поток
        """
        centered = flux - np.mean(flux)
        return centered
    
    def segment_around_transits(self, 
                               time: np.ndarray, 
                               flux: np.ndarray,
                               period: float,
                               epoch: float,
                               duration: float,
                               segment_length: int = 64,
                               n_transits: int = 5) -> List[np.ndarray]:
        """
        Сегментация кривой блеска вокруг транзитов для ML
        
        Args:
            time: Временные метки
            flux: Поток
            period: Орбитальный период
            epoch: Эпоха первого транзита
            duration: Длительность транзита
            segment_length: Длина сегмента в точках
            n_transits: Максимальное количество транзитов
            
        Returns:
            Список сегментов flux
        """
        segments = []
        half_segment = segment_length // 2
        
        # Находим времена транзитов
        transit_times = []
        t_start, t_end = time[0], time[-1]
        
        # Генерируем времена транзитов в диапазоне наблюдений
        n = 0
        while len(transit_times) < n_transits:
            transit_time = epoch + n * period
            if transit_time > t_end:
                break
            if transit_time >= t_start:
                transit_times.append(transit_time)
            n += 1
        
        logger.info(f"Found {len(transit_times)} transits for segmentation")
        
        for transit_time in transit_times:
            # Находим ближайший индекс к времени транзита
            transit_idx = np.argmin(np.abs(time - transit_time))
            
            # Проверяем, что у нас достаточно точек с обеих сторон
            start_idx = max(0, transit_idx - half_segment)
            end_idx = min(len(flux), transit_idx + half_segment)
            
            if end_idx - start_idx >= segment_length * 0.8:  # Минимум 80% от желаемой длины
                segment = flux[start_idx:end_idx]
                
                # Интерполируем до нужной длины если необходимо
                if len(segment) != segment_length:
                    x_old = np.linspace(0, 1, len(segment))
                    x_new = np.linspace(0, 1, segment_length)
                    segment = np.interp(x_new, x_old, segment)
                
                segments.append(segment)
        
        return segments
    
    def preprocess_lightcurve(self, 
                             time: np.ndarray,
                             flux: np.ndarray,
                             flux_err: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Основной метод предобработки кривой блеска
        
        Args:
            time: Временные данные
            flux: Данные потока
            flux_err: Ошибки потока
            
        Returns:
            Dict с обработанными данными
        """
        try:
            # Используем полный пайплайн предобработки
            processed_data = self.full_preprocessing_pipeline(
                time, flux, flux_err
            )
            processed_time = processed_data['time']
            processed_flux = processed_data['flux']
            processed_flux_err = processed_data['flux_err']
            
            return {
                'time': processed_time,
                'flux': processed_flux,
                'flux_err': processed_flux_err
            }
            
        except Exception as e:
            logger.error(f"Lightcurve preprocessing failed: {e}")
            # Возвращаем исходные данные при ошибке
            return {
                'time': time,
                'flux': flux,
                'flux_err': flux_err
            }
    
    def full_preprocessing_pipeline(self, 
                                   time: np.ndarray,
                                   flux: np.ndarray,
                                   flux_err: np.ndarray,
                                   quality: Optional[np.ndarray] = None,
                                   smooth_method: str = 'savgol',
                                   use_wavelet: bool = False,
                                   normalize_method: str = 'median') -> Dict[str, np.ndarray]:
        """
        Полный пайплайн предобработки
        
        Args:
            time, flux, flux_err: Исходные данные
            quality: Quality flags
            smooth_method: 'savgol', 'median', или 'none'
            use_wavelet: Использовать wavelet denoising
            normalize_method: Метод нормализации
            
        Returns:
            Словарь с предобработанными данными
        """
        logger.info("Starting full preprocessing pipeline")
        
        # 1. Очистка по quality flags
        time_clean, flux_clean, flux_err_clean = self.clean_by_quality_flags(
            time, flux, flux_err, quality
        )
        
        # 2. Sigma-clipping
        time_clean, flux_clean, flux_err_clean = self.sigma_clipping(
            time_clean, flux_clean, flux_err_clean
        )
        
        # 3. Сглаживание
        if smooth_method == 'savgol':
            flux_smooth = self.savitzky_golay_smooth(flux_clean)
        elif smooth_method == 'median':
            flux_smooth = self.median_filter_smooth(flux_clean)
        else:
            flux_smooth = flux_clean.copy()
        
        # 4. Wavelet denoising (опционально)
        if use_wavelet:
            flux_smooth = self.wavelet_denoise(flux_smooth)
        
        # 5. Нормализация
        flux_normalized = self.normalize_flux(flux_smooth, method=normalize_method)
        
        # 6. Центрирование
        flux_centered = self.center_data(flux_normalized)
        
        result = {
            'time': time_clean,
            'flux_raw': flux_clean,
            'flux_smooth': flux_smooth,
            'flux_normalized': flux_normalized,
            'flux_centered': flux_centered,
            'flux_err': flux_err_clean,
            'n_points_original': len(time),
            'n_points_final': len(time_clean),
            'data_quality': len(time_clean) / len(time)
        }
        
        logger.info(f"Preprocessing complete: {len(time)} -> {len(time_clean)} points")
        return result


class TransitSegmenter:
    """
    Специализированный класс для сегментации транзитов
    """
    
    def __init__(self, segment_length: int = 64):
        self.segment_length = segment_length
    
    def create_training_segments(self, 
                                lightcurves: List[Dict],
                                labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание сегментов для обучения ML моделей
        
        Args:
            lightcurves: Список предобработанных кривых блеска
            labels: Метки классов ('CANDIDATE', 'PC', 'FP')
            
        Returns:
            X: Массив сегментов (n_samples, segment_length)
            y: Массив меток
        """
        all_segments = []
        all_labels = []
        
        label_map = {'CANDIDATE': 0, 'PC': 1, 'FP': 2}
        
        for lc, label in zip(lightcurves, labels):
            if 'segments' in lc:
                segments = lc['segments']
                for segment in segments:
                    if len(segment) == self.segment_length:
                        all_segments.append(segment)
                        all_labels.append(label_map[label])
        
        X = np.array(all_segments)
        y = np.array(all_labels)
        
        logger.info(f"Created {len(X)} training segments")
        return X, y


# Утилитарные функции
def calculate_transit_metrics(time: np.ndarray, 
                             flux: np.ndarray, 
                             period: float,
                             epoch: float,
                             duration: float) -> Dict[str, float]:
    """
    Расчет базовых метрик транзита
    
    Args:
        time, flux: Данные кривой блеска
        period: Орбитальный период
        epoch: Эпоха транзита
        duration: Длительность транзита
        
    Returns:
        Словарь с метриками
    """
    # Фазовая кривая
    phase = ((time - epoch) % period) / period
    phase[phase > 0.5] -= 1.0
    
    # Находим точки в транзите
    in_transit = np.abs(phase) < (duration / period / 2)
    out_transit = np.abs(phase) > (duration / period)
    
    if np.sum(in_transit) == 0 or np.sum(out_transit) == 0:
        return {'depth': 0, 'snr': 0, 'duration_observed': 0}
    
    # Глубина транзита
    flux_in = np.median(flux[in_transit])
    flux_out = np.median(flux[out_transit])
    depth = (flux_out - flux_in) / flux_out
    
    # SNR
    noise = np.std(flux[out_transit])
    snr = depth / noise if noise > 0 else 0
    
    # Наблюдаемая длительность
    transit_phases = phase[in_transit]
    duration_observed = (np.max(transit_phases) - np.min(transit_phases)) * period
    
    return {
        'depth': depth,
        'snr': snr,
        'duration_observed': duration_observed,
        'n_points_in_transit': np.sum(in_transit),
        'n_points_out_transit': np.sum(out_transit)
    }
