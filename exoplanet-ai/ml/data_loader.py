"""
Enhanced ML Data Loader for Exoplanet Detection
Улучшенный загрузчик данных для ML моделей обнаружения экзопланет
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import asyncio
import aiohttp
from astropy.io import fits
from astropy.time import Time
import logging
from dataclasses import dataclass
from enum import Enum
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
import time

from ..core.logging_config import get_logger
from ..core.telemetry import trace_function, trace_nasa_api_call, add_span_attributes
from ..core.metrics import metrics_collector

logger = get_logger(__name__)

class DataSource(Enum):
    """Источники данных"""
    TESS = "tess"
    KEPLER = "kepler"
    K2 = "k2"
    SIMULATED = "simulated"

class ProcessingStep(Enum):
    """Этапы обработки данных"""
    RAW = "raw"
    NORMALIZED = "normalized"
    DETRENDED = "detrended"
    CLEANED = "cleaned"
    AUGMENTED = "augmented"

@dataclass
class LightcurveMetadata:
    """Метаданные кривой блеска"""
    target_id: str
    source: DataSource
    mission: str
    sector_quarter_campaign: Optional[int]
    cadence_minutes: float
    total_points: int
    time_span_days: float
    noise_level_ppm: float
    has_gaps: bool
    gap_fraction: float
    processing_steps: List[ProcessingStep]
    created_at: float
    
class LightcurveData:
    """Класс для хранения данных кривой блеска"""
    
    def __init__(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray] = None,
        metadata: Optional[LightcurveMetadata] = None
    ):
        self.time = np.asarray(time)
        self.flux = np.asarray(flux)
        self.flux_err = np.asarray(flux_err) if flux_err is not None else np.ones_like(flux) * 0.001
        self.metadata = metadata
        
        # Валидация данных
        self._validate_data()
    
    def _validate_data(self):
        """Валидация данных"""
        if len(self.time) != len(self.flux):
            raise ValueError("Time and flux arrays must have the same length")
        
        if len(self.flux_err) != len(self.flux):
            raise ValueError("Flux error array must have the same length as flux")
        
        if len(self.time) < 10:
            raise ValueError("Lightcurve must have at least 10 data points")
        
        # Проверяем на NaN и inf
        if np.any(~np.isfinite(self.time)):
            raise ValueError("Time array contains non-finite values")
        
        if np.any(~np.isfinite(self.flux)):
            logger.warning("Flux array contains non-finite values, will be cleaned")
        
    def copy(self) -> 'LightcurveData':
        """Создание копии"""
        return LightcurveData(
            time=self.time.copy(),
            flux=self.flux.copy(),
            flux_err=self.flux_err.copy(),
            metadata=self.metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'time': self.time.tolist(),
            'flux': self.flux.tolist(),
            'flux_err': self.flux_err.tolist(),
            'metadata': self.metadata.__dict__ if self.metadata else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LightcurveData':
        """Создание из словаря"""
        metadata = None
        if data.get('metadata'):
            metadata = LightcurveMetadata(**data['metadata'])
        
        return cls(
            time=np.array(data['time']),
            flux=np.array(data['flux']),
            flux_err=np.array(data['flux_err']),
            metadata=metadata
        )

class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger("ml.preprocessor")
    
    @trace_function("preprocess_lightcurve")
    def preprocess(
        self, 
        lightcurve: LightcurveData,
        target_length: int = 1024,
        normalize_method: str = "median",
        remove_outliers: bool = True,
        detrend: bool = True,
        fill_gaps: bool = True
    ) -> LightcurveData:
        """
        Полная предобработка кривой блеска
        
        Args:
            lightcurve: Исходная кривая блеска
            target_length: Целевая длина после обработки
            normalize_method: Метод нормализации (median, mean, minmax)
            remove_outliers: Удалять выбросы
            detrend: Удалять тренд
            fill_gaps: Заполнять пропуски
        
        Returns:
            Обработанная кривая блеска
        """
        processed = lightcurve.copy()
        steps = [ProcessingStep.RAW]
        
        self.logger.info(
            f"Starting preprocessing: {len(processed.flux)} points",
            extra={
                "target_length": target_length,
                "normalize_method": normalize_method,
                "remove_outliers": remove_outliers,
                "detrend": detrend
            }
        )
        
        # 1. Очистка от NaN и inf
        processed = self._clean_invalid_values(processed)
        
        # 2. Удаление выбросов
        if remove_outliers:
            processed = self._remove_outliers(processed)
            steps.append(ProcessingStep.CLEANED)
        
        # 3. Заполнение пропусков
        if fill_gaps:
            processed = self._fill_gaps(processed)
        
        # 4. Детрендинг
        if detrend:
            processed = self._detrend(processed)
            steps.append(ProcessingStep.DETRENDED)
        
        # 5. Нормализация
        processed = self._normalize(processed, method=normalize_method)
        steps.append(ProcessingStep.NORMALIZED)
        
        # 6. Ресемплинг до целевой длины
        if len(processed.flux) != target_length:
            processed = self._resample(processed, target_length)
        
        # Обновляем метаданные
        if processed.metadata:
            processed.metadata.processing_steps = steps
            processed.metadata.total_points = len(processed.flux)
        
        self.logger.info(
            f"Preprocessing completed: {len(processed.flux)} points",
            extra={"processing_steps": [step.value for step in steps]}
        )
        
        return processed
    
    def _clean_invalid_values(self, lightcurve: LightcurveData) -> LightcurveData:
        """Очистка от NaN и inf значений"""
        valid_mask = (
            np.isfinite(lightcurve.time) & 
            np.isfinite(lightcurve.flux) & 
            np.isfinite(lightcurve.flux_err)
        )
        
        if not np.all(valid_mask):
            invalid_count = np.sum(~valid_mask)
            self.logger.warning(f"Removing {invalid_count} invalid data points")
            
            return LightcurveData(
                time=lightcurve.time[valid_mask],
                flux=lightcurve.flux[valid_mask],
                flux_err=lightcurve.flux_err[valid_mask],
                metadata=lightcurve.metadata
            )
        
        return lightcurve
    
    def _remove_outliers(
        self, 
        lightcurve: LightcurveData, 
        sigma_threshold: float = 5.0
    ) -> LightcurveData:
        """Удаление выбросов методом sigma clipping"""
        flux = lightcurve.flux.copy()
        
        # Итеративное удаление выбросов
        for _ in range(3):  # Максимум 3 итерации
            median_flux = np.median(flux)
            mad = np.median(np.abs(flux - median_flux))
            sigma = 1.4826 * mad  # Robust sigma estimate
            
            outlier_mask = np.abs(flux - median_flux) > sigma_threshold * sigma
            
            if not np.any(outlier_mask):
                break
            
            # Заменяем выбросы медианными значениями
            flux[outlier_mask] = median_flux
        
        outliers_removed = np.sum(flux != lightcurve.flux)
        if outliers_removed > 0:
            self.logger.info(f"Removed/corrected {outliers_removed} outliers")
        
        return LightcurveData(
            time=lightcurve.time,
            flux=flux,
            flux_err=lightcurve.flux_err,
            metadata=lightcurve.metadata
        )
    
    def _fill_gaps(self, lightcurve: LightcurveData) -> LightcurveData:
        """Заполнение пропусков в данных"""
        time = lightcurve.time
        flux = lightcurve.flux
        
        # Определяем пропуски по времени
        dt = np.median(np.diff(time))
        gap_threshold = 3 * dt
        
        gaps = np.diff(time) > gap_threshold
        if not np.any(gaps):
            return lightcurve  # Пропусков нет
        
        # Интерполяция для заполнения небольших пропусков
        from scipy.interpolate import interp1d
        
        # Создаем равномерную временную сетку
        time_uniform = np.arange(time[0], time[-1], dt)
        
        # Интерполируем только для небольших пропусков
        f_interp = interp1d(
            time, flux, 
            kind='linear', 
            bounds_error=False, 
            fill_value=np.median(flux)
        )
        
        flux_uniform = f_interp(time_uniform)
        flux_err_uniform = np.full_like(flux_uniform, np.median(lightcurve.flux_err))
        
        self.logger.info(f"Filled gaps: {len(time_uniform)} uniform points")
        
        return LightcurveData(
            time=time_uniform,
            flux=flux_uniform,
            flux_err=flux_err_uniform,
            metadata=lightcurve.metadata
        )
    
    def _detrend(self, lightcurve: LightcurveData) -> LightcurveData:
        """Удаление долгосрочного тренда"""
        from scipy.signal import savgol_filter
        
        flux = lightcurve.flux.copy()
        
        # Определяем окно для фильтра Савицкого-Голея
        window_length = min(len(flux) // 10, 101)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 5:
            # Применяем фильтр для получения тренда
            trend = savgol_filter(flux, window_length, polyorder=2)
            
            # Удаляем тренд
            detrended_flux = flux - trend + np.median(flux)
            
            self.logger.info("Applied Savitzky-Golay detrending")
            
            return LightcurveData(
                time=lightcurve.time,
                flux=detrended_flux,
                flux_err=lightcurve.flux_err,
                metadata=lightcurve.metadata
            )
        
        return lightcurve
    
    def _normalize(
        self, 
        lightcurve: LightcurveData, 
        method: str = "median"
    ) -> LightcurveData:
        """Нормализация потока"""
        flux = lightcurve.flux.copy()
        
        if method == "median":
            norm_factor = np.median(flux)
        elif method == "mean":
            norm_factor = np.mean(flux)
        elif method == "minmax":
            flux_min, flux_max = np.min(flux), np.max(flux)
            flux = (flux - flux_min) / (flux_max - flux_min)
            norm_factor = 1.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if method != "minmax":
            flux = flux / norm_factor
        
        # Нормализуем ошибки тоже
        flux_err = lightcurve.flux_err / norm_factor
        
        self.logger.info(f"Applied {method} normalization")
        
        return LightcurveData(
            time=lightcurve.time,
            flux=flux,
            flux_err=flux_err,
            metadata=lightcurve.metadata
        )
    
    def _resample(
        self, 
        lightcurve: LightcurveData, 
        target_length: int
    ) -> LightcurveData:
        """Ресемплинг до целевой длины"""
        current_length = len(lightcurve.flux)
        
        if current_length == target_length:
            return lightcurve
        
        # Создаем индексы для интерполяции
        old_indices = np.linspace(0, current_length - 1, current_length)
        new_indices = np.linspace(0, current_length - 1, target_length)
        
        # Интерполируем
        new_time = np.interp(new_indices, old_indices, lightcurve.time)
        new_flux = np.interp(new_indices, old_indices, lightcurve.flux)
        new_flux_err = np.interp(new_indices, old_indices, lightcurve.flux_err)
        
        self.logger.info(f"Resampled from {current_length} to {target_length} points")
        
        return LightcurveData(
            time=new_time,
            flux=new_flux,
            flux_err=new_flux_err,
            metadata=lightcurve.metadata
        )

class DataAugmentor:
    """Класс для аугментации данных"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger("ml.augmentor")
    
    def augment_dataset(
        self,
        lightcurves: List[LightcurveData],
        augmentation_factor: int = 2
    ) -> List[LightcurveData]:
        """Аугментация датасета"""
        augmented = []
        
        for lc in lightcurves:
            augmented.append(lc)  # Оригинал
            
            # Генерируем аугментированные версии
            for i in range(augmentation_factor - 1):
                aug_lc = self._augment_single(lc, seed=i)
                augmented.append(aug_lc)
        
        self.logger.info(
            f"Dataset augmented: {len(lightcurves)} -> {len(augmented)} samples"
        )
        
        return augmented
    
    def _augment_single(
        self, 
        lightcurve: LightcurveData, 
        seed: int = None
    ) -> LightcurveData:
        """Аугментация одной кривой блеска"""
        if seed is not None:
            np.random.seed(seed)
        
        flux = lightcurve.flux.copy()
        
        # 1. Добавление шума
        noise_level = np.std(flux) * 0.1
        flux += np.random.normal(0, noise_level, len(flux))
        
        # 2. Масштабирование
        scale_factor = np.random.uniform(0.95, 1.05)
        flux *= scale_factor
        
        # 3. Сдвиг по времени (циклический)
        shift = np.random.randint(0, len(flux))
        flux = np.roll(flux, shift)
        
        # 4. Добавление систематических эффектов
        if np.random.random() < 0.3:  # 30% вероятность
            # Линейный тренд
            trend = np.linspace(-0.001, 0.001, len(flux))
            flux += trend
        
        return LightcurveData(
            time=lightcurve.time,
            flux=flux,
            flux_err=lightcurve.flux_err,
            metadata=lightcurve.metadata
        )

class MLDataLoader:
    """Основной класс загрузчика данных для ML"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.preprocessor = DataPreprocessor(config)
        self.augmentor = DataAugmentor(config)
        self.logger = get_logger("ml.data_loader")
        
        # Кэш для загруженных данных
        self._cache = {}
        self._cache_size_limit = self.config.get("cache_size", 1000)
    
    @trace_function("load_lightcurve_from_nasa")
    async def load_lightcurve_from_nasa(
        self,
        target_id: str,
        mission: str = "TESS",
        sector: Optional[int] = None
    ) -> Optional[LightcurveData]:
        """
        Загрузка кривой блеска из NASA MAST
        
        Args:
            target_id: ID цели
            mission: Миссия (TESS, Kepler, K2)
            sector: Сектор/квартал/кампания
        
        Returns:
            Данные кривой блеска или None
        """
        cache_key = f"{mission}_{target_id}_{sector}"
        
        # Проверяем кэш
        if cache_key in self._cache:
            self.logger.info(f"Loading from cache: {cache_key}")
            metrics_collector.record_cache_operation("get", "hit")
            return self._cache[cache_key]
        
        metrics_collector.record_cache_operation("get", "miss")
        
        try:
            with trace_nasa_api_call("lightcurve", target_id):
                # Здесь должна быть реальная загрузка из NASA MAST
                # Пока используем симуляцию
                lightcurve = await self._simulate_nasa_lightcurve(
                    target_id, mission, sector
                )
                
                # Сохраняем в кэш
                self._add_to_cache(cache_key, lightcurve)
                
                return lightcurve
                
        except Exception as e:
            self.logger.error(f"Failed to load lightcurve: {e}")
            metrics_collector.record_error("nasa_api_error", "data_loader")
            return None
    
    async def _simulate_nasa_lightcurve(
        self,
        target_id: str,
        mission: str,
        sector: Optional[int]
    ) -> LightcurveData:
        """Симуляция загрузки из NASA (для демонстрации)"""
        
        # Параметры для разных миссий
        mission_params = {
            "TESS": {"duration": 27, "cadence": 2, "points": 13000, "noise": 100e-6},
            "Kepler": {"duration": 90, "cadence": 30, "points": 4320, "noise": 50e-6},
            "K2": {"duration": 80, "cadence": 30, "points": 3840, "noise": 80e-6}
        }
        
        params = mission_params.get(mission, mission_params["TESS"])
        
        # Генерируем данные
        np.random.seed(hash(target_id) % 2**32)
        
        time = np.linspace(0, params["duration"], params["points"])
        flux = np.ones(params["points"])
        
        # Добавляем реалистичный шум
        flux += np.random.normal(0, params["noise"], params["points"])
        
        # Добавляем звездную активность
        rotation_period = np.random.uniform(10, 30)
        flux += 0.001 * np.sin(2 * np.pi * time / rotation_period)
        
        # Ошибки потока
        flux_err = np.full(params["points"], params["noise"])
        
        # Метаданные
        metadata = LightcurveMetadata(
            target_id=target_id,
            source=DataSource(mission.lower()),
            mission=mission,
            sector_quarter_campaign=sector,
            cadence_minutes=params["cadence"],
            total_points=params["points"],
            time_span_days=params["duration"],
            noise_level_ppm=params["noise"] * 1e6,
            has_gaps=False,
            gap_fraction=0.0,
            processing_steps=[ProcessingStep.RAW],
            created_at=time.time()
        )
        
        # Симулируем задержку API
        await asyncio.sleep(0.1)
        
        return LightcurveData(time, flux, flux_err, metadata)
    
    def _add_to_cache(self, key: str, data: LightcurveData):
        """Добавление в кэш с ограничением размера"""
        if len(self._cache) >= self._cache_size_limit:
            # Удаляем самый старый элемент (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = data
    
    @trace_function("prepare_training_data")
    def prepare_training_data(
        self,
        lightcurves: List[LightcurveData],
        labels: List[int],
        target_length: int = 1024,
        validation_split: float = 0.2,
        augment: bool = True,
        augmentation_factor: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Подготовка данных для обучения
        
        Args:
            lightcurves: Список кривых блеска
            labels: Метки классов (0 - нет планеты, 1 - есть планета)
            target_length: Целевая длина последовательности
            validation_split: Доля валидационных данных
            augment: Применять аугментацию
            augmentation_factor: Фактор аугментации
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        self.logger.info(
            f"Preparing training data: {len(lightcurves)} samples",
            extra={
                "target_length": target_length,
                "validation_split": validation_split,
                "augment": augment
            }
        )
        
        # Предобработка
        processed_lightcurves = []
        for lc in lightcurves:
            try:
                processed = self.preprocessor.preprocess(
                    lc, target_length=target_length
                )
                processed_lightcurves.append(processed)
            except Exception as e:
                self.logger.warning(f"Failed to preprocess lightcurve: {e}")
                continue
        
        # Аугментация
        if augment and len(processed_lightcurves) > 0:
            augmented_lightcurves = self.augmentor.augment_dataset(
                processed_lightcurves, augmentation_factor
            )
            
            # Расширяем метки
            augmented_labels = []
            for i, label in enumerate(labels[:len(processed_lightcurves)]):
                for _ in range(augmentation_factor):
                    augmented_labels.append(label)
            
            processed_lightcurves = augmented_lightcurves
            labels = augmented_labels
        
        # Конвертируем в numpy массивы
        X = np.array([lc.flux for lc in processed_lightcurves])
        y = np.array(labels[:len(X)])
        
        # Разделение на train/validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        
        # Перемешиваем данные
        indices = np.random.permutation(n_samples)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        self.logger.info(
            f"Training data prepared: train={len(X_train)}, val={len(X_val)}",
            extra={
                "train_positive": np.sum(y_train),
                "val_positive": np.sum(y_val)
            }
        )
        
        return X_train, X_val, y_train, y_val
    
    def save_processed_data(
        self, 
        data: Dict[str, Any], 
        filepath: Union[str, Path]
    ):
        """Сохранение обработанных данных"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Загрузка обработанных данных"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.logger.info(f"Processed data loaded from {filepath}")
        return data

# Глобальный экземпляр
ml_data_loader = MLDataLoader()
