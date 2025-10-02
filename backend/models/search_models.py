"""
Search Models for Exoplanet Detection
Модели данных для поиска экзопланет
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass
import numpy as np


class LightCurveData(BaseModel):
    """Данные кривой блеска"""
    target_name: str = Field(..., min_length=1, max_length=100, description="Название цели")
    time: List[float] = Field(..., description="Временные метки")
    flux: List[float] = Field(..., description="Значения потока")
    flux_err: Optional[List[float]] = Field(None, description="Ошибки потока")
    quality: Optional[List[int]] = Field(None, description="Качество данных")
    mission: str = Field("TESS", pattern="^(TESS|Kepler|K2)$", description="Миссия")
    sector: Optional[int] = Field(None, description="Сектор/квартал/кампания")
    quarter: Optional[int] = Field(None, description="Квартал (Kepler/K2)")
    campaign: Optional[int] = Field(None, description="Кампания (K2)")
    duration_days: float = Field(0.0, ge=0.0, description="Продолжительность наблюдений (дни)")
    cadence: float = Field(2.0, ge=0.1, le=180.0, description="Каденс (минуты)")
    data_points: int = Field(0, ge=0, description="Количество точек данных")


class TargetInfo(BaseModel):
    """Информация о цели"""
    target_name: str = Field(..., min_length=1, max_length=100, description="Название цели")
    catalog_id: str = Field(..., min_length=1, max_length=50, description="ID в каталоге")
    coordinates: Dict[str, float] = Field(..., description="Координаты {'ra': float, 'dec': float}")
    magnitude: float = Field(..., ge=0.0, le=30.0, description="Звездная величина")
    available_data: List[str] = Field([], description="Доступные данные ['TESS', 'Kepler', ...]")


class TransitCandidate(BaseModel):
    """Кандидат в транзиты"""
    period: float = Field(..., gt=0, description="Период (дни)")
    t0: float = Field(..., description="Время центра транзита (BJD)")
    duration: float = Field(..., gt=0, description="Длительность транзита (часы)")
    depth: float = Field(..., gt=0, description="Глубина транзита (ppm)")
    snr: float = Field(..., ge=0, description="Отношение сигнал/шум")
    sde: float = Field(..., ge=0, description="Спектральная плотность энергии")
    bls_power: float = Field(..., ge=0, description="Мощность BLS")
    planet_radius: Optional[float] = Field(None, gt=0, description="Радиус планеты (R_earth)")
    semi_major_axis: Optional[float] = Field(None, gt=0, description="Большая полуось (AU)")
    equilibrium_temp: Optional[float] = Field(None, gt=0, description="Равновесная температура (K)")
    false_alarm_probability: float = Field(0.5, ge=0, le=1, description="Вероятность ложного срабатывания")
    planet_probability: float = Field(0.5, ge=0, le=1, description="Вероятность наличия планеты")


class BLSResult(BaseModel):
    """Результат BLS анализа"""
    periods: List[float] = Field(..., description="Тестовые периоды")
    power: List[float] = Field(..., description="Значения мощности BLS")
    best_period: float = Field(..., gt=0, description="Лучший период (дни)")
    best_power: float = Field(..., ge=0, description="Лучшая мощность BLS")
    best_t0: float = Field(..., description="Лучшее время центра транзита (BJD)")
    best_duration: float = Field(..., gt=0, description="Лучшая длительность транзита (дни)")
    best_depth: float = Field(..., ge=0, description="Лучшая глубина транзита")
    snr: float = Field(..., ge=0, description="Отношение сигнал/шум")
    depth: float = Field(..., ge=0, description="Глубина транзита")
    depth_err: float = Field(..., ge=0, description="Ошибка глубины транзита")
    significance: float = Field(..., ge=0, le=1, description="Статистическая значимость")
    is_significant: bool = Field(..., description="Является ли результат значимым")


class SearchRequest(BaseModel):
    """Запрос на поиск"""
    target_name: str = Field(..., min_length=1, max_length=100, description="Название цели")
    catalog: str = Field("TIC", pattern="^(TIC|KIC|EPIC)$", description="Каталог")
    mission: str = Field("TESS", pattern="^(TESS|Kepler|K2)$", description="Миссия")
    period_min: float = Field(0.5, ge=0.1, le=100.0, description="Минимальный период (дни)")
    period_max: float = Field(20.0, ge=0.1, le=100.0, description="Максимальный период (дни)")
    duration_min: float = Field(0.05, ge=0.01, le=1.0, description="Минимальная длительность (дни)")
    duration_max: float = Field(0.3, ge=0.01, le=1.0, description="Максимальная длительность (дни)")
    snr_threshold: float = Field(7.0, ge=3.0, le=20.0, description="Порог SNR")
    use_bls: bool = Field(True, description="Использовать BLS анализ")
    use_ai: bool = Field(True, description="Использовать ИИ анализ")
    use_ensemble: bool = Field(True, description="Использовать ансамбль моделей")


class SearchResult(BaseModel):
    """Результат поиска"""
    target_name: str = Field(..., min_length=1, max_length=100, description="Название цели")
    lightcurve: LightCurveData = Field(..., description="Данные кривой блеска")
    bls_result: Optional[BLSResult] = Field(None, description="Результат BLS анализа")
    candidates: List[TransitCandidate] = Field([], description="Найденные кандидаты")
    total_candidates: int = Field(0, ge=0, description="Общее количество кандидатов")
    processing_time: float = Field(..., ge=0, description="Время обработки (секунды)")
    search_parameters: SearchRequest = Field(..., description="Параметры поиска")


# Dataclasses for internal use
@dataclass
class InternalLightCurveData:
    """Internal lightcurve data structure"""
    target_name: str
    time: np.ndarray
    flux: np.ndarray
    flux_err: Optional[np.ndarray] = None
    quality: Optional[np.ndarray] = None
    mission: str = "TESS"
    sector: Optional[int] = None
    quarter: Optional[int] = None
    campaign: Optional[int] = None
    duration_days: float = 0.0
    cadence: float = 2.0
    data_points: int = 0


@dataclass
class InternalTargetInfo:
    """Internal target information"""
    target_name: str
    catalog_id: str
    ra: float
    dec: float
    magnitude: float
    available_data: List[str]


@dataclass
class InternalTransitCandidate:
    """Internal transit candidate"""
    period: float
    t0: float
    duration: float
    depth: float
    snr: float
    sde: float
    bls_power: float
    planet_radius: Optional[float] = None
    semi_major_axis: Optional[float] = None
    equilibrium_temp: Optional[float] = None
    false_alarm_probability: float = 0.5
    planet_probability: float = 0.5


@dataclass
class InternalBLSResult:
    """Internal BLS result"""
    periods: np.ndarray
    power: np.ndarray
    best_period: float
    best_power: float
    best_t0: float
    best_duration: float
    best_depth: float
    snr: float
    depth: float
    depth_err: float
    significance: float
    is_significant: bool


@dataclass
class InternalSearchRequest:
    """Internal search request"""
    target_name: str
    catalog: str
    mission: str
    period_min: float
    period_max: float
    duration_min: float
    duration_max: float
    snr_threshold: float
    use_bls: bool
    use_ai: bool
    use_ensemble: bool


@dataclass
class InternalSearchResult:
    """Internal search result"""
    target_name: str
    lightcurve: InternalLightCurveData
    bls_result: Optional[InternalBLSResult]
    candidates: List[InternalTransitCandidate]
    total_candidates: int
    processing_time: float
    search_parameters: InternalSearchRequest