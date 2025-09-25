from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime

class SearchRequest(BaseModel):
    """Модель запроса для поиска экзопланет"""
    target_name: str = Field(..., description="Имя цели (TIC ID, KIC ID и т.д.)")
    catalog: str = Field(default="TIC", description="Каталог (TIC, KIC, EPIC)")
    mission: str = Field(default="TESS", description="Миссия (TESS, Kepler, K2)")
    
    # Параметры BLS поиска
    period_min: float = Field(default=0.5, description="Минимальный период (дни)")
    period_max: float = Field(default=50.0, description="Максимальный период (дни)")
    duration_min: float = Field(default=0.01, description="Минимальная длительность транзита (дни)")
    duration_max: float = Field(default=0.5, description="Максимальная длительность транзита (дни)")
    snr_threshold: float = Field(default=7.0, description="Пороговое значение SNR")
    
    class Config:
        schema_extra = {
            "example": {
                "target_name": "TIC 441420236",
                "catalog": "TIC",
                "mission": "TESS",
                "period_min": 1.0,
                "period_max": 30.0,
                "duration_min": 0.05,
                "duration_max": 0.3,
                "snr_threshold": 8.0
            }
        }

class LightCurveData(BaseModel):
    """Модель данных кривой блеска"""
    target_name: str
    time: List[float] = Field(..., description="Время в днях")
    flux: List[float] = Field(..., description="Поток")
    flux_err: Optional[List[float]] = Field(None, description="Ошибки потока")
    quality: Optional[List[int]] = Field(None, description="Флаги качества")
    
    # Метаданные
    mission: str
    sector: Optional[int] = None
    quarter: Optional[int] = None
    campaign: Optional[int] = None
    
    # Статистика
    duration_days: float
    cadence: float
    data_points: int
    
    class Config:
        arbitrary_types_allowed = True

class TransitCandidate(BaseModel):
    """Модель кандидата в экзопланеты"""
    period: float = Field(..., description="Орбитальный период (дни)")
    t0: float = Field(..., description="Время первого транзита")
    duration: float = Field(..., description="Длительность транзита (часы)")
    depth: float = Field(..., description="Глубина транзита (ppm)")
    
    # BLS статистика
    snr: float = Field(..., description="Отношение сигнал/шум")
    sde: float = Field(..., description="Signal Detection Efficiency")
    bls_power: float = Field(..., description="BLS мощность")
    
    # Физические параметры
    planet_radius: Optional[float] = Field(None, description="Радиус планеты (R_Earth)")
    semi_major_axis: Optional[float] = Field(None, description="Большая полуось (AU)")
    equilibrium_temp: Optional[float] = Field(None, description="Равновесная температура (K)")
    
    # Вероятности
    false_alarm_probability: float = Field(..., description="Вероятность ложного срабатывания")
    planet_probability: float = Field(..., description="Вероятность планеты")
    
    class Config:
        schema_extra = {
            "example": {
                "period": 3.52,
                "t0": 1325.67,
                "duration": 2.4,
                "depth": 1250.0,
                "snr": 12.5,
                "sde": 15.2,
                "bls_power": 0.85,
                "planet_radius": 1.2,
                "semi_major_axis": 0.045,
                "equilibrium_temp": 850,
                "false_alarm_probability": 0.001,
                "planet_probability": 0.92
            }
        }

class BLSResult(BaseModel):
    """Результат BLS анализа"""
    periods: List[float] = Field(..., description="Массив периодов")
    power: List[float] = Field(..., description="BLS мощность")
    best_period: float = Field(..., description="Лучший период")
    best_power: float = Field(..., description="Максимальная мощность")
    
    # Параметры лучшего кандидата
    best_t0: float
    best_duration: float
    best_depth: float
    
class SearchResult(BaseModel):
    """Результат поиска экзопланет"""
    target_name: str
    search_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Исходные данные
    lightcurve: LightCurveData
    
    # Результаты BLS
    bls_result: BLSResult
    
    # Найденные кандидаты
    candidates: List[TransitCandidate] = Field(default_factory=list)
    
    # Статистика поиска
    total_candidates: int = Field(default=0)
    processing_time: float = Field(..., description="Время обработки в секундах")
    
    # Параметры поиска
    search_parameters: SearchRequest
    
    class Config:
        arbitrary_types_allowed = True

class TargetInfo(BaseModel):
    """Информация о цели наблюдения"""
    target_name: str
    catalog_id: str
    coordinates: Dict[str, float]  # ra, dec
    magnitude: Optional[float] = None
    stellar_parameters: Optional[Dict[str, Any]] = None
    available_data: List[str] = Field(default_factory=list)
    
class ValidationResult(BaseModel):
    """Результат валидации кандидата"""
    candidate_id: str
    validation_score: float = Field(..., ge=0, le=1)
    validation_flags: List[str] = Field(default_factory=list)
    comparison_with_known: Optional[Dict[str, Any]] = None
