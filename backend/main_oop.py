#!/usr/bin/env python3
"""
🌌 Exoplanet AI - Оптимизированный FastAPI Backend
Веб-платформа для поиска экзопланет с использованием ИИ
Архитектура: ООП + Clean Code + SOLID принципы
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
import json
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn


# ================================
# 📊 DATA MODELS (Pydantic)
# ================================

class TICRequest(BaseModel):
    """Запрос на загрузку данных TESS по TIC ID"""
    tic_id: str = Field(..., description="TESS Input Catalog ID")
    sectors: Optional[List[int]] = Field(None, description="Список секторов для загрузки")
    
    @validator('tic_id')
    def validate_tic_id(cls, v):
        if not v.strip():
            raise ValueError('TIC ID не может быть пустым')
        return v.strip()


class LightcurveData(BaseModel):
    """Данные кривой блеска"""
    tic_id: str = Field(..., description="TIC ID звезды")
    times: List[float] = Field(..., description="Временные метки")
    fluxes: List[float] = Field(..., description="Значения потока")
    
    @validator('times', 'fluxes')
    def validate_arrays(cls, v):
        if len(v) < 10:
            raise ValueError('Недостаточно данных для анализа (минимум 10 точек)')
        return v


class ExoplanetCandidate(BaseModel):
    """Кандидат в экзопланеты"""
    id: str = Field(..., description="Уникальный ID кандидата")
    period: float = Field(..., description="Период обращения (дни)")
    depth: float = Field(..., description="Глубина транзита")
    duration: float = Field(..., description="Длительность транзита (часы)")
    confidence: float = Field(..., ge=0, le=1, description="Уровень уверенности")
    start_time: float = Field(..., description="Время начала транзита")
    end_time: float = Field(..., description="Время окончания транзита")
    method: str = Field(..., description="Метод обнаружения")


class AnalysisRequest(BaseModel):
    """Запрос на анализ кривой блеска"""
    lightcurve_data: LightcurveData
    model_type: str = Field(..., description="Тип модели для анализа")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Дополнительные параметры")


class AnalysisResponse(BaseModel):
    """Результат анализа кривой блеска"""
    success: bool = Field(..., description="Успешность анализа")
    candidates: List[ExoplanetCandidate] = Field(..., description="Найденные кандидаты")
    processing_time: float = Field(..., description="Время обработки (секунды)")
    model_used: str = Field(..., description="Использованная модель")
    statistics: Dict[str, Any] = Field(..., description="Статистика анализа")
    error: Optional[str] = Field(None, description="Сообщение об ошибке")


# ================================
# 🔬 BUSINESS LOGIC CLASSES
# ================================

class TransitCandidate:
    """Внутренний класс для кандидата транзита"""
    
    def __init__(self, period: float, depth: float, duration: float, 
                 confidence: float, start_time: float, end_time: float, method: str):
        self.period = period
        self.depth = depth
        self.duration = duration
        self.confidence = confidence
        self.start_time = start_time
        self.end_time = end_time
        self.method = method
    
    def to_api_model(self, candidate_id: str) -> ExoplanetCandidate:
        """Конвертация в API модель"""
        return ExoplanetCandidate(
            id=candidate_id,
            period=self.period,
            depth=self.depth,
            duration=self.duration,
            confidence=self.confidence,
            start_time=self.start_time,
            end_time=self.end_time,
            method=self.method
        )


class ITransitDetector(ABC):
    """Интерфейс для детекторов транзитов"""
    
    @abstractmethod
    def detect_transits(self, times: np.ndarray, fluxes: np.ndarray) -> List[TransitCandidate]:
        """Обнаружение транзитов в кривой блеска"""
        pass


class SimpleTransitDetector(ITransitDetector):
    """Простой детектор транзитов для демонстрации"""
    
    def __init__(self, threshold_sigma: float = 2.0, min_points: int = 5):
        self.threshold_sigma = threshold_sigma
        self.min_points = min_points
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_transits(self, times: np.ndarray, fluxes: np.ndarray) -> List[TransitCandidate]:
        """
        Простой алгоритм детекции транзитов:
        1. Находим области с пониженным потоком
        2. Группируем соседние точки
        3. Создаем кандидатов
        """
        self.logger.info(f"Анализ {len(times)} точек данных")
        
        # Статистики потока
        mean_flux = np.mean(fluxes)
        std_flux = np.std(fluxes)
        threshold = mean_flux - self.threshold_sigma * std_flux
        
        # Поиск областей транзитов
        in_transit = fluxes < threshold
        transit_groups = self._group_consecutive_points(in_transit)
        
        # Создание кандидатов
        candidates = []
        for i, group in enumerate(transit_groups):
            if len(group) >= self.min_points:
                candidate = self._create_candidate_from_group(
                    group, times, fluxes, mean_flux, f"simple_{i}"
                )
                candidates.append(candidate)
        
        self.logger.info(f"Найдено {len(candidates)} кандидатов")
        return candidates[:3]  # Максимум 3 кандидата для демо
    
    def _group_consecutive_points(self, in_transit: np.ndarray) -> List[List[int]]:
        """Группировка последовательных точек транзита"""
        groups = []
        current_group = []
        
        for i, is_transit in enumerate(in_transit):
            if is_transit:
                current_group.append(i)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        
        # Добавляем последнюю группу
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _create_candidate_from_group(
        self, 
        group: List[int], 
        times: np.ndarray, 
        fluxes: np.ndarray, 
        mean_flux: float,
        method: str
    ) -> TransitCandidate:
        """Создание кандидата из группы точек"""
        start_idx, end_idx = group[0], group[-1]
        
        # Вычисляем параметры транзита
        period = np.random.uniform(5, 20)  # Демо: случайный период
        depth = (mean_flux - np.mean(fluxes[group])) / mean_flux
        duration = (times[end_idx] - times[start_idx]) * 24  # в часах
        confidence = min(0.95, len(group) / 20.0)  # Простая оценка уверенности
        
        return TransitCandidate(
            period=period,
            depth=abs(depth),
            duration=duration,
            confidence=confidence,
            start_time=times[start_idx],
            end_time=times[end_idx],
            method=method
        )


class TESSDataService:
    """Сервис для работы с данными TESS"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache = {}  # Простой кэш
    
    async def load_lightcurve(self, tic_id: str, sectors: Optional[List[int]] = None) -> LightcurveData:
        """Загрузка кривой блеска по TIC ID"""
        self.logger.info(f"Загрузка данных для TIC {tic_id}")
        
        # Проверяем кэш
        cache_key = f"{tic_id}_{sectors}"
        if cache_key in self.cache:
            self.logger.info("Данные найдены в кэше")
            return self.cache[cache_key]
        
        # Генерируем демо данные (в реальности здесь был бы запрос к MAST)
        demo_data = self._generate_demo_lightcurve(tic_id)
        
        # Сохраняем в кэш
        self.cache[cache_key] = demo_data
        
        return demo_data
    
    def _generate_demo_lightcurve(self, tic_id: str) -> LightcurveData:
        """Генерация демо кривой блеска"""
        np.random.seed(hash(tic_id) % 2**32)  # Детерминированная генерация
        
        # Создаем базовую кривую блеска
        n_points = 1000
        times = np.linspace(0, 27.4, n_points)  # 27.4 дня (сектор TESS)
        
        # Базовый поток с шумом
        base_flux = 1.0 + np.random.normal(0, 0.001, n_points)
        
        # Добавляем несколько транзитов для демонстрации
        if hash(tic_id) % 3 == 0:  # 1/3 звезд имеют транзиты
            base_flux = self._add_demo_transits(times, base_flux)
        
        return LightcurveData(
            tic_id=tic_id,
            times=times.tolist(),
            fluxes=base_flux.tolist()
        )
    
    def _add_demo_transits(self, times: np.ndarray, fluxes: np.ndarray) -> np.ndarray:
        """Добавление демо транзитов"""
        period = np.random.uniform(8, 15)  # Период в днях
        depth = np.random.uniform(0.001, 0.01)  # Глубина транзита
        duration = np.random.uniform(2, 6) / 24  # Длительность в днях
        
        # Добавляем транзиты
        for phase in np.arange(0, times[-1], period):
            transit_mask = (times >= phase) & (times <= phase + duration)
            fluxes[transit_mask] *= (1 - depth)
        
        return fluxes


class ExoplanetAnalysisService:
    """Основной сервис анализа экзопланет"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.detector = SimpleTransitDetector()
        self.results_cache = {}
    
    async def analyze_lightcurve(self, request: AnalysisRequest) -> AnalysisResponse:
        """Анализ кривой блеска"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Начало анализа с моделью {request.model_type}")
            
            # Конвертируем данные
            times = np.array(request.lightcurve_data.times)
            fluxes = np.array(request.lightcurve_data.fluxes)
            
            # Выполняем детекцию транзитов
            transit_candidates = self.detector.detect_transits(times, fluxes)
            
            # Конвертируем в API модели
            api_candidates = [
                candidate.to_api_model(f"{request.model_type}_{i}")
                for i, candidate in enumerate(transit_candidates)
            ]
            
            # Вычисляем статистики
            processing_time = (datetime.now() - start_time).total_seconds()
            statistics = self._calculate_statistics(api_candidates, times, processing_time)
            
            # Сохраняем результаты
            self._cache_results(request.lightcurve_data.tic_id, api_candidates, statistics)
            
            return AnalysisResponse(
                success=True,
                candidates=api_candidates,
                processing_time=processing_time,
                model_used=request.model_type,
                statistics=statistics
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResponse(
                success=False,
                candidates=[],
                processing_time=processing_time,
                model_used=request.model_type,
                statistics={},
                error=str(e)
            )
    
    def _calculate_statistics(
        self, 
        candidates: List[ExoplanetCandidate], 
        times: np.ndarray, 
        processing_time: float
    ) -> Dict[str, Any]:
        """Вычисление статистик анализа"""
        return {
            "total_candidates": len(candidates),
            "average_confidence": np.mean([c.confidence for c in candidates]) if candidates else 0,
            "processing_time": processing_time,
            "data_points": len(times),
            "time_span": float(times[-1] - times[0]) if len(times) > 1 else 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _cache_results(
        self, 
        tic_id: str, 
        candidates: List[ExoplanetCandidate], 
        statistics: Dict[str, Any]
    ):
        """Кэширование результатов анализа"""
        self.results_cache[tic_id] = {
            "candidates": [candidate.dict() for candidate in candidates],
            "statistics": statistics,
            "timestamp": datetime.now().isoformat()
        }


# ================================
# 🚀 FASTAPI APPLICATION
# ================================

class ExoplanetAPI:
    """Главный класс API приложения"""
    
    def __init__(self):
        self.app = FastAPI(
            title="🌌 Exoplanet AI API",
            description="Веб-платформа для поиска экзопланет с использованием ИИ",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tess_service = TESSDataService()
        self.analysis_service = ExoplanetAnalysisService()
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_startup_events()
    
    def _setup_middleware(self):
        """Настройка middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Настройка маршрутов"""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """🏠 Главная страница API"""
            return {
                "message": "🌌 Exoplanet AI API",
                "version": "2.0.0",
                "status": "active",
                "timestamp": datetime.now().isoformat(),
                "endpoints": {
                    "health": "/health",
                    "nasa_stats": "/api/nasa/stats",
                    "load_data": "/load-tic",
                    "analyze": "/analyze",
                    "docs": "/docs"
                }
            }
        
        @self.app.get("/health", tags=["System"])
        async def health_check():
            """🔍 Проверка состояния системы"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "services": {
                    "tess_data": "active",
                    "analysis": "active",
                    "cache": f"{len(self.analysis_service.results_cache)} results cached"
                }
            }
        
        @self.app.get("/api/nasa/stats", tags=["Data"])
        async def get_nasa_stats():
            """📊 Статистика NASA для лендинга"""
            return {
                "totalPlanets": 5635,  # Актуальные данные NASA
                "totalHosts": 4143,
                "lastUpdated": "2024-01-15",
                "source": "NASA Exoplanet Archive"
            }
        
        @self.app.post("/load-tic", response_model=Dict[str, Any], tags=["Data"])
        async def load_tic_data(request: TICRequest):
            """🛰️ Загрузка данных TESS по TIC ID"""
            try:
                self.logger.info(f"Запрос данных для TIC {request.tic_id}")
                
                lightcurve_data = await self.tess_service.load_lightcurve(
                    request.tic_id, 
                    request.sectors
                )
                
                return {
                    "success": True,
                    "data": lightcurve_data.dict(),
                    "message": f"Данные для TIC {request.tic_id} успешно загружены"
                }
                
            except Exception as e:
                self.logger.error(f"Ошибка загрузки TIC {request.tic_id}: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Ошибка загрузки данных: {str(e)}"
                )
        
        @self.app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
        async def analyze_lightcurve(request: AnalysisRequest):
            """🔬 Анализ кривой блеска для поиска экзопланет"""
            return await self.analysis_service.analyze_lightcurve(request)
    
    def _setup_startup_events(self):
        """Настройка событий запуска"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """🚀 Инициализация при запуске"""
            self.logger.info("🌌 Запуск Exoplanet AI API v2.0.0")
            self.logger.info("✅ Все сервисы инициализированы")


# ================================
# 🎯 APPLICATION ENTRY POINT
# ================================

# Создаем экземпляр API
api_instance = ExoplanetAPI()
app = api_instance.app

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    uvicorn.run(
        "main_optimized:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
