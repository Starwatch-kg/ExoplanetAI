"""
GPI Analysis API routes
Gaussian Process Inference для анализа экзопланет
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class GPIAnalysisRequest(BaseModel):
    """Запрос на GPI анализ"""
    target_name: Optional[str] = Field(None, description="Название цели")
    period_min: float = Field(1.0, description="Минимальный период (дни)")
    period_max: float = Field(20.0, description="Максимальный период (дни)")
    duration_min: float = Field(0.05, description="Минимальная длительность транзита")
    duration_max: float = Field(0.2, description="Максимальная длительность транзита")
    snr_threshold: float = Field(5.0, description="Порог SNR")
    significance_threshold: float = Field(0.7, description="Порог значимости")


class GPIResult(BaseModel):
    """Результат GPI анализа"""
    predicted_class: str
    confidence_score: float
    planet_parameters: Dict[str, float]
    gpi_parameters: Dict[str, float]
    plot_data: Dict[str, List[float]]
    processing_time_ms: float
    method: str = "GPI"
    significance: float


def generate_demo_lightcurve(target_name: str, period: float = None) -> Dict[str, np.ndarray]:
    """
    Генерация демонстрационной кривой блеска для GPI анализа
    """
    np.random.seed(hash(target_name) % 2**32)
    
    # Параметры
    if period is None:
        period = np.random.uniform(2.0, 15.0)
    
    transit_depth = np.random.uniform(0.001, 0.01)  # 0.1% - 1%
    transit_duration = period * np.random.uniform(0.05, 0.15)  # 5-15% от периода
    
    # Временной ряд (30 дней наблюдений)
    time = np.linspace(0, 30, 2000)
    
    # Базовый уровень с небольшим шумом
    flux = np.ones_like(time) + np.random.normal(0, 0.0005, len(time))
    
    # Добавляем транзиты
    for i in range(int(30 / period) + 1):
        transit_center = i * period
        if transit_center > 30:
            break
            
        # Простая модель транзита (трапеция)
        transit_mask = np.abs(time - transit_center) < transit_duration / 2
        if np.any(transit_mask):
            # Глубина транзита с плавными краями
            transit_profile = np.exp(-((time - transit_center) / (transit_duration / 4))**2)
            flux -= transit_depth * transit_profile * (transit_profile > 0.1)
    
    # Добавляем долгосрочные вариации
    flux += 0.001 * np.sin(2 * np.pi * time / 5.0)  # 5-дневные вариации
    
    return {
        'time': time,
        'flux': flux,
        'period': period,
        'depth': transit_depth,
        'duration': transit_duration
    }


def perform_gpi_analysis(
    time: np.ndarray,
    flux: np.ndarray,
    params: GPIAnalysisRequest
) -> Dict[str, Any]:
    """
    Выполнение GPI анализа
    """
    start_time = time[0] if len(time) > 0 else 0
    
    # Простая имитация GPI анализа
    # В реальной реализации здесь был бы Gaussian Process
    
    # Поиск периодичности
    periods = np.linspace(params.period_min, params.period_max, 1000)
    
    # Простой BLS-подобный поиск
    best_period = 0
    best_power = 0
    best_depth = 0
    
    for period in periods:
        # Фолдинг данных
        phase = (time % period) / period
        sorted_indices = np.argsort(phase)
        sorted_flux = flux[sorted_indices]
        
        # Поиск транзита (минимум в фолдированной кривой)
        window_size = int(len(sorted_flux) * 0.1)  # 10% от периода
        if window_size < 5:
            window_size = 5
            
        smoothed = np.convolve(sorted_flux, np.ones(window_size)/window_size, mode='same')
        min_idx = np.argmin(smoothed)
        
        # Оценка глубины и мощности сигнала
        depth = 1.0 - np.min(smoothed)
        noise_std = np.std(sorted_flux - smoothed)
        power = depth / noise_std if noise_std > 0 else 0
        
        if power > best_power:
            best_power = power
            best_period = period
            best_depth = depth
    
    # Оценка параметров
    snr = best_power
    significance = min(0.99, snr / 20.0)  # Нормализация
    
    # Классификация на основе GPI критериев
    if snr > params.snr_threshold * 2 and significance > 0.8 and best_depth > 0.002:
        predicted_class = "Confirmed"
        confidence = min(0.95, 0.7 + snr * 0.03)
    elif snr > params.snr_threshold and significance > params.significance_threshold:
        predicted_class = "Candidate"
        confidence = min(0.85, 0.5 + snr * 0.04)
    else:
        predicted_class = "False Positive"
        confidence = max(0.60, min(0.80, 0.6 + significance * 0.25))
    
    # Создание модельной кривой для визуализации
    model_flux = np.ones_like(flux)
    if best_period > 0:
        for i in range(int((time[-1] - time[0]) / best_period) + 1):
            transit_center = start_time + i * best_period
            transit_mask = np.abs(time - transit_center) < best_period * 0.05
            if np.any(transit_mask):
                transit_profile = np.exp(-((time - transit_center) / (best_period * 0.02))**2)
                model_flux -= best_depth * transit_profile * (transit_profile > 0.1)
    
    return {
        'predicted_class': predicted_class,
        'confidence_score': confidence,
        'planet_parameters': {
            'orbital_period_days': float(best_period),
            'transit_depth_ppm': float(best_depth * 1e6),
            'transit_duration_hours': float(best_period * 0.1 * 24),
            'snr': float(snr)
        },
        'gpi_parameters': {
            'best_period': float(best_period),
            'signal_power': float(best_power),
            'noise_level': float(np.std(flux)),
            'data_points': len(time)
        },
        'plot_data': {
            'time': time.tolist(),
            'flux': flux.tolist(),
            'model': model_flux.tolist()
        },
        'significance': float(significance),
        'method': 'GPI'
    }


@router.post("/analyze", response_model=GPIResult)
async def gpi_analyze(
    target_name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    period_min: float = Form(1.0),
    period_max: float = Form(20.0),
    duration_min: float = Form(0.05),
    duration_max: float = Form(0.2),
    snr_threshold: float = Form(5.0),
    significance_threshold: float = Form(0.7)
):
    """
    GPI анализ экзопланет
    """
    start_time = time.time()
    
    try:
        # Создаем объект параметров
        params = GPIAnalysisRequest(
            target_name=target_name,
            period_min=period_min,
            period_max=period_max,
            duration_min=duration_min,
            duration_max=duration_max,
            snr_threshold=snr_threshold,
            significance_threshold=significance_threshold
        )
        
        # Получение данных
        if file:
            # Загрузка файла
            content = await file.read()
            
            try:
                # Попытка парсинга CSV/TXT
                import io
                import pandas as pd
                
                content_str = content.decode('utf-8')
                data = pd.read_csv(io.StringIO(content_str))
                
                # Попытка найти колонки времени и потока
                time_col = None
                flux_col = None
                
                for col in data.columns:
                    col_lower = col.lower()
                    if 'time' in col_lower or 'bjd' in col_lower or 'mjd' in col_lower:
                        time_col = col
                    elif 'flux' in col_lower or 'mag' in col_lower or 'brightness' in col_lower:
                        flux_col = col
                
                if time_col is None or flux_col is None:
                    # Используем первые две колонки
                    time_col = data.columns[0]
                    flux_col = data.columns[1]
                
                time_array = data[time_col].values
                flux_array = data[flux_col].values
                
                # Нормализация
                flux_array = flux_array / np.median(flux_array)
                
            except Exception as e:
                logger.error(f"Error parsing uploaded file: {e}")
                raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
                
        elif target_name:
            # Генерация демо данных
            logger.info(f"Generating demo lightcurve for GPI analysis: {target_name}")
            demo_data = generate_demo_lightcurve(target_name)
            time_array = demo_data['time']
            flux_array = demo_data['flux']
            
        else:
            raise HTTPException(status_code=400, detail="Either target_name or file must be provided")
        
        # Выполнение GPI анализа
        logger.info("Starting GPI analysis")
        result = perform_gpi_analysis(time_array, flux_array, params)
        
        # Добавляем время обработки
        processing_time = (time.time() - start_time) * 1000
        result['processing_time_ms'] = processing_time
        
        logger.info(f"GPI analysis completed in {processing_time:.1f}ms")
        
        return GPIResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GPI analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/parameters")
async def get_gpi_parameters():
    """
    Получить доступные параметры GPI анализа
    """
    return {
        "parameters": {
            "period_min": {
                "description": "Minimum orbital period to search (days)",
                "default": 1.0,
                "min": 0.1,
                "max": 100.0
            },
            "period_max": {
                "description": "Maximum orbital period to search (days)", 
                "default": 20.0,
                "min": 1.0,
                "max": 1000.0
            },
            "duration_min": {
                "description": "Minimum transit duration fraction",
                "default": 0.05,
                "min": 0.01,
                "max": 0.5
            },
            "duration_max": {
                "description": "Maximum transit duration fraction",
                "default": 0.2,
                "min": 0.1,
                "max": 0.5
            },
            "snr_threshold": {
                "description": "Signal-to-noise ratio threshold",
                "default": 5.0,
                "min": 1.0,
                "max": 50.0
            },
            "significance_threshold": {
                "description": "Statistical significance threshold",
                "default": 0.7,
                "min": 0.1,
                "max": 0.99
            }
        },
        "method": "GPI",
        "description": "Gaussian Process Inference for exoplanet detection"
    }
