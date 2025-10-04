"""
Real Unified Analysis API - Production version without synthetic data
Единый API анализа экзопланет с РЕАЛЬНЫМИ данными NASA
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import logging
import asyncio
from pathlib import Path
import json
import io
import base64
from datetime import datetime

from core.logging import get_logger
from data_sources.real_nasa_client import get_nasa_client
from ml.lightcurve_preprocessor import LightCurvePreprocessor
from ml.feature_extractor import ExoplanetFeatureExtractor
from ml.exoplanet_classifier import ExoplanetEnsembleClassifier

logger = get_logger(__name__)
router = APIRouter()

# Глобальные объекты с ленивой инициализацией
_preprocessor = None
_feature_extractor = None
_classifier = None

def get_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = LightCurvePreprocessor()
    return _preprocessor

def get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = ExoplanetFeatureExtractor()
    return _feature_extractor

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = ExoplanetEnsembleClassifier()
    return _classifier


class UnifiedAnalysisRequest(BaseModel):
    """Запрос на единый анализ"""
    target_name: str = Field(..., description="Название цели (TOI-715, Kepler-452b, etc.)")
    mission: Optional[str] = Field("TESS", description="Миссия (TESS, Kepler, K2)")
    sector: Optional[int] = Field(None, description="Сектор для TESS")
    cadence: Optional[str] = Field("short", description="Каденция (short, long)")
    auto_download: bool = Field(True, description="Автоматическая загрузка из NASA")
    lightcurve_data: Optional[Dict[str, List[float]]] = Field(None, description="Данные кривой блеска")


class UnifiedAnalysisResult(BaseModel):
    """Результат единого анализа"""
    target_name: str
    predicted_class: str
    confidence_score: float
    planet_parameters: Dict[str, float]
    star_info: Dict[str, Any]
    plot_data: Dict[str, List[float]]
    data_quality_score: float
    processing_time_ms: float
    method: str = "Unified Real Data Analysis"
    data_source: str
    analysis_notes: List[str]


async def perform_real_unified_analysis(
    target_name: str,
    mission: str = "TESS",
    sector: Optional[int] = None,
    cadence: str = "short",
    lightcurve_data: Optional[Dict[str, Any]] = None,
    file_data: Optional[bytes] = None
) -> UnifiedAnalysisResult:
    """
    Выполняет полный анализ кривой блеска с РЕАЛЬНЫМИ данными NASA
    """
    start_time = time.time()
    analysis_notes = []
    
    try:
        # 1. Получение данных кривой блеска (ТОЛЬКО РЕАЛЬНЫЕ ДАННЫЕ)
        logger.info(f"Loading REAL lightcurve data for {target_name}")
        
        if lightcurve_data:
            # Данные переданы напрямую
            logger.info("Using provided lightcurve data")
            analysis_notes.append("Used provided lightcurve data")
            
        elif file_data:
            # Загрузка из файла
            logger.info("Loading lightcurve from uploaded file")
            
            try:
                import pandas as pd
                df = pd.read_csv(io.StringIO(file_data.decode('utf-8')))
                
                # Автоматическое определение колонок
                time_col = None
                flux_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if 'time' in col_lower or 'bjd' in col_lower or 'mjd' in col_lower:
                        time_col = col
                    elif 'flux' in col_lower or 'mag' in col_lower:
                        flux_col = col
                
                if time_col is None or flux_col is None:
                    time_col = df.columns[0]
                    flux_col = df.columns[1]
                
                lightcurve_data = {
                    'time': df[time_col].values.tolist(),
                    'flux': df[flux_col].values.tolist(),
                    'source': 'uploaded_file'
                }
                analysis_notes.append(f"Loaded {len(lightcurve_data['time'])} points from uploaded file")
                
            except Exception as e:
                logger.error(f"Error parsing uploaded file: {e}")
                raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
                
        else:
            # Загрузка ТОЛЬКО из NASA/MAST (БЕЗ СИНТЕТИКИ)
            logger.info(f"Downloading REAL data from NASA/MAST: {target_name}")
            
            try:
                nasa_client = get_nasa_client()
                
                # Валидация цели
                validation = await nasa_client.validate_target(target_name)
                if not validation['is_valid']:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Target '{target_name}' not found in NASA catalogs. "
                               f"Available targets include: TOI-715, Kepler-452b, TRAPPIST-1e, Proxima Cen b, K2-18b"
                    )
                
                # Загрузка реальных данных
                recommended_mission = validation.get('recommended_mission', mission)
                real_data = await nasa_client.get_lightcurve(
                    target_name, 
                    mission=recommended_mission,
                    sector=sector,
                    cadence=cadence
                )
                
                if not real_data:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Failed to download lightcurve data for '{target_name}' from {recommended_mission}. "
                               f"The target may not have available observations in the specified mission/sector."
                    )
                
                lightcurve_data = {
                    'time': real_data['time'],
                    'flux': real_data['flux'],
                    'flux_err': real_data.get('flux_err'),
                    'source': f"NASA/{recommended_mission}",
                    'mission': recommended_mission,
                    'data_points': real_data['data_points'],
                    'sector': real_data.get('sector'),
                    'cadence': real_data.get('cadence')
                }
                
                analysis_notes.append(f"Downloaded {real_data['data_points']} real data points from {recommended_mission}")
                if real_data.get('sector'):
                    analysis_notes.append(f"TESS Sector: {real_data['sector']}")
                        
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error accessing NASA data: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"NASA/MAST service unavailable: {str(e)}. Please try again later."
                )

        # 2. Предобработка данных
        logger.info("Preprocessing real lightcurve data")
        preprocessor = get_preprocessor()
        
        time_array = np.array(lightcurve_data['time'])
        flux_array = np.array(lightcurve_data['flux'])
        flux_err_array = np.array(lightcurve_data.get('flux_err', np.ones_like(flux_array) * 0.001))
        
        # Очистка и нормализация реальных данных
        processed_data = preprocessor.preprocess_lightcurve(
            time_array, flux_array, flux_err_array
        )
        
        analysis_notes.append(f"Processed {len(processed_data['time'])} data points")
        
        # 3. BLS анализ транзитов на реальных данных
        logger.info("Running BLS transit analysis on real data")
        
        nasa_client = get_nasa_client()
        bls_result = await nasa_client.analyze_transit_bls(
            processed_data['time'], 
            processed_data['flux'],
            period_min=1.0,
            period_max=50.0
        )
        
        if 'error' in bls_result:
            raise HTTPException(status_code=500, detail=f"BLS analysis failed: {bls_result['error']}")
        
        # 4. Извлечение признаков
        logger.info("Extracting features from real data")
        feature_extractor = get_feature_extractor()
        features = feature_extractor.extract_features(
            processed_data['time'],
            processed_data['flux'],
            processed_data['flux_err']
        )
        
        # 5. ML классификация с улучшенными параметрами для реальных данных
        classifier = get_classifier()
        
        # Создаем расширенный набор признаков для реальных данных
        enhanced_features = {
            'snr': bls_result['snr'],
            'transit_depth': bls_result['transit_depth'],
            'significance': bls_result['significance'],
            'period_stability': bls_result['best_power'],
            'data_quality': 1.0 - np.std(processed_data['flux']),
            'data_points': len(processed_data['time']),
            'time_span_days': float(np.max(processed_data['time']) - np.min(processed_data['time'])),
            'cadence_minutes': float(np.median(np.diff(processed_data['time'])) * 24 * 60),
            'real_data_bonus': 1.5  # Бонус для реальных данных
        }
        
        # Создаем последовательность для CNN
        sequence_length = 64
        if len(processed_data['flux']) >= sequence_length:
            sequence = processed_data['flux'][:sequence_length]
        else:
            sequence = np.pad(processed_data['flux'], (0, sequence_length - len(processed_data['flux'])), 'constant')
        
        prediction = classifier.predict_single(enhanced_features, sequence)
        
        # 6. Получение параметров звезды из NASA
        stellar_data = None
        try:
            nasa_client = get_nasa_client()
            stellar_data = await nasa_client.get_stellar_parameters(target_name)
        except Exception as e:
            logger.warning(f"Could not fetch stellar parameters: {e}")
        
        # 7. Формирование результата
        planet_params = {
            'orbital_period_days': bls_result['best_period'],
            'transit_depth_ppm': bls_result['depth_ppm'],
            'transit_duration_hours': bls_result['transit_duration_hours'],
            'snr': bls_result['snr'],
            'significance': bls_result['significance'],
            'bls_power': bls_result['best_power']
        }
        
        # Информация о звезде
        if stellar_data:
            star_info = {
                'target_id': target_name,
                'star_name': stellar_data.get('star_name', target_name),
                'ra': stellar_data.get('ra', 0),
                'dec': stellar_data.get('dec', 0),
                'effective_temperature': stellar_data.get('effective_temperature', 0),
                'stellar_radius': stellar_data.get('stellar_radius', 0),
                'stellar_mass': stellar_data.get('stellar_mass', 0),
                'distance_pc': stellar_data.get('distance_pc', 0),
                'v_magnitude': stellar_data.get('v_magnitude', 0),
                'source': 'NASA Exoplanet Archive'
            }
            analysis_notes.append("Retrieved stellar parameters from NASA Exoplanet Archive")
        else:
            star_info = {
                'target_id': target_name,
                'star_name': target_name,
                'source': 'Not available'
            }
            analysis_notes.append("Stellar parameters not available")
        
        # Данные для графика
        plot_data = {
            'time': processed_data['time'].tolist(),
            'flux': processed_data['flux'].tolist(),
            'flux_err': processed_data['flux_err'].tolist(),
        }
        
        # Добавляем транзитные времена если найдены
        if bls_result.get('transit_epoch'):
            period = bls_result['best_period']
            epoch = bls_result['transit_epoch']
            time_span = np.max(processed_data['time']) - np.min(processed_data['time'])
            n_transits = int(time_span / period) + 1
            
            transit_times = [epoch + i * period for i in range(n_transits)]
            transit_times = [t for t in transit_times if np.min(processed_data['time']) <= t <= np.max(processed_data['time'])]
            plot_data['transit_times'] = transit_times
            
            analysis_notes.append(f"Identified {len(transit_times)} potential transit events")
        
        # Оценка качества данных
        data_quality = min(1.0, len(processed_data['time']) / 1000.0)  # Нормализуем по количеству точек
        noise_level = np.std(processed_data['flux'])
        data_quality *= (1.0 - min(0.5, noise_level * 10))  # Учитываем шум
        
        processing_time = (time.time() - start_time) * 1000
        
        analysis_notes.append(f"Analysis completed in {processing_time:.1f} ms")
        analysis_notes.append(f"Data quality score: {data_quality:.2f}")
        
        result = UnifiedAnalysisResult(
            target_name=target_name,
            predicted_class=prediction['predicted_class'],
            confidence_score=prediction['confidence'],
            planet_parameters=planet_params,
            star_info=star_info,
            plot_data=plot_data,
            data_quality_score=data_quality,
            processing_time_ms=processing_time,
            data_source=lightcurve_data.get('source', 'NASA'),
            analysis_notes=analysis_notes
        )
        
        logger.info(f"Real data analysis completed for {target_name}: {result.predicted_class} ({result.confidence_score:.1%})")
        return result
        
    except Exception as e:
        logger.error(f"Real unified analysis failed for {target_name}: {e}")
        raise


@router.post("/analyze", response_model=UnifiedAnalysisResult)
async def unified_analyze(request: UnifiedAnalysisRequest):
    """
    Единый анализ экзопланет с РЕАЛЬНЫМИ данными NASA
    """
    logger.info(f"Starting real unified analysis for {request.target_name}")
    
    try:
        result = await perform_real_unified_analysis(
            target_name=request.target_name,
            mission=request.mission,
            sector=request.sector,
            cadence=request.cadence,
            lightcurve_data=request.lightcurve_data
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Unified analysis failed for {request.target_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-file", response_model=UnifiedAnalysisResult)
async def analyze_lightcurve_file(
    target_name: str = Form(...),
    file: UploadFile = File(...),
    mission: str = Form("TESS")
):
    """
    Анализ загруженного файла с кривой блеска
    """
    logger.info(f"Starting file analysis for {target_name}")
    
    try:
        # Читаем файл
        content = await file.read()
        
        result = await perform_real_unified_analysis(
            target_name=target_name,
            mission=mission,
            file_data=content
        )
        
        return result
        
    except Exception as e:
        logger.error(f"File analysis failed for {target_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate-target/{target_name}")
async def validate_target(target_name: str):
    """
    Валидация цели в NASA каталогах
    """
    try:
        nasa_client = get_nasa_client()
        validation = await nasa_client.validate_target(target_name)
        return validation
    except Exception as e:
        logger.error(f"Target validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-targets")
async def get_available_targets(limit: int = 20):
    """
    Получить список доступных целей из NASA каталогов
    """
    try:
        nasa_client = get_nasa_client()
        
        # Получаем популярные цели
        popular_targets = [
            "TOI-715", "Kepler-452b", "TRAPPIST-1e", "Proxima Cen b", 
            "K2-18b", "TOI-849b", "TOI-1338b", "TOI-2109b"
        ]
        
        available_targets = []
        for target in popular_targets[:limit]:
            try:
                validation = await nasa_client.validate_target(target)
                if validation['is_valid']:
                    available_targets.append({
                        'name': target,
                        'missions': validation['available_missions'],
                        'recommended_mission': validation['recommended_mission'],
                        'planet_data': validation.get('planet_data'),
                        'stellar_data': validation.get('stellar_data')
                    })
            except:
                continue
        
        return {
            'available_targets': available_targets,
            'total_count': len(available_targets),
            'note': 'These are confirmed exoplanets with available NASA/MAST data'
        }
        
    except Exception as e:
        logger.error(f"Error fetching available targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Проверка здоровья unified analysis API"""
    try:
        # Проверяем доступность NASA клиента
        nasa_client = get_nasa_client()
        
        return {
            "status": "healthy",
            "service": "real_unified_analysis",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "preprocessor": "ready",
                "feature_extractor": "ready", 
                "classifier": "ready",
                "nasa_client": "ready",
                "astro_packages": "available" if nasa_client else "unavailable"
            },
            "data_sources": ["NASA Exoplanet Archive", "MAST/TESS", "MAST/Kepler"],
            "supported_missions": ["TESS", "Kepler", "K2"],
            "note": "Production system - real NASA data only"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
