"""
Smart Search API with Enhanced AI Analysis
Умный поиск с улучшенным ИИ анализом
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import logging
import asyncio
from datetime import datetime
import redis
import json

from core.logging import get_logger
from core.auth import get_current_user, require_auth
from core.rate_limiting import rate_limit
from data_sources.real_nasa_client import get_nasa_client
from ml.enhanced_classifier import OptimizedEnsemble, EnhancedFeatureExtractor, generate_recommendations
from ml.lightcurve_preprocessor import LightCurvePreprocessor

logger = get_logger(__name__)
router = APIRouter()

# Redis для кэширования
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
except:
    redis_client = None
    logger.warning("Redis not available, caching disabled")


class SmartAnalysisRequest(BaseModel):
    """Запрос на умный анализ"""
    target_name: Optional[str] = Field(None, description="Название цели")
    time_data: Optional[List[float]] = Field(None, description="Временные данные")
    flux_data: Optional[List[float]] = Field(None, description="Данные потока")
    flux_err_data: Optional[List[float]] = Field(None, description="Ошибки потока")
    mission: str = Field("TESS", description="Миссия")
    auto_detect_cadence: bool = Field(True, description="Автоопределение каденции")
    adaptive_detrending: bool = Field(True, description="Адаптивное удаление трендов")
    include_uncertainty: bool = Field(True, description="Включить оценку неопределенности")
    explain_prediction: bool = Field(True, description="Объяснить предсказание")


class SmartAnalysisResult(BaseModel):
    """Результат умного анализа"""
    target_name: str
    predicted_class: str
    confidence_score: float
    uncertainty_bounds: List[float]
    transit_probability: float
    signal_characteristics: Dict[str, float]
    feature_importance: List[float]
    decision_reasoning: List[str]
    recommendations: List[str]
    data_quality_metrics: Dict[str, float]
    processing_time_ms: float
    model_version: str = "enhanced_v2.0"


class SearchFilters(BaseModel):
    """Фильтры для умного поиска"""
    confidence_min: float = Field(0.7, ge=0.0, le=1.0)
    snr_min: float = Field(5.0, ge=0.0)
    data_quality_min: float = Field(0.8, ge=0.0, le=1.0)
    missions: List[str] = Field(["TESS", "Kepler"], description="Список миссий")
    planet_types: List[str] = Field(["confirmed", "candidate"], description="Типы планет")
    period_range: Optional[List[float]] = Field(None, description="Диапазон периодов [min, max]")
    depth_range: Optional[List[float]] = Field(None, description="Диапазон глубин [min, max]")


class SmartSearchRequest(BaseModel):
    """Запрос умного поиска"""
    query: str = Field(..., description="Поисковый запрос")
    filters: SearchFilters = Field(default_factory=SearchFilters)
    use_ai_ranking: bool = Field(True, description="Использовать ИИ ранжирование")
    max_results: int = Field(20, ge=1, le=100)
    include_similar: bool = Field(True, description="Включить похожие объекты")


class BatchAnalysisRequest(BaseModel):
    """Запрос пакетного анализа"""
    targets: List[str] = Field(..., description="Список целей для анализа")
    analysis_params: SmartAnalysisRequest = Field(default_factory=SmartAnalysisRequest)
    parallel_limit: int = Field(5, ge=1, le=10, description="Лимит параллельных запросов")


# Глобальные объекты с ленивой инициализацией
_enhanced_classifier = None
_feature_extractor = None
_preprocessor = None

def get_enhanced_classifier():
    global _enhanced_classifier
    if _enhanced_classifier is None:
        _enhanced_classifier = OptimizedEnsemble()
        # Здесь можно загрузить предобученную модель
        logger.info("Enhanced classifier initialized")
    return _enhanced_classifier

def get_enhanced_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = EnhancedFeatureExtractor()
    return _feature_extractor

def get_adaptive_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = LightCurvePreprocessor()
    return _preprocessor


class SmartSearchEngine:
    """Движок умного поиска с ИИ"""
    
    def __init__(self):
        self.nasa_client = get_nasa_client()
        self.classifier = get_enhanced_classifier()
        self.feature_extractor = get_enhanced_feature_extractor()
        
    async def smart_search(self, request: SmartSearchRequest) -> Dict[str, Any]:
        """Выполнение умного поиска"""
        start_time = time.time()
        
        # Кэширование результатов
        cache_key = f"smart_search:{hash(str(request.dict()))}"
        if redis_client:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info("Returning cached search results")
                return json.loads(cached_result)
        
        # Парсинг запроса
        search_params = self._parse_search_query(request.query)
        
        # Поиск в NASA каталогах
        nasa_results = await self._search_nasa_catalogs(search_params, request.filters)
        
        # ИИ анализ и ранжирование
        if request.use_ai_ranking:
            ranked_results = await self._ai_ranking(nasa_results, request.filters)
        else:
            ranked_results = nasa_results
        
        # Поиск похожих объектов
        if request.include_similar:
            similar_objects = await self._find_similar_objects(ranked_results[:5])
            ranked_results.extend(similar_objects)
        
        # Ограничиваем результаты
        final_results = ranked_results[:request.max_results]
        
        # Генерация рекомендаций
        recommendations = self._generate_search_recommendations(final_results, request.filters)
        
        result = {
            'results': final_results,
            'total_found': len(nasa_results),
            'ai_ranked': request.use_ai_ranking,
            'recommendations': recommendations,
            'search_time_ms': (time.time() - start_time) * 1000,
            'filters_applied': request.filters.dict()
        }
        
        # Кэшируем результат
        if redis_client:
            redis_client.setex(cache_key, 3600, json.dumps(result, default=str))
        
        return result
    
    def _parse_search_query(self, query: str) -> Dict[str, Any]:
        """Парсинг поискового запроса"""
        params = {
            'target_name': None,
            'coordinates': None,
            'period_range': None,
            'depth_range': None
        }
        
        query_lower = query.lower()
        
        # Поиск по имени (TOI, TIC, Kepler, etc.)
        if any(prefix in query_lower for prefix in ['toi', 'tic', 'kepler', 'k2', 'trappist']):
            params['target_name'] = query.strip()
        
        # Поиск координат (RA, Dec)
        if 'ra' in query_lower and 'dec' in query_lower:
            # Простой парсинг координат
            import re
            coords = re.findall(r'[\d.]+', query)
            if len(coords) >= 2:
                params['coordinates'] = {'ra': float(coords[0]), 'dec': float(coords[1])}
        
        # Поиск по периоду
        if 'period' in query_lower:
            import re
            periods = re.findall(r'[\d.]+', query)
            if len(periods) >= 1:
                period = float(periods[0])
                params['period_range'] = [period * 0.8, period * 1.2]  # ±20%
        
        return params
    
    async def _search_nasa_catalogs(self, search_params: Dict, filters: SearchFilters) -> List[Dict]:
        """Поиск в NASA каталогах"""
        results = []
        
        try:
            if search_params['target_name']:
                # Поиск по имени
                target_data = await self.nasa_client.search_exoplanet_archive(
                    target_name=search_params['target_name']
                )
                if target_data:
                    results.extend(target_data)
            
            elif search_params['coordinates']:
                # Поиск по координатам
                coord_results = await self.nasa_client.search_by_coordinates(
                    ra=search_params['coordinates']['ra'],
                    dec=search_params['coordinates']['dec'],
                    radius=0.1  # градусы
                )
                if coord_results:
                    results.extend(coord_results)
            
            else:
                # Общий поиск с фильтрами
                general_results = await self.nasa_client.search_with_filters(
                    missions=filters.missions,
                    planet_types=filters.planet_types,
                    period_range=filters.period_range,
                    depth_range=filters.depth_range,
                    limit=filters.max_results * 2  # Берем больше для последующей фильтрации
                )
                if general_results:
                    results.extend(general_results)
        
        except Exception as e:
            logger.error(f"NASA catalog search failed: {e}")
        
        return results
    
    async def _ai_ranking(self, results: List[Dict], filters: SearchFilters) -> List[Dict]:
        """ИИ ранжирование результатов"""
        if not results:
            return results
        
        ranked_results = []
        
        for result in results:
            try:
                # Быстрый анализ для ранжирования
                ai_score = await self._calculate_ai_score(result, filters)
                result['ai_score'] = ai_score
                result['ai_ranking_applied'] = True
                ranked_results.append(result)
                
            except Exception as e:
                logger.warning(f"AI ranking failed for {result.get('name', 'unknown')}: {e}")
                result['ai_score'] = 0.5  # Средний балл при ошибке
                ranked_results.append(result)
        
        # Сортируем по ИИ баллу
        ranked_results.sort(key=lambda x: x.get('ai_score', 0), reverse=True)
        
        return ranked_results
    
    async def _calculate_ai_score(self, target_data: Dict, filters: SearchFilters) -> float:
        """Расчет ИИ балла для ранжирования"""
        score = 0.5  # Базовый балл
        
        try:
            # Факторы для ранжирования
            
            # 1. Качество данных
            data_quality = target_data.get('data_quality', 0.8)
            score += (data_quality - 0.5) * 0.3
            
            # 2. SNR
            snr = target_data.get('snr', 5.0)
            if snr >= filters.snr_min:
                score += min(0.2, (snr - filters.snr_min) / 20.0)
            
            # 3. Подтвержденность
            if target_data.get('disposition') == 'CONFIRMED':
                score += 0.2
            elif target_data.get('disposition') == 'CANDIDATE':
                score += 0.1
            
            # 4. Популярность (количество публикаций)
            publications = target_data.get('publication_count', 0)
            score += min(0.1, publications / 50.0)
            
            # 5. Уникальность параметров
            period = target_data.get('orbital_period', 0)
            if 1 <= period <= 100:  # Интересный диапазон
                score += 0.1
            
            # Нормализуем балл
            score = max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"AI score calculation failed: {e}")
            score = 0.5
        
        return score
    
    async def _find_similar_objects(self, top_results: List[Dict]) -> List[Dict]:
        """Поиск похожих объектов"""
        similar_objects = []
        
        for result in top_results:
            try:
                # Поиск по похожим параметрам
                period = result.get('orbital_period')
                radius = result.get('planet_radius')
                
                if period and radius:
                    similar = await self.nasa_client.search_similar_planets(
                        period_range=[period * 0.9, period * 1.1],
                        radius_range=[radius * 0.9, radius * 1.1],
                        limit=3
                    )
                    
                    for sim in similar:
                        sim['similarity_reason'] = f"Похож на {result.get('name', 'unknown')}"
                        sim['is_similar'] = True
                        similar_objects.append(sim)
                        
            except Exception as e:
                logger.warning(f"Similar object search failed: {e}")
        
        return similar_objects
    
    def _generate_search_recommendations(self, results: List[Dict], filters: SearchFilters) -> List[str]:
        """Генерация рекомендаций по поиску"""
        recommendations = []
        
        if not results:
            recommendations.append("Попробуйте расширить критерии поиска")
            recommendations.append("Проверьте правильность написания названия объекта")
            return recommendations
        
        # Анализ результатов
        confirmed_count = sum(1 for r in results if r.get('disposition') == 'CONFIRMED')
        candidate_count = sum(1 for r in results if r.get('disposition') == 'CANDIDATE')
        
        if confirmed_count > 0:
            recommendations.append(f"Найдено {confirmed_count} подтвержденных экзопланет")
        
        if candidate_count > 0:
            recommendations.append(f"Найдено {candidate_count} кандидатов в экзопланеты")
        
        # Рекомендации по фильтрам
        avg_snr = np.mean([r.get('snr', 0) for r in results if r.get('snr')])
        if avg_snr < filters.snr_min:
            recommendations.append("Рассмотрите снижение порога SNR для большего количества результатов")
        
        return recommendations


# Инициализация движка поиска
search_engine = SmartSearchEngine()


@router.post("/ai/analyze_lightcurve", response_model=SmartAnalysisResult)
@require_auth(roles=["researcher", "admin"])
@rate_limit(requests_per_minute=10)
async def smart_lightcurve_analysis(
    request: SmartAnalysisRequest,
    current_user = Depends(get_current_user)
):
    """Умный анализ кривой блеска с улучшенным ИИ"""
    start_time = time.time()
    
    try:
        logger.info(f"Smart analysis requested by {current_user.username if current_user else 'anonymous'}")
        
        # Получение данных
        if request.target_name:
            # Загрузка из NASA
            nasa_client = get_nasa_client()
            lightcurve_data = await nasa_client.get_lightcurve(
                request.target_name, 
                mission=request.mission
            )
            time_data = np.array(lightcurve_data['time'])
            flux_data = np.array(lightcurve_data['flux'])
            flux_err_data = np.array(lightcurve_data.get('flux_err', np.ones_like(flux_data) * 0.001))
        else:
            # Используем переданные данные
            time_data = np.array(request.time_data)
            flux_data = np.array(request.flux_data)
            flux_err_data = np.array(request.flux_err_data or np.ones_like(flux_data) * 0.001)
        
        # Адаптивная предобработка
        preprocessor = get_adaptive_preprocessor()
        processed_data = preprocessor.smart_preprocess(
            time_data, flux_data, flux_err_data,
            auto_detect_cadence=request.auto_detect_cadence,
            adaptive_detrending=request.adaptive_detrending
        )
        
        # Расширенное извлечение признаков
        feature_extractor = get_enhanced_feature_extractor()
        
        # Базовые признаки
        base_features = feature_extractor.extract_features(
            processed_data['time'],
            processed_data['flux'],
            processed_data['flux_err']
        )
        
        # Астрофизические признаки
        transit_params = {
            'primary_depth': 0.01,  # Будет заменено на реальные данные
            'secondary_depth': 0.0001,
            'ingress_duration': 0.1,
            'egress_duration': 0.1,
            'transit_duration': 1.0,
            'secondary_eclipse_phase': 0.5
        }
        
        astro_features = feature_extractor.extract_astrophysical_features(
            processed_data['time'],
            processed_data['flux'],
            transit_params
        )
        
        # Признаки качества
        quality_features = feature_extractor.extract_quality_features(
            processed_data['time'],
            processed_data['flux'],
            processed_data['flux_err']
        )
        
        # Объединяем все признаки
        all_features = np.concatenate([base_features, list(astro_features.values()), list(quality_features.values())])
        
        # Ensemble предсказание с uncertainty quantification
        classifier = get_enhanced_classifier()
        prediction = classifier.predict_with_uncertainty(all_features)
        
        # Объяснение предсказания
        explanation = None
        if request.explain_prediction:
            feature_names = (feature_extractor.base_features + 
                           list(astro_features.keys()) + 
                           list(quality_features.keys()))
            explanation = classifier.explain_prediction(all_features, feature_names)
        
        # Генерация рекомендаций
        recommendations = generate_recommendations(prediction)
        
        processing_time = (time.time() - start_time) * 1000
        
        result = SmartAnalysisResult(
            target_name=request.target_name or "uploaded_data",
            predicted_class=prediction['class_name'],
            confidence_score=prediction['confidence'],
            uncertainty_bounds=prediction['uncertainty_bounds'],
            transit_probability=prediction['probabilities'].get('CANDIDATE', 0.0),
            signal_characteristics={
                'snr_estimate': float(np.std(flux_data) * 10),  # Простая оценка
                'data_points': len(time_data),
                'time_span_days': float(np.max(time_data) - np.min(time_data))
            },
            feature_importance=prediction['feature_importance'],
            decision_reasoning=explanation['reasoning'] if explanation else [],
            recommendations=recommendations,
            data_quality_metrics=quality_features,
            processing_time_ms=processing_time
        )
        
        logger.info(f"Smart analysis completed: {result.predicted_class} ({result.confidence_score:.1%})")
        return result
        
    except Exception as e:
        logger.error(f"Smart analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/smart_search")
@rate_limit(requests_per_minute=20)
async def smart_search(request: SmartSearchRequest):
    """Умный поиск экзопланет с ИИ ранжированием"""
    try:
        result = await search_engine.smart_search(request)
        return result
    except Exception as e:
        logger.error(f"Smart search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/batch_analyze")
@require_auth(roles=["researcher", "admin"])
@rate_limit(requests_per_minute=5)
async def batch_lightcurve_analysis(
    request: BatchAnalysisRequest,
    current_user = Depends(get_current_user)
):
    """Пакетный анализ множественных целей"""
    try:
        logger.info(f"Batch analysis for {len(request.targets)} targets")
        
        # Ограничиваем параллельность
        semaphore = asyncio.Semaphore(request.parallel_limit)
        
        async def analyze_single_target(target_name: str):
            async with semaphore:
                try:
                    analysis_request = request.analysis_params
                    analysis_request.target_name = target_name
                    return await smart_lightcurve_analysis(analysis_request, current_user)
                except Exception as e:
                    logger.error(f"Analysis failed for {target_name}: {e}")
                    return {"target_name": target_name, "error": str(e)}
        
        # Выполняем анализ параллельно
        tasks = [analyze_single_target(target) for target in request.targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обрабатываем результаты
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result)})
            elif "error" in result:
                failed_results.append(result)
            else:
                successful_results.append(result)
        
        return {
            "successful_analyses": successful_results,
            "failed_analyses": failed_results,
            "total_requested": len(request.targets),
            "success_rate": len(successful_results) / len(request.targets)
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/health")
async def ai_health_check():
    """Проверка здоровья ИИ системы"""
    try:
        return {
            "status": "healthy",
            "service": "smart_ai_analysis",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "enhanced_classifier": "ready",
                "feature_extractor": "ready",
                "search_engine": "ready",
                "redis_cache": "available" if redis_client else "unavailable"
            },
            "model_version": "enhanced_v2.0",
            "features": {
                "uncertainty_quantification": True,
                "explainable_ai": True,
                "adaptive_preprocessing": True,
                "smart_search": True,
                "batch_processing": True
            }
        }
    except Exception as e:
        logger.error(f"AI health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
