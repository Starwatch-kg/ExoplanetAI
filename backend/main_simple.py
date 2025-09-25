"""
Simple Exoplanet AI Backend - Stable Version
Упрощенная стабильная версия backend сервера
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Модели данных для API
class SearchRequest(BaseModel):
    """Запрос на поиск экзопланет"""
    target_name: str = Field(..., min_length=1, max_length=100, description="Название цели")
    catalog: str = Field("TIC", pattern="^(TIC|KIC|EPIC)$", description="Каталог")
    mission: str = Field("TESS", pattern="^(TESS|Kepler|K2)$", description="Миссия")
    use_bls: bool = Field(True, description="Использовать BLS анализ")
    use_ai: bool = Field(True, description="Использовать ИИ анализ")
    use_ensemble: bool = Field(True, description="Использовать ensemble поиск")
    search_mode: str = Field("ensemble", pattern="^(single|ensemble|comprehensive)$", description="Режим поиска")
    period_min: float = Field(0.5, ge=0.1, le=100.0, description="Минимальный период (дни)")
    period_max: float = Field(20.0, ge=0.1, le=100.0, description="Максимальный период (дни)")
    snr_threshold: float = Field(7.0, ge=3.0, le=20.0, description="Порог SNR")

class SearchResponse(BaseModel):
    """Ответ поиска экзопланет"""
    target_name: str
    catalog: str
    mission: str
    bls_result: Optional[Dict[str, Any]] = None
    ai_result: Optional[Dict[str, Any]] = None
    lightcurve_info: Dict[str, Any]
    star_info: Dict[str, Any]
    candidates_found: int
    processing_time_ms: float
    status: str
    request_id: Optional[str] = None

class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str
    timestamp: str
    version: str
    environment: str
    services: Dict[str, str]

class BLSRequest(BaseModel):
    """Запрос на BLS анализ"""
    target_name: str = Field(..., min_length=1, max_length=100, description="Название цели")
    catalog: str = Field("TIC", pattern="^(TIC|KIC|EPIC)$", description="Каталог")
    mission: str = Field("TESS", pattern="^(TESS|Kepler|K2)$", description="Миссия")
    period_min: float = Field(0.5, ge=0.1, le=100.0, description="Минимальный период (дни)")
    period_max: float = Field(20.0, ge=0.1, le=100.0, description="Максимальный период (дни)")
    duration_min: float = Field(0.05, ge=0.01, le=1.0, description="Минимальная длительность (дни)")
    duration_max: float = Field(0.3, ge=0.01, le=1.0, description="Максимальная длительность (дни)")
    snr_threshold: float = Field(7.0, ge=3.0, le=20.0, description="Порог SNR")
    use_enhanced: bool = Field(True, description="Использовать расширенный анализ")

class BLSResponse(BaseModel):
    """Ответ BLS анализа"""
    target_name: str
    best_period: float
    best_t0: float
    best_duration: float
    best_power: float
    snr: float
    depth: float
    depth_err: float
    significance: float
    is_significant: bool
    enhanced_analysis: bool
    ml_confidence: float
    physical_validation: bool
    processing_time_ms: float
    request_id: Optional[str] = None

class CatalogResponse(BaseModel):
    """Ответ каталога"""
    catalogs: List[str]
    missions: List[str]
    description: Dict[str, str]

# Создание приложения
app = FastAPI(
    title="Exoplanet AI - Simple Stable Backend",
    description="""
    🌌 **Стабильная версия системы обнаружения экзопланет**
    
    Упрощенная версия для надежной работы:
    - 🔍 Поиск экзопланет с BLS анализом
    - 🤖 Симуляция ИИ анализа
    - 📊 Генерация реалистичных данных
    - 🛡️ Полная поддержка CORS
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - максимально открытая конфигурация для разработки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],  # Разрешаем все заголовки
    expose_headers=["*"]
)

# Gzip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ===== ENDPOINTS =====

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "service": "Exoplanet AI Simple",
        "version": "1.0.0",
        "status": "active",
        "environment": "development",
        "message": "🌌 Stable Exoplanet Detection System"
    }

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния системы"""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        version="1.0.0",
        environment="development",
        services={
            "api": "healthy",
            "bls": "healthy",
            "ai": "simulated",
            "cors": "enabled"
        }
    )

@app.get("/api/v1/test-cors")
async def test_cors(request: Request):
    """Тест CORS"""
    return {
        "message": "CORS работает отлично! ✅",
        "timestamp": time.time(),
        "origin": request.headers.get("origin", "unknown"),
        "method": request.method,
        "url": str(request.url),
        "status": "success"
    }

@app.options("/api/v1/{path:path}")
async def options_handler():
    """Обработчик OPTIONS запросов для CORS preflight"""
    return {"message": "OK"}

@app.post("/api/v1/search", response_model=SearchResponse)
async def search_exoplanets(request_data: SearchRequest):
    """
    🔍 ПОИСК ЭКЗОПЛАНЕТ
    
    Стабильный поиск с симуляцией BLS и ИИ анализа
    """
    print(f"🔍 Поиск экзопланет для цели: {request_data.target_name}")
    
    start_time = time.time()
    
    try:
        # Генерируем детерминированные, но реалистичные результаты
        np.random.seed(hash(request_data.target_name) % 2**32)
        
        bls_result = None
        ai_result = None
        candidates_found = 0
        
        # BLS анализ (симуляция)
        if request_data.use_bls:
            best_period = np.random.uniform(request_data.period_min, request_data.period_max)
            snr = np.random.uniform(5.0, 15.0)
            depth = np.random.uniform(0.0005, 0.005)
            significance = np.random.uniform(0.8, 0.99)
            is_significant = snr > request_data.snr_threshold
            
            bls_result = {
                "best_period": round(best_period, 4),
                "best_t0": round(np.random.uniform(0.0, best_period), 4),
                "best_duration": round(np.random.uniform(0.05, 0.2), 4),
                "snr": round(snr, 2),
                "depth": round(depth, 6),
                "significance": round(significance, 3),
                "is_significant": is_significant,
                "ml_confidence": round(np.random.uniform(0.6, 0.95), 3)
            }
            
            if is_significant:
                candidates_found += 1
        
        # ИИ анализ (симуляция)
        if request_data.use_ai:
            prediction = np.random.uniform(0.3, 0.9)
            confidence = np.random.uniform(0.6, 0.95)
            is_candidate = prediction > 0.7
            
            ai_result = {
                "prediction": round(prediction, 3),
                "confidence": round(confidence, 3),
                "is_candidate": is_candidate,
                "model_used": "ensemble_simulation",
                "inference_time_ms": round(np.random.uniform(50, 200), 1)
            }
            
            if is_candidate:
                candidates_found += 1
        
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Формируем ответ
        response = SearchResponse(
            target_name=request_data.target_name,
            catalog=request_data.catalog,
            mission=request_data.mission,
            bls_result=bls_result,
            ai_result=ai_result,
            lightcurve_info={
                "points_count": 1000,
                "time_span_days": 30.0,
                "cadence_minutes": 30.0,
                "noise_level_ppm": 1000.0,
                "data_source": "simulation"
            },
            star_info={
                "target_id": request_data.target_name,
                "ra": round(np.random.uniform(0, 360), 3),
                "dec": round(np.random.uniform(-90, 90), 3),
                "magnitude": round(np.random.uniform(8, 16), 2),
                "temperature": round(np.random.uniform(3500, 7000), 0),
                "radius": round(np.random.uniform(0.5, 2.0), 2),
                "mass": round(np.random.uniform(0.5, 1.5), 2),
                "stellar_type": np.random.choice(["G", "K", "M", "F"])
            },
            candidates_found=candidates_found,
            processing_time_ms=processing_time_ms,
            status="success",
            request_id=f"req_{int(time.time())}"
        )
        
        print(f"✅ Поиск завершен: найдено {candidates_found} кандидатов за {processing_time_ms}ms")
        return response
        
    except Exception as e:
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        print(f"❌ Ошибка поиска: {e}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/api/v1/bls", response_model=BLSResponse)
async def analyze_bls(request_data: BLSRequest):
    """
    📊 BLS АНАЛИЗ
    
    Box Least Squares анализ для поиска транзитов
    """
    print(f"📊 BLS анализ для цели: {request_data.target_name}")
    
    start_time = time.time()
    
    try:
        # Генерируем детерминированные результаты BLS
        np.random.seed(hash(request_data.target_name) % 2**32)
        
        best_period = np.random.uniform(request_data.period_min, request_data.period_max)
        best_power = np.random.uniform(15.0, 30.0)
        snr = np.random.uniform(5.0, 15.0)
        depth = np.random.uniform(0.001, 0.008)
        significance = np.random.uniform(0.85, 0.99)
        is_significant = snr > request_data.snr_threshold
        
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        response = BLSResponse(
            target_name=request_data.target_name,
            best_period=round(best_period, 6),
            best_t0=round(np.random.uniform(0.0, best_period), 6),
            best_duration=round(np.random.uniform(request_data.duration_min, request_data.duration_max), 6),
            best_power=round(best_power, 6),
            snr=round(snr, 2),
            depth=round(depth, 6),
            depth_err=round(depth * 0.1, 6),
            significance=round(significance, 3),
            is_significant=is_significant,
            enhanced_analysis=request_data.use_enhanced,
            ml_confidence=round(np.random.uniform(0.7, 0.95), 3),
            physical_validation=is_significant,
            processing_time_ms=processing_time_ms,
            request_id=f"bls_{int(time.time())}"
        )
        
        print(f"✅ BLS анализ завершен: P={response.best_period:.3f}d, SNR={response.snr:.1f}")
        return response
        
    except Exception as e:
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        print(f"❌ Ошибка BLS анализа: {e}")
        
        raise HTTPException(
            status_code=500,
            detail=f"BLS analysis failed: {str(e)}"
        )

@app.get("/api/v1/catalogs", response_model=CatalogResponse)
async def get_catalogs():
    """
    📚 КАТАЛОГИ
    
    Получение списка доступных каталогов и миссий
    """
    return CatalogResponse(
        catalogs=["TIC", "KIC", "EPIC"],
        missions=["TESS", "Kepler", "K2"],
        description={
            "TIC": "TESS Input Catalog - каталог целей для миссии TESS",
            "KIC": "Kepler Input Catalog - каталог целей для миссии Kepler",
            "EPIC": "Ecliptic Plane Input Catalog - каталог целей для миссии K2",
            "TESS": "Transiting Exoplanet Survey Satellite",
            "Kepler": "Kepler Space Telescope",
            "K2": "K2 Mission (extended Kepler mission)"
        }
    )

# Запуск приложения
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 STARTING EXOPLANET AI - SIMPLE STABLE VERSION")
    print("=" * 80)
    print("🌐 Host: 0.0.0.0")
    print("🔌 Port: 8000")
    print("🔄 Reload: True")
    print("📊 Docs: http://localhost:8000/docs")
    print("🔍 API: http://localhost:8000/api/v1/")
    print("🧪 CORS Test: http://localhost:8000/api/v1/test-cors")
    print("=" * 80)
    
    try:
        uvicorn.run(
            "main_simple:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        raise
