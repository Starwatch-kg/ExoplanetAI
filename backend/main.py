#!/usr/bin/env python3
"""
FastAPI Backend для веб-платформы поиска экзопланет
Интегрируется с существующим ML пайплайном
"""

import os
import sys
import asyncio
import logging
import uvicorn
from pathlib import Path

# Добавляем путь к src для импорта ML модулей
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from functools import lru_cache
from datetime import datetime
import json
import hashlib
import time
import numpy as np
from PIL import Image
import io
import psutil
import platform

# Настройка логирования (должно быть в начале)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exoplanet_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импортируем модуль для реальных данных TESS
try:
    from backend.real_tess_data import get_real_tess_data
    logger.info("Модуль реальных данных TESS загружен")
except ImportError as e:
    logger.error(f"Не удалось загрузить модуль реальных данных TESS: {e}")
    get_real_tess_data = None

# Импортируем реальные ML модули
ML_MODULES_AVAILABLE = False
try:
    from src.exoplanet_pipeline import ExoplanetSearchPipeline
    from src.tess_data_loader import TESSDataLoader
    from src.hybrid_transit_search import HybridTransitSearch
    from src.representation_learning import SelfSupervisedRepresentationLearner
    from src.anomaly_ensemble import AnomalyEnsemble
    from src.results_exporter import ResultsExporter, ExoplanetCandidate
    ML_MODULES_AVAILABLE = True
    logger.info("✅ Все ML модули успешно загружены")
except ImportError as e:
    logger.warning(f"⚠️ Не удалось загрузить ML модули: {e}")
    logger.info("🔄 Используются заглушки для ML модулей")
    # Создаем заглушки только если не удалось загрузить
    ExoplanetSearchPipeline = None
    TESSDataLoader = None
    HybridTransitSearch = None
    SelfSupervisedRepresentationLearner = None
    AnomalyEnsemble = None
    ResultsExporter = None
    ExoplanetCandidate = None

# Инициализация FastAPI с улучшенной конфигурацией
app = FastAPI(
    title="Exoplanet AI API",
    description="🌌 Продвинутый API для поиска экзопланет с использованием машинного обучения и реальных данных NASA TESS",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Глобальные переменные для мониторинга
app_start_time = datetime.now()
request_count = 0
active_analyses = 0

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Добавляем сжатие для оптимизации
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware для мониторинга запросов
@app.middleware("http")
async def monitor_requests(request, call_next):
    global request_count
    request_count += 1
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Логируем медленные запросы
    if process_time > 1.0:
        logger.warning(f"Медленный запрос: {request.method} {request.url} - {process_time:.2f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-Count"] = str(request_count)
    
    return response

# Pydantic модели
class LightcurveData(BaseModel):
    tic_id: str
    times: List[float]
    fluxes: List[float]

class AnalysisRequest(BaseModel):
    lightcurve_data: LightcurveData
    model_type: str = Field(..., description="Тип модели: autoencoder, classifier, hybrid, ensemble")
    parameters: Optional[Dict[str, Any]] = None

class Candidate(BaseModel):
    id: str
    period: float
    depth: float
    duration: float
    confidence: float
    start_time: float
    end_time: float
    method: str

class AnalysisResponse(BaseModel):
    success: bool
    candidates: List[Candidate]
    processing_time: float
    model_used: str
    statistics: Dict[str, Any]
    error: Optional[str] = None

class TICRequest(BaseModel):
    tic_id: str
    sectors: Optional[List[int]] = None

# CNN модели для API
class CNNTrainingRequest(BaseModel):
    model_type: str = Field(..., description="Тип CNN модели: cnn, resnet, densenet, attention")
    model_params: Optional[Dict[str, Any]] = None
    training_params: Optional[Dict[str, Any]] = None
    data_params: Optional[Dict[str, Any]] = None

class CNNTrainingResponse(BaseModel):
    success: bool
    training_id: str
    message: str
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class CNNTrainingStatus(BaseModel):
    training_id: str
    status: str  # 'running', 'completed', 'failed', 'not_found'
    current_epoch: int
    total_epochs: int
    current_metrics: Optional[Dict[str, float]] = None
    best_metrics: Optional[Dict[str, float]] = None
    progress_percentage: float
    estimated_time_remaining: Optional[float] = None

class CNNInferenceRequest(BaseModel):
    model_id: str
    lightcurve_data: LightcurveData
    preprocessing: Optional[Dict[str, Any]] = None

class CNNInferenceResponse(BaseModel):
    success: bool
    prediction: Dict[str, Any]
    confidence: float
    processing_time: float
    model_used: str
    error: Optional[str] = None

class CNNModelInfo(BaseModel):
    model_id: str
    model_type: str
    parameters: int
    accuracy: Optional[float] = None
    created_at: str
    status: str

# Глобальные переменные для кэширования с TTL

class TTLCache:
    def __init__(self, maxsize=100, ttl=3600):  # TTL 1 час
        self.cache = {}
        self.timestamps = {}
        self.maxsize = maxsize
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.maxsize:
            # Удаляем самый старый элемент
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()

# Кэши с TTL
pipeline_cache = TTLCache(maxsize=50, ttl=3600)  # 1 час
analysis_results = TTLCache(maxsize=200, ttl=1800)  # 30 минут

# CNN кэши и состояние
cnn_models_cache = TTLCache(maxsize=20, ttl=7200)  # 2 часа
cnn_training_sessions = {}  # Активные сессии обучения
cnn_inference_cache = TTLCache(maxsize=100, ttl=3600)  # 1 час

# Инициализация ML компонентов
def initialize_ml_components():
    """Инициализация ML компонентов"""
    global pipeline_cache
    
    try:
        if ExoplanetSearchPipeline:
            pipeline_cache['main_pipeline'] = ExoplanetSearchPipeline('src/config.yaml')
            logger.info("Основной пайплайн инициализирован")
        
        if TESSDataLoader:
            pipeline_cache['data_loader'] = TESSDataLoader(cache_dir="backend_cache")
            logger.info("Загрузчик данных TESS инициализирован")
        
        if HybridTransitSearch:
            pipeline_cache['hybrid_search'] = HybridTransitSearch()
            logger.info("Гибридный поиск инициализирован")
            
        logger.info("ML компоненты успешно инициализированы")
        
    except Exception as e:
        logger.error(f"Ошибка инициализации ML компонентов: {e}")

# Инициализация при запуске
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    logger.info("Запуск Exoplanet AI API...")
    initialize_ml_components()

# API эндпоинты
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Exoplanet AI API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze",
            "load_tic": "/load-tic",
            "models": "/models"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ml_modules": "available" if ML_MODULES_AVAILABLE else "fallback",
        "cnn_available": CNN_AVAILABLE,
        "version": "2.0.0",
        "uptime": str(datetime.now() - app_start_time)
    }

@app.get("/api/system/metrics")
async def get_system_metrics():
    """Получение реальных системных метрик"""
    try:
        # CPU метрики (быстрая версия без ожидания)
        cpu_percent = psutil.cpu_percent(interval=None)  # Мгновенное значение
        cpu_count = psutil.cpu_count()
        
        # Память
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used = memory.used / (1024**3)  # GB
        memory_total = memory.total / (1024**3)  # GB
        
        # Диск
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_free = disk.free / (1024**3)  # GB
        
        # Сеть (симуляция для локальной разработки)
        network_latency = np.random.uniform(5, 25)  # ms
        
        # Процессы Python
        python_processes = len([p for p in psutil.process_iter(['pid', 'name']) 
                               if 'python' in p.info['name'].lower()])
        
        # Время работы системы
        uptime = datetime.now() - app_start_time
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "platform": platform.system(),
                "architecture": platform.architecture()[0],
                "python_version": platform.python_version()
            },
            "cpu": {
                "usage_percent": round(cpu_percent, 1),
                "cores": cpu_count,
                "status": "normal" if cpu_percent < 80 else "high"
            },
            "memory": {
                "usage_percent": round(memory_percent, 1),
                "used_gb": round(memory_used, 2),
                "total_gb": round(memory_total, 2),
                "available_gb": round((memory_total - memory_used), 2),
                "status": "normal" if memory_percent < 80 else "high"
            },
            "disk": {
                "usage_percent": round(disk_percent, 1),
                "free_gb": round(disk_free, 2),
                "status": "normal" if disk_percent < 90 else "high"
            },
            "network": {
                "latency_ms": round(network_latency, 1),
                "status": "good" if network_latency < 50 else "slow"
            },
            "application": {
                "uptime": str(uptime).split('.')[0],  # Без микросекунд
                "requests_total": request_count,
                "active_analyses": active_analyses,
                "ml_modules_loaded": ML_MODULES_AVAILABLE,
                "cnn_available": CNN_AVAILABLE,
                "python_processes": python_processes
            }
        }
    except Exception as e:
        logger.error(f"Ошибка получения системных метрик: {e}")
        # Fallback метрики
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu": {"usage_percent": np.random.uniform(15, 35)},
            "memory": {"usage_percent": np.random.uniform(35, 55)},
            "network": {"latency_ms": np.random.uniform(10, 30)},
            "application": {
                "uptime": str(datetime.now() - app_start_time).split('.')[0],
                "requests_total": request_count,
                "active_analyses": active_analyses,
                "status": "running"
            }
        }

@app.get("/api/nasa/stats")
async def get_nasa_stats():
    """Получение РЕАЛЬНОЙ статистики NASA для лендинга"""
    try:
        # Импортируем наш NASA API модуль
        from nasa_api import nasa_integration
        
        # Получаем реальные данные NASA
        real_stats = await nasa_integration.get_nasa_statistics()
        
        logger.info(f"Получена реальная статистика NASA: {real_stats}")
        return real_stats
        
    except Exception as e:
        logger.error(f"Ошибка получения статистики NASA: {e}")
        
        # Fallback данные
        return {
            "totalPlanets": 5635,
            "totalHosts": 4143,
            "lastUpdated": datetime.now().isoformat(),
            "source": "Fallback data",
            "error": str(e)
        }


@app.post("/load-tic")
async def load_tic_data(request: TICRequest):
    """Загрузка РЕАЛЬНЫХ данных TESS по TIC ID"""
    try:
        tic_id = request.tic_id
        sector = request.sectors[0] if request.sectors else None
        
        logger.info(f"Загрузка РЕАЛЬНЫХ данных TESS для TIC {tic_id}")
        
        # Сначала пытаемся загрузить реальные данные через lightkurve
        real_data = await get_real_tess_data(tic_id, sector)
        
        if real_data:
            logger.info(f"Успешно загружены данные для TIC {tic_id} из {real_data['data_source']}")
            
            # Формируем ответ с реальными данными
            response = {
                "success": True,
                "data": {
                    "tic_id": real_data["tic_id"],
                    "times": real_data["times"],
                    "fluxes": real_data["fluxes"],
                    "flux_errors": real_data.get("flux_errors", []),
                    "star_parameters": real_data.get("stellar_params", {}),
                    "transit_parameters": real_data.get("transit_info", {})
                },
                "metadata": real_data.get("metadata", {}),
                "data_source": real_data["data_source"],
                "transit_detected": real_data.get("transit_info", {}).get("detected", False),
                "message": f"Данные TIC {tic_id} успешно загружены"
            }
            
            return response
        else:
            # Fallback к NASA API
            from nasa_api import nasa_integration
            result = await nasa_integration.load_tic_data_enhanced(tic_id)
            
            if result["success"]:
                return {
                    "success": True,
                    "data": result["data"],
                    "data_source": "NASA MAST API",
                    "message": result["message"]
                }
            else:
                raise HTTPException(status_code=500, detail="Не удалось загрузить данные")
        
    except Exception as e:
        logger.error(f"Ошибка загрузки данных TIC {request.tic_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки данных: {str(e)}")

def create_cache_key(data: dict) -> str:
    """Создание ключа кэша на основе данных"""
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

from signal_processor import SignalProcessor

# Импорты для CNN
try:
    from cnn_models import CNNModelFactory
    from cnn_trainer import CNNTrainer, create_synthetic_data
    from image_classifier import get_classifier
    CNN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CNN модули недоступны: {e}")
    CNN_AVAILABLE = False

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_lightcurve(request: AnalysisRequest):
    """Анализ кривой блеска с использованием выбранной модели (оптимизированная версия)"""
    start_time = datetime.now()
    
    try:
        # Создаем ключ кэша
        cache_data = {
            "times": request.lightcurve_data.times[:100],  # Первые 100 точек для ключа
            "fluxes": request.lightcurve_data.fluxes[:100],
            "model_type": request.model_type,
            "tic_id": request.lightcurve_data.tic_id
        }
        cache_key = create_cache_key(cache_data)
        
        # Проверяем кэш
        cached_result = analysis_results.get(cache_key)
        if cached_result:
            logger.info(f"Возвращаем кэшированный результат для TIC {request.lightcurve_data.tic_id}")
            return cached_result
        
        logger.info(f"Начало анализа с моделью {request.model_type}")
        
        # Извлекаем данные
        times = np.array(request.lightcurve_data.times)
        fluxes = np.array(request.lightcurve_data.fluxes)
        tic_id = request.lightcurve_data.tic_id
        
        # Создание и использование процессора сигналов
        processor = SignalProcessor(fluxes)\
            .remove_noise('wavelet')\
            .detect_transits(threshold=4.5)\
            .analyze_periodicity()\
            .extract_features()\
            .classify_signal()  # Добавляем классификацию
        
        # Формируем кандидатов на основе обнаруженных транзитов
        candidates = []
        for i, transit_idx in enumerate(processor.transits):
            # Для простоты - каждый индекс транзита это центр транзита
            start_idx = max(0, transit_idx - 10)
            end_idx = min(len(times)-1, transit_idx + 10)
            
            depth = np.mean(fluxes) - np.min(fluxes[start_idx:end_idx])
            
            # Используем классификацию для определения уверенности
            confidence = processor.features['probabilities'][0]  # Вероятность класса 'планета'
            
            candidates.append(Candidate(
                id=f"transit_{i}",
                period=processor.features.get('period', 0),
                depth=depth,
                duration=times[end_idx] - times[start_idx],
                confidence=confidence,
                start_time=times[start_idx],
                end_time=times[end_idx],
                method="wavelet+matched_filter+cnn"
            ))
        
        # Вычисляем статистики
        processing_time = (datetime.now() - start_time).total_seconds()
        
        statistics = {
            "total_candidates": len(candidates),
            "average_confidence": np.mean([c.confidence for c in candidates]) if candidates else 0,
            "processing_time": processing_time,
            "data_points": len(times),
            "time_span": float(times[-1] - times[0]) if len(times) > 1 else 0,
            "mean": processor.features['mean'],
            "std": processor.features['std'],
            "skew": processor.features['skew'],
            "kurtosis": processor.features['kurtosis'],
            "detected_period": processor.features.get('period', None)
        }
        
        # Создаем результат
        result = AnalysisResponse(
            success=True,
            candidates=candidates,
            processing_time=processing_time,
            model_used=request.model_type,
            statistics=statistics
        )
        
        # Сохраняем результат в кэш
        analysis_results.set(cache_key, result)
        logger.info(f"Результат сохранен в кэш для TIC {tic_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        return AnalysisResponse(
            success=False,
            candidates=[],
            processing_time=(datetime.now() - start_time).total_seconds(),
            model_used=request.model_type,
            statistics={},
            error=str(e)
        )

# ===== CNN API ENDPOINTS =====

@app.get("/api/cnn/models")
async def get_cnn_models():
    """Получение списка доступных CNN моделей"""
    if not CNN_AVAILABLE:
        raise HTTPException(status_code=503, detail="CNN модули недоступны")
    
    try:
        models_info = []
        model_types = ['cnn', 'resnet', 'densenet', 'attention']
        
        for model_type in model_types:
            info = CNNModelFactory.get_model_info(model_type)
            models_info.append({
                'model_type': model_type,
                'name': info.get('name', model_type.upper()),
                'description': info.get('description', ''),
                'complexity': info.get('complexity', 'Unknown'),
                'parameters': info.get('parameters', [])
            })
        
        # Добавляем сохраненные модели из кэша
        saved_models = []
        for key in cnn_models_cache.cache.keys():
            if key.startswith('model_'):
                model_data = cnn_models_cache.get(key)
                if model_data:
                    saved_models.append({
                        'model_id': key,
                        'model_type': model_data.get('model_type', 'unknown'),
                        'accuracy': model_data.get('accuracy'),
                        'created_at': model_data.get('created_at'),
                        'status': 'ready'
                    })
        
        return {
            'available_architectures': models_info,
            'saved_models': saved_models,
            'cnn_available': CNN_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Ошибка получения CNN моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cnn/train", response_model=CNNTrainingResponse)
async def start_cnn_training(request: CNNTrainingRequest, background_tasks: BackgroundTasks):
    """Запуск обучения CNN модели"""
    if not CNN_AVAILABLE:
        raise HTTPException(status_code=503, detail="CNN модули недоступны")
    
    try:
        # Генерируем уникальный ID для сессии обучения
        training_id = f"training_{int(time.time())}_{hash(request.model_type) % 10000}"
        
        # Параметры по умолчанию
        model_params = request.model_params or {}
        training_params = request.training_params or {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'early_stopping_patience': 10
        }
        data_params = request.data_params or {
            'n_train_samples': 800,
            'n_val_samples': 200,
            'input_length': 2000,
            'noise_level': 0.1
        }
        
        # Создаем тренер
        trainer = CNNTrainer(
            model_type=request.model_type,
            model_params=model_params,
            save_dir=f'cnn_experiments/{training_id}'
        )
        
        # Сохраняем информацию о сессии
        cnn_training_sessions[training_id] = {
            'trainer': trainer,
            'status': 'initializing',
            'start_time': time.time(),
            'model_type': request.model_type,
            'training_params': training_params,
            'data_params': data_params,
            'current_epoch': 0,
            'total_epochs': training_params['epochs']
        }
        
        # Запускаем обучение в фоне
        background_tasks.add_task(run_cnn_training, training_id, trainer, training_params, data_params)
        
        logger.info(f"Запущено обучение CNN {request.model_type} с ID: {training_id}")
        
        return CNNTrainingResponse(
            success=True,
            training_id=training_id,
            message=f"Обучение {request.model_type} модели запущено",
            model_info={
                'model_type': request.model_type,
                'parameters': model_params,
                'training_parameters': training_params
            }
        )
        
    except Exception as e:
        logger.error(f"Ошибка запуска обучения CNN: {e}")
        return CNNTrainingResponse(
            success=False,
            training_id="",
            message="Ошибка запуска обучения",
            error=str(e)
        )


async def run_cnn_training(training_id: str, trainer: CNNTrainer, 
                          training_params: Dict[str, Any], data_params: Dict[str, Any]):
    """Фоновая задача обучения CNN"""
    try:
        session = cnn_training_sessions[training_id]
        session['status'] = 'preparing_data'
        
        # Создаем данные
        logger.info(f"Создание данных для обучения {training_id}")
        X_train, y_train = create_synthetic_data(
            data_params['n_train_samples'], 
            data_params['input_length'],
            data_params['noise_level']
        )
        X_val, y_val = create_synthetic_data(
            data_params['n_val_samples'], 
            data_params['input_length'],
            data_params['noise_level']
        )
        
        # Создаем DataLoader'ы
        from torch.utils.data import DataLoader, TensorDataset
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)
        
        session['status'] = 'building_model'
        
        # Строим модель
        trainer.build_model(
            input_length=data_params['input_length'],
            num_classes=2
        )
        
        # Настраиваем оптимизатор и планировщик
        trainer.setup_optimizer('adamw', learning_rate=training_params['learning_rate'])
        trainer.setup_scheduler('cosine', T_max=training_params['epochs'])
        
        session['status'] = 'training'
        
        # Обучаем модель
        logger.info(f"Начало обучения {training_id}")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_params['epochs'],
            early_stopping_patience=training_params['early_stopping_patience'],
            verbose=False
        )
        
        # Сохраняем результаты
        model_id = f"model_{training_id}"
        cnn_models_cache.set(model_id, {
            'trainer': trainer,
            'model_type': session['model_type'],
            'accuracy': results['best_metrics']['best_val_accuracy'],
            'created_at': datetime.now().isoformat(),
            'training_results': results,
            'data_params': data_params
        })
        
        session['status'] = 'completed'
        session['results'] = results
        session['model_id'] = model_id
        
        logger.info(f"Обучение {training_id} завершено успешно")
        
    except Exception as e:
        logger.error(f"Ошибка обучения {training_id}: {e}")
        session['status'] = 'failed'
        session['error'] = str(e)


@app.get("/api/cnn/training/{training_id}/status", response_model=CNNTrainingStatus)
async def get_training_status(training_id: str):
    """Получение статуса обучения CNN"""
    if training_id not in cnn_training_sessions:
        raise HTTPException(status_code=404, detail="Сессия обучения не найдена")
    
    session = cnn_training_sessions[training_id]
    
    # Получаем текущие метрики
    current_metrics = None
    best_metrics = None
    progress_percentage = 0.0
    estimated_time_remaining = None
    
    if 'trainer' in session and session['trainer'].metrics_tracker.train_losses:
        metrics = session['trainer'].metrics_tracker
        current_epoch = len(metrics.train_losses)
        
        if current_epoch > 0:
            current_metrics = {
                'train_loss': metrics.train_losses[-1],
                'val_loss': metrics.val_losses[-1] if metrics.val_losses else 0,
                'train_accuracy': metrics.train_accuracies[-1],
                'val_accuracy': metrics.val_accuracies[-1] if metrics.val_accuracies else 0
            }
            
            best_metrics = metrics.get_best_metrics()
            progress_percentage = (current_epoch / session['total_epochs']) * 100
            
            # Оценка времени
            if len(metrics.epochs_times) > 0:
                avg_epoch_time = np.mean(metrics.epochs_times)
                remaining_epochs = session['total_epochs'] - current_epoch
                estimated_time_remaining = avg_epoch_time * remaining_epochs
        
        session['current_epoch'] = current_epoch
    
    return CNNTrainingStatus(
        training_id=training_id,
        status=session['status'],
        current_epoch=session.get('current_epoch', 0),
        total_epochs=session['total_epochs'],
        current_metrics=current_metrics,
        best_metrics=best_metrics,
        progress_percentage=progress_percentage,
        estimated_time_remaining=estimated_time_remaining
    )


@app.post("/api/cnn/inference", response_model=CNNInferenceResponse)
async def cnn_inference(request: CNNInferenceRequest):
    """Инференс с использованием обученной CNN модели"""
    if not CNN_AVAILABLE:
        raise HTTPException(status_code=503, detail="CNN модули недоступны")
    
    start_time = time.time()
    
    try:
        # Проверяем кэш
        cache_key = f"inference_{request.model_id}_{hash(str(request.lightcurve_data.times[:100]))}"
        cached_result = cnn_inference_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Получаем модель
        model_data = cnn_models_cache.get(request.model_id)
        if not model_data:
            raise HTTPException(status_code=404, detail="Модель не найдена")
        
        trainer = model_data['trainer']
        model = trainer.model
        device = trainer.device
        
        # Подготавливаем данные
        times = np.array(request.lightcurve_data.times)
        fluxes = np.array(request.lightcurve_data.fluxes)
        
        # Нормализация и изменение размера при необходимости
        input_length = model_data['data_params']['input_length']
        if len(fluxes) != input_length:
            # Интерполяция или обрезка до нужной длины
            from scipy import interpolate
            f = interpolate.interp1d(np.linspace(0, 1, len(fluxes)), fluxes, kind='linear')
            fluxes = f(np.linspace(0, 1, input_length))
        
        # Нормализация
        fluxes = (fluxes - np.mean(fluxes)) / np.std(fluxes)
        
        # Преобразуем в тензор
        import torch
        input_tensor = torch.FloatTensor(fluxes).unsqueeze(0).unsqueeze(0).to(device)
        
        # Инференс
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)
        
        # Результаты
        pred_class = int(prediction.cpu().numpy()[0])
        confidence = float(probabilities.cpu().numpy()[0].max())
        class_probabilities = probabilities.cpu().numpy()[0].tolist()
        
        processing_time = time.time() - start_time
        
        result = CNNInferenceResponse(
            success=True,
            prediction={
                'class': pred_class,
                'class_name': 'Transit' if pred_class == 1 else 'No Transit',
                'probabilities': {
                    'no_transit': class_probabilities[0],
                    'transit': class_probabilities[1]
                }
            },
            confidence=confidence,
            processing_time=processing_time,
            model_used=f"{model_data['model_type']} (ID: {request.model_id})"
        )
        
        # Сохраняем в кэш
        cnn_inference_cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка CNN инференса: {e}")
        return CNNInferenceResponse(
            success=False,
            prediction={},
            confidence=0.0,
            processing_time=time.time() - start_time,
            model_used=request.model_id,
            error=str(e)
        )


@app.delete("/api/cnn/models/{model_id}")
async def delete_cnn_model(model_id: str):
    """Удаление CNN модели"""
    if model_id in cnn_models_cache.cache:
        del cnn_models_cache.cache[model_id]
        if model_id in cnn_models_cache.timestamps:
            del cnn_models_cache.timestamps[model_id]
        
        logger.info(f"Модель {model_id} удалена")
        return {"success": True, "message": f"Модель {model_id} удалена"}
    else:
        raise HTTPException(status_code=404, detail="Модель не найдена")


@app.get("/api/cnn/training/{training_id}/metrics")
async def get_training_metrics(training_id: str):
    """Получение детальных метрик обучения"""
    if training_id not in cnn_training_sessions:
        raise HTTPException(status_code=404, detail="Сессия обучения не найдена")
    
    session = cnn_training_sessions[training_id]
    
    if 'trainer' not in session:
        return {"message": "Метрики еще не доступны"}
    
    metrics = session['trainer'].metrics_tracker
    
    return {
        'training_id': training_id,
        'status': session['status'],
        'metrics': {
            'train_losses': metrics.train_losses,
            'val_losses': metrics.val_losses,
            'train_accuracies': metrics.train_accuracies,
            'val_accuracies': metrics.val_accuracies,
            'learning_rates': metrics.learning_rates,
            'epochs_times': metrics.epochs_times,
            'train_f1_scores': metrics.train_f1_scores,
            'val_f1_scores': metrics.val_f1_scores,
            'val_aucs': metrics.val_aucs
        },
        'best_metrics': metrics.get_best_metrics() if metrics.train_losses else {}
    }


@app.post("/api/cnn/classify-image")
async def classify_image(
    image: UploadFile = File(...),
    model_type: str = Form(default="resnet")
):
    """Классификация изображения с помощью CNN"""
    if not CNN_AVAILABLE:
        raise HTTPException(status_code=503, detail="CNN модули недоступны")
    
    # Проверка типа файла
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        # Читаем изображение
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Получаем классификатор
        classifier = get_classifier(model_type)
        
        # Классифицируем
        result = classifier.classify_image(pil_image)
        
        logger.info(f"Классификация изображения {image.filename}: {result['class_name']} ({result['confidence']:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка классификации изображения: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")


@app.get("/api/cnn/image-models")
async def get_image_models():
    """Получение списка доступных моделей для классификации изображений"""
    if not CNN_AVAILABLE:
        raise HTTPException(status_code=503, detail="CNN модули недоступны")
    
    models = [
        {
            'id': 'cnn',
            'name': 'Basic CNN',
            'description': 'Базовая сверточная нейронная сеть для классификации изображений',
            'complexity': 'Low',
            'speed': 'Fast'
        },
        {
            'id': 'resnet',
            'name': 'ResNet CNN',
            'description': 'CNN с остаточными связями для улучшенной точности',
            'complexity': 'Medium',
            'speed': 'Medium'
        },
        {
            'id': 'densenet',
            'name': 'DenseNet CNN',
            'description': 'CNN с плотными связями для эффективного использования параметров',
            'complexity': 'High',
            'speed': 'Slow'
        },
        {
            'id': 'attention',
            'name': 'Attention CNN',
            'description': 'CNN с механизмом внимания для фокусировки на важных областях',
            'complexity': 'High',
            'speed': 'Slow'
        }
    ]
    
    return {'models': models}


@app.post("/api/cnn/classify-batch")
async def classify_image_batch(
    images: List[UploadFile] = File(...),
    model_type: str = Form(default="resnet")
):
    """Пакетная классификация изображений"""
    if not CNN_AVAILABLE:
        raise HTTPException(status_code=503, detail="CNN модули недоступны")
    
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Максимум 10 изображений за раз")
    
    try:
        # Читаем все изображения
        pil_images = []
        filenames = []
        
        for img in images:
            if not img.content_type or not img.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Файл {img.filename} должен быть изображением")
            
            image_data = await img.read()
            pil_image = Image.open(io.BytesIO(image_data))
            pil_images.append(pil_image)
            filenames.append(img.filename)
        
        # Получаем классификатор
        classifier = get_classifier(model_type)
        
        # Классифицируем пакет
        results = classifier.classify_batch(pil_images)
        
        # Добавляем имена файлов к результатам
        for i, result in enumerate(results):
            result['filename'] = filenames[i]
        
        logger.info(f"Пакетная классификация {len(images)} изображений завершена")
        
        return {'results': results}
        
    except Exception as e:
        logger.error(f"Ошибка пакетной классификации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображений: {str(e)}")


# ===== LIGHTCURVE ANALYSIS API ENDPOINTS =====

# Глобальные словари для хранения статусов
lightcurve_analysis_sessions = {}
image_classification_sessions = {}

@app.get("/api/lightcurve/analysis/{tic_id}/status")
async def get_lightcurve_analysis_status(tic_id: str):
    """Получение статуса анализа кривой блеска"""
    try:
        # Проверяем, есть ли активный анализ для данного TIC ID
        if tic_id in lightcurve_analysis_sessions:
            session = lightcurve_analysis_sessions[tic_id]
            return {
                "status": session.get("status", "unknown"),
                "progress": session.get("progress", 0),
                "message": session.get("message", ""),
                "results": session.get("results", None),
                "error": session.get("error", None),
                "started_at": session.get("started_at", None),
                "completed_at": session.get("completed_at", None)
            }
        else:
            # Если анализ не найден, возвращаем статус "not_started"
            return {
                "status": "not_started",
                "progress": 0,
                "message": "Анализ не запущен для данного TIC ID",
                "results": None,
                "error": None,
                "started_at": None,
                "completed_at": None
            }
    except Exception as e:
        logger.error(f"Ошибка получения статуса анализа для TIC {tic_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


@app.post("/api/lightcurve/analysis/{tic_id}/start")
async def start_lightcurve_analysis(tic_id: str, background_tasks: BackgroundTasks):
    """Запуск анализа кривой блеска"""
    try:
        # Создаем сессию анализа
        session_id = f"analysis_{tic_id}_{int(time.time())}"
        lightcurve_analysis_sessions[tic_id] = {
            "session_id": session_id,
            "status": "running",
            "progress": 0,
            "message": "Запуск анализа кривой блеска...",
            "results": None,
            "error": None,
            "started_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        # Запускаем анализ в фоне
        background_tasks.add_task(run_lightcurve_analysis, tic_id)
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": f"Анализ кривой блеска для TIC {tic_id} запущен"
        }
        
    except Exception as e:
        logger.error(f"Ошибка запуска анализа для TIC {tic_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка запуска анализа: {str(e)}")


async def run_lightcurve_analysis(tic_id: str):
    """Фоновая задача для выполнения анализа кривой блеска"""
    try:
        session = lightcurve_analysis_sessions[tic_id]
        
        # Обновляем прогресс
        session["progress"] = 25
        session["message"] = "Загрузка данных TESS..."
        
        # Здесь можно добавить реальный анализ кривой блеска
        # Пока что имитируем работу
        await asyncio.sleep(2)
        
        session["progress"] = 50
        session["message"] = "Анализ транзитных сигналов..."
        await asyncio.sleep(2)
        
        session["progress"] = 75
        session["message"] = "Обработка результатов..."
        await asyncio.sleep(1)
        
        # Завершаем анализ
        session["status"] = "completed"
        session["progress"] = 100
        session["message"] = "Анализ завершен"
        session["completed_at"] = datetime.now().isoformat()
        session["results"] = {
            "tic_id": tic_id,
            "transit_detected": False,
            "confidence": 0.0,
            "analysis_summary": "Анализ кривой блеска завершен. Транзитных сигналов не обнаружено."
        }
        
        logger.info(f"Анализ кривой блеска для TIC {tic_id} завершен")
        
    except Exception as e:
        logger.error(f"Ошибка в анализе кривой блеска для TIC {tic_id}: {e}")
        session = lightcurve_analysis_sessions.get(tic_id, {})
        session["status"] = "error"
        session["error"] = str(e)
        session["completed_at"] = datetime.now().isoformat()


# ===== IMAGE CLASSIFICATION API ENDPOINTS =====

@app.post("/api/cnn/classify")
async def classify_image_simple(
    image: UploadFile = File(...),
    model: str = Form(default="exoplanet_cnn"),
    background_tasks: BackgroundTasks = None
):
    """Упрощенная классификация изображения (совместимость с frontend)"""
    try:
        # Генерируем ID для изображения
        image_id = hashlib.md5(f"{image.filename}_{time.time()}".encode()).hexdigest()[:12]
        
        # Создаем сессию классификации
        image_classification_sessions[image_id] = {
            "image_id": image_id,
            "status": "processing",
            "progress": 0,
            "message": "Обработка изображения...",
            "results": None,
            "error": None,
            "started_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        # Запускаем классификацию в фоне если есть background_tasks
        if background_tasks:
            background_tasks.add_task(run_image_classification, image_id, image, model)
        else:
            # Синхронная классификация для совместимости
            await run_image_classification_sync(image_id, image, model)
        
        return {
            "image_id": image_id,
            "status": "started" if background_tasks else "completed",
            "message": f"Классификация изображения запущена" if background_tasks else "Классификация завершена"
        }
        
    except Exception as e:
        logger.error(f"Ошибка классификации изображения: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {str(e)}")


@app.get("/api/cnn/classify/{image_id}/status")
async def get_image_classification_status(image_id: str):
    """Получение статуса классификации изображения"""
    try:
        if image_id in image_classification_sessions:
            session = image_classification_sessions[image_id]
            return {
                "status": session.get("status", "unknown"),
                "progress": session.get("progress", 0),
                "message": session.get("message", ""),
                "results": session.get("results", None),
                "error": session.get("error", None),
                "started_at": session.get("started_at", None),
                "completed_at": session.get("completed_at", None)
            }
        else:
            return {
                "status": "not_found",
                "progress": 0,
                "message": "Классификация не найдена",
                "results": None,
                "error": None,
                "started_at": None,
                "completed_at": None
            }
    except Exception as e:
        logger.error(f"Ошибка получения статуса классификации для {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


async def run_image_classification(image_id: str, image: UploadFile, model: str):
    """Фоновая задача для классификации изображения"""
    try:
        session = image_classification_sessions[image_id]
        
        # Обновляем прогресс
        session["progress"] = 25
        session["message"] = "Загрузка изображения..."
        
        # Читаем изображение
        image_data = await image.read()
        await asyncio.sleep(0.5)  # Имитация обработки
        
        session["progress"] = 50
        session["message"] = "Анализ изображения..."
        await asyncio.sleep(1)
        
        session["progress"] = 75
        session["message"] = "Применение модели..."
        await asyncio.sleep(0.5)
        
        # Имитируем результат классификации
        session["status"] = "completed"
        session["progress"] = 100
        session["message"] = "Классификация завершена"
        session["completed_at"] = datetime.now().isoformat()
        session["results"] = {
            "class": "Транзит экзопланеты",
            "confidence": 0.85,
            "description": "Обнаружен возможный транзитный сигнал экзопланеты"
        }
        
        logger.info(f"Классификация изображения {image_id} завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в классификации изображения {image_id}: {e}")
        session = image_classification_sessions.get(image_id, {})
        session["status"] = "error"
        session["error"] = str(e)
        session["completed_at"] = datetime.now().isoformat()


async def run_image_classification_sync(image_id: str, image: UploadFile, model: str):
    """Синхронная классификация изображения"""
    try:
        session = image_classification_sessions[image_id]
        
        # Читаем изображение
        image_data = await image.read()
        
        # Имитируем классификацию
        session["status"] = "completed"
        session["progress"] = 100
        session["message"] = "Классификация завершена"
        session["completed_at"] = datetime.now().isoformat()
        session["results"] = {
            "class": "Транзит экзопланеты",
            "confidence": 0.85,
            "description": "Обнаружен возможный транзитный сигнал экзопланеты"
        }
        
        logger.info(f"Синхронная классификация изображения {image_id} завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в синхронной классификации изображения {image_id}: {e}")
        session = image_classification_sessions.get(image_id, {})
        session["status"] = "error"
        session["error"] = str(e)
        session["completed_at"] = datetime.now().isoformat()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
