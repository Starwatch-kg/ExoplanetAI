"""
Prometheus Metrics System
Система метрик Prometheus для мониторинга приложения
"""

import time
from typing import Dict, Any, Optional, List
from functools import wraps
from contextlib import contextmanager
from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logging_config import get_logger

logger = get_logger(__name__)

# Создаем отдельный registry для изоляции метрик
REGISTRY = CollectorRegistry()

# ===== HTTP МЕТРИКИ =====

# Счетчик HTTP запросов
http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

# Гистограмма времени ответа HTTP
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
    registry=REGISTRY
)

# Размер HTTP запросов
http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=REGISTRY
)

# Размер HTTP ответов
http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

# Активные соединения
http_active_connections = Gauge(
    'http_active_connections',
    'Number of active HTTP connections',
    registry=REGISTRY
)

# ===== БИЗНЕС МЕТРИКИ =====

# Анализы экзопланет
exoplanet_analyses_total = Counter(
    'exoplanet_analyses_total',
    'Total number of exoplanet analyses',
    ['catalog', 'mission', 'status'],
    registry=REGISTRY
)

# Время выполнения BLS анализа
bls_analysis_duration_seconds = Histogram(
    'bls_analysis_duration_seconds',
    'BLS analysis duration in seconds',
    ['target_catalog'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
    registry=REGISTRY
)

# Найденные кандидаты
planet_candidates_found = Counter(
    'planet_candidates_found_total',
    'Total number of planet candidates found',
    ['catalog', 'mission', 'significance_level'],
    registry=REGISTRY
)

# SNR распределение
candidate_snr_distribution = Histogram(
    'candidate_snr_distribution',
    'Distribution of candidate SNR values',
    buckets=[3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0],
    registry=REGISTRY
)

# ===== ML МЕТРИКИ =====

# ML инференсы
ml_inferences_total = Counter(
    'ml_inferences_total',
    'Total number of ML inferences',
    ['model_name', 'model_version', 'status'],
    registry=REGISTRY
)

# Время ML инференса
ml_inference_duration_seconds = Histogram(
    'ml_inference_duration_seconds',
    'ML inference duration in seconds',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY
)

# Уверенность ML модели
ml_model_confidence = Histogram(
    'ml_model_confidence',
    'ML model confidence distribution',
    ['model_name'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    registry=REGISTRY
)

# Загруженные модели
ml_models_loaded = Gauge(
    'ml_models_loaded',
    'Number of loaded ML models',
    ['model_type'],
    registry=REGISTRY
)

# ===== СИСТЕМНЫЕ МЕТРИКИ =====

# Использование памяти
memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['type'],  # rss, vms, shared
    registry=REGISTRY
)

# Использование CPU
cpu_usage_percent = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    registry=REGISTRY
)

# Активные задачи
active_tasks = Gauge(
    'active_tasks',
    'Number of active background tasks',
    ['task_type'],
    registry=REGISTRY
)

# Ошибки приложения
application_errors_total = Counter(
    'application_errors_total',
    'Total number of application errors',
    ['error_type', 'component'],
    registry=REGISTRY
)

# ===== DATA SOURCE МЕТРИКИ =====

# NASA API запросы
nasa_api_requests_total = Counter(
    'nasa_api_requests_total',
    'Total number of NASA API requests',
    ['api_endpoint', 'status_code'],
    registry=REGISTRY
)

# Время ответа NASA API
nasa_api_duration_seconds = Histogram(
    'nasa_api_duration_seconds',
    'NASA API response time in seconds',
    ['api_endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY
)

# Кэш попадания
cache_operations_total = Counter(
    'cache_operations_total',
    'Total number of cache operations',
    ['operation', 'result'],  # get/set, hit/miss
    registry=REGISTRY
)

# ===== ИНФОРМАЦИОННЫЕ МЕТРИКИ =====

# Информация о приложении
app_info = Info(
    'exoplanet_ai_app_info',
    'Information about the Exoplanet AI application',
    registry=REGISTRY
)

# Статус приложения
app_status = Enum(
    'exoplanet_ai_app_status',
    'Current status of the application',
    states=['starting', 'healthy', 'degraded', 'unhealthy'],
    registry=REGISTRY
)

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware для сбора HTTP метрик"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("metrics.middleware")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Увеличиваем счетчик активных соединений
        http_active_connections.inc()
        
        start_time = time.time()
        method = request.method
        path = self._get_route_path(request)
        
        # Измеряем размер запроса
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_size = int(content_length)
                http_request_size_bytes.labels(method=method, endpoint=path).observe(request_size)
            except ValueError:
                pass
        
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            
            # Записываем метрики успешного запроса
            duration = time.time() - start_time
            
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            # Измеряем размер ответа
            response_size = response.headers.get("content-length")
            if response_size:
                try:
                    size = int(response_size)
                    http_response_size_bytes.labels(
                        method=method,
                        endpoint=path,
                        status_code=status_code
                    ).observe(size)
                except ValueError:
                    pass
            
            return response
            
        except Exception as e:
            # Записываем метрики ошибки
            duration = time.time() - start_time
            
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status_code="500"
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            application_errors_total.labels(
                error_type=type(e).__name__,
                component="http_middleware"
            ).inc()
            
            raise
        
        finally:
            # Уменьшаем счетчик активных соединений
            http_active_connections.dec()
    
    def _get_route_path(self, request: Request) -> str:
        """Получение пути маршрута (без параметров)"""
        # Пытаемся получить путь маршрута из FastAPI
        if hasattr(request, 'scope') and 'route' in request.scope:
            route = request.scope['route']
            if hasattr(route, 'path'):
                return route.path
        
        # Fallback на обычный путь
        path = request.url.path
        
        # Нормализуем некоторые пути
        if path.startswith('/api/v1/'):
            return path
        elif path.startswith('/api/'):
            return path
        elif path in ['/', '/docs', '/redoc', '/openapi.json', '/metrics', '/health']:
            return path
        else:
            return '/other'

class MetricsCollector:
    """Класс для сбора и управления метриками"""
    
    def __init__(self):
        self.logger = get_logger("metrics.collector")
        self._initialize_app_info()
    
    def _initialize_app_info(self):
        """Инициализация информационных метрик"""
        app_info.info({
            'version': '2.0.0',
            'service': 'exoplanet-ai',
            'description': 'AI-powered exoplanet detection system'
        })
        app_status.state('starting')
    
    def set_app_status(self, status: str):
        """Установка статуса приложения"""
        app_status.state(status)
        self.logger.info(f"Application status changed to: {status}")
    
    def record_exoplanet_analysis(
        self, 
        catalog: str, 
        mission: str, 
        status: str,
        duration: float,
        candidates_found: int = 0,
        max_snr: float = 0.0
    ):
        """Запись метрик анализа экзопланет"""
        exoplanet_analyses_total.labels(
            catalog=catalog,
            mission=mission,
            status=status
        ).inc()
        
        bls_analysis_duration_seconds.labels(
            target_catalog=catalog
        ).observe(duration)
        
        if candidates_found > 0:
            significance_level = "high" if max_snr > 10 else "medium" if max_snr > 7 else "low"
            planet_candidates_found.labels(
                catalog=catalog,
                mission=mission,
                significance_level=significance_level
            ).inc(candidates_found)
            
            candidate_snr_distribution.observe(max_snr)
    
    def record_ml_inference(
        self,
        model_name: str,
        model_version: str,
        duration: float,
        confidence: float,
        status: str = "success"
    ):
        """Запись метрик ML инференса"""
        ml_inferences_total.labels(
            model_name=model_name,
            model_version=model_version,
            status=status
        ).inc()
        
        ml_inference_duration_seconds.labels(
            model_name=model_name
        ).observe(duration)
        
        ml_model_confidence.labels(
            model_name=model_name
        ).observe(confidence)
    
    def record_nasa_api_call(
        self,
        endpoint: str,
        duration: float,
        status_code: int
    ):
        """Запись метрик NASA API"""
        nasa_api_requests_total.labels(
            api_endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        nasa_api_duration_seconds.labels(
            api_endpoint=endpoint
        ).observe(duration)
    
    def record_cache_operation(self, operation: str, result: str):
        """Запись метрик кэша"""
        cache_operations_total.labels(
            operation=operation,
            result=result
        ).inc()
    
    def record_error(self, error_type: str, component: str):
        """Запись метрик ошибок"""
        application_errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def update_system_metrics(self):
        """Обновление системных метрик"""
        try:
            import psutil
            
            # Память
            memory = psutil.virtual_memory()
            memory_usage_bytes.labels(type='total').set(memory.total)
            memory_usage_bytes.labels(type='available').set(memory.available)
            memory_usage_bytes.labels(type='used').set(memory.used)
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage_percent.set(cpu_percent)
            
        except ImportError:
            self.logger.warning("psutil not available, skipping system metrics")
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
    
    def set_ml_models_loaded(self, model_type: str, count: int):
        """Установка количества загруженных моделей"""
        ml_models_loaded.labels(model_type=model_type).set(count)
    
    def set_active_tasks(self, task_type: str, count: int):
        """Установка количества активных задач"""
        active_tasks.labels(task_type=task_type).set(count)

# Глобальный экземпляр коллектора
metrics_collector = MetricsCollector()

# Декораторы для автоматического сбора метрик
def track_time(metric_name: str, labels: Dict[str, str] = None):
    """Декоратор для отслеживания времени выполнения"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                # Здесь можно добавить логику записи в соответствующую метрику
                logger.debug(f"{metric_name} took {duration:.3f}s", extra={"duration": duration})
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(f"{metric_name} took {duration:.3f}s", extra={"duration": duration})
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator

@contextmanager
def track_ml_inference(model_name: str, model_version: str = "unknown"):
    """Context manager для отслеживания ML инференса"""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        metrics_collector.record_ml_inference(
            model_name=model_name,
            model_version=model_version,
            duration=duration,
            confidence=0.0,  # Будет обновлено извне
            status="success"
        )
    except Exception as e:
        duration = time.time() - start_time
        metrics_collector.record_ml_inference(
            model_name=model_name,
            model_version=model_version,
            duration=duration,
            confidence=0.0,
            status="error"
        )
        metrics_collector.record_error(type(e).__name__, "ml_inference")
        raise

def get_metrics() -> str:
    """Получение метрик в формате Prometheus"""
    return generate_latest(REGISTRY)

def get_metrics_content_type() -> str:
    """Получение content-type для метрик"""
    return CONTENT_TYPE_LATEST
