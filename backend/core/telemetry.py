"""
OpenTelemetry Integration
Интеграция OpenTelemetry для трассировки и мониторинга
"""

import os
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.trace.status import Status, StatusCode

from .logging_config import get_logger

logger = get_logger(__name__)

class TelemetryConfig:
    """Конфигурация телеметрии"""
    
    def __init__(self):
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "exoplanet-ai")
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", "2.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # OTLP endpoints
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.otlp_traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", self.otlp_endpoint)
        self.otlp_metrics_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", self.otlp_endpoint)
        
        # Headers для аутентификации
        self.otlp_headers = self._parse_headers(os.getenv("OTEL_EXPORTER_OTLP_HEADERS", ""))
        
        # Настройки экспорта
        self.traces_enabled = os.getenv("OTEL_TRACES_ENABLED", "true").lower() == "true"
        self.metrics_enabled = os.getenv("OTEL_METRICS_ENABLED", "true").lower() == "true"
        
        # Sampling
        self.trace_sample_rate = float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "1.0"))
        
    def _parse_headers(self, headers_str: str) -> Dict[str, str]:
        """Парсинг заголовков из строки"""
        headers = {}
        if headers_str:
            for header in headers_str.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    headers[key.strip()] = value.strip()
        return headers

class TelemetryManager:
    """Менеджер телеметрии"""
    
    def __init__(self, config: TelemetryConfig = None):
        self.config = config or TelemetryConfig()
        self.tracer = None
        self.meter = None
        self._initialized = False
    
    def initialize(self):
        """Инициализация OpenTelemetry"""
        if self._initialized:
            return
        
        try:
            # Создаем ресурс с метаданными сервиса
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
            })
            
            # Настраиваем трассировку
            if self.config.traces_enabled:
                self._setup_tracing(resource)
            
            # Настраиваем метрики
            if self.config.metrics_enabled:
                self._setup_metrics(resource)
            
            # Настраиваем пропагацию
            set_global_textmap(B3MultiFormat())
            
            self._initialized = True
            logger.info("OpenTelemetry initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            # Продолжаем работу без телеметрии
    
    def _setup_tracing(self, resource: Resource):
        """Настройка трассировки"""
        # Создаем провайдер трассировки
        tracer_provider = TracerProvider(resource=resource)
        
        # Настраиваем OTLP экспортер для трассировки
        if self.config.otlp_traces_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_traces_endpoint,
                headers=self.config.otlp_headers
            )
            
            # Добавляем batch processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
        
        # Устанавливаем глобальный провайдер
        trace.set_tracer_provider(tracer_provider)
        
        # Получаем tracer
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
        
        logger.info(f"Tracing configured with endpoint: {self.config.otlp_traces_endpoint}")
    
    def _setup_metrics(self, resource: Resource):
        """Настройка метрик"""
        # Создаем OTLP экспортер для метрик
        if self.config.otlp_metrics_endpoint:
            otlp_exporter = OTLPMetricExporter(
                endpoint=self.config.otlp_metrics_endpoint,
                headers=self.config.otlp_headers
            )
            
            # Создаем reader с периодическим экспортом
            metric_reader = PeriodicExportingMetricReader(
                exporter=otlp_exporter,
                export_interval_millis=30000  # 30 секунд
            )
            
            # Создаем провайдер метрик
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            )
            
            # Устанавливаем глобальный провайдер
            metrics.set_meter_provider(meter_provider)
            
            # Получаем meter
            self.meter = metrics.get_meter(
                self.config.service_name,
                self.config.service_version
            )
            
            logger.info(f"Metrics configured with endpoint: {self.config.otlp_metrics_endpoint}")
    
    def instrument_fastapi(self, app):
        """Инструментация FastAPI приложения"""
        if not self._initialized:
            self.initialize()
        
        try:
            # Инструментируем FastAPI
            FastAPIInstrumentor.instrument_app(
                app,
                tracer_provider=trace.get_tracer_provider() if self.config.traces_enabled else None
            )
            
            # Инструментируем HTTP клиенты
            RequestsInstrumentor().instrument()
            AioHttpClientInstrumentor().instrument()
            
            logger.info("FastAPI instrumentation completed")
            
        except Exception as e:
            logger.error(f"Failed to instrument FastAPI: {e}")
    
    def instrument_sqlalchemy(self, engine):
        """Инструментация SQLAlchemy"""
        try:
            SQLAlchemyInstrumentor().instrument(
                engine=engine,
                tracer_provider=trace.get_tracer_provider() if self.config.traces_enabled else None
            )
            logger.info("SQLAlchemy instrumentation completed")
        except Exception as e:
            logger.error(f"Failed to instrument SQLAlchemy: {e}")

# Глобальный менеджер телеметрии
telemetry_manager = TelemetryManager()

# Декораторы для трассировки
def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """Декоратор для трассировки функций"""
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not telemetry_manager.tracer:
                return await func(*args, **kwargs)
            
            with telemetry_manager.tracer.start_as_current_span(span_name) as span:
                # Добавляем атрибуты
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Добавляем информацию о функции
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not telemetry_manager.tracer:
                return func(*args, **kwargs)
            
            with telemetry_manager.tracer.start_as_current_span(span_name) as span:
                # Добавляем атрибуты
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        # Определяем, асинхронная ли функция
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: Optional[trace.SpanKind] = None
):
    """Context manager для создания span"""
    if not telemetry_manager.tracer:
        yield None
        return
    
    with telemetry_manager.tracer.start_as_current_span(
        name,
        kind=kind or trace.SpanKind.INTERNAL
    ) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

# Специализированные трассировщики
@contextmanager
def trace_ml_inference(
    model_name: str,
    model_version: str = "unknown",
    input_shape: Optional[tuple] = None
):
    """Трассировка ML инференса"""
    attributes = {
        "ml.model.name": model_name,
        "ml.model.version": model_version,
        "ml.task.type": "inference"
    }
    
    if input_shape:
        attributes["ml.input.shape"] = str(input_shape)
    
    with trace_span(
        f"ml.inference.{model_name}",
        attributes=attributes,
        kind=trace.SpanKind.INTERNAL
    ) as span:
        start_time = time.time()
        try:
            yield span
            duration = time.time() - start_time
            if span:
                span.set_attribute("ml.inference.duration_ms", duration * 1000)
        except Exception as e:
            duration = time.time() - start_time
            if span:
                span.set_attribute("ml.inference.duration_ms", duration * 1000)
                span.set_attribute("ml.inference.error", str(e))
            raise

@contextmanager
def trace_bls_analysis(
    target_name: str,
    catalog: str,
    mission: str
):
    """Трассировка BLS анализа"""
    attributes = {
        "exoplanet.target.name": target_name,
        "exoplanet.catalog": catalog,
        "exoplanet.mission": mission,
        "exoplanet.analysis.type": "bls"
    }
    
    with trace_span(
        "exoplanet.analysis.bls",
        attributes=attributes,
        kind=trace.SpanKind.INTERNAL
    ) as span:
        start_time = time.time()
        try:
            yield span
            duration = time.time() - start_time
            if span:
                span.set_attribute("exoplanet.analysis.duration_ms", duration * 1000)
        except Exception as e:
            duration = time.time() - start_time
            if span:
                span.set_attribute("exoplanet.analysis.duration_ms", duration * 1000)
                span.set_attribute("exoplanet.analysis.error", str(e))
            raise

@contextmanager
def trace_nasa_api_call(
    endpoint: str,
    target_name: Optional[str] = None
):
    """Трассировка вызовов NASA API"""
    attributes = {
        "http.client.name": "nasa_api",
        "nasa.api.endpoint": endpoint
    }
    
    if target_name:
        attributes["nasa.api.target"] = target_name
    
    with trace_span(
        f"nasa.api.{endpoint}",
        attributes=attributes,
        kind=trace.SpanKind.CLIENT
    ) as span:
        start_time = time.time()
        try:
            yield span
            duration = time.time() - start_time
            if span:
                span.set_attribute("nasa.api.duration_ms", duration * 1000)
        except Exception as e:
            duration = time.time() - start_time
            if span:
                span.set_attribute("nasa.api.duration_ms", duration * 1000)
                span.set_attribute("nasa.api.error", str(e))
            raise

def add_span_attributes(attributes: Dict[str, Any]):
    """Добавление атрибутов к текущему span"""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        for key, value in attributes.items():
            current_span.set_attribute(key, value)

def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Добавление события к текущему span"""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes or {})

def record_exception(exception: Exception, attributes: Optional[Dict[str, Any]] = None):
    """Запись исключения в текущий span"""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.record_exception(exception, attributes or {})
        current_span.set_status(Status(StatusCode.ERROR, str(exception)))

# Функции для настройки
def setup_telemetry(app, config: Optional[TelemetryConfig] = None):
    """Настройка телеметрии для приложения"""
    global telemetry_manager
    
    if config:
        telemetry_manager = TelemetryManager(config)
    
    # Инициализируем телеметрию
    telemetry_manager.initialize()
    
    # Инструментируем приложение
    telemetry_manager.instrument_fastapi(app)
    
    logger.info("Telemetry setup completed")

def get_trace_id() -> Optional[str]:
    """Получение текущего trace ID"""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        trace_id = current_span.get_span_context().trace_id
        return format(trace_id, '032x')
    return None

def get_span_id() -> Optional[str]:
    """Получение текущего span ID"""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        span_id = current_span.get_span_context().span_id
        return format(span_id, '016x')
    return None
