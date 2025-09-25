"""
Enhanced Structured JSON Logging Configuration
Улучшенная система структурированного JSON логирования
"""

import json
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar
from pathlib import Path

# Context variables для request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
trace_id_var: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
span_id_var: ContextVar[Optional[str]] = ContextVar('span_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)

class StructuredJSONFormatter(logging.Formatter):
    """
    Structured JSON formatter для логов
    Создает структурированные JSON логи с полной трассировкой
    """
    
    def __init__(self, service_name: str = "exoplanet-ai", environment: str = "development"):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
    
    def format(self, record: logging.LogRecord) -> str:
        """Форматирование лог записи в JSON"""
        
        # Базовая структура лога
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "environment": self.environment,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Добавляем request tracking
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id
            
        trace_id = trace_id_var.get()
        if trace_id:
            log_entry["trace_id"] = trace_id
            
        span_id = span_id_var.get()
        if span_id:
            log_entry["span_id"] = span_id
            
        user_id = user_id_var.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        # Добавляем информацию о файле и строке
        log_entry.update({
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "module": record.module,
        })
        
        # Добавляем дополнительные поля из record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry["extra"] = extra_fields
        
        # Обработка исключений
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Маскирование секретов
        log_entry = self._mask_secrets(log_entry)
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)
    
    def _mask_secrets(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Маскирование секретных данных в логах"""
        secret_keys = [
            'password', 'token', 'key', 'secret', 'auth', 'credential',
            'api_key', 'access_token', 'refresh_token', 'jwt', 'bearer'
        ]
        
        def mask_recursive(obj):
            if isinstance(obj, dict):
                return {
                    k: "***MASKED***" if any(secret in k.lower() for secret in secret_keys)
                    else mask_recursive(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [mask_recursive(item) for item in obj]
            elif isinstance(obj, str) and len(obj) > 20:
                # Маскируем длинные строки, которые могут быть токенами
                for secret in secret_keys:
                    if secret in obj.lower():
                        return "***MASKED***"
            return obj
        
        return mask_recursive(log_entry)

class RequestContextLogger:
    """
    Менеджер контекста для логирования запросов
    Автоматически устанавливает и очищает request_id
    """
    
    def __init__(self, request_id: Optional[str] = None, trace_id: Optional[str] = None, 
                 span_id: Optional[str] = None, user_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.trace_id = trace_id
        self.span_id = span_id
        self.user_id = user_id
        self.tokens = []
    
    def __enter__(self):
        self.tokens.append(request_id_var.set(self.request_id))
        if self.trace_id:
            self.tokens.append(trace_id_var.set(self.trace_id))
        if self.span_id:
            self.tokens.append(span_id_var.set(self.span_id))
        if self.user_id:
            self.tokens.append(user_id_var.set(self.user_id))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in reversed(self.tokens):
            if token:
                try:
                    token.var.set(token.old_value)
                except LookupError:
                    pass

def setup_logging(
    service_name: str = "exoplanet-ai",
    environment: str = "development",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> None:
    """
    Настройка системы логирования
    
    Args:
        service_name: Имя сервиса
        environment: Окружение (development/staging/production)
        log_level: Уровень логирования
        log_file: Путь к файлу логов (опционально)
        enable_console: Включить вывод в консоль
    """
    
    # Создаем форматтер
    formatter = StructuredJSONFormatter(service_name, environment)
    
    # Настраиваем root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Очищаем существующие handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Настраиваем логгеры библиотек
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Получение логгера с именем"""
    return logging.getLogger(name)

def log_performance(operation: str, duration_ms: float, **kwargs):
    """Логирование производительности операций"""
    logger = get_logger("performance")
    logger.info(
        f"Performance metric: {operation}",
        extra={
            "operation": operation,
            "duration_ms": duration_ms,
            "performance_data": kwargs
        }
    )

def log_ml_inference(model_name: str, input_shape: tuple, prediction: float, 
                    confidence: float, duration_ms: float, **kwargs):
    """Специализированное логирование ML инференса"""
    logger = get_logger("ml.inference")
    logger.info(
        f"ML inference completed: {model_name}",
        extra={
            "model_name": model_name,
            "input_shape": input_shape,
            "prediction": prediction,
            "confidence": confidence,
            "duration_ms": duration_ms,
            "ml_metadata": kwargs
        }
    )

def log_api_request(method: str, path: str, status_code: int, 
                   duration_ms: float, **kwargs):
    """Логирование API запросов"""
    logger = get_logger("api")
    logger.info(
        f"API request: {method} {path} -> {status_code}",
        extra={
            "http_method": method,
            "http_path": path,
            "http_status": status_code,
            "duration_ms": duration_ms,
            "api_metadata": kwargs
        }
    )

def log_error_with_context(error: Exception, context: Dict[str, Any] = None):
    """Логирование ошибок с контекстом"""
    logger = get_logger("error")
    logger.error(
        f"Error occurred: {str(error)}",
        exc_info=True,
        extra={
            "error_type": type(error).__name__,
            "error_context": context or {}
        }
    )

# Примеры использования в комментариях
"""
Примеры использования:

# Настройка логирования при старте приложения
setup_logging(
    service_name="exoplanet-ai",
    environment="production",
    log_level="INFO",
    log_file="logs/app.log"
)

# Использование в middleware
with RequestContextLogger(request_id="req-123", trace_id="trace-456"):
    logger = get_logger(__name__)
    logger.info("Processing request")

# Логирование производительности
log_performance("bls_analysis", 1250.5, target_name="TIC123", periods_tested=1000)

# Логирование ML инференса
log_ml_inference("cnn_classifier", (1024,), 0.85, 0.92, 45.2, 
                target_id="TIC123", model_version="v1.2")

# Логирование API запросов
log_api_request("POST", "/api/v1/predict", 200, 1250.5, 
               target_name="TIC123", model_used="ensemble")
"""
