"""
Centralized Error Handling System
Централизованная система обработки ошибок с маскированием секретов
"""

import traceback
import re
from typing import Dict, Any, Optional, Union
from datetime import datetime
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

from .logging_config import get_logger, log_error_with_context
from .middleware import get_request_context

logger = get_logger(__name__)

class ExoplanetAIException(Exception):
    """Базовое исключение для приложения"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationException(ExoplanetAIException):
    """Исключение валидации данных"""
    
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details or {}
        )
        self.field = field

class DataSourceException(ExoplanetAIException):
    """Исключение источников данных (NASA API, etc.)"""
    
    def __init__(self, message: str, source: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="DATA_SOURCE_ERROR",
            status_code=503,
            details=details or {}
        )
        self.source = source

class MLModelException(ExoplanetAIException):
    """Исключение ML моделей"""
    
    def __init__(self, message: str, model_name: str, details: Dict[str, Any] = None):
        super().__init__(
            message=message,
            error_code="ML_MODEL_ERROR",
            status_code=500,
            details=details or {}
        )
        self.model_name = model_name

class RateLimitException(ExoplanetAIException):
    """Исключение превышения лимитов"""
    
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429
        )
        self.retry_after = retry_after

class SecretMasker:
    """Класс для маскирования секретных данных"""
    
    # Паттерны для поиска секретов
    SECRET_PATTERNS = [
        # API ключи
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})', 'API_KEY'),
        # Токены
        (r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{20,})', 'TOKEN'),
        # Пароли
        (r'password["\']?\s*[:=]\s*["\']?([^\s"\']{8,})', 'PASSWORD'),
        # JWT токены
        (r'eyJ[a-zA-Z0-9_\-]*\.eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*', 'JWT_TOKEN'),
        # Bearer токены
        (r'Bearer\s+([a-zA-Z0-9_\-\.]{20,})', 'BEARER_TOKEN'),
        # Секретные ключи
        (r'secret[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})', 'SECRET_KEY'),
        # Database URLs с паролями
        (r'://[^:]+:([^@]+)@', 'DB_PASSWORD'),
    ]
    
    @classmethod
    def mask_secrets(cls, text: str) -> str:
        """Маскирование секретов в тексте"""
        if not isinstance(text, str):
            return text
        
        masked_text = text
        for pattern, secret_type in cls.SECRET_PATTERNS:
            masked_text = re.sub(
                pattern, 
                lambda m: m.group(0).replace(m.group(1), '***MASKED***'),
                masked_text,
                flags=re.IGNORECASE
            )
        
        return masked_text
    
    @classmethod
    def mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Маскирование секретов в словаре"""
        if not isinstance(data, dict):
            return data
        
        masked_data = {}
        secret_keys = {
            'password', 'token', 'key', 'secret', 'auth', 'credential',
            'api_key', 'access_token', 'refresh_token', 'jwt', 'bearer',
            'authorization', 'x-api-key'
        }
        
        for key, value in data.items():
            if any(secret in key.lower() for secret in secret_keys):
                masked_data[key] = '***MASKED***'
            elif isinstance(value, dict):
                masked_data[key] = cls.mask_dict(value)
            elif isinstance(value, list):
                masked_data[key] = [
                    cls.mask_dict(item) if isinstance(item, dict) else cls.mask_secrets(str(item))
                    for item in value
                ]
            elif isinstance(value, str):
                masked_data[key] = cls.mask_secrets(value)
            else:
                masked_data[key] = value
        
        return masked_data

class ErrorResponseBuilder:
    """Построитель ответов об ошибках"""
    
    @staticmethod
    def build_error_response(
        error: Exception,
        request: Request = None,
        include_traceback: bool = False
    ) -> Dict[str, Any]:
        """Построение стандартного ответа об ошибке"""
        
        # Получаем контекст запроса
        request_context = {}
        if request:
            try:
                request_context = get_request_context(request)
            except:
                pass
        
        # Базовая структура ответа
        error_response = {
            "error": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_context.get("request_id", "unknown"),
            "trace_id": request_context.get("trace_id", "unknown"),
        }
        
        # Обработка различных типов ошибок
        if isinstance(error, ExoplanetAIException):
            error_response.update({
                "error_code": error.error_code,
                "message": error.message,
                "status_code": error.status_code,
                "details": SecretMasker.mask_dict(error.details)
            })
            
            # Специальная обработка для RateLimitException
            if isinstance(error, RateLimitException):
                error_response["retry_after"] = error.retry_after
                
        elif isinstance(error, HTTPException):
            error_response.update({
                "error_code": "HTTP_ERROR",
                "message": error.detail,
                "status_code": error.status_code
            })
            
        elif isinstance(error, RequestValidationError):
            error_response.update({
                "error_code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "status_code": 422,
                "validation_errors": [
                    {
                        "field": ".".join(str(loc) for loc in err["loc"]),
                        "message": err["msg"],
                        "type": err["type"]
                    }
                    for err in error.errors()
                ]
            })
            
        else:
            # Общие исключения
            error_response.update({
                "error_code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
                "status_code": 500,
                "error_type": type(error).__name__
            })
        
        # Добавляем traceback в development режиме
        if include_traceback:
            error_response["traceback"] = SecretMasker.mask_secrets(
                traceback.format_exc()
            )
        
        # Маскируем секреты в сообщении
        if "message" in error_response:
            error_response["message"] = SecretMasker.mask_secrets(error_response["message"])
        
        return error_response

# Обработчики исключений для FastAPI
async def exoplanet_ai_exception_handler(request: Request, exc: ExoplanetAIException) -> JSONResponse:
    """Обработчик пользовательских исключений"""
    
    # Логируем ошибку
    log_error_with_context(
        exc, 
        context={
            "request_path": request.url.path,
            "request_method": request.method,
            "error_code": exc.error_code,
            **get_request_context(request)
        }
    )
    
    # Строим ответ
    error_response = ErrorResponseBuilder.build_error_response(exc, request)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers={
            "X-Error-Code": exc.error_code,
            "X-Request-ID": error_response.get("request_id", "unknown")
        }
    )

async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Обработчик HTTP исключений"""
    
    logger.warning(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "detail": exc.detail,
            "request_path": request.url.path,
            **get_request_context(request)
        }
    )
    
    error_response = ErrorResponseBuilder.build_error_response(exc, request)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers={
            "X-Request-ID": error_response.get("request_id", "unknown")
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Обработчик ошибок валидации"""
    
    logger.warning(
        f"Validation error: {exc.errors()}",
        extra={
            "validation_errors": exc.errors(),
            "request_path": request.url.path,
            **get_request_context(request)
        }
    )
    
    error_response = ErrorResponseBuilder.build_error_response(exc, request)
    
    return JSONResponse(
        status_code=422,
        content=error_response,
        headers={
            "X-Request-ID": error_response.get("request_id", "unknown")
        }
    )

async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Обработчик общих исключений"""
    
    # Логируем критическую ошибку
    log_error_with_context(
        exc,
        context={
            "request_path": request.url.path,
            "request_method": request.method,
            **get_request_context(request)
        }
    )
    
    # В production не показываем детали внутренних ошибок
    include_traceback = False  # Можно настроить через конфигурацию
    
    error_response = ErrorResponseBuilder.build_error_response(
        exc, 
        request, 
        include_traceback=include_traceback
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response,
        headers={
            "X-Request-ID": error_response.get("request_id", "unknown")
        }
    )

def setup_error_handlers(app):
    """Настройка всех обработчиков ошибок"""
    
    # Пользовательские исключения
    app.add_exception_handler(ExoplanetAIException, exoplanet_ai_exception_handler)
    
    # HTTP исключения
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Ошибки валидации
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Общие исключения (должен быть последним)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Error handlers configured successfully")

# Утилиты для создания исключений
def raise_validation_error(message: str, field: str = None, details: Dict[str, Any] = None):
    """Создание исключения валидации"""
    raise ValidationException(message, field, details)

def raise_data_source_error(message: str, source: str, details: Dict[str, Any] = None):
    """Создание исключения источника данных"""
    raise DataSourceException(message, source, details)

def raise_ml_model_error(message: str, model_name: str, details: Dict[str, Any] = None):
    """Создание исключения ML модели"""
    raise MLModelException(message, model_name, details)

def raise_rate_limit_error(message: str = "Rate limit exceeded", retry_after: int = 60):
    """Создание исключения превышения лимитов"""
    raise RateLimitException(message, retry_after)
