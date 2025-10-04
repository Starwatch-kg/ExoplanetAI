"""
Централизованная система обработки исключений для ExoplanetAI
"""
import logging
from typing import Any, Dict, Optional, Type, Union

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class ExoplanetAIException(Exception):
    """Базовое исключение для ExoplanetAI"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)


class DataSourceError(ExoplanetAIException):
    """Ошибка источника данных"""
    pass


class AnalysisError(ExoplanetAIException):
    """Ошибка анализа данных"""
    pass


class ValidationError(ExoplanetAIException):
    """Ошибка валидации входных данных"""
    pass


class AuthenticationError(ExoplanetAIException):
    """Ошибка аутентификации"""
    pass


class RateLimitError(ExoplanetAIException):
    """Превышение лимита запросов"""
    pass


def handle_service_error(
    error: Exception,
    context: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> HTTPException:
    """
    Централизованная обработка ошибок сервисов
    
    Args:
        error: Исключение для обработки
        context: Контекст где произошла ошибка
        user_id: ID пользователя (для логирования)
        request_id: ID запроса (для трассировки)
    
    Returns:
        HTTPException с соответствующим статусом и сообщением
    """
    
    # Контекст для логирования
    log_context = {
        "context": context,
        "error_type": type(error).__name__,
        "user_id": user_id,
        "request_id": request_id
    }
    
    if isinstance(error, DataSourceError):
        logger.error(f"Data source error: {error.message}", extra=log_context)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "data_source_unavailable",
                "message": "External data source is temporarily unavailable",
                "request_id": request_id
            }
        )
    
    elif isinstance(error, AnalysisError):
        logger.error(f"Analysis error: {error.message}", extra=log_context)
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "analysis_failed",
                "message": "Data analysis could not be completed",
                "request_id": request_id
            }
        )
    
    elif isinstance(error, ValidationError):
        logger.warning(f"Validation error: {error.message}", extra=log_context)
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "validation_failed",
                "message": error.message,
                "request_id": request_id
            }
        )
    
    elif isinstance(error, AuthenticationError):
        logger.warning(f"Authentication error: {error.message}", extra=log_context)
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "authentication_failed",
                "message": "Authentication required or invalid",
                "request_id": request_id
            }
        )
    
    elif isinstance(error, RateLimitError):
        logger.warning(f"Rate limit error: {error.message}", extra=log_context)
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": "Too many requests, please try again later",
                "request_id": request_id
            }
        )
    
    elif isinstance(error, ValueError):
        logger.warning(f"Value error: {str(error)}", extra=log_context)
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_input",
                "message": "Invalid input data provided",
                "request_id": request_id
            }
        )
    
    elif isinstance(error, (ConnectionError, TimeoutError)):
        logger.error(f"Network error: {str(error)}", extra=log_context)
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "network_error",
                "message": "Network connectivity issue",
                "request_id": request_id
            }
        )
    
    else:
        # Неожиданная ошибка - логируем с полным стеком
        logger.error(
            f"Unexpected error in {context}: {str(error)}", 
            extra=log_context,
            exc_info=True
        )
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": request_id
            }
        )


# Декоратор для автоматической обработки ошибок
def handle_errors(context: str):
    """Декоратор для автоматической обработки ошибок в endpoint'ах"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Извлекаем request_id если доступен
                request_id = getattr(kwargs.get('request'), 'state', {}).get('request_id')
                raise handle_service_error(e, context, request_id=request_id)
        return wrapper
    return decorator
