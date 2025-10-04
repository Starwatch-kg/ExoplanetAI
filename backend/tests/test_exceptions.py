"""
Тесты для системы обработки исключений
"""
import pytest
from fastapi import HTTPException, status

from core.exceptions import (
    ExoplanetAIException,
    DataSourceError,
    AnalysisError,
    ValidationError,
    AuthenticationError,
    RateLimitError,
    handle_service_error
)


class TestExoplanetAIExceptions:
    """Тесты базовых исключений"""
    
    def test_base_exception_creation(self):
        """Тест создания базового исключения"""
        context = {"target": "Kepler-452b", "mission": "TESS"}
        exc = ExoplanetAIException("Test error", context)
        
        assert exc.message == "Test error"
        assert exc.context == context
        assert str(exc) == "Test error"
    
    def test_specialized_exceptions(self):
        """Тест специализированных исключений"""
        data_error = DataSourceError("NASA API unavailable")
        analysis_error = AnalysisError("BLS analysis failed")
        validation_error = ValidationError("Invalid target name")
        
        assert isinstance(data_error, ExoplanetAIException)
        assert isinstance(analysis_error, ExoplanetAIException)
        assert isinstance(validation_error, ExoplanetAIException)


class TestServiceErrorHandler:
    """Тесты обработчика ошибок сервисов"""
    
    def test_data_source_error_handling(self):
        """Тест обработки ошибок источников данных"""
        error = DataSourceError("NASA API timeout")
        http_exc = handle_service_error(error, "test_context", "user123", "req456")
        
        assert isinstance(http_exc, HTTPException)
        assert http_exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert http_exc.detail["error"] == "data_source_unavailable"
        assert http_exc.detail["request_id"] == "req456"
    
    def test_analysis_error_handling(self):
        """Тест обработки ошибок анализа"""
        error = AnalysisError("Transit detection failed")
        http_exc = handle_service_error(error, "transit_analysis")
        
        assert http_exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert http_exc.detail["error"] == "analysis_failed"
    
    def test_validation_error_handling(self):
        """Тест обработки ошибок валидации"""
        error = ValidationError("Target name too long")
        http_exc = handle_service_error(error, "validation")
        
        assert http_exc.status_code == status.HTTP_400_BAD_REQUEST
        assert http_exc.detail["error"] == "validation_failed"
        assert "Target name too long" in http_exc.detail["message"]
    
    def test_authentication_error_handling(self):
        """Тест обработки ошибок аутентификации"""
        error = AuthenticationError("Invalid JWT token")
        http_exc = handle_service_error(error, "auth")
        
        assert http_exc.status_code == status.HTTP_401_UNAUTHORIZED
        assert http_exc.detail["error"] == "authentication_failed"
    
    def test_rate_limit_error_handling(self):
        """Тест обработки ошибок превышения лимитов"""
        error = RateLimitError("Too many requests")
        http_exc = handle_service_error(error, "rate_limit")
        
        assert http_exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert http_exc.detail["error"] == "rate_limit_exceeded"
    
    def test_value_error_handling(self):
        """Тест обработки ValueError"""
        error = ValueError("Invalid input format")
        http_exc = handle_service_error(error, "input_validation")
        
        assert http_exc.status_code == status.HTTP_400_BAD_REQUEST
        assert http_exc.detail["error"] == "invalid_input"
    
    def test_network_error_handling(self):
        """Тест обработки сетевых ошибок"""
        error = ConnectionError("Connection refused")
        http_exc = handle_service_error(error, "external_api")
        
        assert http_exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert http_exc.detail["error"] == "network_error"
    
    def test_unexpected_error_handling(self):
        """Тест обработки неожиданных ошибок"""
        error = RuntimeError("Unexpected runtime error")
        http_exc = handle_service_error(error, "unknown_context")
        
        assert http_exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert http_exc.detail["error"] == "internal_server_error"
        assert "unexpected error" in http_exc.detail["message"].lower()


@pytest.mark.asyncio
class TestErrorHandlerDecorator:
    """Тесты декоратора обработки ошибок"""
    
    async def test_successful_execution(self):
        """Тест успешного выполнения функции"""
        from core.exceptions import handle_errors
        
        @handle_errors("test_function")
        async def test_func():
            return {"result": "success"}
        
        result = await test_func()
        assert result == {"result": "success"}
    
    async def test_error_handling_in_decorator(self):
        """Тест обработки ошибок в декораторе"""
        from core.exceptions import handle_errors
        
        @handle_errors("test_function")
        async def failing_func():
            raise ValidationError("Test validation error")
        
        with pytest.raises(HTTPException) as exc_info:
            await failing_func()
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
