"""
Enhanced Middleware with OpenTelemetry Integration
Улучшенный middleware с интеграцией OpenTelemetry и трассировкой
"""

import time
import uuid
import json
from typing import Callable, Dict, Any, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.types import ASGIApp
import logging
from collections import defaultdict, deque

from .logging_config import RequestContextLogger, log_api_request, get_logger

logger = get_logger(__name__)

class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для трассировки запросов с поддержкой OpenTelemetry
    Добавляет request_id, trace_id, span_id в заголовки и логи
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger("middleware.request_tracking")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Генерируем или извлекаем идентификаторы
        request_id = self._get_or_generate_request_id(request)
        trace_id = self._get_or_generate_trace_id(request)
        span_id = self._generate_span_id()
        
        # Устанавливаем контекст логирования
        with RequestContextLogger(
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id
        ):
            start_time = time.time()
            
            # Логируем начало запроса
            self.logger.info(
                "Request started",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_query": str(request.url.query) if request.url.query else None,
                    "client_ip": self._get_client_ip(request),
                    "user_agent": request.headers.get("user-agent"),
                    "content_type": request.headers.get("content-type"),
                    "content_length": request.headers.get("content-length")
                }
            )
            
            try:
                # Добавляем идентификаторы в state запроса
                request.state.request_id = request_id
                request.state.trace_id = trace_id
                request.state.span_id = span_id
                
                # Выполняем запрос
                response = await call_next(request)
                
                # Вычисляем время выполнения
                duration_ms = (time.time() - start_time) * 1000
                
                # Добавляем заголовки трассировки
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Trace-ID"] = trace_id
                response.headers["X-Span-ID"] = span_id
                response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
                
                # Логируем успешное завершение
                log_api_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    client_ip=self._get_client_ip(request),
                    response_size=response.headers.get("content-length")
                )
                
                return response
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Логируем ошибку
                self.logger.error(
                    f"Request failed: {str(e)}",
                    exc_info=True,
                    extra={
                        "http_method": request.method,
                        "http_path": request.url.path,
                        "duration_ms": duration_ms,
                        "error_type": type(e).__name__
                    }
                )
                
                # Возвращаем ошибку с трассировкой
                error_response = JSONResponse(
                    status_code=500,
                    content={
                        "error": "Internal Server Error",
                        "message": "An unexpected error occurred",
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "timestamp": time.time()
                    }
                )
                
                error_response.headers["X-Request-ID"] = request_id
                error_response.headers["X-Trace-ID"] = trace_id
                error_response.headers["X-Span-ID"] = span_id
                
                return error_response
    
    def _get_or_generate_request_id(self, request: Request) -> str:
        """Получение или генерация request_id"""
        # Проверяем заголовки
        request_id = (
            request.headers.get("X-Request-ID") or
            request.headers.get("x-request-id") or
            request.headers.get("Request-ID")
        )
        
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:16]}"
        
        return request_id
    
    def _get_or_generate_trace_id(self, request: Request) -> str:
        """Получение или генерация trace_id"""
        # Поддержка W3C Trace Context
        traceparent = request.headers.get("traceparent")
        if traceparent:
            try:
                # Формат: 00-{trace_id}-{span_id}-{flags}
                parts = traceparent.split("-")
                if len(parts) >= 2:
                    return parts[1]
            except:
                pass
        
        # Альтернативные заголовки
        trace_id = (
            request.headers.get("X-Trace-ID") or
            request.headers.get("x-trace-id") or
            request.headers.get("Trace-ID")
        )
        
        if not trace_id:
            trace_id = f"trace_{uuid.uuid4().hex[:16]}"
        
        return trace_id
    
    def _generate_span_id(self) -> str:
        """Генерация span_id"""
        return f"span_{uuid.uuid4().hex[:8]}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Получение IP адреса клиента"""
        # Проверяем заголовки прокси
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback на client из scope
        client = getattr(request, "client", None)
        if client:
            return client.host
        
        return "unknown"

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Улучшенный Rate Limiting Middleware
    Поддерживает различные стратегии лимитирования
    """
    
    def __init__(
        self, 
        app: ASGIApp,
        calls: int = 100,
        period: int = 60,
        burst_calls: int = 20,
        burst_period: int = 1,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.burst_calls = burst_calls
        self.burst_period = burst_period
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/redoc"]
        
        # Хранилища для отслеживания запросов
        self.clients = defaultdict(lambda: deque())
        self.burst_clients = defaultdict(lambda: deque())
        
        self.logger = get_logger("middleware.rate_limit")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Пропускаем исключенные пути
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Проверяем burst лимит
        if not self._check_burst_limit(client_ip, current_time):
            self.logger.warning(
                f"Burst rate limit exceeded for {client_ip}",
                extra={
                    "client_ip": client_ip,
                    "limit_type": "burst",
                    "path": request.url.path
                }
            )
            return self._create_rate_limit_response(
                "Burst rate limit exceeded",
                self.burst_period
            )
        
        # Проверяем основной лимит
        if not self._check_main_limit(client_ip, current_time):
            self.logger.warning(
                f"Rate limit exceeded for {client_ip}",
                extra={
                    "client_ip": client_ip,
                    "limit_type": "main",
                    "path": request.url.path
                }
            )
            return self._create_rate_limit_response(
                f"Rate limit exceeded: {self.calls} requests per {self.period} seconds",
                self.period
            )
        
        # Записываем запрос
        self._record_request(client_ip, current_time)
        
        return await call_next(request)
    
    def _check_burst_limit(self, client_ip: str, current_time: float) -> bool:
        """Проверка burst лимита"""
        client_requests = self.burst_clients[client_ip]
        
        # Удаляем старые запросы
        while client_requests and client_requests[0] <= current_time - self.burst_period:
            client_requests.popleft()
        
        return len(client_requests) < self.burst_calls
    
    def _check_main_limit(self, client_ip: str, current_time: float) -> bool:
        """Проверка основного лимита"""
        client_requests = self.clients[client_ip]
        
        # Удаляем старые запросы
        while client_requests and client_requests[0] <= current_time - self.period:
            client_requests.popleft()
        
        return len(client_requests) < self.calls
    
    def _record_request(self, client_ip: str, current_time: float):
        """Записываем запрос в оба счетчика"""
        self.clients[client_ip].append(current_time)
        self.burst_clients[client_ip].append(current_time)
    
    def _get_client_ip(self, request: Request) -> str:
        """Получение IP адреса клиента"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        client = getattr(request, "client", None)
        if client:
            return client.host
        
        return "unknown"
    
    def _create_rate_limit_response(self, message: str, retry_after: int) -> JSONResponse:
        """Создание ответа о превышении лимита"""
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate Limit Exceeded",
                "message": message,
                "retry_after": retry_after,
                "timestamp": time.time()
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(self.calls),
                "X-RateLimit-Period": str(self.period)
            }
        )

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware для добавления заголовков безопасности
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Добавляем заголовки безопасности
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response

class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware для ограничения размера запросов
    """
    
    def __init__(self, app: ASGIApp, max_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_size = max_size
        self.logger = get_logger("middleware.request_size")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")
        
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    self.logger.warning(
                        f"Request size {size} exceeds limit {self.max_size}",
                        extra={
                            "content_length": size,
                            "max_size": self.max_size,
                            "path": request.url.path
                        }
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request Entity Too Large",
                            "message": f"Request size {size} exceeds maximum {self.max_size} bytes",
                            "max_size": self.max_size
                        }
                    )
            except ValueError:
                pass
        
        return await call_next(request)

# Функция для настройки всех middleware
def setup_middleware(app, config: Dict[str, Any] = None):
    """
    Настройка всех middleware для приложения
    
    Args:
        app: FastAPI приложение
        config: Конфигурация middleware
    """
    config = config or {}
    
    # Request tracking (должен быть первым)
    app.add_middleware(RequestTrackingMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request size limiting
    max_request_size = config.get("max_request_size", 10 * 1024 * 1024)
    app.add_middleware(RequestSizeMiddleware, max_size=max_request_size)
    
    # Rate limiting
    rate_limit_config = config.get("rate_limit", {})
    app.add_middleware(
        RateLimitMiddleware,
        calls=rate_limit_config.get("calls", 100),
        period=rate_limit_config.get("period", 60),
        burst_calls=rate_limit_config.get("burst_calls", 20),
        burst_period=rate_limit_config.get("burst_period", 1),
        exclude_paths=rate_limit_config.get("exclude_paths")
    )
    
    logger.info("All middleware configured successfully")

# Утилиты для работы с трассировкой
def get_request_context(request: Request) -> Dict[str, str]:
    """Получение контекста трассировки из запроса"""
    return {
        "request_id": getattr(request.state, "request_id", "unknown"),
        "trace_id": getattr(request.state, "trace_id", "unknown"),
        "span_id": getattr(request.state, "span_id", "unknown")
    }

def create_child_span_id() -> str:
    """Создание дочернего span_id"""
    return f"span_{uuid.uuid4().hex[:8]}"
