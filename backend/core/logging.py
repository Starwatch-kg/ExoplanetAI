"""
Structured logging configuration for ExoplanetAI
Конфигурация структурированного логирования
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import colorama
import structlog
from rich.console import Console
from rich.logging import RichHandler
from structlog.stdlib import LoggerFactory

# Initialize colorama for Windows compatibility
colorama.init()

# Rich console for beautiful output
console = Console()


def configure_structlog(
    service_name: str = "exoplanet-ai",
    environment: str = "development",
    log_level: str = "INFO",
    enable_json: bool = False,
    enable_console: bool = True,
):
    """
    Configure structured logging with structlog

    Args:
        service_name: Name of the service
        environment: Environment (development, production, etc.)
        log_level: Logging level
        enable_json: Enable JSON output for production
        enable_console: Enable console output
    """

    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt="ISO")

    # Shared processors
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if enable_json:
        # JSON output for production
        structlog.configure(
            processors=shared_processors
            + [structlog.processors.add_log_level, structlog.processors.JSONRenderer()],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Pretty console output for development
        structlog.configure(
            processors=shared_processors + [structlog.dev.ConsoleRenderer(colors=True)],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    # Configure standard library logging
    handlers = [
        RichHandler(
            console=console,
            show_time=False,  # structlog handles time
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
    ]

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=handlers if enable_console else [],
    )

    # Add service context to all logs
    structlog.configure(
        processors=shared_processors
        + [
            _add_service_context(service_name, environment),
            (
                structlog.processors.JSONRenderer()
                if enable_json
                else structlog.dev.ConsoleRenderer(colors=True)
            ),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_service_context(service_name: str, environment: str):
    """Add service context to all log entries"""

    def processor(logger, method_name, event_dict):
        event_dict["service"] = service_name
        event_dict["environment"] = environment
        return event_dict

    return processor


class ExoplanetLogger:
    """
    Enhanced logger for ExoplanetAI with domain-specific methods
    """

    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.name = name

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)

    # Domain-specific logging methods

    def log_api_request(
        self,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs,
    ):
        """Log API request"""
        self.logger.info(
            "API request",
            event_type="api_request",
            method=method,
            path=path,
            user_id=user_id,
            ip_address=ip_address,
            **kwargs,
        )

    def log_api_response(
        self,
        method: str,
        path: str,
        status_code: int,
        processing_time_ms: float,
        user_id: Optional[str] = None,
        **kwargs,
    ):
        """Log API response"""
        self.logger.info(
            "API response",
            event_type="api_response",
            method=method,
            path=path,
            status_code=status_code,
            processing_time_ms=processing_time_ms,
            user_id=user_id,
            **kwargs,
        )

    def log_data_source_request(
        self,
        source_name: str,
        target: str,
        operation: str,
        success: bool,
        processing_time_ms: Optional[float] = None,
        **kwargs,
    ):
        """Log data source request"""
        level = "info" if success else "error"
        getattr(self.logger, level)(
            f"Data source {operation}",
            event_type="data_source_request",
            source_name=source_name,
            target=target,
            operation=operation,
            success=success,
            processing_time_ms=processing_time_ms,
            **kwargs,
        )

    def log_cache_operation(
        self,
        operation: str,
        key: str,
        hit: Optional[bool] = None,
        ttl: Optional[int] = None,
        **kwargs,
    ):
        """Log cache operation"""
        self.logger.debug(
            f"Cache {operation}",
            event_type="cache_operation",
            operation=operation,
            key=key,
            hit=hit,
            ttl=ttl,
            **kwargs,
        )

    def log_authentication(
        self,
        username: str,
        success: bool,
        method: str = "jwt",
        ip_address: Optional[str] = None,
        **kwargs,
    ):
        """Log authentication attempt"""
        level = "info" if success else "warning"
        getattr(self.logger, level)(
            f"Authentication {'successful' if success else 'failed'}",
            event_type="authentication",
            username=username,
            success=success,
            method=method,
            ip_address=ip_address,
            **kwargs,
        )

    def log_ml_inference(
        self,
        target: str,
        model: str,
        result: Dict[str, Any],
        processing_time_ms: float,
        **kwargs,
    ):
        """Log ML inference"""
        self.logger.info(
            "ML inference completed",
            event_type="ml_inference",
            target=target,
            model=model,
            result=result,
            processing_time_ms=processing_time_ms,
            **kwargs,
        )

    def log_error_with_context(
        self, error: Exception, context: Dict[str, Any], **kwargs
    ):
        """Log error with full context"""
        self.logger.error(
            f"Error: {str(error)}",
            event_type="error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            **kwargs,
        )

    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        tags: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Log performance metric"""
        self.logger.info(
            f"Performance metric: {metric_name}",
            event_type="performance_metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            **kwargs,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ):
        """Log security event"""
        level = "critical" if severity == "high" else "warning"
        getattr(self.logger, level)(
            f"Security event: {description}",
            event_type="security_event",
            security_event_type=event_type,
            severity=severity,
            description=description,
            ip_address=ip_address,
            user_id=user_id,
            **kwargs,
        )


def get_logger(name: str) -> ExoplanetLogger:
    """Get enhanced logger instance"""
    return ExoplanetLogger(name)


# Request logging middleware
class StructuredLoggingMiddleware:
    """Middleware for structured request/response logging"""

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("middleware.logging")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract request info
        method = scope["method"]
        path = scope["path"]
        query_string = scope.get("query_string", b"").decode()
        client_ip = None

        # Get client IP
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"x-forwarded-for":
                client_ip = header_value.decode().split(",")[0].strip()
                break
            elif header_name == b"x-real-ip":
                client_ip = header_value.decode()
                break

        if not client_ip and scope.get("client"):
            client_ip = scope["client"][0]

        start_time = datetime.now()

        # Log request
        self.logger.log_api_request(
            method=method, path=path, query_string=query_string, ip_address=client_ip
        )

        # Capture response
        status_code = 500  # Default to error

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Log unhandled errors
            self.logger.log_error_with_context(
                error=e,
                context={"method": method, "path": path, "ip_address": client_ip},
            )
            raise
        finally:
            # Log response
            end_time = datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            self.logger.log_api_response(
                method=method,
                path=path,
                status_code=status_code,
                processing_time_ms=processing_time_ms,
                ip_address=client_ip,
            )


# Export main functions
__all__ = [
    "configure_structlog",
    "get_logger",
    "ExoplanetLogger",
    "StructuredLoggingMiddleware",
]
