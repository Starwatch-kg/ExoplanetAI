"""
Clean Logging Configuration for Exoplanet AI
Централизованное логирование для production
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class ExoplanetLogger:
    """Enhanced logger for Exoplanet AI"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    def info(self, message: str, **kwargs):
        """Log info message with extra fields"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs):
        """Log warning message with extra fields"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message with extra fields"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.error(message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **kwargs):
        """Log debug message with extra fields"""
        extra = {"extra_fields": kwargs} if kwargs else {}
        self.logger.debug(message, extra=extra)


def setup_logging(
    service_name: str = "exoplanet-ai",
    environment: str = "development",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_json: bool = False,
) -> None:
    """
    Setup centralized logging configuration

    Args:
        service_name: Name of the service
        environment: Environment (development/production)
        log_level: Logging level
        log_file: Path to log file (optional)
        enable_console: Enable console logging
        enable_json: Enable JSON formatting
    """

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Create formatters
    if enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # Configure structlog if using JSON
    if enable_json:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    # Log startup message
    logger = get_logger("exoplanet.startup")
    logger.info(
        f"🚀 Logging initialized for {service_name}",
        service=service_name,
        environment=environment,
        log_level=log_level,
        console_enabled=enable_console,
        file_enabled=bool(log_file),
        json_enabled=enable_json,
    )


def get_logger(name: str) -> ExoplanetLogger:
    """
    Get enhanced logger instance

    Args:
        name: Logger name

    Returns:
        Enhanced logger instance
    """
    return ExoplanetLogger(name)


# Context managers for request logging
class RequestContextLogger:
    """Context manager for request-specific logging"""

    def __init__(self, request_id: str, endpoint: str):
        self.request_id = request_id
        self.endpoint = endpoint
        self.logger = get_logger("exoplanet.request")
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(
            f"🔄 Request started: {self.endpoint}",
            request_id=self.request_id,
            endpoint=self.endpoint,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()

        if exc_type is None:
            self.logger.info(
                f"✅ Request completed: {self.endpoint}",
                request_id=self.request_id,
                endpoint=self.endpoint,
                duration_seconds=duration,
            )
        else:
            self.logger.error(
                f"❌ Request failed: {self.endpoint}",
                request_id=self.request_id,
                endpoint=self.endpoint,
                duration_seconds=duration,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                exc_info=True,
            )


# Decorators for function logging
def log_function_call(logger_name: Optional[str] = None):
    """Decorator to log function calls"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or f"exoplanet.{func.__module__}")

            logger.debug(
                f"🔧 Calling function: {func.__name__}",
                function=func.__name__,
                module=func.__module__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )

            try:
                result = func(*args, **kwargs)
                logger.debug(
                    f"✅ Function completed: {func.__name__}",
                    function=func.__name__,
                    module=func.__module__,
                )
                return result
            except Exception as e:
                logger.error(
                    f"❌ Function failed: {func.__name__}",
                    function=func.__name__,
                    module=func.__module__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# Specialized loggers for different components
def get_api_logger() -> ExoplanetLogger:
    """Get API-specific logger"""
    return get_logger("exoplanet.api")


def get_ml_logger() -> ExoplanetLogger:
    """Get ML-specific logger"""
    return get_logger("exoplanet.ml")


def get_data_logger() -> ExoplanetLogger:
    """Get data service logger"""
    return get_logger("exoplanet.data")


def get_bls_logger() -> ExoplanetLogger:
    """Get BLS service logger"""
    return get_logger("exoplanet.bls")


# Performance logging utilities
def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics"""
    logger = get_logger("exoplanet.performance")

    status = "⚡" if duration < 1.0 else "🐌" if duration > 10.0 else "⏱️"

    logger.info(
        f"{status} Performance: {operation}",
        operation=operation,
        duration_seconds=duration,
        **kwargs,
    )


# Error logging utilities
def log_error_with_context(
    error: Exception, context: Dict[str, Any], logger_name: str = "exoplanet.error"
):
    """Log error with additional context"""
    logger = get_logger(logger_name)

    logger.error(
        f"❌ Error occurred: {type(error).__name__}",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context,
        exc_info=True,
    )
