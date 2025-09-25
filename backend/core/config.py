"""
Centralized Configuration Management
Централизованное управление конфигурацией приложения
"""

import os
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from pydantic.env_settings import SettingsSourceCallable
import json

class DatabaseConfig(BaseSettings):
    """Конфигурация базы данных"""
    
    url: str = Field(
        default="sqlite:///./exoplanet_ai.db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    
    max_connections: int = Field(
        default=10,
        env="DATABASE_MAX_CONNECTIONS",
        description="Maximum number of database connections"
    )
    
    connection_timeout: int = Field(
        default=30,
        env="DATABASE_CONNECTION_TIMEOUT",
        description="Database connection timeout in seconds"
    )
    
    enable_query_logging: bool = Field(
        default=False,
        env="DATABASE_ENABLE_QUERY_LOGGING",
        description="Enable SQL query logging"
    )
    
    pool_pre_ping: bool = Field(
        default=True,
        env="DATABASE_POOL_PRE_PING",
        description="Enable connection pool pre-ping"
    )

class SecurityConfig(BaseSettings):
    """Конфигурация безопасности"""
    
    secret_key: str = Field(
        default="exoplanet_ai_production_secret_key_2024_v2_secure",
        env="SECRET_KEY",
        description="Secret key for JWT and encryption"
    )
    
    algorithm: str = Field(
        default="HS256",
        env="ALGORITHM",
        description="JWT algorithm"
    )
    
    access_token_expire_minutes: int = Field(
        default=1440,  # 24 hours
        env="ACCESS_TOKEN_EXPIRE_MINUTES",
        description="Access token expiration time in minutes"
    )
    
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:5174", 
            "http://localhost:3000",
            "https://exoplanet-ai.netlify.app"
        ],
        env="ALLOWED_ORIGINS",
        description="Allowed CORS origins"
    )
    
    @validator('allowed_origins', pre=True)
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

class LoggingConfig(BaseSettings):
    """Конфигурация логирования"""
    
    level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    
    format: str = Field(
        default="json",
        env="LOG_FORMAT",
        description="Log format: json or text"
    )
    
    file_path: Optional[str] = Field(
        default=None,
        env="LOG_FILE_PATH",
        description="Path to log file"
    )
    
    max_file_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        env="LOG_MAX_FILE_SIZE",
        description="Maximum log file size in bytes"
    )
    
    backup_count: int = Field(
        default=5,
        env="LOG_BACKUP_COUNT",
        description="Number of log file backups to keep"
    )
    
    enable_console: bool = Field(
        default=True,
        env="LOG_ENABLE_CONSOLE",
        description="Enable console logging"
    )

class APIConfig(BaseSettings):
    """Конфигурация внешних API"""
    
    # NASA APIs
    mast_api_url: str = Field(
        default="https://mast.stsci.edu/api/v0.1",
        env="MAST_API_URL",
        description="NASA MAST API URL"
    )
    
    exoplanet_archive_url: str = Field(
        default="https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        env="EXOPLANET_ARCHIVE_URL",
        description="NASA Exoplanet Archive URL"
    )
    
    simbad_api_url: str = Field(
        default="http://simbad.u-strasbg.fr/simbad/sim-tap",
        env="SIMBAD_API_URL",
        description="SIMBAD API URL"
    )
    
    # API Keys (если потребуются)
    nasa_api_key: Optional[str] = Field(
        default=None,
        env="NASA_API_KEY",
        description="NASA API key (if required)"
    )
    
    # Timeouts and limits
    api_timeout: int = Field(
        default=30,
        env="API_TIMEOUT",
        description="API request timeout in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        env="API_MAX_RETRIES",
        description="Maximum number of API retries"
    )
    
    rate_limit_per_minute: int = Field(
        default=60,
        env="API_RATE_LIMIT_PER_MINUTE",
        description="API rate limit per minute"
    )

class MLConfig(BaseSettings):
    """Конфигурация ML моделей"""
    
    models_path: str = Field(
        default="./ai/models",
        env="ML_MODELS_PATH",
        description="Path to ML models directory"
    )
    
    cache_size: int = Field(
        default=5000,
        env="ML_CACHE_SIZE",
        description="ML cache size"
    )
    
    batch_size: int = Field(
        default=32,
        env="ML_BATCH_SIZE",
        description="ML batch size"
    )
    
    device: str = Field(
        default="auto",
        env="ML_DEVICE",
        description="ML device: auto, cpu, cuda"
    )
    
    enable_gpu: bool = Field(
        default=True,
        env="ML_ENABLE_GPU",
        description="Enable GPU acceleration"
    )
    
    model_timeout: int = Field(
        default=300,
        env="ML_MODEL_TIMEOUT",
        description="ML model timeout in seconds"
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        env="ML_CONFIDENCE_THRESHOLD",
        description="ML confidence threshold"
    )

class PerformanceConfig(BaseSettings):
    """Конфигурация производительности"""
    
    max_concurrent_requests: int = Field(
        default=10,
        env="MAX_CONCURRENT_REQUESTS",
        description="Maximum concurrent requests"
    )
    
    request_timeout: int = Field(
        default=300,
        env="REQUEST_TIMEOUT",
        description="Request timeout in seconds"
    )
    
    bls_max_periods: int = Field(
        default=5000,
        env="BLS_MAX_PERIODS",
        description="Maximum periods for BLS analysis"
    )
    
    lightcurve_max_points: int = Field(
        default=50000,
        env="LIGHTCURVE_MAX_POINTS",
        description="Maximum points in lightcurve"
    )
    
    max_request_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        env="MAX_REQUEST_SIZE",
        description="Maximum request size in bytes"
    )
    
    worker_threads: int = Field(
        default=4,
        env="WORKER_THREADS",
        description="Number of worker threads"
    )

class CacheConfig(BaseSettings):
    """Конфигурация кэширования"""
    
    redis_url: Optional[str] = Field(
        default=None,
        env="REDIS_URL",
        description="Redis connection URL"
    )
    
    cache_ttl: int = Field(
        default=3600,
        env="CACHE_TTL",
        description="Cache TTL in seconds"
    )
    
    enable_memory_cache: bool = Field(
        default=True,
        env="ENABLE_MEMORY_CACHE",
        description="Enable in-memory caching"
    )
    
    memory_cache_size: int = Field(
        default=1000,
        env="MEMORY_CACHE_SIZE",
        description="Memory cache size"
    )

class MonitoringConfig(BaseSettings):
    """Конфигурация мониторинга"""
    
    # Prometheus
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS",
        description="Enable Prometheus metrics"
    )
    
    metrics_path: str = Field(
        default="/metrics",
        env="METRICS_PATH",
        description="Metrics endpoint path"
    )
    
    # OpenTelemetry
    enable_tracing: bool = Field(
        default=True,
        env="ENABLE_TRACING",
        description="Enable OpenTelemetry tracing"
    )
    
    otlp_endpoint: Optional[str] = Field(
        default=None,
        env="OTEL_EXPORTER_OTLP_ENDPOINT",
        description="OTLP endpoint URL"
    )
    
    service_name: str = Field(
        default="exoplanet-ai",
        env="OTEL_SERVICE_NAME",
        description="Service name for tracing"
    )
    
    service_version: str = Field(
        default="2.0.0",
        env="OTEL_SERVICE_VERSION",
        description="Service version for tracing"
    )
    
    # Health checks
    health_check_interval: int = Field(
        default=30,
        env="HEALTH_CHECK_INTERVAL",
        description="Health check interval in seconds"
    )

class RateLimitConfig(BaseSettings):
    """Конфигурация rate limiting"""
    
    enabled: bool = Field(
        default=True,
        env="RATE_LIMIT_ENABLED",
        description="Enable rate limiting"
    )
    
    requests_per_minute: int = Field(
        default=100,
        env="RATE_LIMIT_REQUESTS_PER_MINUTE",
        description="Requests per minute limit"
    )
    
    burst_requests: int = Field(
        default=20,
        env="RATE_LIMIT_BURST_REQUESTS",
        description="Burst requests limit"
    )
    
    burst_window: int = Field(
        default=1,
        env="RATE_LIMIT_BURST_WINDOW",
        description="Burst window in seconds"
    )
    
    exclude_paths: List[str] = Field(
        default=["/health", "/metrics", "/docs", "/redoc"],
        env="RATE_LIMIT_EXCLUDE_PATHS",
        description="Paths to exclude from rate limiting"
    )
    
    @validator('exclude_paths', pre=True)
    def parse_exclude_paths(cls, v):
        if isinstance(v, str):
            return [path.strip() for path in v.split(',')]
        return v

class AppConfig(BaseSettings):
    """Основная конфигурация приложения"""
    
    # Основные настройки
    app_name: str = Field(
        default="Exoplanet AI",
        env="APP_NAME",
        description="Application name"
    )
    
    version: str = Field(
        default="2.0.0",
        env="APP_VERSION",
        description="Application version"
    )
    
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Environment: development, staging, production"
    )
    
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    
    # Функциональность
    enable_ai_features: bool = Field(
        default=False,
        env="ENABLE_AI_FEATURES",
        description="Enable AI features"
    )
    
    enable_database: bool = Field(
        default=True,
        env="ENABLE_DATABASE",
        description="Enable database"
    )
    
    enable_real_nasa_data: bool = Field(
        default=True,
        env="ENABLE_REAL_NASA_DATA",
        description="Enable real NASA data fetching"
    )
    
    # Вложенные конфигурации
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    api: APIConfig = APIConfig()
    ml: MLConfig = MLConfig()
    performance: PerformanceConfig = PerformanceConfig()
    cache: CacheConfig = CacheConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )
    
    def is_production(self) -> bool:
        """Проверка production окружения"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Проверка development окружения"""
        return self.environment.lower() == "development"
    
    def get_log_level(self) -> str:
        """Получение уровня логирования"""
        if self.debug:
            return "DEBUG"
        return self.logging.level
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь (без секретов)"""
        config_dict = self.dict()
        
        # Маскируем секретные данные
        secret_keys = ['secret_key', 'nasa_api_key', 'redis_url', 'database_url']
        
        def mask_secrets(obj, path=""):
            if isinstance(obj, dict):
                return {
                    k: "***MASKED***" if any(secret in k.lower() for secret in secret_keys)
                    else mask_secrets(v, f"{path}.{k}" if path else k)
                    for k, v in obj.items()
                }
            return obj
        
        return mask_secrets(config_dict)
    
    def validate_config(self) -> List[str]:
        """Валидация конфигурации"""
        errors = []
        
        # Проверяем обязательные настройки для production
        if self.is_production():
            if self.security.secret_key == "exoplanet_ai_production_secret_key_2024_v2_secure":
                errors.append("Default secret key should not be used in production")
            
            if self.debug:
                errors.append("Debug mode should be disabled in production")
            
            if not self.monitoring.enable_metrics:
                errors.append("Metrics should be enabled in production")
        
        # Проверяем пути
        if self.ml.models_path:
            models_path = Path(self.ml.models_path)
            if not models_path.exists():
                errors.append(f"ML models path does not exist: {self.ml.models_path}")
        
        # Проверяем URL базы данных
        if self.enable_database and not self.database.url:
            errors.append("Database URL is required when database is enabled")
        
        return errors

# Глобальная конфигурация
config = AppConfig()

# Функции для работы с конфигурацией
def get_config() -> AppConfig:
    """Получение глобальной конфигурации"""
    return config

def reload_config() -> AppConfig:
    """Перезагрузка конфигурации"""
    global config
    config = AppConfig()
    return config

def validate_environment():
    """Валидация окружения"""
    errors = config.validate_config()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        raise ValueError(error_msg)

def create_directories():
    """Создание необходимых директорий"""
    directories = [
        config.ml.models_path,
        "logs",
        "data/cache",
        "data/embeddings"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Инициализация при импорте
try:
    validate_environment()
    create_directories()
except Exception as e:
    print(f"Warning: Configuration validation failed: {e}")
    print("Continuing with default configuration...")

# Экспорт для удобства
__all__ = [
    'config',
    'get_config',
    'reload_config',
    'validate_environment',
    'create_directories',
    'AppConfig',
    'DatabaseConfig',
    'SecurityConfig',
    'LoggingConfig',
    'APIConfig',
    'MLConfig',
    'PerformanceConfig',
    'CacheConfig',
    'MonitoringConfig',
    'RateLimitConfig'
]
