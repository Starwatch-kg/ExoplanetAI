"""
Unified Configuration System
Объединенная система конфигурации для всего приложения
"""

import os
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field, validator
import torch

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    
    url: str = Field(
        default="sqlite:///./exoplanet_ai.db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    
    max_connections: int = Field(
        default=10,
        env="DATABASE_MAX_CONNECTIONS"
    )
    
    connection_timeout: int = Field(
        default=30,
        env="DATABASE_CONNECTION_TIMEOUT"
    )
    
    enable_query_logging: bool = Field(
        default=False,
        env="DATABASE_ENABLE_QUERY_LOGGING"
    )

class SecuritySettings(BaseSettings):
    """Security configuration"""
    
    secret_key: str = Field(
        default="exoplanet_ai_production_secret_key_2024_v2_secure",
        env="SECRET_KEY"
    )
    
    algorithm: str = Field(
        default="HS256",
        env="ALGORITHM"
    )
    
    access_token_expire_minutes: int = Field(
        default=1440,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    allowed_origins: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:5174", 
            "http://localhost:3000"
        ],
        env="ALLOWED_ORIGINS"
    )
    
    @validator('allowed_origins', pre=True)
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    
    level: str = Field(
        default="INFO",
        env="LOG_LEVEL"
    )
    
    format: str = Field(
        default="json",
        env="LOG_FORMAT"
    )
    
    file_path: Optional[str] = Field(
        default="logs/exoplanet_ai.log",
        env="LOG_FILE_PATH"
    )
    
    enable_console: bool = Field(
        default=True,
        env="LOG_ENABLE_CONSOLE"
    )

class APISettings(BaseSettings):
    """External API configuration"""
    
    mast_api_url: str = Field(
        default="https://mast.stsci.edu/api/v0.1",
        env="MAST_API_URL"
    )
    
    exoplanet_archive_url: str = Field(
        default="https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        env="EXOPLANET_ARCHIVE_URL"
    )
    
    nasa_api_key: Optional[str] = Field(
        default=None,
        env="NASA_API_KEY"
    )
    
    api_timeout: int = Field(
        default=30,
        env="API_TIMEOUT"
    )
    
    max_retries: int = Field(
        default=3,
        env="API_MAX_RETRIES"
    )

class MLSettings(BaseSettings):
    """Machine Learning configuration"""
    
    models_path: str = Field(
        default="./models",
        env="ML_MODELS_PATH"
    )
    
    cache_size: int = Field(
        default=5000,
        env="ML_CACHE_SIZE"
    )
    
    batch_size: int = Field(
        default=32,
        env="ML_BATCH_SIZE"
    )
    
    device: str = Field(
        default="auto",
        env="ML_DEVICE"
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        env="ML_CONFIDENCE_THRESHOLD"
    )
    
    @validator('device')
    def validate_device(cls, v):
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

class BLSSettings(BaseSettings):
    """BLS Analysis configuration"""
    
    max_periods: int = Field(
        default=1000,
        env="BLS_MAX_PERIODS"
    )
    
    max_durations: int = Field(
        default=10,
        env="BLS_MAX_DURATIONS"
    )
    
    snr_threshold: float = Field(
        default=7.0,
        env="BLS_SNR_THRESHOLD"
    )
    
    use_enhanced: bool = Field(
        default=True,
        env="BLS_USE_ENHANCED"
    )

class PerformanceSettings(BaseSettings):
    """Performance configuration"""
    
    max_concurrent_requests: int = Field(
        default=10,
        env="MAX_CONCURRENT_REQUESTS"
    )
    
    request_timeout: int = Field(
        default=300,
        env="REQUEST_TIMEOUT"
    )
    
    max_request_size: int = Field(
        default=10 * 1024 * 1024,
        env="MAX_REQUEST_SIZE"
    )
    
    worker_threads: int = Field(
        default=4,
        env="WORKER_THREADS"
    )

class CacheSettings(BaseSettings):
    """Cache configuration"""
    
    redis_url: Optional[str] = Field(
        default=None,
        env="REDIS_URL"
    )
    
    cache_ttl: int = Field(
        default=3600,
        env="CACHE_TTL"
    )
    
    enable_memory_cache: bool = Field(
        default=True,
        env="ENABLE_MEMORY_CACHE"
    )
    
    memory_cache_size: int = Field(
        default=1000,
        env="MEMORY_CACHE_SIZE"
    )

class MonitoringSettings(BaseSettings):
    """Monitoring configuration"""
    
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS"
    )
    
    metrics_path: str = Field(
        default="/metrics",
        env="METRICS_PATH"
    )
    
    enable_tracing: bool = Field(
        default=True,
        env="ENABLE_TRACING"
    )
    
    otlp_endpoint: Optional[str] = Field(
        default=None,
        env="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    
    service_name: str = Field(
        default="exoplanet-ai",
        env="OTEL_SERVICE_NAME"
    )
    
    service_version: str = Field(
        default="2.0.0",
        env="OTEL_SERVICE_VERSION"
    )

class RateLimitSettings(BaseSettings):
    """Rate limiting configuration"""
    
    enabled: bool = Field(
        default=True,
        env="RATE_LIMIT_ENABLED"
    )
    
    requests_per_minute: int = Field(
        default=100,
        env="RATE_LIMIT_REQUESTS_PER_MINUTE"
    )
    
    burst_requests: int = Field(
        default=20,
        env="RATE_LIMIT_BURST_REQUESTS"
    )
    
    burst_window: int = Field(
        default=1,
        env="RATE_LIMIT_BURST_WINDOW"
    )
    
    exclude_paths: List[str] = Field(
        default=["/health", "/metrics", "/docs", "/redoc"],
        env="RATE_LIMIT_EXCLUDE_PATHS"
    )
    
    @validator('exclude_paths', pre=True)
    def parse_exclude_paths(cls, v):
        if isinstance(v, str):
            return [path.strip() for path in v.split(',')]
        return v

class Settings(BaseSettings):
    """Main application settings"""
    
    # Application info
    app_name: str = Field(
        default="Exoplanet AI",
        env="APP_NAME"
    )
    
    version: str = Field(
        default="2.0.0",
        env="APP_VERSION"
    )
    
    environment: str = Field(
        default="development",
        env="ENVIRONMENT"
    )
    
    debug: bool = Field(
        default=False,
        env="DEBUG"
    )
    
    # Feature flags
    enable_ai_features: bool = Field(
        default=True,
        env="ENABLE_AI_FEATURES"
    )
    
    enable_database: bool = Field(
        default=True,
        env="ENABLE_DATABASE"
    )
    
    enable_real_nasa_data: bool = Field(
        default=True,
        env="ENABLE_REAL_NASA_DATA"
    )
    
    # Nested settings
    database: DatabaseSettings = DatabaseSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()
    api: APISettings = APISettings()
    ml: MLSettings = MLSettings()
    bls: BLSSettings = BLSSettings()
    performance: PerformanceSettings = PerformanceSettings()
    cache: CacheSettings = CacheSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"
    
    def get_log_level(self) -> str:
        """Get effective log level"""
        if self.debug:
            return "DEBUG"
        return self.logging.level
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (masking secrets)"""
        config_dict = self.dict()
        
        # Mask secret keys
        secret_keys = ['secret_key', 'nasa_api_key', 'redis_url']
        
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
        """Validate configuration"""
        errors = []
        
        # Production checks
        if self.is_production():
            if self.security.secret_key == "exoplanet_ai_production_secret_key_2024_v2_secure":
                errors.append("Default secret key should not be used in production")
            
            if self.debug:
                errors.append("Debug mode should be disabled in production")
        
        # Path checks
        models_path = Path(self.ml.models_path)
        if not models_path.exists():
            try:
                models_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create models directory: {e}")
        
        return errors
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.ml.models_path,
            "logs",
            "data/cache"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()

# Validate and create directories on import
try:
    errors = settings.validate_config()
    if errors:
        print(f"Configuration warnings: {errors}")
    
    settings.create_directories()
except Exception as e:
    print(f"Configuration setup failed: {e}")

# Export for convenience
__all__ = [
    'settings',
    'Settings',
    'DatabaseSettings',
    'SecuritySettings',
    'LoggingSettings',
    'APISettings',
    'MLSettings',
    'BLSSettings',
    'PerformanceSettings',
    'CacheSettings',
    'MonitoringSettings',
    'RateLimitSettings'
]
