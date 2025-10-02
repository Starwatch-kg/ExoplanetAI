"""
Clean Configuration Module for Exoplanet AI
Централизованная конфигурация для production
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path


class ServerConfig(BaseSettings):
    """Server configuration"""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8001, description="Server port")
    reload: bool = Field(default=False, description="Auto-reload in development")
    workers: int = Field(default=1, description="Number of worker processes")
    
    class Config:
        env_prefix = "SERVER_"


class DatabaseConfig(BaseSettings):
    """Database configuration (if needed in future)"""
    enabled: bool = Field(default=False, description="Enable database")
    url: Optional[str] = Field(default=None, description="Database URL")
    
    class Config:
        env_prefix = "DB_"


class NASAConfig(BaseSettings):
    """NASA API configuration"""
    mast_api_url: str = Field(
        default="https://mast.stsci.edu/api/v0.1",
        description="MAST API base URL"
    )
    exoplanet_archive_url: str = Field(
        default="https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        description="NASA Exoplanet Archive URL"
    )
    api_timeout: int = Field(default=60, description="API timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum API retries")
    
    class Config:
        env_prefix = "NASA_"


class MLConfig(BaseSettings):
    """Machine Learning configuration"""
    enabled: bool = Field(default=True, description="Enable ML features")
    models_path: str = Field(default="./models", description="Path to ML models")
    device: str = Field(default="auto", description="ML device (cpu/cuda/auto)")
    batch_size: int = Field(default=32, description="ML batch size")
    
    class Config:
        env_prefix = "ML_"


class CacheConfig(BaseSettings):
    """Cache configuration"""
    enabled: bool = Field(default=True, description="Enable caching")
    ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_size: int = Field(default=1000, description="Maximum cache entries")
    
    class Config:
        env_prefix = "CACHE_"


class SecurityConfig(BaseSettings):
    """Security configuration"""
    allowed_origins: List[str] = Field(
        default_factory=lambda: get_allowed_origins("development"),
        description="Allowed CORS origins"
    )
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    
    class Config:
        env_prefix = "SECURITY_"

def get_allowed_origins(env: str):
    # FIX: [FINDING_005] make CORS origins environment-specific
    if env == "production":
        return ["https://exoplanet-ai.example.com"]
    return ["http://localhost:5173", "http://localhost:3000"]


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    enable_console: bool = Field(default=True, description="Enable console logging")
    enable_json: bool = Field(default=False, description="Enable JSON logging")
    
    class Config:
        env_prefix = "LOG_"


class MonitoringConfig(BaseSettings):
    """Monitoring and metrics configuration"""
    enabled: bool = Field(default=False, description="Enable monitoring")
    service_name: str = Field(default="exoplanet-ai", description="Service name")
    enable_tracing: bool = Field(default=False, description="Enable tracing")
    enable_metrics: bool = Field(default=False, description="Enable metrics")
    
    class Config:
        env_prefix = "MONITORING_"


class AppConfig(BaseSettings):
    """Main application configuration"""
    
    # Basic app settings
    name: str = Field(default="ExoplanetAI", description="Application name")
    version: str = Field(default="2.0.0-clean", description="Application version")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Feature flags
    enable_ai_features: bool = Field(default=True, description="Enable AI features")
    enable_real_nasa_data: bool = Field(default=True, description="Use real NASA data")
    
    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    nasa: NASAConfig = Field(default_factory=NASAConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    def model_post_init(self, __context):
        # Update security config based on environment
        self.security.allowed_origins = get_allowed_origins(self.environment)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_log_level(self) -> str:
        """Get log level"""
        return self.logging.level
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "environment": self.environment,
            "debug": self.debug,
            "enable_ai_features": self.enable_ai_features,
            "enable_real_nasa_data": self.enable_real_nasa_data,
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "workers": self.server.workers
            },
            "features": {
                "database": self.database.enabled,
                "ml": self.ml.enabled,
                "cache": self.cache.enabled,
                "monitoring": self.monitoring.enabled
            }
        }
    
    def create_directories(self):
        """Create necessary directories"""
        if self.logging.file_path:
            log_dir = Path(self.logging.file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.ml.enabled:
            models_dir = Path(self.ml.models_path)
            models_dir.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = AppConfig()

# Create necessary directories on import
config.create_directories()
