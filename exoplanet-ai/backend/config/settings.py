"""
Configuration settings for ExoplanetAI backend.
Uses Pydantic for validation and environment variable management.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, validator
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with validation and environment variable support.
    """

    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = True
    RELOAD: bool = True

    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./exoplanet_ai.db"
    DATABASE_TEST_URL: str = "sqlite:///./test.db"

    # Redis Configuration (for caching)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # AI Model Configuration
    ENABLE_AI_FEATURES: bool = True
    MODEL_CACHE_DIR: str = "./ml_models"
    MAX_MODEL_MEMORY_MB: int = 2048

    # Data Processing Configuration
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_FORMATS: List[str] = [".fits", ".csv", ".json", ".txt"]
    TEMP_DATA_DIR: str = "./temp_data"

    # External APIs
    NASA_API_KEY: Optional[str] = None
    MAST_API_TOKEN: Optional[str] = None

    # Monitoring and Logging
    SENTRY_DSN: Optional[str] = None
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Performance Settings
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 300  # seconds
    CACHE_TTL: int = 3600  # seconds

    # Feature Flags
    ENABLE_DATABASE: bool = True
    ENABLE_CACHING: bool = True
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = False

    # BLS Algorithm Configuration
    BLS_CONFIG = {
        "default_min_period": 0.5,
        "default_max_period": 50.0,
        "default_duration": 0.1,
        "default_frequency_factor": 1.0,
        "max_iterations": 1000,
        "convergence_threshold": 1e-6,
    }

    # Transit Detection Configuration
    TRANSIT_CONFIG = {
        "snr_threshold": 7.0,
        "min_transit_duration": 0.01,
        "max_transit_duration": 0.5,
        "phase_coverage": 0.8,
    }

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        Path(settings.MODEL_CACHE_DIR),
        Path(settings.TEMP_DATA_DIR),
        Path("./logs"),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
create_directories()
