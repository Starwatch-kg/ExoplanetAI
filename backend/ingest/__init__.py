"""
Data ingestion and management system for ExoplanetAI
Система ингеста и управления данными для ExoplanetAI
"""

from .data_manager import DataManager
from .validator import DataValidator
from .storage import StorageManager
from .versioning import VersionManager

__all__ = [
    "DataManager",
    "DataValidator", 
    "StorageManager",
    "VersionManager"
]
