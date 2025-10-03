"""
Modular data sources package for ExoplanetAI
Модульная система источников астрономических данных
"""

from .base import BaseDataSource, DataSourceError
from .esa import ESADataSource
from .kepler import KeplerDataSource
from .nasa import NASADataSource
from .registry import DataSourceRegistry, get_registry
from .tess import TESSDataSource

__all__ = [
    "BaseDataSource",
    "DataSourceError",
    "DataSourceRegistry",
    "get_registry",
    "NASADataSource",
    "ESADataSource",
    "KeplerDataSource",
    "TESSDataSource",
]
