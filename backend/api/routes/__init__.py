"""
API routes module
Модуль API маршрутов
"""

from fastapi import APIRouter

from .admin import router as admin_router
from .auth import router as auth_router
# Temporarily disabled due to missing astroquery dependencies
# from .data_management import router as data_management_router
from .lightcurves import router as lightcurves_router
from .ml_classification import router as ml_classification_router
from .planets import router as planets_router
from .statistics import router as statistics_router
from .system import router as system_router
from .unified_analysis import router as unified_analysis_router
from .real_unified_analysis import router as real_unified_analysis_router
from .gpi_analysis import router as gpi_analysis_router
from .ai_training import router as ai_training_router
from .auto_discovery import router as auto_discovery_router
from .monitoring import router as monitoring_router
from .scheduler import router as scheduler_router


def create_api_router() -> APIRouter:
    """Create main API router with all sub-routes"""
    api_router = APIRouter(prefix="/api/v1")

    # Public routes
    api_router.include_router(system_router, tags=["System"])
    api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    
    # Real Unified Analysis - главный эндпоинт с реальными данными NASA
    api_router.include_router(real_unified_analysis_router, prefix="/analyze", tags=["Real NASA Analysis"])
    
    # Legacy Unified Analysis (с синтетикой)
    api_router.include_router(unified_analysis_router, prefix="/analyze/legacy", tags=["Legacy Analysis"])
    
    # GPI Analysis - специализированный метод
    api_router.include_router(gpi_analysis_router, prefix="/analyze/gpi", tags=["GPI Analysis"])
    
    # AI Training - обучение моделей
    api_router.include_router(ai_training_router, prefix="/ai", tags=["AI Training"])
    
    # Exoplanets endpoints
    api_router.include_router(planets_router, prefix="/exoplanets", tags=["Exoplanets"])
    
    # Compatibility endpoints (создаем отдельные роутеры для избежания конфликтов)
    from .planets import router as catalog_router
    from .planets import router as database_router
    api_router.include_router(catalog_router, prefix="/catalog", tags=["Catalog"])
    api_router.include_router(database_router, prefix="/database", tags=["Database"])

    # Protected routes
    api_router.include_router(
        lightcurves_router, prefix="/lightcurve", tags=["Light Curves"]
    )
    api_router.include_router(
        ml_classification_router, prefix="/ml", tags=["Machine Learning"]
    )
    api_router.include_router(
        statistics_router, prefix="/statistics", tags=["Statistics"]
    )
    # Temporarily disabled due to missing astroquery dependencies
    # api_router.include_router(
    #     data_management_router, tags=["Data Management"]
    # )

    # Admin routes
    api_router.include_router(admin_router, prefix="/admin", tags=["Administration"])
    
    # Automated Discovery System
    api_router.include_router(auto_discovery_router, tags=["Auto Discovery"])
    api_router.include_router(monitoring_router, tags=["Monitoring"])
    api_router.include_router(scheduler_router, tags=["Scheduler"])

    return api_router
