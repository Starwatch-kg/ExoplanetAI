"""
API routes module
Модуль API маршрутов
"""

from fastapi import APIRouter

from .admin import router as admin_router
from .auth import router as auth_router
from .lightcurves import router as lightcurves_router
from .planets import router as planets_router
from .statistics import router as statistics_router


def create_api_router() -> APIRouter:
    """Create main API router with all sub-routes"""
    api_router = APIRouter(prefix="/api/v1")

    # Public routes
    api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    api_router.include_router(planets_router, prefix="/exoplanets", tags=["Exoplanets"])

    # Protected routes
    api_router.include_router(
        lightcurves_router, prefix="/lightcurve", tags=["Light Curves"]
    )
    api_router.include_router(
        statistics_router, prefix="/statistics", tags=["Statistics"]
    )

    # Admin routes
    api_router.include_router(admin_router, prefix="/admin", tags=["Administration"])

    return api_router
