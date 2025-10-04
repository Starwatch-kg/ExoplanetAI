"""
System endpoints for health checks and monitoring
Системные эндпоинты для проверки здоровья и мониторинга
"""

import time
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.cache import get_cache
from core.logging import get_logger
from data_sources.registry import get_registry

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    System health check endpoint
    Эндпоинт проверки здоровья системы
    """
    try:
        # Check data sources
        registry = get_registry()
        registry_info = registry.get_registry_info()

        # Check cache
        cache = get_cache()
        cache_health = await cache.health_check()

        # Determine overall health
        overall_status = "healthy"
        if registry_info["initialized_sources"] == 0:
            overall_status = "degraded"
        if cache_health.get("status") == "unhealthy":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "timestamp": time.time(),
            "version": "2.0.0",
            "components": {
                "data_sources": {
                    "status": (
                        "healthy"
                        if registry_info["initialized_sources"] > 0
                        else "unhealthy"
                    ),
                    "initialized": registry_info["initialized_sources"],
                    "total": registry_info["total_sources"],
                },
                "cache": {
                    "status": cache_health.get("status", "unknown"),
                    "redis_connected": cache_health.get("redis_connected", False),
                },
                "authentication": {"status": "healthy"},
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "timestamp": time.time()},
        )


@router.get("/status")
async def system_status():
    """
    Detailed system status
    Детальный статус системы
    """
    try:
        # Get cache stats
        cache = get_cache()
        cache_stats = await cache.get_stats()

        # Get registry info
        registry = get_registry()
        registry_info = registry.get_registry_info()

        return {
            "status": "operational",
            "timestamp": time.time(),
            "version": "2.0.0",
            "data_sources": {
                "total": registry_info["total_sources"],
                "initialized": registry_info["initialized_sources"],
                "available_sources": registry_info.get("available_sources", []),
            },
            "cache": {
                "status": cache_stats.get("status", "unknown"),
                "hits": cache_stats.get("hits", 0),
                "misses": cache_stats.get("misses", 0),
                "hit_rate": cache_stats.get("hit_rate", 0.0),
            },
            "features": {
                "real_data_only": True,
                "synthetic_data": False,
                "authentication": "JWT with role-based access",
                "rate_limiting": True,
            },
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "timestamp": time.time()},
        )
