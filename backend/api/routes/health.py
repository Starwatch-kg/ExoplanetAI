"""
Health check API routes
"""
import time
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.cache import get_cache
from data_sources.registry import get_registry

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    """
    System health check endpoint for API v1
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
            "uptime_seconds": 0,  # Will be calculated by the main app
        }

    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "error": str(e), 
                "timestamp": time.time()
            },
        )
