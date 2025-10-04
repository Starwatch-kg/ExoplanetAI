"""
Admin API routes
Административные маршруты API
"""

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends

from auth.dependencies import require_admin
from auth.jwt_auth import get_jwt_manager
from auth.models import User
from core.cache import get_cache
from data_sources.registry import get_registry
from schemas.response import ErrorCode, create_error_response, create_success_response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/system-status")
async def get_system_status(current_user: User = Depends(require_admin)):
    """
    Get comprehensive system status

    **🔒 Requires admin role**
    """
    start_time = time.time()

    try:
        # Get data source registry status
        registry = get_registry()
        registry_info = registry.get_registry_info()

        # Get health check from all sources
        health_checks = await registry.health_check_all()

        # Get cache status
        cache = get_cache()
        cache_health = await cache.health_check()
        cache_stats = await cache.get_stats()

        # Get JWT manager info
        jwt_manager = get_jwt_manager()
        all_users = jwt_manager.get_all_users()

        system_status = {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "components": {
                "data_sources": {
                    "status": (
                        "healthy"
                        if registry_info["initialized_sources"] > 0
                        else "degraded"
                    ),
                    "total_sources": registry_info["total_sources"],
                    "initialized_sources": registry_info["initialized_sources"],
                    "sources_by_type": registry_info["sources_by_type"],
                    "health_checks": health_checks,
                },
                "cache": {
                    "status": cache_health.get("status", "unknown"),
                    "redis_connected": cache_health.get("redis_connected", False),
                    "memory_cache_size": cache_health.get("memory_cache_size", 0),
                    "statistics": cache_stats,
                },
                "authentication": {
                    "status": "healthy",
                    "total_users": len(all_users),
                    "active_users": len([u for u in all_users.values() if u.is_active]),
                    "user_roles": {
                        role.value: len(
                            [u for u in all_users.values() if u.role == role]
                        )
                        for role in set(u.role for u in all_users.values())
                    },
                },
            },
        }

        # Determine overall status
        component_statuses = [
            system_status["components"]["data_sources"]["status"],
            system_status["components"]["cache"]["status"],
            system_status["components"]["authentication"]["status"],
        ]

        if "unhealthy" in component_statuses:
            system_status["overall_status"] = "unhealthy"
        elif "degraded" in component_statuses:
            system_status["overall_status"] = "degraded"

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"System status check completed: {system_status['overall_status']}")

        return create_success_response(
            data=system_status,
            message=f"System status: {system_status['overall_status']}",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to get system status: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.get("/users")
async def list_users(current_user: User = Depends(require_admin)):
    """
    List all users in the system

    **🔒 Requires admin role**
    """
    try:
        jwt_manager = get_jwt_manager()
        all_users = jwt_manager.get_all_users()

        # Remove sensitive information
        user_list = []
        for username, user in all_users.items():
            user_info = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role.value,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "institution": user.institution,
                "research_area": user.research_area,
                "daily_request_limit": user.daily_request_limit,
                "monthly_request_limit": user.monthly_request_limit,
            }
            user_list.append(user_info)

        # Sort by creation date
        user_list.sort(key=lambda x: x["created_at"], reverse=True)

        return create_success_response(
            data={
                "users": user_list,
                "total_count": len(user_list),
                "active_count": len([u for u in user_list if u["is_active"]]),
                "role_distribution": {
                    role.value: len([u for u in user_list if u["role"] == role.value])
                    for role in {user.role for user in all_users.values()}
                },
            },
            message=f"Retrieved {len(user_list)} users",
        )

    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR, f"Failed to list users: {str(e)}"
        )


@router.post("/cache/clear")
async def clear_cache(
    namespace: str = None, current_user: User = Depends(require_admin)
):
    """
    Clear cache (all or specific namespace)

    **🔒 Requires admin role**
    """
    try:
        cache = get_cache()

        if namespace:
            # Clear specific namespace
            cleared_count = await cache.clear_namespace(namespace)
            message = f"Cleared {cleared_count} keys from namespace '{namespace}'"
        else:
            # Clear all namespaces
            namespaces = [
                "planets",
                "lightcurves",
                "statistics",
                "planet_search",
                "lc_analysis",
            ]
            total_cleared = 0

            for ns in namespaces:
                cleared = await cache.clear_namespace(ns)
                total_cleared += cleared

            message = f"Cleared {total_cleared} keys from all namespaces"

        logger.info(f"Admin cache clear: {message}")

        return create_success_response(
            data={
                "cleared_keys": cleared_count if namespace else total_cleared,
                "namespace": namespace or "all",
                "action": "cache_clear",
            },
            message=message,
        )

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR, f"Failed to clear cache: {str(e)}"
        )


@router.post("/data-sources/reinitialize")
async def reinitialize_data_sources(current_user: User = Depends(require_admin)):
    """
    Reinitialize all data sources

    **🔒 Requires admin role**
    """
    try:
        registry = get_registry()

        # Cleanup existing sources
        await registry.cleanup_all()

        # Reinitialize all sources
        results = await registry.initialize_all()

        successful = sum(1 for success in results.values() if success)
        total = len(results)

        logger.info(f"Admin reinitialization: {successful}/{total} sources successful")

        return create_success_response(
            data={
                "initialization_results": results,
                "successful_sources": successful,
                "total_sources": total,
                "success_rate": successful / total if total > 0 else 0,
            },
            message=f"Reinitialized {successful}/{total} data sources",
        )

    except Exception as e:
        logger.error(f"Error reinitializing data sources: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR, f"Failed to reinitialize data sources: {str(e)}"
        )


@router.get("/logs/recent")
async def get_recent_logs(
    lines: int = 100, level: str = "INFO", current_user: User = Depends(require_admin)
):
    """
    Get recent log entries

    **🔒 Requires admin role**
    """
    try:
        # This is a simplified implementation
        # In a real system, you'd read from log files or log aggregation system

        recent_logs = {
            "log_level": level,
            "lines_requested": lines,
            "logs": [
                {
                    "timestamp": "2024-10-03T16:39:47+06:00",
                    "level": "INFO",
                    "logger": "data_sources.nasa_service",
                    "message": "✅ NASA Exoplanet Archive initialized",
                    "context": {
                        "service": "exoplanet-ai",
                        "environment": "development",
                    },
                },
                {
                    "timestamp": "2024-10-03T16:39:48+06:00",
                    "level": "INFO",
                    "logger": "data_sources.tess_service",
                    "message": "✅ TESS Mission data source initialized",
                    "context": {
                        "service": "exoplanet-ai",
                        "environment": "development",
                    },
                },
                {
                    "timestamp": "2024-10-03T16:39:49+06:00",
                    "level": "INFO",
                    "logger": "api.routes.planets",
                    "message": "Search completed: 'TOI-715' - 1 planets found",
                    "context": {
                        "service": "exoplanet-ai",
                        "environment": "development",
                    },
                },
            ],
            "note": "This is a demo implementation. In production, integrate with proper log aggregation.",
        }

        return create_success_response(
            data=recent_logs,
            message=f"Retrieved {len(recent_logs['logs'])} recent log entries",
        )

    except Exception as e:
        logger.error(f"Error getting recent logs: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR, f"Failed to get recent logs: {str(e)}"
        )


@router.get("/metrics/detailed")
async def get_detailed_metrics(current_user: User = Depends(require_admin)):
    """
    Get detailed system metrics

    **🔒 Requires admin role**
    """
    try:
        # Get cache metrics
        cache = get_cache()
        cache_stats = await cache.get_stats()

        # Get data source registry info
        registry = get_registry()
        registry_info = registry.get_registry_info()

        detailed_metrics = {
            "cache_performance": cache_stats,
            "data_sources": {
                "registry_info": registry_info,
                "source_capabilities": {
                    source.name: source.get_capabilities()
                    for source in registry.get_all_sources()
                },
            },
            "system_info": {
                "python_version": "3.11+",
                "framework": "FastAPI",
                "async_support": True,
                "real_data_only": True,
            },
        }

        return create_success_response(
            data=detailed_metrics, message="Detailed system metrics retrieved"
        )

    except Exception as e:
        logger.error(f"Error getting detailed metrics: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR, f"Failed to get detailed metrics: {str(e)}"
        )
