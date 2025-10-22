"""
Database API routes - Demo database endpoints
Маршруты API базы данных - Demo эндпоинты базы данных
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Query
from schemas.response import create_success_response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/statistics")
async def get_database_statistics():
    """
    Get database statistics
    Получить статистику базы данных
    """
    try:
        return create_success_response({
            "total_planets": 1247,
            "confirmed_planets": 892,
            "candidate_planets": 355,
            "total_stars": 1089,
            "habitable_planets": 156,
            "recent_discoveries": 23,
            "data_sources": {
                "NASA": 567,
                "TESS": 423,
                "Kepler": 234,
                "K2": 23
            },
            "last_updated": datetime.now().isoformat(),
            "source": "Demo Database"
        })
        
    except Exception as e:
        logger.error(f"Error getting database statistics: {e}")
        return create_success_response({
            "error": str(e),
            "total_planets": 0
        })


@router.get("/search-history")
async def get_search_history(
    limit: int = Query(50, description="Number of search records", ge=1, le=200)
):
    """
    Get search history
    Получить историю поиска
    """
    try:
        # Generate demo search history
        searches = []
        search_terms = [
            "TOI-715", "Kepler-452b", "WASP-12b", "HD 209458b", "TIC 307210830",
            "habitable planets", "hot jupiter", "super earth", "transit", "radial velocity"
        ]
        
        for i in range(min(limit, 20)):
            search = {
                "id": i + 1,
                "query": search_terms[i % len(search_terms)],
                "timestamp": (datetime.now() - timedelta(hours=i*2)).isoformat(),
                "results_count": 1 + (i % 5),
                "user_id": f"demo_user_{(i % 3) + 1}",
                "execution_time_ms": 150 + (i % 300)
            }
            searches.append(search)
        
        return create_success_response({
            "searches": searches,
            "total_count": 1234,
            "limit": limit,
            "source": "Demo Database"
        })
        
    except Exception as e:
        logger.error(f"Error getting search history: {e}")
        return create_success_response({
            "searches": [],
            "error": str(e)
        })


@router.get("/metrics")
async def get_database_metrics(
    hours: int = Query(24, description="Time range in hours", ge=1, le=168)
):
    """
    Get database metrics
    Получить метрики базы данных
    """
    try:
        return create_success_response({
            "time_range_hours": hours,
            "total_queries": 1547,
            "successful_queries": 1423,
            "failed_queries": 124,
            "average_response_time_ms": 245,
            "cache_hit_rate": 0.78,
            "unique_users": 89,
            "popular_searches": [
                {"query": "TOI-715", "count": 156},
                {"query": "habitable", "count": 134},
                {"query": "Kepler-452b", "count": 98},
                {"query": "transit", "count": 87},
                {"query": "TESS", "count": 76}
            ],
            "api_endpoints_usage": {
                "/api/v1/exoplanets/search": 567,
                "/api/v1/lightcurve/demo": 423,
                "/api/v1/analyze/analyze": 234,
                "/api/v1/catalog/exoplanets": 189
            },
            "timestamp": datetime.now().isoformat(),
            "source": "Demo Database"
        })
        
    except Exception as e:
        logger.error(f"Error getting database metrics: {e}")
        return create_success_response({
            "error": str(e),
            "total_queries": 0
        })
