"""
Exoplanets API routes
Маршруты API экзопланет
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from auth.dependencies import get_optional_user, require_researcher
from auth.models import User
from core.cache import get_cache
from data_sources.registry import get_registry
from schemas.response import ErrorCode, create_error_response, create_success_response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search")
async def search_exoplanets(
    q: str = Query(
        ..., description="Search query (planet name, star name, etc.)", min_length=1
    ),
    limit: int = Query(50, description="Maximum results per source", ge=1, le=200),
    sources: Optional[str] = Query(
        None, description="Comma-separated source types (nasa,tess,kepler)"
    ),
    discovery_year_min: Optional[int] = Query(
        None, description="Minimum discovery year", ge=1990
    ),
    discovery_year_max: Optional[int] = Query(
        None, description="Maximum discovery year", le=2030
    ),
    confirmed_only: bool = Query(False, description="Only confirmed planets"),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Search for exoplanets across multiple data sources

    **Public endpoint** - no authentication required

    **Example queries:**
    - `TOI-715` - Search for specific planet
    - `Kepler` - Search for Kepler planets
    - `TRAPPIST` - Search for TRAPPIST system
    """
    start_time = time.time()

    try:
        # Get registry
        registry = get_registry()

        # Parse source types
        source_types = None
        if sources:
            from data_sources.base import DataSourceType

            try:
                source_types = [
                    DataSourceType(s.strip().lower()) for s in sources.split(",")
                ]
            except ValueError as e:
                return create_error_response(
                    ErrorCode.VALIDATION_ERROR, f"Invalid source type: {e}"
                )

        # Build filters
        filters = {}
        if discovery_year_min:
            filters["discovery_year_min"] = discovery_year_min
        if discovery_year_max:
            filters["discovery_year_max"] = discovery_year_max
        if confirmed_only:
            filters["confirmed_only"] = True

        # Check cache first
        cache = get_cache()
        cache_key = f"search:{q}:{limit}:{sources}:{discovery_year_min}:{discovery_year_max}:{confirmed_only}"
        cached_result = await cache.get("planet_search", cache_key)

        if cached_result:
            logger.info(f"Cache hit for search: {q}")
            cached_result["cached"] = True
            cached_result["search_time_ms"] = (time.time() - start_time) * 1000
            return create_success_response(
                data=cached_result, message="Search results (cached)"
            )

        # Search across sources
        search_results = await registry.search_all_sources(
            query=q, limit=limit, source_types=source_types
        )

        # Cache results for 1 hour
        await cache.set("planet_search", cache_key, search_results, ttl=3600)

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Search completed: '{q}' - {search_results['total_planets_found']} planets found"
        )

        return create_success_response(
            data=search_results,
            message=f"Found {search_results['total_planets_found']} planets",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Search error for '{q}': {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Search failed: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.get("/{planet_name}")
async def get_planet_info(
    planet_name: str = Path(
        ..., description="Planet name (e.g., 'TOI-715 b', 'Kepler-452b')"
    ),
    source: Optional[str] = Query(None, description="Preferred data source"),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get detailed information about a specific exoplanet

    **Public endpoint** - no authentication required

    **Examples:**
    - `/api/v1/exoplanets/TOI-715%20b`
    - `/api/v1/exoplanets/Kepler-452b`
    - `/api/v1/exoplanets/TRAPPIST-1%20e`
    """
    start_time = time.time()

    try:
        # Check cache first
        cache = get_cache()
        cache_key = f"{planet_name}:{source}"
        cached_planet = await cache.get("planets", cache_key)

        if cached_planet:
            logger.info(f"Cache hit for planet: {planet_name}")
            return create_success_response(
                data={"planet": cached_planet, "cached": True},
                message=f"Planet information for {planet_name}",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Get registry and sources
        registry = get_registry()
        sources = registry.get_available_sources()

        if not sources:
            return create_error_response(
                ErrorCode.SERVICE_UNAVAILABLE, "No data sources available"
            )

        # Try to find planet in sources
        planet_info = None
        source_used = None

        # If specific source requested, try it first
        if source:
            specific_source = registry.get_source(source)
            if specific_source and specific_source.is_initialized:
                try:
                    planet_info = await specific_source.fetch_planet_info(planet_name)
                    if planet_info:
                        source_used = source
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source}: {e}")

        # If not found, try all sources
        if not planet_info:
            for src in sources:
                try:
                    planet_info = await src.fetch_planet_info(planet_name)
                    if planet_info:
                        source_used = src.name
                        break
                except Exception as e:
                    logger.debug(f"Planet not found in {src.name}: {e}")
                    continue

        if not planet_info:
            return create_error_response(
                ErrorCode.DATA_NOT_FOUND,
                f"Planet '{planet_name}' not found in any data source",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Convert to dict and cache
        planet_dict = planet_info.__dict__
        planet_dict["source_used"] = source_used

        # Cache for 6 hours
        await cache.set("planets", cache_key, planet_dict, ttl=21600)

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Planet info retrieved: {planet_name} from {source_used}")

        return create_success_response(
            data={"planet": planet_dict, "cached": False},
            message=f"Planet information for {planet_name}",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error fetching planet info for '{planet_name}': {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to fetch planet information: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.get("/{planet_name}/validate")
async def validate_planet(
    planet_name: str = Path(..., description="Planet name to validate"),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Validate if a planet exists in astronomical databases

    **Public endpoint** - no authentication required
    """
    start_time = time.time()

    try:
        registry = get_registry()
        sources = registry.get_available_sources()

        validation_results = {}
        found_in_sources = []

        for source in sources:
            try:
                exists = await source.validate_target(planet_name)
                validation_results[source.name] = {
                    "exists": exists,
                    "source_type": source.source_type.value,
                }
                if exists:
                    found_in_sources.append(source.name)
            except Exception as e:
                validation_results[source.name] = {"exists": False, "error": str(e)}

        is_valid = len(found_in_sources) > 0

        return create_success_response(
            data={
                "planet_name": planet_name,
                "is_valid": is_valid,
                "found_in_sources": found_in_sources,
                "validation_details": validation_results,
            },
            message=f"Validation {'successful' if is_valid else 'failed'} for {planet_name}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        logger.error(f"Validation error for '{planet_name}': {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Validation failed: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.get("/")
async def list_recent_discoveries(
    limit: int = Query(20, description="Number of recent discoveries", ge=1, le=100),
    discovery_year: Optional[int] = Query(None, description="Filter by discovery year"),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get list of recent exoplanet discoveries

    **Public endpoint** - no authentication required
    """
    start_time = time.time()

    try:
        # This is a simplified implementation
        # In a real system, you'd query for recent discoveries from your database

        cache = get_cache()
        cache_key = f"recent:{limit}:{discovery_year}"
        cached_results = await cache.get("recent_discoveries", cache_key)

        if cached_results:
            return create_success_response(
                data=cached_results,
                message="Recent discoveries (cached)",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # For now, return some example recent discoveries
        recent_discoveries = [
            {
                "name": "TOI-715 b",
                "host_star": "TOI-715",
                "discovery_year": 2024,
                "discovery_method": "Transit",
                "discovery_facility": "TESS",
                "status": "confirmed",
                "radius_earth_radii": 1.55,
                "orbital_period_days": 19.3,
                "habitable_zone": True,
            },
            {
                "name": "K2-18 b",
                "host_star": "K2-18",
                "discovery_year": 2015,
                "discovery_method": "Transit",
                "discovery_facility": "K2",
                "status": "confirmed",
                "radius_earth_radii": 2.3,
                "orbital_period_days": 33.0,
                "habitable_zone": True,
                "atmosphere_detected": True,
            },
        ]

        # Filter by year if specified
        if discovery_year:
            recent_discoveries = [
                p
                for p in recent_discoveries
                if p.get("discovery_year") == discovery_year
            ]

        # Limit results
        recent_discoveries = recent_discoveries[:limit]

        result = {
            "discoveries": recent_discoveries,
            "total_count": len(recent_discoveries),
            "filters": {"limit": limit, "discovery_year": discovery_year},
        }

        # Cache for 1 hour
        await cache.set("recent_discoveries", cache_key, result, ttl=3600)

        return create_success_response(
            data=result,
            message=f"Found {len(recent_discoveries)} recent discoveries",
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        logger.error(f"Error fetching recent discoveries: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to fetch recent discoveries: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )
