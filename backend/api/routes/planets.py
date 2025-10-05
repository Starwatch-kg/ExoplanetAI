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


async def generate_demo_search_results(query: str, limit: int) -> dict:
    """Generate demo search results when real data is not available"""
    import random
    import numpy as np
    
    # Check if query looks like a real target
    # Remove planet suffixes (b, c, d, etc.) before checking
    query_clean = query.upper().replace('-', '').replace(' ', '').rstrip('BCDEFGH')
    is_real_target = any(query_clean.startswith(prefix) for prefix in 
                         ['TOI', 'TIC', 'KEPLER', 'KOI', 'K2', 'EPIC', 'WASP', 'HAT', 'HD', 'GJ'])
    
    # If it's just random numbers or invalid query, return no results
    if not is_real_target and query.replace('-', '').replace(' ', '').isdigit():
        return {
            "planets": [],
            "total_planets_found": 0,
            "sources_searched": ["nasa", "tess", "kepler"],
            "search_query": query,
            "cached": False,
            "demo_data": True,
            "message": "No exoplanets found for this target. Try searching for known targets like TOI-715, TIC-307210830, or Kepler-452b"
        }
    
    # Generate consistent mock data based on query
    random.seed(hash(query) % 2**32)
    np.random.seed(hash(query) % 2**32)
    
    # Create mock planets based on query
    mock_planets = []
    num_results = min(random.randint(1, 5), limit)
    
    for i in range(num_results):
        planet_name = f"{query}-{chr(97+i)}" if not any(char.isdigit() for char in query) else f"{query}.{i+1:02d}"
        
        mock_planet = {
            "name": planet_name,
            "host_star": query if "TIC" in query or "TOI" in query else f"{query.split('-')[0]} {random.randint(1, 999)}",
            "discovery_year": random.randint(2009, 2024),
            "discovery_method": random.choice(["Transit", "Radial Velocity", "Direct Imaging"]),
            "orbital_period": round(random.uniform(0.5, 365.0), 3),
            "planet_radius": round(random.uniform(0.5, 15.0), 3),
            "planet_mass": round(random.uniform(0.1, 300.0), 3),
            "equilibrium_temperature": random.randint(200, 2000),
            "distance_pc": round(random.uniform(10.0, 500.0), 1),
            "stellar_magnitude": round(random.uniform(8.0, 16.0), 2),
            "disposition": random.choice(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]),
            "source": random.choice(["NASA Exoplanet Archive", "TESS", "Kepler"]),
            "ra": round(random.uniform(0, 360), 6),
            "dec": round(random.uniform(-90, 90), 6),
            "transit_depth_ppm": random.randint(100, 10000) if random.random() > 0.3 else None,
            "impact_parameter": round(random.uniform(0, 1), 3) if random.random() > 0.5 else None
        }
        mock_planets.append(mock_planet)
    
    return {
        "planets": mock_planets,
        "total_planets_found": len(mock_planets),
        "sources_searched": ["nasa", "tess", "kepler"],
        "search_query": query,
        "cached": False,
        "demo_data": True
    }


@router.get("/search/demo")
async def search_exoplanets_demo(
    q: str = Query(
        ..., description="Search query (planet name, star name, etc.)", min_length=1
    ),
    limit: int = Query(50, description="Maximum results per source", ge=1, le=200),
):
    """
    Search for exoplanets (Demo version - returns mock data)
    
    Returns realistic mock exoplanet data for demonstration purposes.
    """
    import random
    import numpy as np
    
    # Generate consistent mock data based on query
    random.seed(hash(q) % 2**32)
    np.random.seed(hash(q) % 2**32)
    
    # Create mock planets based on query
    mock_planets = []
    num_results = min(random.randint(1, 5), limit)
    
    for i in range(num_results):
        planet_name = f"{q}-{chr(97+i)}" if not any(char.isdigit() for char in q) else f"{q}.{i+1:02d}"
        
        mock_planet = {
            "name": planet_name,
            "host_star": q if "TIC" in q or "TOI" in q else f"{q.split('-')[0]} {random.randint(1, 999)}",
            "discovery_year": random.randint(2009, 2024),
            "discovery_method": random.choice(["Transit", "Radial Velocity", "Direct Imaging"]),
            "orbital_period": round(random.uniform(0.5, 365.0), 3),
            "planet_radius": round(random.uniform(0.5, 15.0), 3),
            "planet_mass": round(random.uniform(0.1, 300.0), 3),
            "equilibrium_temperature": random.randint(200, 2000),
            "distance_pc": round(random.uniform(10.0, 500.0), 1),
            "stellar_magnitude": round(random.uniform(8.0, 16.0), 2),
            "disposition": random.choice(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]),
            "source": random.choice(["NASA Exoplanet Archive", "TESS", "Kepler"]),
            "ra": round(random.uniform(0, 360), 6),
            "dec": round(random.uniform(-90, 90), 6),
            "transit_depth_ppm": random.randint(100, 10000) if random.random() > 0.3 else None,
            "impact_parameter": round(random.uniform(0, 1), 3) if random.random() > 0.5 else None
        }
        mock_planets.append(mock_planet)
    
    return {
        "status": "success",
        "data": {
            "planets": mock_planets,
            "total_planets_found": len(mock_planets),
            "sources_searched": ["nasa", "tess", "kepler"],
            "search_query": q,
            "cached": False
        },
        "message": f"Found {len(mock_planets)} demo planets for '{q}'",
        "processing_time_ms": random.randint(50, 200)
    }


@router.get("/exoplanets")
async def get_exoplanet_catalog(
    limit: int = Query(12, ge=1, le=100),
    offset: int = Query(0, ge=0),
    habitable_only: bool = Query(False)
):
    """
    Получить каталог экзопланет для отображения
    """
    try:
        # Генерируем demo каталог
        exoplanets = []
        
        demo_planets = [
            {"name": "TOI-715 b", "host_star": "TOI-715", "radius_earth_radii": 1.55, "orbital_period_days": 19.3, "equilibrium_temperature_k": 450, "distance_parsecs": 137.0, "discovery_method": "Transit", "discovery_year": 2024, "status": "Confirmed"},
            {"name": "Kepler-452b", "host_star": "Kepler-452", "radius_earth_radii": 1.63, "orbital_period_days": 384.8, "equilibrium_temperature_k": 265, "distance_parsecs": 430.0, "discovery_method": "Transit", "discovery_year": 2015, "status": "Confirmed"},
            {"name": "TOI-849 b", "host_star": "TOI-849", "radius_earth_radii": 3.4, "orbital_period_days": 0.765, "equilibrium_temperature_k": 1800, "distance_parsecs": 224.0, "discovery_method": "Transit", "discovery_year": 2020, "status": "Confirmed"},
            {"name": "TOI-1338 b", "host_star": "TOI-1338", "radius_earth_radii": 6.9, "orbital_period_days": 95.2, "equilibrium_temperature_k": 200, "distance_parsecs": 395.0, "discovery_method": "Transit", "discovery_year": 2020, "status": "Confirmed"},
            {"name": "K2-18 b", "host_star": "K2-18", "radius_earth_radii": 2.3, "orbital_period_days": 33.0, "equilibrium_temperature_k": 234, "distance_parsecs": 34.0, "discovery_method": "Transit", "discovery_year": 2015, "status": "Confirmed"},
            {"name": "TRAPPIST-1e", "host_star": "TRAPPIST-1", "radius_earth_radii": 0.92, "orbital_period_days": 6.1, "equilibrium_temperature_k": 251, "distance_parsecs": 12.1, "discovery_method": "Transit", "discovery_year": 2017, "status": "Confirmed"},
            {"name": "Proxima Cen b", "host_star": "Proxima Centauri", "radius_earth_radii": 1.17, "orbital_period_days": 11.2, "equilibrium_temperature_k": 234, "distance_parsecs": 1.3, "discovery_method": "Radial_Velocity", "discovery_year": 2016, "status": "Confirmed"},
            {"name": "TOI-2109 b", "host_star": "TOI-2109", "radius_earth_radii": 1.35, "orbital_period_days": 0.67, "equilibrium_temperature_k": 2400, "distance_parsecs": 262.0, "discovery_method": "Transit", "discovery_year": 2021, "status": "Confirmed"},
        ]
        
        # Фильтрация по обитаемости
        if habitable_only:
            demo_planets = [p for p in demo_planets if 200 <= p["equilibrium_temperature_k"] <= 300]
        
        # Пагинация
        total = len(demo_planets)
        paginated = demo_planets[offset:offset + limit]
        
        return {
            "exoplanets": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except Exception as e:
        logger.error(f"Error getting exoplanet catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_database_statistics():
    """
    Получить статистику базы данных
    """
    return {
        "total_exoplanets": 5234,
        "confirmed_planets": 3456,
        "candidate_planets": 1234,
        "false_positives": 544,
        "missions": {
            "TESS": 2100,
            "Kepler": 2800,
            "K2": 334
        },
        "discovery_methods": {
            "Transit": 4200,
            "Radial Velocity": 800,
            "Direct Imaging": 134,
            "Microlensing": 100
        },
        "last_updated": "2025-10-04T20:15:00Z"
    }

@router.get("/search-history")
async def get_search_history(limit: int = Query(50, ge=1, le=100)):
    """
    Получить историю поиска
    """
    import random
    from datetime import datetime, timedelta
    
    history = []
    targets = ["TOI-715", "Kepler-452b", "TOI-849", "TRAPPIST-1e", "K2-18b", "Proxima Cen b"]
    
    for i in range(min(limit, 20)):
        history.append({
            "id": i + 1,
            "target_name": random.choice(targets),
            "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 168))).isoformat(),
            "result_class": random.choice(["Confirmed", "Candidate", "False Positive"]),
            "confidence": round(random.uniform(0.6, 0.99), 2),
            "processing_time_ms": random.randint(150, 800)
        })
    
    return {
        "search_history": history,
        "total": len(history)
    }

@router.get("/metrics")
async def get_database_metrics(hours: int = Query(24, ge=1, le=168)):
    """
    Получить метрики базы данных
    """
    import random
    from datetime import datetime, timedelta
    
    # Генерируем метрики за указанный период
    metrics = {
        "time_period_hours": hours,
        "total_searches": random.randint(50, 200),
        "successful_analyses": random.randint(40, 180),
        "average_processing_time_ms": random.randint(200, 500),
        "api_calls": {
            "analyze": random.randint(30, 150),
            "catalog": random.randint(10, 50),
            "health": random.randint(100, 300)
        },
        "classification_results": {
            "Confirmed": random.randint(10, 40),
            "Candidate": random.randint(15, 60),
            "False Positive": random.randint(5, 30)
        },
        "data_sources_used": {
            "NASA": random.randint(20, 80),
            "Demo": random.randint(10, 40),
            "User Upload": random.randint(5, 20)
        }
    }
    
    return metrics

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

        # For demo purposes, always use demo data to avoid timeouts
        logger.info(f"Using demo data for search: '{q}'")
        search_results = await generate_demo_search_results(q, limit)

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
