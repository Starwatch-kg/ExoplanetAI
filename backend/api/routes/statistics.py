"""
Statistics API routes
Маршруты API статистики
"""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query

from auth.dependencies import require_researcher
from auth.models import User
from core.cache import get_cache
from data_sources.registry import get_registry
from schemas.response import ErrorCode, create_error_response, create_success_response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def get_global_statistics(
    include_sources: bool = Query(True, description="Include per-source statistics"),
    current_user: User = Depends(require_researcher),
):
    """
    Get global exoplanet statistics across all data sources

    **🔒 Requires researcher role or higher**

    Returns comprehensive statistics about exoplanet discoveries,
    methods, years, and data source availability.
    """
    start_time = time.time()

    try:
        # Check cache first
        cache = get_cache()
        cache_key = f"global_stats:{include_sources}"
        cached_stats = await cache.get("statistics", cache_key)

        if cached_stats:
            logger.info("Cache hit for global statistics")
            return create_success_response(
                data=cached_stats,
                message="Global statistics (cached)",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Get registry and collect statistics
        registry = get_registry()

        # Get aggregated statistics from all sources
        aggregated_stats = await registry.get_aggregated_statistics()

        # Add registry information
        registry_info = registry.get_registry_info()

        # Compile global statistics
        global_stats = {
            "overview": {
                "total_exoplanets": aggregated_stats.get("total_planets", 0),
                "data_sources_available": registry_info["initialized_sources"],
                "data_sources_total": registry_info["total_sources"],
                "last_updated": aggregated_stats.get("timestamp"),
            },
            "data_sources": registry_info["sources_by_type"],
            "registry_status": {
                "initialized": registry_info["initialized"],
                "available_sources": registry_info["source_names"],
            },
        }

        # Add per-source details if requested
        if include_sources:
            global_stats["source_details"] = aggregated_stats.get("sources", {})

        # Add discovery trends (example data)
        global_stats["discovery_trends"] = {
            "by_year": {
                "2020": 1200,
                "2021": 1300,
                "2022": 1400,
                "2023": 1500,
                "2024": 800,  # Partial year
            },
            "by_method": {
                "Transit": 3200,
                "Radial Velocity": 800,
                "Direct Imaging": 50,
                "Gravitational Microlensing": 100,
                "Astrometry": 20,
            },
            "by_status": {"Confirmed": 4000, "Candidate": 1200, "Disputed": 50},
        }

        # Add mission statistics
        global_stats["missions"] = {
            "TESS": {"status": "Active", "discoveries": 2000, "launch_year": 2018},
            "Kepler": {
                "status": "Completed",
                "discoveries": 2600,
                "launch_year": 2009,
                "end_year": 2013,
            },
            "K2": {
                "status": "Completed",
                "discoveries": 400,
                "start_year": 2014,
                "end_year": 2018,
            },
        }

        # Cache for 2 hours
        await cache.set("statistics", cache_key, global_stats, ttl=7200)

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Global statistics compiled: {global_stats['overview']['total_exoplanets']} planets"
        )

        return create_success_response(
            data=global_stats,
            message="Global exoplanet statistics",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error compiling global statistics: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to compile statistics: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.get("/discoveries")
async def get_discovery_statistics(
    year_min: Optional[int] = Query(
        None, description="Minimum discovery year", ge=1990
    ),
    year_max: Optional[int] = Query(
        None, description="Maximum discovery year", le=2030
    ),
    method: Optional[str] = Query(None, description="Discovery method filter"),
    current_user: User = Depends(require_researcher),
):
    """
    Get exoplanet discovery statistics with filters

    **🔒 Requires researcher role or higher**
    """
    start_time = time.time()

    try:
        # Build cache key
        cache_key = f"discoveries:{year_min}:{year_max}:{method}"
        cache = get_cache()
        cached_data = await cache.get("discovery_stats", cache_key)

        if cached_data:
            return create_success_response(
                data=cached_data,
                message="Discovery statistics (cached)",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Generate discovery statistics (example implementation)
        discovery_stats = {
            "filters": {"year_min": year_min, "year_max": year_max, "method": method},
            "timeline": _generate_discovery_timeline(year_min, year_max, method),
            "methods": _generate_method_statistics(year_min, year_max),
            "facilities": _generate_facility_statistics(year_min, year_max),
            "summary": {
                "total_discoveries": 0,
                "average_per_year": 0,
                "peak_year": None,
                "most_productive_method": None,
            },
        }

        # Calculate summary statistics
        timeline = discovery_stats["timeline"]
        if timeline:
            discovery_stats["summary"]["total_discoveries"] = sum(timeline.values())
            discovery_stats["summary"]["average_per_year"] = discovery_stats["summary"][
                "total_discoveries"
            ] / len(timeline)
            discovery_stats["summary"]["peak_year"] = max(
                timeline.keys(), key=lambda k: timeline[k]
            )

        methods = discovery_stats["methods"]
        if methods:
            discovery_stats["summary"]["most_productive_method"] = max(
                methods.keys(), key=lambda k: methods[k]
            )

        # Cache for 4 hours
        await cache.set("discovery_stats", cache_key, discovery_stats, ttl=14400)

        processing_time = (time.time() - start_time) * 1000

        return create_success_response(
            data=discovery_stats,
            message="Discovery statistics compiled",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error compiling discovery statistics: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to compile discovery statistics: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.get("/physical-properties")
async def get_physical_property_statistics(
    property_type: str = Query(
        "radius", description="Property type (radius, mass, period, temperature)"
    ),
    current_user: User = Depends(require_researcher),
):
    """
    Get statistics on exoplanet physical properties

    **🔒 Requires researcher role or higher**
    """
    start_time = time.time()

    try:
        cache_key = f"physical_props:{property_type}"
        cache = get_cache()
        cached_data = await cache.get("physical_stats", cache_key)

        if cached_data:
            return create_success_response(
                data=cached_data,
                message=f"Physical property statistics for {property_type} (cached)",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Generate property statistics
        property_stats = _generate_property_statistics(property_type)

        # Cache for 6 hours
        await cache.set("physical_stats", cache_key, property_stats, ttl=21600)

        processing_time = (time.time() - start_time) * 1000

        return create_success_response(
            data=property_stats,
            message=f"Physical property statistics for {property_type}",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error compiling physical property statistics: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to compile property statistics: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


@router.get("/habitability")
async def get_habitability_statistics(current_user: User = Depends(require_researcher)):
    """
    Get statistics on potentially habitable exoplanets

    **🔒 Requires researcher role or higher**
    """
    start_time = time.time()

    try:
        cache_key = "habitability_stats"
        cache = get_cache()
        cached_data = await cache.get("habitability", cache_key)

        if cached_data:
            return create_success_response(
                data=cached_data,
                message="Habitability statistics (cached)",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Generate habitability statistics
        habitability_stats = {
            "potentially_habitable": {
                "total_count": 60,
                "earth_size": 15,
                "super_earth": 30,
                "mini_neptune": 15,
            },
            "habitable_zone": {
                "confirmed_in_hz": 45,
                "candidates_in_hz": 120,
                "optimistic_hz": 200,
            },
            "earth_similarity": {
                "esi_0_8_to_1_0": 5,  # Very Earth-like
                "esi_0_6_to_0_8": 15,  # Earth-like
                "esi_0_4_to_0_6": 40,  # Potentially habitable
                "esi_below_0_4": 2000,  # Less likely habitable
            },
            "notable_targets": [
                {
                    "name": "Kepler-452b",
                    "esi": 0.83,
                    "status": "confirmed",
                    "distance_ly": 1400,
                    "note": "Earth's cousin",
                },
                {
                    "name": "TOI-715 b",
                    "esi": 0.85,
                    "status": "confirmed",
                    "distance_ly": 137,
                    "note": "Recent discovery in habitable zone",
                },
                {
                    "name": "TRAPPIST-1 e",
                    "esi": 0.85,
                    "status": "confirmed",
                    "distance_ly": 40,
                    "note": "Part of 7-planet system",
                },
            ],
            "atmospheric_studies": {
                "with_atmosphere_detected": 8,
                "water_vapor_detected": 3,
                "biosignature_candidates": 0,
                "jwst_targets": 25,
            },
        }

        # Cache for 8 hours
        await cache.set("habitability", cache_key, habitability_stats, ttl=28800)

        processing_time = (time.time() - start_time) * 1000

        return create_success_response(
            data=habitability_stats,
            message="Habitability statistics compiled",
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Error compiling habitability statistics: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Failed to compile habitability statistics: {str(e)}",
            processing_time_ms=(time.time() - start_time) * 1000,
        )


def _generate_discovery_timeline(
    year_min: Optional[int], year_max: Optional[int], method: Optional[str]
) -> Dict[str, int]:
    """Generate discovery timeline data"""
    # Example implementation - in real system, query database
    timeline = {}

    start_year = year_min or 1995
    end_year = year_max or 2024

    for year in range(start_year, end_year + 1):
        # Simulate discovery counts with realistic growth
        if year < 2009:  # Pre-Kepler
            count = min(50, max(1, year - 1994) * 5)
        elif year < 2018:  # Kepler era
            count = 200 + (year - 2009) * 100
        else:  # TESS era
            count = 800 + (year - 2018) * 200

        # Apply method filter (simplified)
        if method:
            if method.lower() == "transit" and year >= 2009:
                count = int(count * 0.8)  # Most are transits
            elif method.lower() == "radial velocity" and year < 2009:
                count = int(count * 0.9)  # Early discoveries
            else:
                count = int(count * 0.1)  # Other methods less common

        timeline[str(year)] = count

    return timeline


def _generate_method_statistics(
    year_min: Optional[int], year_max: Optional[int]
) -> Dict[str, int]:
    """Generate discovery method statistics"""
    return {
        "Transit": 3200,
        "Radial Velocity": 800,
        "Direct Imaging": 50,
        "Gravitational Microlensing": 100,
        "Astrometry": 20,
        "Pulsar Timing": 10,
        "Transit Timing Variations": 30,
    }


def _generate_facility_statistics(
    year_min: Optional[int], year_max: Optional[int]
) -> Dict[str, int]:
    """Generate discovery facility statistics"""
    return {
        "Kepler": 2600,
        "TESS": 2000,
        "K2": 400,
        "WASP": 200,
        "HAT": 150,
        "KELT": 100,
        "CoRoT": 80,
        "Ground-based RV": 500,
        "Other": 300,
    }


def _generate_property_statistics(property_type: str) -> Dict[str, Any]:
    """Generate physical property statistics"""

    if property_type == "radius":
        return {
            "property": "radius",
            "unit": "Earth radii",
            "distribution": {
                "sub_earth": {"range": "< 0.8", "count": 200, "percentage": 5},
                "earth_size": {"range": "0.8 - 1.25", "count": 800, "percentage": 20},
                "super_earth": {"range": "1.25 - 2.0", "count": 1200, "percentage": 30},
                "mini_neptune": {"range": "2.0 - 4.0", "count": 1000, "percentage": 25},
                "neptune_size": {"range": "4.0 - 6.0", "count": 400, "percentage": 10},
                "jupiter_size": {"range": "> 6.0", "count": 400, "percentage": 10},
            },
            "statistics": {
                "median": 2.1,
                "mean": 3.2,
                "std_dev": 2.8,
                "min": 0.3,
                "max": 25.0,
            },
        }

    elif property_type == "mass":
        return {
            "property": "mass",
            "unit": "Earth masses",
            "distribution": {
                "sub_earth": {"range": "< 0.8", "count": 150, "percentage": 8},
                "earth_mass": {"range": "0.8 - 2.0", "count": 600, "percentage": 32},
                "super_earth": {"range": "2.0 - 10.0", "count": 700, "percentage": 37},
                "neptune_mass": {
                    "range": "10.0 - 50.0",
                    "count": 300,
                    "percentage": 16,
                },
                "jupiter_mass": {"range": "> 50.0", "count": 150, "percentage": 8},
            },
            "statistics": {
                "median": 8.5,
                "mean": 45.2,
                "std_dev": 120.5,
                "min": 0.02,
                "max": 4000.0,
            },
        }

    elif property_type == "period":
        return {
            "property": "orbital_period",
            "unit": "days",
            "distribution": {
                "ultra_short": {"range": "< 1", "count": 300, "percentage": 8},
                "short": {"range": "1 - 10", "count": 1500, "percentage": 40},
                "medium": {"range": "10 - 100", "count": 1200, "percentage": 32},
                "long": {"range": "100 - 1000", "count": 600, "percentage": 16},
                "very_long": {"range": "> 1000", "count": 150, "percentage": 4},
            },
            "statistics": {
                "median": 15.2,
                "mean": 85.6,
                "std_dev": 245.8,
                "min": 0.09,
                "max": 5000.0,
            },
        }

    else:  # temperature
        return {
            "property": "equilibrium_temperature",
            "unit": "Kelvin",
            "distribution": {
                "frozen": {"range": "< 200", "count": 400, "percentage": 15},
                "cold": {"range": "200 - 400", "count": 600, "percentage": 22},
                "temperate": {"range": "400 - 600", "count": 500, "percentage": 18},
                "hot": {"range": "600 - 1000", "count": 700, "percentage": 26},
                "ultra_hot": {"range": "> 1000", "count": 500, "percentage": 19},
            },
            "statistics": {
                "median": 650.0,
                "mean": 850.5,
                "std_dev": 520.2,
                "min": 50.0,
                "max": 3000.0,
            },
        }
