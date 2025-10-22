"""
Catalog API routes - Demo catalog endpoints
Маршруты API каталога - Demo эндпоинты каталога
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Query
from schemas.response import create_success_response

logger = logging.getLogger(__name__)
router = APIRouter()


async def generate_demo_catalog_exoplanets(limit: int = 12, offset: int = 0, habitable_only: bool = False) -> List[dict]:
    """Generate demo catalog exoplanets"""
    import random
    
    # Base demo planets
    base_planets = [
        {
            "id": 1,
            "name": "TOI-715.01",
            "host_star": "TOI-715",
            "discovery_year": 2020,
            "discovery_method": "Transit",
            "orbital_period": 19.3,
            "planet_radius": 1.55,
            "planet_mass": 2.3,
            "equilibrium_temperature": 450,
            "distance_pc": 137.0,
            "habitable": True,
            "confirmed": True,
            "source": "TESS"
        },
        {
            "id": 2,
            "name": "Kepler-452b",
            "host_star": "Kepler-452",
            "discovery_year": 2015,
            "discovery_method": "Transit",
            "orbital_period": 384.8,
            "planet_radius": 1.63,
            "planet_mass": 5.0,
            "equilibrium_temperature": 265,
            "distance_pc": 1402.0,
            "habitable": True,
            "confirmed": True,
            "source": "Kepler"
        },
        {
            "id": 3,
            "name": "WASP-12b",
            "host_star": "WASP-12",
            "discovery_year": 2008,
            "discovery_method": "Transit",
            "orbital_period": 1.09,
            "planet_radius": 1.79,
            "planet_mass": 1.41,
            "equilibrium_temperature": 2516,
            "distance_pc": 427.0,
            "habitable": False,
            "confirmed": True,
            "source": "Ground-based"
        }
    ]
    
    # Generate more planets if needed
    planets = []
    for i in range(limit):
        if i < len(base_planets):
            planet = base_planets[i].copy()
        else:
            # Generate synthetic planet
            planet = {
                "id": i + 1,
                "name": f"Demo-Planet-{i+1:03d}",
                "host_star": f"Demo-Star-{i+1:03d}",
                "discovery_year": 2015 + (i % 10),
                "discovery_method": random.choice(["Transit", "Radial Velocity", "Direct Imaging"]),
                "orbital_period": round(random.uniform(1, 500), 2),
                "planet_radius": round(random.uniform(0.5, 3.0), 2),
                "planet_mass": round(random.uniform(0.1, 10.0), 2),
                "equilibrium_temperature": round(random.uniform(200, 2000)),
                "distance_pc": round(random.uniform(10, 1000), 1),
                "habitable": random.choice([True, False]),
                "confirmed": random.choice([True, False]),
                "source": random.choice(["TESS", "Kepler", "K2", "Ground-based"])
            }
        
        if habitable_only and not planet["habitable"]:
            continue
            
        planets.append(planet)
    
    return planets[offset:offset+limit]


@router.get("/exoplanets")
async def get_catalog_exoplanets(
    limit: int = Query(12, description="Number of planets to return", ge=1, le=100),
    offset: int = Query(0, description="Offset for pagination", ge=0),
    habitable_only: bool = Query(False, description="Return only habitable planets")
):
    """
    Get catalog of exoplanets
    Получить каталог экзопланет
    """
    try:
        planets = await generate_demo_catalog_exoplanets(limit, offset, habitable_only)
        
        return create_success_response({
            "planets": planets,
            "total_count": 1000,  # Mock total
            "limit": limit,
            "offset": offset,
            "habitable_only": habitable_only,
            "source": "Demo Catalog"
        })
        
    except Exception as e:
        logger.error(f"Error getting catalog exoplanets: {e}")
        return create_success_response({
            "planets": [],
            "total_count": 0,
            "error": str(e)
        })
