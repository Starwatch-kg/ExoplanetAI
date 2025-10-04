"""
ESA (European Space Agency) data source implementation
Источник данных ESA (Европейское космическое агентство)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

from .base import (
    BaseDataSource,
    DataNotFoundError,
    DataSourceError,
    DataSourceType,
    DataSourceUnavailableError,
    LightCurveData,
    PlanetInfo,
    PlanetStatus,
    SearchResult,
)

logger = logging.getLogger(__name__)


class ESADataSource(BaseDataSource):
    """ESA Archive data source"""

    def __init__(self):
        super().__init__("ESA Archive", DataSourceType.ESA)
        self.base_url = "https://archives.esac.esa.int"
        self.session: Optional[aiohttp.ClientSession] = None

        # ESA missions and their endpoints
        self.missions = {
            "gaia": {
                "url": "https://gea.esac.esa.int/archive",
                "description": "Gaia astrometry mission",
            },
            "cheops": {
                "url": "https://cheops.unige.ch/pht",
                "description": "CHEOPS exoplanet characterization",
            },
            "plato": {
                "url": "https://plato.esa.int",
                "description": "PLATO exoplanet mission (future)",
            },
        }

    async def initialize(self) -> bool:
        """Initialize ESA data source"""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "ExoplanetAI/2.0 (Scientific Research)",
                    "Accept": "application/json",
                },
            )

            # Test connection
            health = await self.health_check()
            if health.get("status") == "healthy":
                self.is_initialized = True
                logger.info("✅ ESA Archive data source initialized")
                return True
            else:
                logger.error("❌ ESA data source health check failed")
                return False

        except Exception as e:
            logger.error(f"ESA data source initialization failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_initialized = False

    async def health_check(self) -> Dict[str, Any]:
        """Check ESA service health"""
        try:
            if not self.session:
                return {
                    "status": "unhealthy",
                    "error": "Session not initialized",
                    "timestamp": datetime.now().isoformat(),
                }

            # Test ESA archives endpoint
            url = f"{self.base_url}/ehst"

            async with self.session.get(url) as response:
                if response.status < 500:
                    return {
                        "status": "healthy",
                        "esa_status": response.status,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"ESA HTTP {response.status}",
                        "timestamp": datetime.now().isoformat(),
                    }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def search_planets(
        self, query: str, limit: int = 100, filters: Optional[Dict[str, Any]] = None
    ) -> SearchResult:
        """Search for planets in ESA archives"""
        start_time = datetime.now()

        try:
            # ESA data access is more complex and mission-specific
            # For now, return example data for known ESA-related targets

            planets = []

            # Check if query matches known ESA targets
            esa_targets = self._get_known_esa_targets()

            for target in esa_targets:
                if (
                    query.lower() in target["name"].lower()
                    or query.lower() in target.get("host_star", "").lower()
                ):
                    planet_info = self._dict_to_planet_info(target)
                    if planet_info:
                        planets.append(planet_info)

                    if len(planets) >= limit:
                        break

            end_time = datetime.now()
            search_time_ms = (end_time - start_time).total_seconds() * 1000

            return SearchResult(
                query=query,
                total_found=len(planets),
                planets=planets,
                search_time_ms=search_time_ms,
                source=self.name,
                cached=False,
            )

        except Exception as e:
            logger.error(f"ESA search failed for '{query}': {e}")
            raise DataSourceError(f"ESA search failed: {e}")

    def _get_known_esa_targets(self) -> List[Dict]:
        """Get known ESA-related exoplanet targets"""
        return [
            {
                "name": "WASP-189 b",
                "host_star": "WASP-189",
                "status": "confirmed",
                "discovery_year": 2018,
                "discovery_method": "Transit",
                "discovery_facility": "WASP",
                "orbital_period_days": 2.724,
                "radius_jupiter_radii": 1.6,
                "mass_jupiter_masses": 2.1,
                "equilibrium_temperature_k": 2641,
                "ra_deg": 308.31,
                "dec_deg": 1.61,
                "stellar_temperature_k": 8000,
                "stellar_mass_solar": 2.0,
                "cheops_observed": True,
                "note": "Ultra-hot Jupiter observed by CHEOPS",
            },
            {
                "name": "TOI-849 b",
                "host_star": "TOI-849",
                "status": "confirmed",
                "discovery_year": 2020,
                "discovery_method": "Transit",
                "discovery_facility": "TESS",
                "orbital_period_days": 0.765,
                "radius_earth_radii": 3.4,
                "mass_earth_masses": 40.8,
                "equilibrium_temperature_k": 1800,
                "cheops_observed": True,
                "note": "Exposed planetary core observed by CHEOPS",
            },
            {
                "name": "K2-141 b",
                "host_star": "K2-141",
                "status": "confirmed",
                "discovery_year": 2018,
                "discovery_method": "Transit",
                "discovery_facility": "K2",
                "orbital_period_days": 0.28,
                "radius_earth_radii": 1.5,
                "equilibrium_temperature_k": 3000,
                "cheops_observed": True,
                "note": "Lava planet with extreme conditions",
            },
        ]

    async def fetch_planet_info(self, planet_name: str) -> Optional[PlanetInfo]:
        """Fetch detailed planet information from ESA archives"""
        try:
            # Search in known ESA targets
            esa_targets = self._get_known_esa_targets()

            for target in esa_targets:
                if planet_name.lower() in target["name"].lower():
                    return self._dict_to_planet_info(target)

            return None

        except Exception as e:
            logger.error(f"Failed to fetch ESA info for '{planet_name}': {e}")
            return None

    def _dict_to_planet_info(self, target_dict: Dict) -> Optional[PlanetInfo]:
        """Convert target dictionary to PlanetInfo"""
        try:
            status_map = {
                "confirmed": PlanetStatus.CONFIRMED,
                "candidate": PlanetStatus.CANDIDATE,
                "disputed": PlanetStatus.DISPUTED,
            }

            return PlanetInfo(
                name=target_dict["name"],
                host_star=target_dict.get("host_star", target_dict["name"]),
                status=status_map.get(
                    target_dict.get("status", "confirmed"), PlanetStatus.CONFIRMED
                ),
                discovery_year=target_dict.get("discovery_year"),
                discovery_method=target_dict.get("discovery_method"),
                discovery_facility=target_dict.get("discovery_facility"),
                # Orbital parameters
                orbital_period_days=target_dict.get("orbital_period_days"),
                # Physical parameters
                radius_earth_radii=target_dict.get("radius_earth_radii"),
                radius_jupiter_radii=target_dict.get("radius_jupiter_radii"),
                mass_earth_masses=target_dict.get("mass_earth_masses"),
                mass_jupiter_masses=target_dict.get("mass_jupiter_masses"),
                # Atmospheric parameters
                equilibrium_temperature_k=target_dict.get("equilibrium_temperature_k"),
                # Host star parameters
                stellar_mass_solar=target_dict.get("stellar_mass_solar"),
                stellar_temperature_k=target_dict.get("stellar_temperature_k"),
                # Coordinates
                ra_deg=target_dict.get("ra_deg"),
                dec_deg=target_dict.get("dec_deg"),
                # Data source info
                source=self.name,
                last_updated=datetime.now(),
                data_quality=f"ESA Archive - {target_dict.get('note', 'Standard quality')}",
            )

        except Exception as e:
            logger.error(f"Error converting ESA target to PlanetInfo: {e}")
            return None

    async def fetch_light_curve(
        self,
        target_name: str,
        mission: Optional[str] = None,
        sector_quarter: Optional[int] = None,
    ) -> Optional[LightCurveData]:
        """Fetch light curve data from ESA missions"""

        # ESA light curve access is complex and mission-specific
        # For CHEOPS, data access requires special permissions
        # For now, return None indicating no direct light curve access

        logger.info(
            f"ESA light curve access for {target_name} requires mission-specific protocols"
        )
        return None

    async def get_statistics(self) -> Dict[str, Any]:
        """Get ESA mission statistics"""
        try:
            return {
                "agency": "European Space Agency (ESA)",
                "missions": {
                    "gaia": {
                        "status": "Active",
                        "launch_date": "2013-12-19",
                        "primary_goal": "Astrometry and stellar characterization",
                        "targets": "1+ billion stars",
                    },
                    "cheops": {
                        "status": "Active",
                        "launch_date": "2019-12-18",
                        "primary_goal": "Exoplanet characterization",
                        "targets": "Known exoplanets for precise measurements",
                    },
                    "plato": {
                        "status": "Development",
                        "planned_launch": "2026",
                        "primary_goal": "Earth-like exoplanet detection",
                        "targets": "1 million stars",
                    },
                    "ariel": {
                        "status": "Development",
                        "planned_launch": "2029",
                        "primary_goal": "Exoplanet atmosphere characterization",
                        "targets": "1000+ exoplanets",
                    },
                },
                "data_access": {
                    "gaia": "Public via ESA Archive",
                    "cheops": "Proprietary + Guest Observer Program",
                    "plato": "Future public access",
                    "ariel": "Future public access",
                },
                "last_updated": datetime.now().isoformat(),
                "source": self.name,
            }
        except Exception as e:
            logger.error(f"Failed to get ESA statistics: {e}")
            return {"error": str(e)}

    def get_supported_missions(self) -> List[str]:
        """ESA supported missions"""
        return ["Gaia", "CHEOPS", "PLATO", "ARIEL"]

    def get_capabilities(self) -> Dict[str, bool]:
        """ESA data source capabilities"""
        return {
            "planet_search": True,
            "planet_info": True,
            "light_curves": False,  # Complex access requirements
            "statistics": True,
            "real_time": False,
        }
