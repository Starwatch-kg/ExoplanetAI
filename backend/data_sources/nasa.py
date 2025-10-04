"""
NASA Exoplanet Archive data source implementation
Источник данных NASA Exoplanet Archive
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

try:
    import astropy.units as u
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False

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


class NASADataSource(BaseDataSource):
    """NASA Exoplanet Archive data source"""

    def __init__(self):
        super().__init__("NASA Exoplanet Archive", DataSourceType.NASA)
        self.base_url = "https://exoplanetarchive.ipac.caltech.edu"
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> bool:
        """Initialize NASA data source"""
        try:
            if not ASTROQUERY_AVAILABLE:
                logger.error("astroquery not available for NASA data source")
                return False

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
                logger.info("✅ NASA Exoplanet Archive initialized")
                return True
            else:
                logger.error("❌ NASA Exoplanet Archive health check failed")
                return False

        except Exception as e:
            logger.error(f"NASA data source initialization failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_initialized = False

    async def health_check(self) -> Dict[str, Any]:
        """Check NASA service health"""
        try:
            if not self.session:
                return {
                    "status": "unhealthy",
                    "error": "Session not initialized",
                    "timestamp": datetime.now().isoformat(),
                }

            # Test API endpoint
            url = f"{self.base_url}/TAP/sync"
            params = {"query": "SELECT TOP 1 pl_name FROM ps", "format": "json"}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "response_time_ms": response.headers.get(
                            "X-Response-Time", "unknown"
                        ),
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}",
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
        """Search for planets in NASA archive"""
        start_time = datetime.now()

        try:
            # Run astroquery in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._sync_search_planets, query, limit, filters
            )

            end_time = datetime.now()
            search_time_ms = (end_time - start_time).total_seconds() * 1000

            planets = []
            if result and len(result) > 0:
                for row in result:
                    planet = self._row_to_planet_info(row)
                    if planet:
                        planets.append(planet)

            return SearchResult(
                query=query,
                total_found=len(planets),
                planets=planets,
                search_time_ms=search_time_ms,
                source=self.name,
                cached=False,
            )

        except Exception as e:
            logger.error(f"NASA search failed for '{query}': {e}")
            raise DataSourceError(f"NASA search failed: {e}")

    def _sync_search_planets(
        self, query: str, limit: int, filters: Optional[Dict]
    ) -> Any:
        """Synchronous planet search using astroquery"""
        try:
            # Build query conditions
            conditions = []

            # Search in planet name and host star name
            conditions.append(
                f"(pl_name LIKE '%{query}%' OR hostname LIKE '%{query}%')"
            )

            # Apply filters if provided
            if filters:
                if "discovery_year_min" in filters:
                    conditions.append(f"disc_year >= {filters['discovery_year_min']}")
                if "discovery_year_max" in filters:
                    conditions.append(f"disc_year <= {filters['discovery_year_max']}")
                if "discovery_method" in filters:
                    conditions.append(
                        f"discoverymethod = '{filters['discovery_method']}'"
                    )
                if "confirmed_only" in filters and filters["confirmed_only"]:
                    conditions.append("pl_controv_flag = 0")

            where_clause = " AND ".join(conditions)

            # Execute query
            result = NasaExoplanetArchive.query_criteria(
                table="ps", where=where_clause, cache=True  # Planetary Systems table
            )

            # Limit results
            if len(result) > limit:
                result = result[:limit]

            return result

        except Exception as e:
            logger.error(f"Astroquery search error: {e}")
            raise

    async def fetch_planet_info(self, planet_name: str) -> Optional[PlanetInfo]:
        """Fetch detailed planet information"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._sync_fetch_planet_info, planet_name
            )

            if result and len(result) > 0:
                return self._row_to_planet_info(result[0])
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to fetch planet info for '{planet_name}': {e}")
            raise DataNotFoundError(f"Planet '{planet_name}' not found in NASA archive")

    def _sync_fetch_planet_info(self, planet_name: str) -> Any:
        """Synchronous planet info fetch"""
        try:
            result = NasaExoplanetArchive.query_criteria(
                table="ps", where=f"pl_name = '{planet_name}'", cache=True
            )
            return result
        except Exception as e:
            logger.error(f"Astroquery fetch error: {e}")
            raise

    def _row_to_planet_info(self, row) -> Optional[PlanetInfo]:
        """Convert astroquery result row to PlanetInfo"""
        try:
            # Determine status
            status = PlanetStatus.CONFIRMED
            if hasattr(row, "pl_controv_flag") and row["pl_controv_flag"] == 1:
                status = PlanetStatus.DISPUTED

            return PlanetInfo(
                name=str(row["pl_name"]) if row["pl_name"] else "Unknown",
                host_star=str(row["hostname"]) if row["hostname"] else "Unknown",
                status=status,
                discovery_year=int(row["disc_year"]) if row["disc_year"] else None,
                discovery_method=(
                    str(row["discoverymethod"]) if row["discoverymethod"] else None
                ),
                discovery_facility=(
                    str(row["disc_facility"]) if row["disc_facility"] else None
                ),
                # Orbital parameters
                orbital_period_days=(
                    float(row["pl_orbper"]) if row["pl_orbper"] else None
                ),
                semi_major_axis_au=(
                    float(row["pl_orbsmax"]) if row["pl_orbsmax"] else None
                ),
                eccentricity=float(row["pl_orbeccen"]) if row["pl_orbeccen"] else None,
                inclination_deg=float(row["pl_orbincl"]) if row["pl_orbincl"] else None,
                # Physical parameters
                radius_earth_radii=float(row["pl_rade"]) if row["pl_rade"] else None,
                radius_jupiter_radii=float(row["pl_radj"]) if row["pl_radj"] else None,
                mass_earth_masses=float(row["pl_masse"]) if row["pl_masse"] else None,
                mass_jupiter_masses=float(row["pl_massj"]) if row["pl_massj"] else None,
                density_g_cm3=float(row["pl_dens"]) if row["pl_dens"] else None,
                # Atmospheric parameters
                equilibrium_temperature_k=(
                    float(row["pl_eqt"]) if row["pl_eqt"] else None
                ),
                # Transit parameters
                transit_depth_ppm=(
                    float(row["pl_trandep"]) if row["pl_trandep"] else None
                ),
                transit_duration_hours=(
                    float(row["pl_trandur"]) if row["pl_trandur"] else None
                ),
                # Host star parameters
                stellar_mass_solar=float(row["st_mass"]) if row["st_mass"] else None,
                stellar_radius_solar=float(row["st_rad"]) if row["st_rad"] else None,
                stellar_temperature_k=float(row["st_teff"]) if row["st_teff"] else None,
                stellar_magnitude=float(row["sy_vmag"]) if row["sy_vmag"] else None,
                # Coordinates
                ra_deg=float(row["ra"]) if row["ra"] else None,
                dec_deg=float(row["dec"]) if row["dec"] else None,
                distance_pc=float(row["sy_dist"]) if row["sy_dist"] else None,
                # Data source info
                source=self.name,
                last_updated=datetime.now(),
                data_quality="NASA Archive",
            )

        except Exception as e:
            logger.error(f"Error converting row to PlanetInfo: {e}")
            return None

    async def fetch_light_curve(
        self,
        target_name: str,
        mission: Optional[str] = None,
        sector_quarter: Optional[int] = None,
    ) -> Optional[LightCurveData]:
        """NASA archive doesn't provide light curves directly"""
        logger.warning("NASA Exoplanet Archive doesn't provide light curve data")
        return None

    async def get_statistics(self) -> Dict[str, Any]:
        """Get NASA archive statistics"""
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(None, self._sync_get_statistics)
            return stats
        except Exception as e:
            logger.error(f"Failed to get NASA statistics: {e}")
            return {"error": str(e)}

    def _sync_get_statistics(self) -> Dict[str, Any]:
        """Synchronous statistics gathering"""
        try:
            # Get total planet count
            total_result = NasaExoplanetArchive.query_criteria(
                table="ps", select="COUNT(*) as total", cache=True
            )
            total_planets = int(total_result[0]["total"]) if total_result else 0

            # Get confirmed vs candidates
            confirmed_result = NasaExoplanetArchive.query_criteria(
                table="ps",
                select="COUNT(*) as confirmed",
                where="pl_controv_flag = 0",
                cache=True,
            )
            confirmed_count = (
                int(confirmed_result[0]["confirmed"]) if confirmed_result else 0
            )

            # Get discovery methods
            methods_result = NasaExoplanetArchive.query_criteria(
                table="ps",
                select="discoverymethod, COUNT(*) as count",
                where="discoverymethod IS NOT NULL",
                cache=True,
            )

            discovery_methods = {}
            if methods_result:
                for row in methods_result:
                    method = str(row["discoverymethod"])
                    count = int(row["count"])
                    discovery_methods[method] = count

            return {
                "total_planets": total_planets,
                "confirmed_planets": confirmed_count,
                "candidate_planets": total_planets - confirmed_count,
                "discovery_methods": discovery_methods,
                "last_updated": datetime.now().isoformat(),
                "source": self.name,
            }

        except Exception as e:
            logger.error(f"Statistics query error: {e}")
            raise

    def get_supported_missions(self) -> List[str]:
        """NASA archive supports multiple missions"""
        return ["Kepler", "TESS", "K2", "CoRoT", "TRAPPIST", "WASP", "HAT", "KELT"]

    def get_capabilities(self) -> Dict[str, bool]:
        """NASA archive capabilities"""
        return {
            "planet_search": True,
            "planet_info": True,
            "light_curves": False,  # No direct light curve access
            "statistics": True,
            "real_time": False,
        }
