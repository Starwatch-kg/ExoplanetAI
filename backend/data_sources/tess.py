"""
TESS Mission data source implementation
Источник данных миссии TESS через MAST API
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

try:
    import astropy.units as u
    import lightkurve as lk
    from astroquery.mast import Catalogs, Observations

    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False

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


class TESSDataSource(BaseDataSource):
    """TESS Mission data source via MAST API"""

    def __init__(self):
        super().__init__("TESS Mission", DataSourceType.TESS)
        self.base_url = "https://mast.stsci.edu"
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> bool:
        """Initialize TESS data source"""
        try:
            if not LIGHTKURVE_AVAILABLE:
                logger.error("lightkurve not available for TESS data source")
                return False

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=60)  # TESS downloads can be slow
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
                logger.info("✅ TESS Mission data source initialized")
                return True
            else:
                logger.error("❌ TESS data source health check failed")
                return False

        except Exception as e:
            logger.error(f"TESS data source initialization failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_initialized = False

    async def health_check(self) -> Dict[str, Any]:
        """Check TESS/MAST service health"""
        try:
            if not self.session:
                return {
                    "status": "unhealthy",
                    "error": "Session not initialized",
                    "timestamp": datetime.now().isoformat(),
                }

            # Test MAST API
            url = f"{self.base_url}/api/v0.1/Download/file"
            params = {"uri": "mast:TESS/product/"}  # Basic endpoint test

            async with self.session.get(url, params=params) as response:
                # MAST returns various status codes, 400+ usually means service is up
                if response.status < 500:
                    return {
                        "status": "healthy",
                        "mast_status": response.status,
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"MAST HTTP {response.status}",
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
        """Search for TESS targets"""
        start_time = datetime.now()

        try:
            # Run lightkurve search in thread pool
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None, self._sync_search_tess_targets, query, limit
            )

            end_time = datetime.now()
            search_time_ms = (end_time - start_time).total_seconds() * 1000

            planets = []
            if search_results:
                for target in search_results:
                    planet = self._target_to_planet_info(target, query)
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
            logger.error(f"TESS search failed for '{query}': {e}")
            raise DataSourceError(f"TESS search failed: {e}")

    def _sync_search_tess_targets(self, query: str, limit: int) -> List[Any]:
        """Synchronous TESS target search using lightkurve"""
        try:
            # Search for TESS observations
            search_result = lk.search_lightcurve(query, mission="TESS")

            if len(search_result) == 0:
                return []

            # Limit results and extract unique targets
            unique_targets = {}
            for i, obs in enumerate(search_result):
                if i >= limit:
                    break

                target_name = obs.target_name
                if target_name not in unique_targets:
                    unique_targets[target_name] = {
                        "target_name": target_name,
                        "mission": obs.mission,
                        "sectors": [obs.sector] if hasattr(obs, "sector") else [],
                        "exptime": obs.exptime if hasattr(obs, "exptime") else None,
                        "distance": getattr(obs, "distance", None),
                    }
                else:
                    # Add sector to existing target
                    if (
                        hasattr(obs, "sector")
                        and obs.sector not in unique_targets[target_name]["sectors"]
                    ):
                        unique_targets[target_name]["sectors"].append(obs.sector)

            return list(unique_targets.values())

        except Exception as e:
            logger.error(f"Lightkurve TESS search error: {e}")
            raise

    async def fetch_planet_info(self, planet_name: str) -> Optional[PlanetInfo]:
        """Fetch TESS target information"""
        try:
            loop = asyncio.get_event_loop()
            target_info = await loop.run_in_executor(
                None, self._sync_fetch_tess_target_info, planet_name
            )

            if target_info:
                return self._target_to_planet_info(target_info, planet_name)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to fetch TESS info for '{planet_name}': {e}")
            return None

    def _sync_fetch_tess_target_info(self, target_name: str) -> Optional[Dict]:
        """Synchronous TESS target info fetch"""
        try:
            # Search for target
            search_result = lk.search_lightcurve(target_name, mission="TESS")

            if len(search_result) == 0:
                return None

            # Get first result for basic info
            first_obs = search_result[0]

            # Try to get additional info from MAST catalogs
            try:
                catalog_result = Catalogs.query_object(target_name, catalog="TIC")
                catalog_info = catalog_result[0] if len(catalog_result) > 0 else None
            except Exception:
                catalog_info = None

            return {
                "target_name": first_obs.target_name,
                "mission": first_obs.mission,
                "sectors": [
                    obs.sector for obs in search_result if hasattr(obs, "sector")
                ],
                "exptime": first_obs.exptime if hasattr(first_obs, "exptime") else None,
                "catalog_info": catalog_info,
            }

        except Exception as e:
            logger.error(f"TESS target info fetch error: {e}")
            return None

    def _target_to_planet_info(
        self, target_data: Dict, query: str
    ) -> Optional[PlanetInfo]:
        """Convert TESS target data to PlanetInfo"""
        try:
            target_name = target_data.get("target_name", query)

            # Extract catalog info if available
            catalog_info = target_data.get("catalog_info")

            ra_deg = None
            dec_deg = None
            stellar_temp = None
            stellar_radius = None
            stellar_mass = None
            magnitude = None

            if catalog_info is not None:
                ra_deg = float(catalog_info["ra"]) if "ra" in catalog_info else None
                dec_deg = float(catalog_info["dec"]) if "dec" in catalog_info else None
                stellar_temp = (
                    float(catalog_info["Teff"]) if "Teff" in catalog_info else None
                )
                stellar_radius = (
                    float(catalog_info["rad"]) if "rad" in catalog_info else None
                )
                stellar_mass = (
                    float(catalog_info["mass"]) if "mass" in catalog_info else None
                )
                magnitude = (
                    float(catalog_info["Tmag"]) if "Tmag" in catalog_info else None
                )

            return PlanetInfo(
                name=target_name,
                host_star=target_name,  # TESS targets are usually stars
                status=PlanetStatus.CANDIDATE,  # TESS finds candidates
                discovery_method="Transit",
                discovery_facility="TESS",
                # TESS-specific info
                stellar_temperature_k=stellar_temp,
                stellar_radius_solar=stellar_radius,
                stellar_mass_solar=stellar_mass,
                stellar_magnitude=magnitude,
                # Coordinates
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                # Data source info
                source=self.name,
                last_updated=datetime.now(),
                data_quality=f"TESS Sectors: {target_data.get('sectors', [])}",
            )

        except Exception as e:
            logger.error(f"Error converting TESS target to PlanetInfo: {e}")
            return None

    async def fetch_light_curve(
        self,
        target_name: str,
        mission: Optional[str] = None,
        sector_quarter: Optional[int] = None,
    ) -> Optional[LightCurveData]:
        """Fetch TESS light curve data"""
        try:
            loop = asyncio.get_event_loop()
            lc_data = await loop.run_in_executor(
                None, self._sync_fetch_light_curve, target_name, sector_quarter
            )

            return lc_data

        except Exception as e:
            logger.error(f"Failed to fetch TESS light curve for '{target_name}': {e}")
            raise DataNotFoundError(f"TESS light curve for '{target_name}' not found")

    def _sync_fetch_light_curve(
        self, target_name: str, sector: Optional[int]
    ) -> Optional[LightCurveData]:
        """Synchronous TESS light curve fetch"""
        try:
            # Search for TESS light curves
            search_result = lk.search_lightcurve(
                target_name, mission="TESS", sector=sector
            )

            if len(search_result) == 0:
                logger.warning(f"No TESS light curves found for {target_name}")
                return None

            # Download light curves
            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                logger.warning(
                    f"Failed to download TESS light curves for {target_name}"
                )
                return None

            # Stitch multiple sectors together
            lc = lc_collection.stitch()

            # Data quality checks
            if len(lc.flux) < 100:
                logger.warning(
                    f"Insufficient TESS data points for {target_name}: {len(lc.flux)}"
                )
                return None

            # Clean and normalize data
            lc = lc.remove_nans().remove_outliers(sigma=5)
            lc = lc.normalize()

            # Extract arrays
            time_bjd = lc.time.btjd  # Barycentric TESS Julian Date
            flux = lc.flux.value
            flux_err = (
                lc.flux_err.value
                if lc.flux_err is not None
                else np.ones_like(flux) * 0.001
            )

            # Get sectors
            sectors = []
            for lc_single in lc_collection:
                if hasattr(lc_single, "sector"):
                    sectors.append(lc_single.sector)

            return LightCurveData(
                target_name=target_name,
                time_bjd=time_bjd,
                flux=flux,
                flux_err=flux_err,
                mission="TESS",
                instrument="TESS Photometer",
                cadence_minutes=2.0,  # TESS 2-minute cadence
                sectors_quarters=sectors,
                detrended=True,
                normalized=True,
                outliers_removed=True,
                source=self.name,
                download_date=datetime.now(),
            )

        except Exception as e:
            logger.error(f"TESS light curve download error: {e}")
            raise

    async def get_statistics(self) -> Dict[str, Any]:
        """Get TESS mission statistics"""
        try:
            # TESS statistics are harder to get programmatically
            # Return basic info about TESS mission
            return {
                "mission": "TESS",
                "launch_date": "2018-04-18",
                "status": "Active",
                "sectors_completed": "60+",  # Approximate as of 2024
                "cadence_modes": ["2-minute", "20-second", "200-second"],
                "sky_coverage": "~85% of sky",
                "primary_targets": "200,000+ stars",
                "note": "Exact planet counts require cross-referencing with NASA archive",
                "last_updated": datetime.now().isoformat(),
                "source": self.name,
            }
        except Exception as e:
            logger.error(f"Failed to get TESS statistics: {e}")
            return {"error": str(e)}

    def get_supported_missions(self) -> List[str]:
        """TESS supports only TESS mission"""
        return ["TESS"]

    def get_capabilities(self) -> Dict[str, bool]:
        """TESS data source capabilities"""
        return {
            "planet_search": True,
            "planet_info": True,
            "light_curves": True,  # Primary capability
            "statistics": True,
            "real_time": False,
        }
