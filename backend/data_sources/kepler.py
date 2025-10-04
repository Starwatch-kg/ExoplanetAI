"""
Kepler Mission data source implementation
Источник данных миссии Kepler через MAST API
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


class KeplerDataSource(BaseDataSource):
    """Kepler Mission data source via MAST API"""

    def __init__(self):
        super().__init__("Kepler Mission", DataSourceType.KEPLER)
        self.base_url = "https://mast.stsci.edu"
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> bool:
        """Initialize Kepler data source"""
        try:
            if not LIGHTKURVE_AVAILABLE:
                logger.error("lightkurve not available for Kepler data source")
                return False

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=60)
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
                logger.info("✅ Kepler Mission data source initialized")
                return True
            else:
                logger.error("❌ Kepler data source health check failed")
                return False

        except Exception as e:
            logger.error(f"Kepler data source initialization failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_initialized = False

    async def health_check(self) -> Dict[str, Any]:
        """Check Kepler/MAST service health"""
        try:
            if not self.session:
                return {
                    "status": "unhealthy",
                    "error": "Session not initialized",
                    "timestamp": datetime.now().isoformat(),
                }

            # Test MAST API
            url = f"{self.base_url}/api/v0.1/Download/file"
            params = {"uri": "mast:KEPLER/url"}  # Basic endpoint test

            async with self.session.get(url, params=params) as response:
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
        """Search for Kepler targets"""
        start_time = datetime.now()

        try:
            # Run lightkurve search in thread pool
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None, self._sync_search_kepler_targets, query, limit
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
            logger.error(f"Kepler search failed for '{query}': {e}")
            raise DataSourceError(f"Kepler search failed: {e}")

    def _sync_search_kepler_targets(self, query: str, limit: int) -> List[Any]:
        """Synchronous Kepler target search using lightkurve"""
        try:
            # Search for Kepler observations
            search_result = lk.search_lightcurve(query, mission="Kepler")

            if len(search_result) == 0:
                # Also try K2
                search_result = lk.search_lightcurve(query, mission="K2")

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
                        "quarters": [obs.quarter] if hasattr(obs, "quarter") else [],
                        "campaigns": [obs.campaign] if hasattr(obs, "campaign") else [],
                        "exptime": obs.exptime if hasattr(obs, "exptime") else None,
                        "distance": getattr(obs, "distance", None),
                    }
                else:
                    # Add quarter/campaign to existing target
                    if (
                        hasattr(obs, "quarter")
                        and obs.quarter not in unique_targets[target_name]["quarters"]
                    ):
                        unique_targets[target_name]["quarters"].append(obs.quarter)
                    if (
                        hasattr(obs, "campaign")
                        and obs.campaign not in unique_targets[target_name]["campaigns"]
                    ):
                        unique_targets[target_name]["campaigns"].append(obs.campaign)

            return list(unique_targets.values())

        except Exception as e:
            logger.error(f"Lightkurve Kepler search error: {e}")
            raise

    async def fetch_planet_info(self, planet_name: str) -> Optional[PlanetInfo]:
        """Fetch Kepler target information"""
        try:
            loop = asyncio.get_event_loop()
            target_info = await loop.run_in_executor(
                None, self._sync_fetch_kepler_target_info, planet_name
            )

            if target_info:
                return self._target_to_planet_info(target_info, planet_name)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to fetch Kepler info for '{planet_name}': {e}")
            return None

    def _sync_fetch_kepler_target_info(self, target_name: str) -> Optional[Dict]:
        """Synchronous Kepler target info fetch"""
        try:
            # Search for target in both Kepler and K2
            search_result = lk.search_lightcurve(target_name, mission="Kepler")

            if len(search_result) == 0:
                search_result = lk.search_lightcurve(target_name, mission="K2")

            if len(search_result) == 0:
                return None

            # Get first result for basic info
            first_obs = search_result[0]

            # Try to get additional info from MAST catalogs
            try:
                if first_obs.mission == "Kepler":
                    catalog_result = Catalogs.query_object(target_name, catalog="KIC")
                else:  # K2
                    catalog_result = Catalogs.query_object(target_name, catalog="EPIC")
                catalog_info = catalog_result[0] if len(catalog_result) > 0 else None
            except Exception:
                catalog_info = None

            return {
                "target_name": first_obs.target_name,
                "mission": first_obs.mission,
                "quarters": [
                    obs.quarter for obs in search_result if hasattr(obs, "quarter")
                ],
                "campaigns": [
                    obs.campaign for obs in search_result if hasattr(obs, "campaign")
                ],
                "exptime": first_obs.exptime if hasattr(first_obs, "exptime") else None,
                "catalog_info": catalog_info,
            }

        except Exception as e:
            logger.error(f"Kepler target info fetch error: {e}")
            return None

    def _target_to_planet_info(
        self, target_data: Dict, query: str
    ) -> Optional[PlanetInfo]:
        """Convert Kepler target data to PlanetInfo"""
        try:
            target_name = target_data.get("target_name", query)
            mission = target_data.get("mission", "Kepler")

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

                # Different catalogs have different column names
                if mission == "Kepler":
                    stellar_temp = (
                        float(catalog_info["teff"]) if "teff" in catalog_info else None
                    )
                    stellar_radius = (
                        float(catalog_info["radius"])
                        if "radius" in catalog_info
                        else None
                    )
                    stellar_mass = (
                        float(catalog_info["mass"]) if "mass" in catalog_info else None
                    )
                    magnitude = (
                        float(catalog_info["kepmag"])
                        if "kepmag" in catalog_info
                        else None
                    )
                else:  # K2
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
                        float(catalog_info["kepmag"])
                        if "kepmag" in catalog_info
                        else None
                    )

            # Determine status - Kepler found many confirmed planets
            status = (
                PlanetStatus.CONFIRMED
                if "kepler" in target_name.lower()
                else PlanetStatus.CANDIDATE
            )

            return PlanetInfo(
                name=target_name,
                host_star=target_name,  # Kepler targets are usually stars
                status=status,
                discovery_method="Transit",
                discovery_facility=mission,
                # Kepler-specific info
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
                data_quality=f"{mission} Quarters: {target_data.get('quarters', [])} Campaigns: {target_data.get('campaigns', [])}",
            )

        except Exception as e:
            logger.error(f"Error converting Kepler target to PlanetInfo: {e}")
            return None

    async def fetch_light_curve(
        self,
        target_name: str,
        mission: Optional[str] = None,
        sector_quarter: Optional[int] = None,
    ) -> Optional[LightCurveData]:
        """Fetch Kepler light curve data"""
        try:
            loop = asyncio.get_event_loop()
            lc_data = await loop.run_in_executor(
                None, self._sync_fetch_light_curve, target_name, mission, sector_quarter
            )

            return lc_data

        except Exception as e:
            logger.error(f"Failed to fetch Kepler light curve for '{target_name}': {e}")
            raise DataNotFoundError(f"Kepler light curve for '{target_name}' not found")

    def _sync_fetch_light_curve(
        self, target_name: str, mission: Optional[str], quarter: Optional[int]
    ) -> Optional[LightCurveData]:
        """Synchronous Kepler light curve fetch"""
        try:
            # Determine mission
            search_mission = mission if mission else "Kepler"

            # Search for light curves
            search_result = lk.search_lightcurve(
                target_name, mission=search_mission, quarter=quarter
            )

            if len(search_result) == 0 and search_mission == "Kepler":
                # Try K2 if Kepler fails
                search_result = lk.search_lightcurve(
                    target_name, mission="K2", campaign=quarter
                )

            if len(search_result) == 0:
                logger.warning(
                    f"No {search_mission} light curves found for {target_name}"
                )
                return None

            # Download light curves
            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                logger.warning(
                    f"Failed to download {search_mission} light curves for {target_name}"
                )
                return None

            # Stitch multiple quarters/campaigns together
            lc = lc_collection.stitch()

            # Data quality checks
            if len(lc.flux) < 100:
                logger.warning(
                    f"Insufficient {search_mission} data points for {target_name}: {len(lc.flux)}"
                )
                return None

            # Clean and normalize data
            lc = lc.remove_nans().remove_outliers(sigma=5)
            lc = lc.normalize()

            # Extract arrays
            time_bjd = lc.time.btjd  # Barycentric Kepler Julian Date
            flux = lc.flux.value
            flux_err = (
                lc.flux_err.value
                if lc.flux_err is not None
                else np.ones_like(flux) * 0.001
            )

            # Get quarters/campaigns
            periods = []
            for lc_single in lc_collection:
                if hasattr(lc_single, "quarter"):
                    periods.append(lc_single.quarter)
                elif hasattr(lc_single, "campaign"):
                    periods.append(lc_single.campaign)

            # Determine cadence
            cadence = (
                29.4 if search_mission == "Kepler" else 29.4
            )  # Long cadence for both
            if hasattr(lc, "cadence"):
                cadence = float(lc.cadence.to(u.minute).value)

            return LightCurveData(
                target_name=target_name,
                time_bjd=time_bjd,
                flux=flux,
                flux_err=flux_err,
                mission=search_mission,
                instrument=f"{search_mission} Photometer",
                cadence_minutes=cadence,
                sectors_quarters=periods,
                detrended=True,
                normalized=True,
                outliers_removed=True,
                source=self.name,
                download_date=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Kepler light curve download error: {e}")
            raise

    async def get_statistics(self) -> Dict[str, Any]:
        """Get Kepler mission statistics"""
        try:
            return {
                "mission": "Kepler",
                "launch_date": "2009-03-07",
                "end_date": "2013-05-11",
                "status": "Completed",
                "k2_mission": {
                    "start_date": "2014-05-30",
                    "end_date": "2018-10-30",
                    "status": "Completed",
                },
                "quarters_completed": 17,
                "k2_campaigns": 19,
                "cadence_modes": ["Long cadence (29.4 min)", "Short cadence (1 min)"],
                "field_of_view": "115 square degrees",
                "primary_targets": "150,000+ stars",
                "confirmed_planets": "2,600+",
                "planet_candidates": "4,000+",
                "note": "Revolutionary exoplanet discovery mission",
                "last_updated": datetime.now().isoformat(),
                "source": self.name,
            }
        except Exception as e:
            logger.error(f"Failed to get Kepler statistics: {e}")
            return {"error": str(e)}

    def get_supported_missions(self) -> List[str]:
        """Kepler supports Kepler and K2 missions"""
        return ["Kepler", "K2"]

    def get_capabilities(self) -> Dict[str, bool]:
        """Kepler data source capabilities"""
        return {
            "planet_search": True,
            "planet_info": True,
            "light_curves": True,  # Primary capability
            "statistics": True,
            "real_time": False,
        }
