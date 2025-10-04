"""
Real Astronomical Data Sources Module
Модуль для работы с реальными астрономическими источниками данных
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import aiohttp
import numpy as np

try:
    import astropy.units as u
    import lightkurve as lk
    from astropy.coordinates import SkyCoord
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    from astroquery.mast import Catalogs, Observations

    ASTRONOMY_LIBS_AVAILABLE = True
except ImportError as e:
    ASTRONOMY_LIBS_AVAILABLE = False
    logging.warning(f"Astronomy libraries not available: {e}")

from core.logging_config import get_logger

logger = get_logger(__name__)


class DataSource(Enum):
    """Supported astronomical data sources"""

    NASA_EXOPLANET_ARCHIVE = "nasa_exoplanet_archive"
    MAST_TESS = "mast_tess"
    MAST_KEPLER = "mast_kepler"
    MAST_K2 = "mast_k2"
    ESA_GAIA = "esa_gaia"
    ESA_PLATO = "esa_plato"


@dataclass
class TargetInfo:
    """Information about astronomical target"""

    name: str
    ra: float
    dec: float
    magnitude: float
    catalog_id: str
    mission: str
    data_quality: str
    observation_days: int
    data_points: int
    source: DataSource


@dataclass
class LightCurveData:
    """Light curve data from real observations"""

    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray
    target_info: TargetInfo
    metadata: Dict[str, Any]


class RealDataSourceManager:
    """Manager for real astronomical data sources"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=6)  # Cache real data for 6 hours

        # API endpoints
        self.endpoints = {
            DataSource.NASA_EXOPLANET_ARCHIVE: "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
            DataSource.MAST_TESS: "https://mast.stsci.edu/api/v0.1",
            DataSource.ESA_GAIA: "https://gea.esac.esa.int/tap-server/tap",
            DataSource.ESA_PLATO: "https://plato.esac.esa.int/pwp-api",
        }

    async def initialize(self):
        """Initialize the data source manager"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)

        if not ASTRONOMY_LIBS_AVAILABLE:
            logger.warning(
                "⚠️ Astronomy libraries not available. Install: pip install lightkurve astroquery"
            )
            logger.info("Some features may be limited without these libraries")
        else:
            logger.info("✅ Real astronomical data sources initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and not expired"""
        if cache_key not in self.cache:
            return False

        if cache_key in self.cache_expiry:
            if datetime.now() > self.cache_expiry[cache_key]:
                del self.cache[cache_key]
                del self.cache_expiry[cache_key]
                return False

        return True

    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with expiry"""
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration

    async def search_nasa_exoplanet_archive(
        self, target_name: str
    ) -> Optional[Dict[str, Any]]:
        """Search NASA Exoplanet Archive for target information"""
        cache_key = f"nasa_archive_{target_name}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        try:
            if not ASTRONOMY_LIBS_AVAILABLE:
                logger.warning("Cannot query NASA Exoplanet Archive without astroquery")
                return None

            # Query confirmed exoplanets
            query = f"""
            SELECT pl_name, hostname, ra, dec, sy_vmag, pl_orbper, pl_radj, pl_massj,
                   st_teff, st_rad, st_mass, disc_year, disc_facility
            FROM ps
            WHERE hostname LIKE '%{target_name}%' OR pl_name LIKE '%{target_name}%'
            """

            result = NasaExoplanetArchive.query_criteria(
                table="ps", where=f"hostname like '%{target_name}%'"
            )

            if len(result) > 0:
                data = {
                    "target_name": target_name,
                    "planets_found": len(result),
                    "host_star": {
                        "ra": float(result[0]["ra"]) if result[0]["ra"] else None,
                        "dec": float(result[0]["dec"]) if result[0]["dec"] else None,
                        "magnitude": (
                            float(result[0]["sy_vmag"])
                            if result[0]["sy_vmag"]
                            else None
                        ),
                        "temperature": (
                            float(result[0]["st_teff"])
                            if result[0]["st_teff"]
                            else None
                        ),
                        "radius": (
                            float(result[0]["st_rad"]) if result[0]["st_rad"] else None
                        ),
                        "mass": (
                            float(result[0]["st_mass"])
                            if result[0]["st_mass"]
                            else None
                        ),
                    },
                    "planets": [],
                }

                for row in result:
                    planet_data = {
                        "name": str(row["pl_name"]),
                        "orbital_period": (
                            float(row["pl_orbper"]) if row["pl_orbper"] else None
                        ),
                        "radius": float(row["pl_radj"]) if row["pl_radj"] else None,
                        "mass": float(row["pl_massj"]) if row["pl_massj"] else None,
                        "discovery_year": (
                            int(row["disc_year"]) if row["disc_year"] else None
                        ),
                        "discovery_facility": (
                            str(row["disc_facility"]) if row["disc_facility"] else None
                        ),
                    }
                    data["planets"].append(planet_data)

                self._cache_data(cache_key, data)
                return data

        except Exception as e:
            logger.error(f"NASA Exoplanet Archive query failed: {e}")

        return None

    async def get_tess_lightcurve(
        self, target_name: str, sector: Optional[int] = None
    ) -> Optional[LightCurveData]:
        """Get TESS light curve data for target"""
        cache_key = f"tess_lc_{target_name}_{sector}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        try:
            if not ASTRONOMY_LIBS_AVAILABLE:
                logger.warning("Cannot download TESS data without lightkurve")
                return None

            # Search for TESS observations
            search_result = lk.search_lightcurve(
                target_name, mission="TESS", sector=sector
            )

            if len(search_result) == 0:
                logger.warning(f"No TESS data found for {target_name}")
                return None

            # Download the light curve
            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                logger.warning(f"Failed to download TESS data for {target_name}")
                return None

            # Combine all sectors
            lc = lc_collection.stitch()

            # Remove NaN values and outliers
            lc = lc.remove_nans().remove_outliers(sigma=5)

            # Normalize flux
            lc = lc.normalize()

            # Create target info
            target_info = TargetInfo(
                name=target_name,
                ra=float(lc.ra.value) if hasattr(lc, "ra") else 0.0,
                dec=float(lc.dec.value) if hasattr(lc, "dec") else 0.0,
                magnitude=float(lc.meta.get("TESSMAG", 0.0)),
                catalog_id=str(lc.meta.get("TICID", "Unknown")),
                mission="TESS",
                data_quality="Good" if len(lc.flux) > 1000 else "Limited",
                observation_days=int((lc.time[-1] - lc.time[0]).value),
                data_points=len(lc.flux),
                source=DataSource.MAST_TESS,
            )

            # Create light curve data object
            lc_data = LightCurveData(
                time=lc.time.value,
                flux=lc.flux.value,
                flux_err=(
                    lc.flux_err.value
                    if lc.flux_err is not None
                    else np.ones_like(lc.flux.value) * 0.001
                ),
                quality=(
                    lc.quality.value
                    if hasattr(lc, "quality")
                    else np.zeros_like(lc.flux.value)
                ),
                target_info=target_info,
                metadata={
                    "mission": "TESS",
                    "sectors": [obs.meta.get("SECTOR", 0) for obs in lc_collection],
                    "cadence": lc.meta.get("TIMEDEL", 0.0),
                    "downloaded_at": datetime.now().isoformat(),
                },
            )

            self._cache_data(cache_key, lc_data)
            logger.info(
                f"✅ Downloaded TESS data for {target_name}: {len(lc.flux)} points"
            )
            return lc_data

        except Exception as e:
            logger.error(f"TESS data download failed for {target_name}: {e}")
            return None

    async def get_kepler_lightcurve(
        self, target_name: str, quarter: Optional[int] = None
    ) -> Optional[LightCurveData]:
        """Get Kepler light curve data for target"""
        cache_key = f"kepler_lc_{target_name}_{quarter}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]

        try:
            if not ASTRONOMY_LIBS_AVAILABLE:
                logger.warning("Cannot download Kepler data without lightkurve")
                return None

            # Search for Kepler observations
            search_result = lk.search_lightcurve(
                target_name, mission="Kepler", quarter=quarter
            )

            if len(search_result) == 0:
                logger.warning(f"No Kepler data found for {target_name}")
                return None

            # Download the light curve
            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                logger.warning(f"Failed to download Kepler data for {target_name}")
                return None

            # Combine all quarters
            lc = lc_collection.stitch()

            # Remove NaN values and outliers
            lc = lc.remove_nans().remove_outliers(sigma=5)

            # Normalize flux
            lc = lc.normalize()

            # Create target info
            target_info = TargetInfo(
                name=target_name,
                ra=float(lc.ra.value) if hasattr(lc, "ra") else 0.0,
                dec=float(lc.dec.value) if hasattr(lc, "dec") else 0.0,
                magnitude=float(lc.meta.get("KEPMAG", 0.0)),
                catalog_id=str(lc.meta.get("KEPLERID", "Unknown")),
                mission="Kepler",
                data_quality="Good" if len(lc.flux) > 1000 else "Limited",
                observation_days=int((lc.time[-1] - lc.time[0]).value),
                data_points=len(lc.flux),
                source=DataSource.MAST_KEPLER,
            )

            # Create light curve data object
            lc_data = LightCurveData(
                time=lc.time.value,
                flux=lc.flux.value,
                flux_err=(
                    lc.flux_err.value
                    if lc.flux_err is not None
                    else np.ones_like(lc.flux.value) * 0.001
                ),
                quality=(
                    lc.quality.value
                    if hasattr(lc, "quality")
                    else np.zeros_like(lc.flux.value)
                ),
                target_info=target_info,
                metadata={
                    "mission": "Kepler",
                    "quarters": [obs.meta.get("QUARTER", 0) for obs in lc_collection],
                    "cadence": lc.meta.get("TIMEDEL", 0.0),
                    "downloaded_at": datetime.now().isoformat(),
                },
            )

            self._cache_data(cache_key, lc_data)
            logger.info(
                f"✅ Downloaded Kepler data for {target_name}: {len(lc.flux)} points"
            )
            return lc_data

        except Exception as e:
            logger.error(f"Kepler data download failed for {target_name}: {e}")
            return None

    async def get_multi_mission_data(
        self, target_name: str
    ) -> Dict[str, Optional[LightCurveData]]:
        """Get data from multiple missions for comparison"""
        results = {}

        # Try TESS first (most recent)
        tess_data = await self.get_tess_lightcurve(target_name)
        results["TESS"] = tess_data

        # Try Kepler
        kepler_data = await self.get_kepler_lightcurve(target_name)
        results["Kepler"] = kepler_data

        # Try K2 (using Kepler function with different mission)
        try:
            if ASTRONOMY_LIBS_AVAILABLE:
                search_result = lk.search_lightcurve(target_name, mission="K2")
                if len(search_result) > 0:
                    k2_data = await self.get_kepler_lightcurve(
                        target_name
                    )  # Similar processing
                    results["K2"] = k2_data
        except Exception as e:
            logger.debug(f"K2 search failed: {e}")
            results["K2"] = None

        return results

    async def validate_target_exists(self, target_name: str) -> bool:
        """Validate that target exists in astronomical databases"""
        try:
            # Check NASA Exoplanet Archive
            nasa_data = await self.search_nasa_exoplanet_archive(target_name)
            if nasa_data:
                return True

            # Check MAST for any observations
            if ASTRONOMY_LIBS_AVAILABLE:
                tess_search = lk.search_lightcurve(target_name, mission="TESS")
                kepler_search = lk.search_lightcurve(target_name, mission="Kepler")

                if len(tess_search) > 0 or len(kepler_search) > 0:
                    return True

            return False

        except Exception as e:
            logger.error(f"Target validation failed: {e}")
            return False

    async def get_target_statistics(self, target_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a target"""
        stats = {
            "target_name": target_name,
            "data_sources": {},
            "total_observations": 0,
            "date_range": {},
            "data_quality": "Unknown",
        }

        try:
            # Get multi-mission data
            multi_data = await self.get_multi_mission_data(target_name)

            for mission, lc_data in multi_data.items():
                if lc_data is not None:
                    stats["data_sources"][mission] = {
                        "data_points": lc_data.target_info.data_points,
                        "observation_days": lc_data.target_info.observation_days,
                        "magnitude": lc_data.target_info.magnitude,
                        "quality": lc_data.target_info.data_quality,
                    }
                    stats["total_observations"] += lc_data.target_info.data_points

            # Get NASA archive info
            nasa_info = await self.search_nasa_exoplanet_archive(target_name)
            if nasa_info:
                stats["nasa_archive"] = nasa_info

            # Determine overall quality
            if stats["total_observations"] > 10000:
                stats["data_quality"] = "Excellent"
            elif stats["total_observations"] > 1000:
                stats["data_quality"] = "Good"
            elif stats["total_observations"] > 100:
                stats["data_quality"] = "Limited"
            else:
                stats["data_quality"] = "Poor"

        except Exception as e:
            logger.error(f"Failed to get target statistics: {e}")

        return stats


# Global instance
real_data_manager = RealDataSourceManager()


async def get_real_lightcurve_data(
    target_name: str, mission: str = "TESS"
) -> Optional[LightCurveData]:
    """
    Get real light curve data for a target from specified mission

    Args:
        target_name: Name of the astronomical target
        mission: Mission name (TESS, Kepler, K2)

    Returns:
        LightCurveData object or None if not found
    """
    await real_data_manager.initialize()

    if mission.upper() == "TESS":
        return await real_data_manager.get_tess_lightcurve(target_name)
    elif mission.upper() in ["KEPLER", "K2"]:
        return await real_data_manager.get_kepler_lightcurve(target_name)
    else:
        logger.warning(f"Unsupported mission: {mission}")
        return None


async def validate_astronomical_target(target_name: str) -> bool:
    """
    Validate that an astronomical target exists in real databases

    Args:
        target_name: Name of the target to validate

    Returns:
        True if target exists, False otherwise
    """
    await real_data_manager.initialize()
    return await real_data_manager.validate_target_exists(target_name)
