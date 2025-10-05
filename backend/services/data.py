"""
Clean Data Service for Real NASA Data Access
Очищенный сервис данных для реального доступа к NASA API
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

try:
    import astropy.units as u
    import lightkurve as lk
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs, Observations
except ImportError as e:
    # Don't raise error immediately, only when the NASA data functions are used
    pass

from core.logging import get_logger

logger = get_logger(__name__)


def normalize_tic_id(raw: str) -> str:
    # FIX: [FINDING_003] Нормализация и валидация TIC id
    if not raw:
        raise ValueError("TIC ID cannot be empty")
    # Очищаем от потенциально опасных символов
    s = raw.upper().replace("TIC", "").strip()
    # Проверяем, что осталась только цифровая строка
    if not re.match(r"^\d{1,10}$", s):
        raise ValueError("Invalid TIC ID format")
    return s


def normalize_kic_id(raw: str) -> str:
    # FIX: [FINDING_003] Нормализация и валидация KIC id
    if not raw:
        raise ValueError("KIC ID cannot be empty")
    # Очищаем от потенциально опасных символов
    s = raw.upper().replace("KIC", "").strip()
    # Проверяем, что осталась только цифровая строка
    if not re.match(r"^\d{1,10}$", s):
        raise ValueError("Invalid KIC ID format")
    return s


def normalize_epic_id(raw: str) -> str:
    # FIX: [FINDING_003] Нормализация и валидация EPIC id
    if not raw:
        raise ValueError("EPIC ID cannot be empty")
    # Очищаем от потенциально опасных символов
    s = raw.upper().replace("EPIC", "").strip()
    # Проверяем, что осталась только цифровая строка
    if not re.match(r"^\d{1,10}$", s):
        raise ValueError("Invalid EPIC ID format")
    return s


class Mission(Enum):
    """Supported space missions"""

    TESS = "TESS"
    KEPLER = "Kepler"
    K2 = "K2"


class Catalog(Enum):
    """Supported star catalogs"""

    TIC = "TIC"
    KIC = "KIC"
    EPIC = "EPIC"


@dataclass
class StarInfo:
    """Star information from catalogs"""

    target_id: str
    catalog: Catalog
    ra: float
    dec: float
    magnitude: float
    temperature: Optional[float] = None
    radius: Optional[float] = None
    mass: Optional[float] = None
    distance: Optional[float] = None
    stellar_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "catalog": self.catalog.value,
            "ra": self.ra,
            "dec": self.dec,
            "magnitude": self.magnitude,
            "temperature": self.temperature,
            "radius": self.radius,
            "mass": self.mass,
            "distance": self.distance,
            "stellar_type": self.stellar_type,
        }


@dataclass
class LightcurveData:
    """Lightcurve data from space missions"""

    target_id: str
    mission: Mission
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    cadence_minutes: float
    noise_level_ppm: float
    data_source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "mission": self.mission.value,
            "time": self.time.tolist(),
            "flux": self.flux.tolist(),
            "flux_err": self.flux_err.tolist(),
            "cadence_minutes": self.cadence_minutes,
            "noise_level_ppm": self.noise_level_ppm,
            "data_source": self.data_source,
        }


class DataService:
    """Real NASA data service using lightkurve and astroquery"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 3600  # 1 hour

    async def initialize(self):
        """Initialize the data service"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        logger.info("✅ DataService initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        logger.info("✅ DataService cleaned up")

    async def get_status(self) -> str:
        """Get service status"""
        try:
            # Test connection to MAST
            if self.session:
                async with self.session.get("https://mast.stsci.edu/") as response:
                    if response.status == 200:
                        return "healthy"
            return "degraded"
        except Exception:
            return "unhealthy"

    async def get_star_info(self, target_name: str, catalog: str) -> StarInfo:
        """Get star information from NASA catalogs"""

        cache_key = f"star_{catalog}_{target_name}"
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return data

        try:
            catalog_enum = Catalog(catalog)

            # Use astroquery to get real star data
            if catalog_enum == Catalog.TIC:
                # Query TIC catalog
                result = await self._query_tic_catalog(target_name)
            elif catalog_enum == Catalog.KIC:
                # Query KIC catalog
                result = await self._query_kic_catalog(target_name)
            elif catalog_enum == Catalog.EPIC:
                # Query EPIC catalog
                result = await self._query_epic_catalog(target_name)
            else:
                raise ValueError(f"Unsupported catalog: {catalog}")

            # Cache the result
            self.cache[cache_key] = (time.time(), result)

            return result

        except Exception as e:
            logger.error(f"Failed to get star info for {target_name}: {e}")
            # Re-raise the exception instead of returning fake data
            raise ValueError(f"No catalog data found for {target_name} in {catalog}")

    async def get_lightcurve(
        self, target_name: str, mission: str
    ) -> Optional[LightcurveData]:
        """Get lightcurve data from NASA missions"""

        cache_key = f"lc_{mission}_{target_name}"
        if cache_key in self.cache:
            cached_time, data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return data

        try:
            mission_enum = Mission(mission)

            # Use lightkurve to download real data
            if mission_enum == Mission.TESS:
                result = await self._download_tess_lightcurve(target_name)
            elif mission_enum == Mission.KEPLER:
                result = await self._download_kepler_lightcurve(target_name)
            elif mission_enum == Mission.K2:
                result = await self._download_k2_lightcurve(target_name)
            else:
                raise ValueError(f"Unsupported mission: {mission}")

            if result:
                # Cache the result
                self.cache[cache_key] = (time.time(), result)

            return result

        except Exception as e:
            logger.error(f"Failed to get lightcurve for {target_name}: {e}")
            return None

    async def _query_tic_catalog(self, tic_id: str) -> StarInfo:
        """Query TIC catalog using astroquery"""

        def _sync_query():
            # Validate and normalize TIC ID
            clean_id = normalize_tic_id(tic_id)

            # Query MAST for TIC data
            catalog_data = Catalogs.query_object(
                f"TIC {clean_id}", radius=0.01 * u.deg, catalog="TIC"
            )

            if len(catalog_data) == 0:
                raise ValueError(f"No TIC data found for {tic_id}")

            row = catalog_data[0]

            return StarInfo(
                target_id=f"TIC {clean_id}",
                catalog=Catalog.TIC,
                ra=float(row.get("ra", 0.0)),
                dec=float(row.get("dec", 0.0)),
                magnitude=float(row.get("Tmag", 12.0)),
                temperature=float(row.get("Teff", 5500)) if row.get("Teff") else None,
                radius=float(row.get("rad", 1.0)) if row.get("rad") else None,
                mass=float(row.get("mass", 1.0)) if row.get("mass") else None,
                stellar_type=(
                    str(row.get("SpType", "Unknown")) if row.get("SpType") else None
                ),
            )

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_query)

    async def _query_kic_catalog(self, kic_id: str) -> StarInfo:
        """Query KIC catalog using astroquery"""

        def _sync_query():
            # Validate and normalize KIC ID
            clean_id = normalize_kic_id(kic_id)

            catalog_data = Catalogs.query_object(
                f"KIC {clean_id}", radius=0.01 * u.deg, catalog="Kepler"
            )

            if len(catalog_data) == 0:
                raise ValueError(f"No KIC data found for {kic_id}")

            row = catalog_data[0]

            return StarInfo(
                target_id=f"KIC {clean_id}",
                catalog=Catalog.KIC,
                ra=float(row.get("ra", 0.0)),
                dec=float(row.get("dec", 0.0)),
                magnitude=float(row.get("kepmag", 12.0)),
                temperature=float(row.get("teff", 5500)) if row.get("teff") else None,
                radius=float(row.get("radius", 1.0)) if row.get("radius") else None,
                mass=float(row.get("mass", 1.0)) if row.get("mass") else None,
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_query)

    async def _query_epic_catalog(self, epic_id: str) -> StarInfo:
        """Query EPIC catalog using astroquery"""

        def _sync_query():
            # Validate and normalize EPIC ID
            clean_id = normalize_epic_id(epic_id)

            catalog_data = Catalogs.query_object(
                f"EPIC {clean_id}", radius=0.01 * u.deg, catalog="K2"
            )

            if len(catalog_data) == 0:
                raise ValueError(f"No EPIC data found for {epic_id}")

            row = catalog_data[0]

            return StarInfo(
                target_id=f"EPIC {clean_id}",
                catalog=Catalog.EPIC,
                ra=float(row.get("ra", 0.0)),
                dec=float(row.get("dec", 0.0)),
                magnitude=float(row.get("kepmag", 12.0)),
                temperature=float(row.get("teff", 5500)) if row.get("teff") else None,
                radius=float(row.get("radius", 1.0)) if row.get("radius") else None,
                mass=float(row.get("mass", 1.0)) if row.get("mass") else None,
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_query)

    async def _download_tess_lightcurve(
        self, target_name: str
    ) -> Optional[LightcurveData]:
        """Download TESS lightcurve using lightkurve - only real data"""

        def _sync_download():
            # Search for TESS data
            logger.info(f"Searching for TESS data for {target_name}")
            search_result = lk.search_lightcurve(target_name, mission="TESS")

            if len(search_result) == 0:
                # Try alternative search methods
                logger.info(
                    f"No direct TESS data found for {target_name}, trying alternative searches..."
                )

                # Try without prefix
                clean_name = target_name.replace("TIC", "").replace("tic", "").strip()
                search_result = lk.search_lightcurve(clean_name, mission="TESS")

                if len(search_result) == 0:
                    # Try with TIC prefix
                    search_result = lk.search_lightcurve(
                        f"TIC {clean_name}", mission="TESS"
                    )

                if len(search_result) == 0:
                    raise ValueError(f"No TESS data found for {target_name}")

            logger.info(
                f"Found {len(search_result)} TESS observations for {target_name}"
            )

            # Download the first available lightcurve
            lc = search_result[0].download()

            if lc is None:
                raise ValueError(
                    f"Failed to download TESS lightcurve for {target_name}"
                )

            # Remove NaN values and normalize
            lc = lc.remove_nans().normalize()

            # Extract time and flux arrays safely
            if hasattr(lc.time, "value"):
                time_array = lc.time.value
            else:
                time_array = np.array(lc.time)

            if hasattr(lc.flux, "value"):
                flux_array = lc.flux.value
            else:
                flux_array = np.array(lc.flux)

            # Handle flux errors
            if lc.flux_err is not None:
                if hasattr(lc.flux_err, "value"):
                    flux_err_array = lc.flux_err.value
                else:
                    flux_err_array = np.array(lc.flux_err)
            else:
                flux_err_array = np.full_like(flux_array, 0.001)

            # Calculate noise level
            noise_ppm = np.std(flux_array) * 1e6

            logger.info(
                f"Successfully downloaded TESS lightcurve: {len(time_array)} points, noise: {noise_ppm:.1f} ppm"
            )

            return LightcurveData(
                target_id=target_name,
                mission=Mission.TESS,
                time=time_array,
                flux=flux_array,
                flux_err=flux_err_array,
                cadence_minutes=2.0,  # TESS 2-minute cadence
                noise_level_ppm=noise_ppm,
                data_source="TESS FFI",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_download)

    async def _download_kepler_lightcurve(
        self, target_name: str
    ) -> Optional[LightcurveData]:
        """Download Kepler lightcurve using lightkurve - only real data"""

        def _sync_download():
            search_result = lk.search_lightcurve(target_name, mission="Kepler")

            if len(search_result) == 0:
                raise ValueError(f"No Kepler data found for {target_name}")

            lc = search_result[0].download()

            if lc is None:
                raise ValueError(
                    f"Failed to download Kepler lightcurve for {target_name}"
                )

            lc = lc.remove_nans().normalize()

            # Extract arrays safely
            if hasattr(lc.time, "value"):
                time_array = lc.time.value
            else:
                time_array = np.array(lc.time)

            if hasattr(lc.flux, "value"):
                flux_array = lc.flux.value
            else:
                flux_array = np.array(lc.flux)

            if lc.flux_err is not None:
                if hasattr(lc.flux_err, "value"):
                    flux_err_array = lc.flux_err.value
                else:
                    flux_err_array = np.array(lc.flux_err)
            else:
                flux_err_array = np.full_like(flux_array, 0.001)

            noise_ppm = np.std(flux_array) * 1e6

            return LightcurveData(
                target_id=target_name,
                mission=Mission.KEPLER,
                time=time_array,
                flux=flux_array,
                flux_err=flux_err_array,
                cadence_minutes=29.4,  # Kepler long cadence
                noise_level_ppm=noise_ppm,
                data_source="Kepler SAP",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_download)

    async def _download_k2_lightcurve(
        self, target_name: str
    ) -> Optional[LightcurveData]:
        """Download K2 lightcurve using lightkurve - only real data"""

        def _sync_download():
            search_result = lk.search_lightcurve(target_name, mission="K2")

            if len(search_result) == 0:
                raise ValueError(f"No K2 data found for {target_name}")

            lc = search_result[0].download()

            if lc is None:
                raise ValueError(f"Failed to download K2 lightcurve for {target_name}")

            lc = lc.remove_nans().normalize()

            # Extract arrays safely
            if hasattr(lc.time, "value"):
                time_array = lc.time.value
            else:
                time_array = np.array(lc.time)

            if hasattr(lc.flux, "value"):
                flux_array = lc.flux.value
            else:
                flux_array = np.array(lc.flux)

            if lc.flux_err is not None:
                if hasattr(lc.flux_err, "value"):
                    flux_err_array = lc.flux_err.value
                else:
                    flux_err_array = np.array(lc.flux_err)
            else:
                flux_err_array = np.full_like(flux_array, 0.001)

            noise_ppm = np.std(flux_array) * 1e6

            return LightcurveData(
                target_id=target_name,
                mission=Mission.K2,
                time=time_array,
                flux=flux_array,
                flux_err=flux_err_array,
                cadence_minutes=29.4,  # K2 long cadence
                noise_level_ppm=noise_ppm,
                data_source="K2 SAP",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_download)


# Global instance
data_service = DataService()
