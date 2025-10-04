"""
Secure NASA Data Service with async operations and comprehensive error handling
Безопасный сервис NASA данных с асинхронными операциями и обработкой ошибок
"""

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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
from core.validators import SecurityError, validate_catalog, validate_target_name

logger = get_logger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data sources"""

    base_url: str
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 1.0
    rate_limit_per_minute: int = 60


class ExponentialBackoff:
    """Exponential backoff for retries"""

    def __init__(
        self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    break

                delay = min(self.base_delay * (2**attempt), self.max_delay)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)

        raise last_exception


class SecureNASAService:
    """Secure NASA data service with comprehensive validation and error handling"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Any] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(hours=6)  # Cache real data for 6 hours
        self.backoff = ExponentialBackoff()

        # Data source configurations
        self.data_sources = {
            "nasa_exoplanet_archive": DataSourceConfig(
                base_url="https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                timeout=60,
                max_retries=3,
            ),
            "mast": DataSourceConfig(
                base_url="https://mast.stsci.edu/api/v0.1", timeout=120, max_retries=3
            ),
        }

        # Rate limiting
        self._request_times: Dict[str, List[datetime]] = {}

    async def initialize(self):
        """Initialize the service with secure session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=120)
            connector = aiohttp.TCPConnector(
                limit=10,  # Connection pool limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                ssl=True,  # Enforce SSL
            )

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": "ExoplanetAI/1.0 (Scientific Research)",
                    "Accept": "application/json",
                },
            )

        if not ASTRONOMY_LIBS_AVAILABLE:
            logger.warning(
                "⚠️ Astronomy libraries not available. Install: pip install lightkurve astroquery"
            )
        else:
            logger.info("✅ Secure NASA service initialized")

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
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries
            oldest_key = min(
                self.cache_expiry.keys(), key=lambda k: self.cache_expiry[k]
            )
            del self.cache[oldest_key]
            del self.cache_expiry[oldest_key]

        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = datetime.now() + self.cache_duration

    def _check_rate_limit(self, service: str) -> bool:
        """Check rate limiting for service"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        if service not in self._request_times:
            self._request_times[service] = []

        # Clean old requests
        self._request_times[service] = [
            req_time
            for req_time in self._request_times[service]
            if req_time > minute_ago
        ]

        # Check limit
        config = self.data_sources.get(service)
        if config and len(self._request_times[service]) >= config.rate_limit_per_minute:
            return False

        # Add current request
        self._request_times[service].append(now)
        return True

    async def _secure_http_request(
        self, url: str, params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make secure HTTP request with validation"""
        if not self.session:
            await self.initialize()

        # Validate URL
        if not url.startswith(
            ("https://exoplanetarchive.ipac.caltech.edu", "https://mast.stsci.edu")
        ):
            raise SecurityError(f"Unauthorized URL: {url}")

        # Rate limiting
        service = "nasa_exoplanet_archive" if "exoplanet" in url else "mast"
        if not self._check_rate_limit(service):
            raise SecurityError(f"Rate limit exceeded for {service}")

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=429,
                        message="Rate limited by NASA API",
                    )
                else:
                    response.raise_for_status()
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise

    async def search_exoplanet_archive(
        self, target_name: str
    ) -> Optional[Dict[str, Any]]:
        """Search NASA Exoplanet Archive for target information"""
        # Input validation
        target_name = validate_target_name(target_name)

        cache_key = f"nasa_archive_{hashlib.md5(target_name.encode()).hexdigest()}"
        if self._is_cached(cache_key):
            logger.info(f"Cache hit for NASA archive: {target_name}")
            return self.cache[cache_key]

        if not ASTRONOMY_LIBS_AVAILABLE:
            logger.error("Cannot query NASA Exoplanet Archive without astroquery")
            return None

        try:
            # Use astroquery with retry logic
            async def _query_archive():
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._sync_query_archive, target_name
                )

            result = await self.backoff.retry(_query_archive)

            if result and len(result) > 0:
                data = self._process_archive_result(result, target_name)
                self._cache_data(cache_key, data)
                logger.info(
                    f"✅ Found {len(result)} entries for {target_name} in NASA archive"
                )
                return data
            else:
                logger.warning(f"No data found for {target_name} in NASA archive")
                return None

        except Exception as e:
            logger.error(f"NASA Exoplanet Archive query failed for {target_name}: {e}")
            return None

    def _sync_query_archive(self, target_name: str):
        """Synchronous query to NASA Exoplanet Archive"""
        try:
            # Query confirmed exoplanets
            result = NasaExoplanetArchive.query_criteria(
                table="ps",
                where=f"hostname like '%{target_name}%' or pl_name like '%{target_name}%'",
                cache=True,
            )
            return result
        except Exception as e:
            logger.error(f"Archive query error: {e}")
            raise

    def _process_archive_result(self, result, target_name: str) -> Dict[str, Any]:
        """Process NASA archive query result"""
        data = {
            "target_name": target_name,
            "planets_found": len(result),
            "query_timestamp": datetime.now().isoformat(),
            "host_star": {},
            "planets": [],
        }

        if len(result) > 0:
            first_row = result[0]

            # Host star information
            data["host_star"] = {
                "ra": float(first_row["ra"]) if first_row["ra"] else None,
                "dec": float(first_row["dec"]) if first_row["dec"] else None,
                "magnitude": (
                    float(first_row["sy_vmag"]) if first_row["sy_vmag"] else None
                ),
                "temperature": (
                    float(first_row["st_teff"]) if first_row["st_teff"] else None
                ),
                "radius": float(first_row["st_rad"]) if first_row["st_rad"] else None,
                "mass": float(first_row["st_mass"]) if first_row["st_mass"] else None,
            }

            # Planet information
            for row in result:
                planet_data = {
                    "name": str(row["pl_name"]) if row["pl_name"] else None,
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

        return data

    async def get_tess_lightcurve(
        self, target_name: str, sector: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get TESS light curve data for target"""
        # Input validation
        target_name = validate_target_name(target_name)

        cache_key = (
            f"tess_lc_{hashlib.md5(f'{target_name}_{sector}'.encode()).hexdigest()}"
        )
        if self._is_cached(cache_key):
            logger.info(f"Cache hit for TESS lightcurve: {target_name}")
            return self.cache[cache_key]

        if not ASTRONOMY_LIBS_AVAILABLE:
            logger.error("Cannot download TESS data without lightkurve")
            return None

        try:

            async def _download_tess():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._sync_download_tess, target_name, sector
                )

            result = await self.backoff.retry(_download_tess)

            if result:
                self._cache_data(cache_key, result)
                time_data, flux_data, flux_err_data = result
                logger.info(
                    f"✅ Downloaded TESS data for {target_name}: {len(time_data)} points"
                )
                return result
            else:
                logger.warning(f"No TESS data found for {target_name}")
                return None

        except Exception as e:
            logger.error(f"TESS data download failed for {target_name}: {e}")
            return None

    def _sync_download_tess(
        self, target_name: str, sector: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Synchronous TESS data download"""
        try:
            # Search for TESS observations
            search_result = lk.search_lightcurve(
                target_name, mission="TESS", sector=sector
            )

            if len(search_result) == 0:
                logger.warning(f"No TESS observations found for {target_name}")
                return None

            # Download the light curve
            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                logger.warning(f"Failed to download TESS data for {target_name}")
                return None

            # Combine all sectors
            lc = lc_collection.stitch()

            # Data quality checks
            if len(lc.flux) < 100:
                logger.warning(
                    f"Insufficient TESS data points for {target_name}: {len(lc.flux)}"
                )
                return None

            # Remove NaN values and outliers
            lc = lc.remove_nans().remove_outliers(sigma=5)

            # Normalize flux
            lc = lc.normalize()

            # Extract data arrays
            time_data = lc.time.value
            flux_data = lc.flux.value
            flux_err_data = (
                lc.flux_err.value
                if lc.flux_err is not None
                else np.ones_like(flux_data) * 0.001
            )

            return time_data, flux_data, flux_err_data

        except Exception as e:
            logger.error(f"TESS download error: {e}")
            raise

    async def get_kepler_lightcurve(
        self, target_name: str, quarter: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get Kepler light curve data for target"""
        # Input validation
        target_name = validate_target_name(target_name)

        cache_key = (
            f"kepler_lc_{hashlib.md5(f'{target_name}_{quarter}'.encode()).hexdigest()}"
        )
        if self._is_cached(cache_key):
            logger.info(f"Cache hit for Kepler lightcurve: {target_name}")
            return self.cache[cache_key]

        if not ASTRONOMY_LIBS_AVAILABLE:
            logger.error("Cannot download Kepler data without lightkurve")
            return None

        try:

            async def _download_kepler():
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._sync_download_kepler, target_name, quarter
                )

            result = await self.backoff.retry(_download_kepler)

            if result:
                self._cache_data(cache_key, result)
                time_data, flux_data, flux_err_data = result
                logger.info(
                    f"✅ Downloaded Kepler data for {target_name}: {len(time_data)} points"
                )
                return result
            else:
                logger.warning(f"No Kepler data found for {target_name}")
                return None

        except Exception as e:
            logger.error(f"Kepler data download failed for {target_name}: {e}")
            return None

    def _sync_download_kepler(
        self, target_name: str, quarter: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Synchronous Kepler data download"""
        try:
            # Search for Kepler observations
            search_result = lk.search_lightcurve(
                target_name, mission="Kepler", quarter=quarter
            )

            if len(search_result) == 0:
                logger.warning(f"No Kepler observations found for {target_name}")
                return None

            # Download the light curve
            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                logger.warning(f"Failed to download Kepler data for {target_name}")
                return None

            # Combine all quarters
            lc = lc_collection.stitch()

            # Data quality checks
            if len(lc.flux) < 100:
                logger.warning(
                    f"Insufficient Kepler data points for {target_name}: {len(lc.flux)}"
                )
                return None

            # Remove NaN values and outliers
            lc = lc.remove_nans().remove_outliers(sigma=5)

            # Normalize flux
            lc = lc.normalize()

            # Extract data arrays
            time_data = lc.time.value
            flux_data = lc.flux.value
            flux_err_data = (
                lc.flux_err.value
                if lc.flux_err is not None
                else np.ones_like(flux_data) * 0.001
            )

            return time_data, flux_data, flux_err_data

        except Exception as e:
            logger.error(f"Kepler download error: {e}")
            raise

    async def validate_target_exists(self, target_name: str) -> bool:
        """Validate that target exists in astronomical databases"""
        try:
            # Check NASA Exoplanet Archive
            archive_data = await self.search_exoplanet_archive(target_name)
            if archive_data and archive_data.get("planets_found", 0) > 0:
                return True

            # Check MAST for any observations
            tess_data = await self.get_tess_lightcurve(target_name)
            if tess_data:
                return True

            kepler_data = await self.get_kepler_lightcurve(target_name)
            if kepler_data:
                return True

            return False

        except Exception as e:
            logger.error(f"Target validation failed for {target_name}: {e}")
            return False


# Global instance
secure_nasa_service = SecureNASAService()
