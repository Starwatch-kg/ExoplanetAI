"""
Base data source interface for astronomical data providers
Базовый интерфейс для источников астрономических данных
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class DataSourceType(str, Enum):
    """Data source types"""

    NASA = "nasa"
    ESA = "esa"
    KEPLER = "kepler"
    TESS = "tess"
    MAST = "mast"


class PlanetStatus(str, Enum):
    """Planet confirmation status"""

    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"
    FALSE_POSITIVE = "false_positive"
    DISPUTED = "disputed"


@dataclass
class PlanetInfo:
    """Standardized planet information"""

    name: str
    host_star: str
    status: PlanetStatus
    discovery_year: Optional[int] = None
    discovery_method: Optional[str] = None
    discovery_facility: Optional[str] = None

    # Orbital parameters
    orbital_period_days: Optional[float] = None
    semi_major_axis_au: Optional[float] = None
    eccentricity: Optional[float] = None
    inclination_deg: Optional[float] = None

    # Physical parameters
    radius_earth_radii: Optional[float] = None
    radius_jupiter_radii: Optional[float] = None
    mass_earth_masses: Optional[float] = None
    mass_jupiter_masses: Optional[float] = None
    density_g_cm3: Optional[float] = None

    # Atmospheric parameters
    equilibrium_temperature_k: Optional[float] = None
    atmosphere_detected: Optional[bool] = None

    # Transit parameters
    transit_depth_ppm: Optional[float] = None
    transit_duration_hours: Optional[float] = None

    # Host star parameters
    stellar_mass_solar: Optional[float] = None
    stellar_radius_solar: Optional[float] = None
    stellar_temperature_k: Optional[float] = None
    stellar_magnitude: Optional[float] = None

    # Coordinates
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    distance_pc: Optional[float] = None

    # Habitability
    habitable_zone: Optional[bool] = None
    earth_similarity_index: Optional[float] = None

    # Data source info
    source: Optional[str] = None
    last_updated: Optional[datetime] = None
    data_quality: Optional[str] = None


@dataclass
class LightCurveData:
    """Standardized light curve data"""

    target_name: str
    time_bjd: np.ndarray  # Barycentric Julian Date
    flux: np.ndarray  # Normalized flux
    flux_err: np.ndarray  # Flux uncertainty

    # Metadata
    mission: str
    instrument: Optional[str] = None
    cadence_minutes: Optional[float] = None
    sectors_quarters: Optional[List[int]] = None
    data_quality_flags: Optional[np.ndarray] = None

    # Processing info
    detrended: bool = False
    normalized: bool = False
    outliers_removed: bool = False

    # Source info
    source: Optional[str] = None
    download_date: Optional[datetime] = None

    def __post_init__(self):
        """Validate data consistency"""
        if len(self.time_bjd) != len(self.flux):
            raise ValueError("Time and flux arrays must have same length")
        if len(self.flux) != len(self.flux_err):
            raise ValueError("Flux and flux_err arrays must have same length")


@dataclass
class SearchResult:
    """Search result container"""

    query: str
    total_found: int
    planets: List[PlanetInfo]
    search_time_ms: float
    source: str
    cached: bool = False


class DataSourceError(Exception):
    """Base exception for data source errors"""

    pass


class DataSourceUnavailableError(DataSourceError):
    """Raised when data source is temporarily unavailable"""

    pass


class DataNotFoundError(DataSourceError):
    """Raised when requested data is not found"""

    pass


class RateLimitError(DataSourceError):
    """Raised when rate limit is exceeded"""

    pass


class BaseDataSource(ABC):
    """
    Abstract base class for all astronomical data sources

    All data sources must implement this interface to ensure
    consistent behavior across different providers.
    """

    def __init__(self, name: str, source_type: DataSourceType):
        self.name = name
        self.source_type = source_type
        self.is_initialized = False
        self._session = None

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the data source (connections, auth, etc.)

        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources (close connections, etc.)"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if data source is healthy and accessible

        Returns:
            Dict with health status information
        """
        pass

    @abstractmethod
    async def search_planets(
        self, query: str, limit: int = 100, filters: Optional[Dict[str, Any]] = None
    ) -> SearchResult:
        """
        Search for exoplanets by name or criteria

        Args:
            query: Search query (planet name, star name, etc.)
            limit: Maximum number of results
            filters: Additional search filters

        Returns:
            SearchResult with found planets
        """
        pass

    @abstractmethod
    async def fetch_planet_info(self, planet_name: str) -> Optional[PlanetInfo]:
        """
        Fetch detailed information about a specific planet

        Args:
            planet_name: Name of the planet

        Returns:
            PlanetInfo object or None if not found
        """
        pass

    @abstractmethod
    async def fetch_light_curve(
        self,
        target_name: str,
        mission: Optional[str] = None,
        sector_quarter: Optional[int] = None,
    ) -> Optional[LightCurveData]:
        """
        Fetch light curve data for a target

        Args:
            target_name: Target identifier
            mission: Specific mission (TESS, Kepler, etc.)
            sector_quarter: Specific sector/quarter

        Returns:
            LightCurveData object or None if not found
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical information from this data source

        Returns:
            Dictionary with statistics (planet counts, etc.)
        """
        pass

    async def validate_target(self, target_name: str) -> bool:
        """
        Validate if target exists in this data source

        Args:
            target_name: Target to validate

        Returns:
            bool: True if target exists
        """
        try:
            result = await self.fetch_planet_info(target_name)
            return result is not None
        except Exception:
            return False

    def get_supported_missions(self) -> List[str]:
        """Get list of supported missions for this data source"""
        return []

    def get_capabilities(self) -> Dict[str, bool]:
        """Get capabilities of this data source"""
        return {
            "planet_search": True,
            "planet_info": True,
            "light_curves": False,
            "statistics": True,
            "real_time": False,
        }

    def __str__(self) -> str:
        return f"{self.name} ({self.source_type.value})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
