"""
Standardized response schemas for ExoplanetAI API
Стандартизированные схемы ответов для API ExoplanetAI
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ResponseStatus(str, Enum):
    """Response status enumeration"""

    SUCCESS = "ok"
    ERROR = "error"
    PARTIAL = "partial"


class ErrorCode(str, Enum):
    """Error code enumeration"""

    VALIDATION_ERROR = "validation_error"
    DATA_NOT_FOUND = "data_not_found"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"
    SECURITY_ERROR = "security_error"


class ErrorDetail(BaseModel):
    """Error detail schema"""

    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseResponse(BaseModel):
    """Base response schema"""

    status: ResponseStatus
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    error: Optional[ErrorDetail] = None


class DataResponse(BaseResponse):
    """Response with data payload"""

    data: Optional[Any] = None


class TargetInfo(BaseModel):
    """Target information schema"""

    name: str
    ra: Optional[float] = None
    dec: Optional[float] = None
    magnitude: Optional[float] = None
    catalog_id: Optional[str] = None
    mission: Optional[str] = None
    data_quality: Optional[str] = None
    observation_days: Optional[int] = None
    data_points: Optional[int] = None


class LightcurveInfo(BaseModel):
    """Light curve information schema"""

    time_points: int
    time_span_days: float
    cadence_minutes: Optional[float] = None
    noise_level_ppm: Optional[float] = None
    data_source: str
    sectors_quarters: Optional[List[int]] = None


class BLSResult(BaseModel):
    """BLS analysis result schema"""

    best_period: float = Field(..., description="Best period in days")
    period_uncertainty: Optional[float] = None
    transit_depth: float = Field(..., description="Transit depth")
    transit_duration: float = Field(..., description="Transit duration in hours")
    snr: float = Field(..., description="Signal-to-noise ratio")
    significance: float = Field(..., description="Statistical significance")
    epoch: Optional[float] = None
    method: str = "BLS"


class GPIResult(BaseModel):
    """GPI analysis result schema"""

    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    orbital_period: Optional[float] = None
    phase_sensitivity: float
    snr: float
    gravitational_signature: Optional[Dict[str, Any]] = None
    method: str = "GPI"


class PlanetaryCharacterization(BaseModel):
    """Planetary characterization schema"""

    radius_earth_radii: Optional[float] = None
    mass_earth_masses: Optional[float] = None
    equilibrium_temperature_k: Optional[float] = None
    orbital_period_days: Optional[float] = None
    semi_major_axis_au: Optional[float] = None
    eccentricity: Optional[float] = None
    habitable_zone: Optional[bool] = None


class SearchResultData(BaseModel):
    """Search result data schema"""

    target_info: TargetInfo
    lightcurve_info: LightcurveInfo
    exoplanet_detected: bool
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    candidates_found: int = Field(..., ge=0)
    bls_result: Optional[BLSResult] = None
    gpi_result: Optional[GPIResult] = None
    planetary_characterization: Optional[PlanetaryCharacterization] = None
    recommendations: Optional[List[str]] = None


class SearchResponse(DataResponse):
    """Search endpoint response"""

    data: Optional[SearchResultData] = None


class ValidationResultData(BaseModel):
    """Target validation result schema"""

    target_name: str
    is_valid: bool
    data_sources_available: List[str] = []
    archive_info: Optional[Dict[str, Any]] = None
    observation_count: Optional[int] = None


class ValidationResponse(DataResponse):
    """Validation endpoint response"""

    data: Optional[ValidationResultData] = None


class DataSourceInfo(BaseModel):
    """Data source information schema"""

    name: str
    type: str
    description: str
    url: str
    status: str = "active"
    last_updated: Optional[datetime] = None


class DataSourcesData(BaseModel):
    """Data sources response data"""

    data_sources: List[DataSourceInfo]
    synthetic_data: bool = False
    real_data_only: bool = True
    total_sources: int


class DataSourcesResponse(DataResponse):
    """Data sources endpoint response"""

    data: Optional[DataSourcesData] = None


class HealthData(BaseModel):
    """Health check data schema"""

    version: str
    uptime_seconds: float
    services: Dict[str, str]
    database_status: str
    external_apis: Dict[str, str]


class HealthResponse(DataResponse):
    """Health endpoint response"""

    data: Optional[HealthData] = None


class MetricsData(BaseModel):
    """System metrics data schema"""

    requests_per_minute: float
    average_response_time_ms: float
    error_rate_percent: float
    cache_hit_rate_percent: Optional[float] = None
    active_connections: int
    memory_usage_mb: float
    cpu_usage_percent: float


class MetricsResponse(DataResponse):
    """Metrics endpoint response"""

    data: Optional[MetricsData] = None


# Response factory functions
def create_success_response(
    data: Any = None,
    message: str = "Success",
    processing_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Create standardized success response"""
    response = {
        "status": ResponseStatus.SUCCESS,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "error": None,
    }

    if processing_time_ms is not None:
        response["processing_time_ms"] = processing_time_ms

    return response


def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        "status": ResponseStatus.ERROR,
        "message": None,
        "timestamp": datetime.now().isoformat(),
        "data": None,
        "error": {
            "code": error_code,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        },
    }

    if processing_time_ms is not None:
        response["processing_time_ms"] = processing_time_ms

    return response


def create_partial_response(
    data: Any,
    message: str,
    warnings: List[str] = None,
    processing_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Create partial success response with warnings"""
    response = {
        "status": ResponseStatus.PARTIAL,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "warnings": warnings or [],
        "error": None,
    }

    if processing_time_ms is not None:
        response["processing_time_ms"] = processing_time_ms

    return response
